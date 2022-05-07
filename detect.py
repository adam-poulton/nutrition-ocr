import argparse
import time
import cv2 as cv
import os
import numpy as np
import re
import requests
import logging

from lib.fast_rcnn.test import get_blobs
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from lib.text_connector.detectors import TextDetector

import process_image

from nutrition_map import NutritionLabelMap
from table_detector import NutritionTableDetector
from text_detector import NutritionTextDetector


class Table:
    """
    represents a collection of data cell rows
    """

    def __init__(self):
        self.data = {}


class Row:
    """
    Represents a row of data cells
    """

    def __init__(self, label):
        self.label = label
        self.data = []

    def add_cell(self, cell):
        self.data.append(cell)

    def as_json(self):
        return {self.label: {'value:'}}


class Cell:
    def __init__(self, bbox, text):
        if len(bbox) != 4:
            raise ValueError("class 'Cell' expected 4 elements in parameter: 'bbox'")
        self.bbox = bbox
        self.text = _clean_string(text.strip())
        self.label = False
        self.processed = False

    def __repr__(self):
        return f'cell: {self.bbox}\t "{self.text}"'

    @property
    def x1(self):
        return self.bbox[0]

    @property
    def x2(self):
        return self.bbox[2]

    @property
    def y1(self):
        return self.bbox[1]

    @property
    def y2(self):
        return self.bbox[3]

    @property
    def center(self):
        x = (self.bbox[0] + self.bbox[2]) // 2
        y = (self.bbox[1] + self.bbox[3]) // 2
        return x, y

    @property
    def text_type(self):
        """
        :return: An integer corresponding to string type:
            0 - name and value
            1 - value only
            2 - name only
        """
        if any(char.isdigit() for char in self.text):
            # the text contains a numerical digit
            # text must contain a value
            if re.search(r'\D{3,}\s', self.text):
                # text contains at least 3 consecutive non-digits as well as digits
                # assume text is a label and value combined i.e 'energy 250kJ'
                return 0
            else:
                # text does not contain 3 consecutive non-digits
                # assume text is a value with units i.e '120mg'
                return 1
        else:
            # text contains no digits
            # assume text is a pure label i.e 'energy'
            return 2

    def is_same_row(self, other):
        if type(other) is type(self):
            _, y = self.center
            return other.y1 < y < other.y2
        raise ValueError(f'Excepted type: {type(self)}. Given type: {type(other)}')

    def is_same_col(self, other):
        if type(other) is type(self):
            x, _ = self.center
            return other.x1 < x < other.x2
        raise ValueError(f'Excepted type: {type(self)}. Given type: {type(other)}')


class DetectionPipeline:
    def __init__(self):
        # initialise the models
        self.table_model = NutritionTableDetector()
        self.text_model = NutritionTextDetector()
        # specify the nutritional keyword file
        basedir = os.getcwd()
        nutrient_path = os.path.join(basedir, 'data', 'nutrition-labels.txt')
        self.nutrient_path = nutrient_path

    def detect(self, img_path, debug=False):
        """
        extracts the nutritional table from a given image by path
        :param img_path: path to an image
        :param debug: debug flag
        :return: dictionary of extracted nutritional information i.e {'energy': {'value': 1670, 'unit': 'kJ'}...}
        """
        image = cv.imread(img_path)
        if debug:
            cv.imwrite(os.path.join('data', 'result', '0-input.jpg'), image)
        image = process_image.threshold(image=image)
        # Get the bounding boxes from the nutritional table model
        dims = self._detect_table_box(image)

        # crop the image to the given bounding box
        # cropped_image should now only contain nutritional table
        cropped_image = process_image.crop(image=image, dims=dims,
                                           save=True, save_loc=os.path.join('data', 'result', '1-preprocessed.jpg'))

        # detect the text in the cropped image
        text_blob_list = self._detect_text(cropped_image)

        cells = []

        for blob_cord in text_blob_list:
            word_image = process_image.crop(image=cropped_image, dims=blob_cord, ext_ratio=0.005)
            if word_image.shape[1] > 0 and word_image.shape[0] > 0:
                text = process_image.ocr(word_image, 1, 7)
                if text:
                    cells.append(Cell(bbox=blob_cord, text=text))

        if debug:
            # print every identified cell on the table
            print(*sorted(cells, key=lambda x: (x.y1, x.x1)), sep="\n")
            for cell in cells:
                # draw a green box around the cells
                cropped_image = cv.rectangle(cropped_image, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 1)
            cv.imwrite(os.path.join('data', 'result', '2-cells.jpg'), cropped_image)

        label_mapper = NutritionLabelMap()  # maps possible aliases of labels into their standard key
        output = {}  # store nutritional labels and values
        mx, my = cropped_image.shape[1], cropped_image.shape[0]  # image max cords

        for cell in cells:
            if cell.text in label_mapper:
                # cell is a label
                cx, cy = cell.center
                label = label_mapper[cell.text]  # get the JSON key for this label
                row = []
                for target in cells:
                    t_min = target.y1
                    t_max = target.y2
                    if _in_bounds(cy, t_min, t_max) and target.text_type != 2 and target.x1 > cell.x1:
                        # label is on same row as the target value
                        row.append(target)
                if row:
                    # get the cell on the far right (quantity per 100g)
                    v = sorted(row, key=lambda c: c.x2, reverse=True)[0]
                    value, unit = _extract_value_unit(v.text)
                    if not unit:
                        unit = label_mapper.default_unit(cell.text)
                    output.update({label: {'value': value, 'unit': unit}})

                    if debug:
                        cropped_image = cv.line(cropped_image, (cx, cy), (mx, cy), (255, 0, 0), 1)
                        cropped_image = cv.rectangle(cropped_image,
                                                     (max(v.x1 - 2, 0),
                                                      max(v.y1 - 2, 0)),
                                                     (min(v.x2 + 2, mx),
                                                      min(v.y2 + 2, my)),
                                                     (255, 0, 0), 2)

                if debug:
                    cropped_image = cv.rectangle(cropped_image,
                                                 (max(cell.x1 - 2, 0),
                                                  max(cell.y1 - 2, 0)),
                                                 (min(cell.x2 + 2, mx),
                                                  min(cell.y2 + 2, my)),
                                                 (255, 0, 0), 2)
                    cv.imwrite(os.path.join('data', 'result', '3-rows.jpg'), cropped_image)

        return output

    def _detect_table_box(self, image):
        # Get the bounding boxes from the nutritional table model
        boxes, scores, classes, num = self.table_model.get_classification(image)
        # Get the dimensions of the image
        width = image.shape[1]
        height = image.shape[0]

        # Select the bounding box with most confident output
        ymin = boxes[0][0][0] * height
        xmin = boxes[0][0][1] * width
        ymax = boxes[0][0][2] * height
        xmax = boxes[0][0][3] * width
        # Package bounding box cords into tuple
        return xmin, ymin, xmax, ymax

    def _detect_text(self, image):
        image, scale = process_image.resize(image=image)
        blobs, scales = get_blobs(image)
        if cfg.TEST.HAS_RPN:
            blob = blobs['data']
            blobs['im_info'] = np.array(
                [[blob.shape[1], blob.shape[2], scales[0]]],
                dtype=np.float32
            )
        cls_prob, box_pred = self.text_model.get_text_classification(blobs=blobs)
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / scales[0]
        text_detector = TextDetector()
        boxes = text_detector.detect(boxes, scores[:, np.newaxis], image.shape[:2])

        blob_list = []
        for box in boxes:
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            blob_list.append((min_x, min_y, max_x, max_y))

        return tuple(blob_list)


def _string_type(string):
    """
    :param string: Type of the string to be checked
    :return: An integer corresponding to type:
        0 - name and value
        1 - value only
        2 - name only
    """
    if any(char.isdigit() for char in string):
        if re.search(r'\D{3,}\s', string):
            # Value and label
            return 0
        else:
            # Value only
            return 1
    # Label only
    return 2


def _in_bounds(target, t_min, t_max):
    return t_min < target < t_max


def _extract_value_unit(string, debug=False):
    """
    returns value-unit pair in given string
      i.e '24.3g' -> (24.3, 'g')
    if there are multiple returns right-most
      i.e '24.3mg 1,650kJ' -> (1650.0, 'kJ')
    """
    # matches value strings i.e '24.5g' or '1,670kJ'
    pat = r'((?:\d+,{0,1}){0,}\d+\.{0,1}\d*)([0-9a-z%]{0,3})'
    items = re.findall(pat, string)
    if not items:
        return [0, 'g']
    value, unit = items[-1]  # get the right most value-unit pair
    if unit == 'cal' and len(items) > 1:
        value, unit = items[-2]  # handles cases like '1670kJ (500Cal)'
    value = float(value.replace(',', '').replace("'", ''))
    # clean up some common OCR errors
    unit = unit.replace('c', 'g').replace('q', 'g').replace('a', 'g')
    unit = re.sub(r'k$|kj$', 'kJ', unit)
    unit = re.sub(r'j$', 'g', unit)
    if debug:
        print(f'  \'{string}\' -> {value} {unit}')
    return value, unit


def _clean_string(string: str):
    text = string.lower().strip()
    text = re.sub(r'\([^()]*\)', '', text)  # remove anything inside brackets
    text = text.replace("nil detected", "0")
    bad_chars = r'[^a-zA-Z0-9 .,%\']'
    text = re.sub(bad_chars, "", text)
    text = re.sub("Omg", "0mg", text)
    text = re.sub("Og", "0g", text)
    text = re.sub(r'(?<=\d) (?=\w)', '', text)
    text = _fix_suffix(text)
    return text.strip()


def _fix_suffix(string: str):
    """
    fixes:
     weight unit 'g' being misread as a '9'
     weight unit 'mg' being misread as 'mq'
    :param string: string to be checked and fixed
    :return: the fixed string
    """
    line = re.search(r"\d\s|\d$", string)
    if line and line.group().strip() == "9":
        index = line.span()[0]
        string = string[:index] + "g" + string[index + 1:]

    line = re.search(r"\dmq\s|\dmq$", string)
    if line:
        index = line.span()[0] + 2
        string = string[:index] + "g" + string[index + 1:]

    return string


def url_to_image(url: str):
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv.imdecode(image, -1)


if __name__ == '__main__':
    import json
    pipeline = DetectionPipeline()
    result = pipeline.detect(os.path.join('images', f'label-{7}.jpg'), debug=True)
    print(json.dumps(result, indent=2))

