import argparse
import time
import cv2 as cv
import os
import numpy as np
import re
import requests

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
        return {self.label: {'value:' }}


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
        x = (self.bbox[0] + self.bbox[2]) / 2
        y = (self.bbox[1] + self.bbox[3]) / 2
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


# Separate the unit from its value. (eg. '24g' to '24' and 'g')
def separate_unit(string):
    string = string.replace(" ", "")
    r1 = re.compile("(\d+[\.\,\']?\d*)([a-zA-Z]+)")
    m1 = r1.match(string)
    r2 = re.compile("(\d+[\.\,\']?\d*)")
    m2 = r2.match(string)
    if m1:
        return float(m1.group(1).replace(',', '').replace("'", '')), m1.group(2).replace('q', 'g')
    elif m2:
        return float(m2.group(1).replace(',', '').replace("'", ''))
    else:
        return ""


class DetectionPipeline:
    def __init__(self):
        # initialise the models
        self.table_model = NutritionTableDetector()
        self.text_model = NutritionTextDetector()
        # specify the nutritional keyword file
        basedir = os.getcwd()
        nutrient_path = os.path.join(basedir, 'data', 'nutrition-labels.txt')
        self.nutrient_path = nutrient_path

    def detect(self, img_path):
        """
        :param img_path: path to an image
        :param debug: debug flag
        :return: dictionary of nutritional OCR captures
        """
        
        image = cv.imread(img_path)
        image = process_image.pre_process(image=image)
        # Get the bounding boxes from the nutritional table model
        dims = self._detect_table_box(image)

        # crop the image to the given bounding box
        # cropped_image should now only contain nutritional table
        cropped_image = process_image.crop(image=image, dims=dims, save=True, save_loc="cropped.jpg")

        # detect the text in the cropped image
        text_blob_list = self._detect_text(cropped_image)

        cells = []

        for blob_cord in text_blob_list:
            word_image = process_image.crop(image=cropped_image, dims=blob_cord, ext_ratio=0.005)
            if word_image.shape[1] > 0 and word_image.shape[0] > 0:
                text = process_image.ocr(word_image, 1, 7)
                if text:
                    cells.append(Cell(bbox=blob_cord, text=text))

        print(*sorted(cells, key=lambda x: (x.y1, x.x1)), sep="\n")

        for cell in cells:
            cropped_image = cv.rectangle(cropped_image, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 1)
        cv.imwrite(os.path.join('images', 'table-boxes.jpg'), cropped_image)
        # Map all boxes according to their location and append the value string to the label

        label_mapper = NutritionLabelMap()  # maps possible aliases of labels into their standard key
        output = {}  # store nutritional labels and values

        for cell in cells:
            if cell.text in label_mapper:
                # cell is a label
                cx, cy = cell.center
                # get the JSON key for this label
                label = label_mapper[cell.text]
                row = []
                for target in cells:
                    t_min = target.y1
                    t_max = target.y2
                    if _in_bounds(cy, t_min, t_max) and target.text_type != 2 and target.x1 > cell.x1:
                        # label is on same row as the target value
                        row.append(target)
                if row:
                    # get the cell on the far left (quantity per 100g)
                    value_cell = sorted(row, key=lambda c: c.x2, reverse=True)[0]
                    value, unit = separate_unit(value_cell.text)
                    output.update({label: {'value': value, 'unit': unit}})

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


def _clean_string(string: str):
    bad_chars = r'[^a-zA-Z0-9 .,%\']'
    text = re.sub(bad_chars, "", string)
    text = re.sub("Omg", "0mg", text)
    text = re.sub("Og", "0g", text)
    text = re.sub('(?<=\d) (?=\w)', '', text)
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
    line = re.search("\d\s|\d$", string)
    if line and line.group().strip() == "9":
        index = line.span()[0]
        string = string[:index] + "g" + string[index + 1:]

    line = re.search("\dmq\s|\dmq$", string)
    if line:
        index = line.span()[0] + 2
        string = string[:index] + "g" + string[index + 1:]

    return string


def url_to_image(url: str):
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv.imdecode(image, -1)


if __name__ == '__main__':
    pipeline = DetectionPipeline()
    pipeline.detect(os.path.join(os.getcwd(), 'images', f'label_{1}.jpg'))
    # for i in range(1, 9):
    #     img = os.path.join(os.getcwd(), 'images', f'label_{i}.jpg')
    #     image = cv.imread(img)
    #     dims = pipeline._detect_table_box(image)
    #     cropped = process_image.crop(image=image, dims=dims, save=True, save_loc=os.path.join('data', 'result', f'cropped_{i}.jpg'))
