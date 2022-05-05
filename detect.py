import argparse
import time
import cv2
import os
import numpy as np
import re

from lib.fast_rcnn.test import get_blobs
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from lib.text_connector.detectors import TextDetector

import process_image

from fuzzydict import FuzzyDict
from table_detector import NutritionTableDetector
from text_detector import NutritionTextDetector

class Cell:
    def __init__(self, bbox, text, is_label=False, parent=None, children=None):
        assert len(bbox) == 4
        self.bbox = bbox
        self.text = text
        self.is_label = is_label
        self.parent = parent
        if children:
            self.children = children
        else:
            self.children = []

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.children):
            current = self.children[self.i]
            self.i += 1
            return current
        else:
            raise StopIteration

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
    def cell_type(self):
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
        # text contains no digits
        # assume text is a pure label i.e 'energy'
        return 2

    def aligned_row(self, other):
        if type(other) is type(self):
            _, y = self.center
            return other.y1 < y < other.y2
        raise ValueError(f'Excepted type: {type(self)}. Given type: {type(other)}')

    def aligned_col(self, other):
        if type(other) is type(self):
            x, _ = self.center
            return other.x1 < x <other.x2
        raise ValueError(f'Excepted type: {type(self)}. Given type: {type(other)}')


# Separate the unit from its value. (eg. '24g' to '24' and 'g')
def separate_unit(string):
    string = string.replace(" ", "")
    r1 = re.compile("(\d+[\.\,\']?\d*)([a-zA-Z]+)")
    m1 = r1.match(string)
    r2 = re.compile("(\d+[\.\,\']?\d*)")
    m2 = r2.match(string)
    if m1:
        return float(m1.group(1).replace(',', '.').replace("'", '.')), m1.group(2)
    elif m2:
        return float(m2.group(1).replace(',', '.').replace("'", '.'))
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

    def detect(self, img_path, debug=False):
        """
        :param img_path: path to an image
        :param debug: debug flag
        :return: dictionary of nutritional OCR captures
        """

        image = cv2.imread(img_path)
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
        dims = (xmin, ymin, xmax, ymax)

        # crop the image to the given bounding box
        # cropped_image should now only contain nutritional table
        cropped_image = process_image.crop(image=image, dims=dims, save=True, save_loc=".data/result/output.jpg")

        # detect the text in the cropped image
        text_blob_list = self._detect_text(cropped_image)
        text_location_list = []  # metadata of detected text boxes
        nutrient_dict = {}  # store nutritional labels and values

        for blob_cord in text_blob_list:
            word_image = process_image.crop(image=cropped_image, dims=blob_cord, ext_ratio=0.005)
            word_image = process_image.pre_process(word_image)
            if word_image.shape[1] > 0 and word_image.shape[0] > 0:
                text = process_image.ocr(word_image, 1, 7)
                if text:
                    center_x = (blob_cord[0] + blob_cord[2]) / 2
                    center_y = (blob_cord[1] + blob_cord[3]) / 2
                    box_center = (center_x, center_y)

                    new_location = {
                        'bbox': blob_cord,
                        'text': text,
                        'box_center': box_center,
                        'string_type': _string_type(text)
                    }
                    text_location_list.append(new_location)

        # Map all boxes according to their location and append the value string to the label
        for location in text_location_list:
            if location['string_type'] == 2:
                # string is a label, find its matching value(s)
                l_center = location['box_center'][1]
                for target in text_location_list:
                    t_min = target['bbox'][1]
                    t_max = target['bbox'][3]
                    if _in_bounds(l_center, t_min, t_max) and target['string_type'] == 1:
                        # label is vertically aligned with the target value, combine strings
                        location['text'] += f' {target["text"]}'
                        location['string_type'] = 0  # flag this box as a combined label-value string

        fuz_dict = FuzzyDict.fromfile(self.nutrient_path)

        # add the labels to the fuzzy dictionary
        for location in text_location_list:
            if location['string_type'] == 0:
                text = _clean_string(location['text'])

                if text in fuz_dict or re.split('[/|I]', text)[0] in fuz_dict:
                    label, value = None, None



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
    pat = "[\|\*\_\'\—\-\{}]".format('"')
    text = re.sub(pat, "", string)
    text = re.sub(" I ", " / ", text)
    text = re.sub("^I ", "", text)
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


