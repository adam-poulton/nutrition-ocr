import cv2 as cv
import os
import numpy as np
import re
import requests
import timeit

from ocr import process_image
from ocr.nutrition_map import NutritionLabelMap
from ocr.table_detector import NutritionTableDetector
from ocr.text_detector import NutritionTextDetector
from ocr.lib.fast_rcnn.test import get_blobs
from ocr.lib.rpn_msr.proposal_layer_tf import proposal_layer
from ocr.lib.text_connector.detectors import TextDetector


class Cell:
    def __init__(self, bbox, text):
        if len(bbox) != 4:
            raise ValueError("class 'Cell' expected 4 elements in parameter: 'bbox'")
        self.bbox = bbox
        self.text = _clean_string(text.strip())

    def __repr__(self):
        return f'{str(self.bbox):24}\t{self.text}'

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
    def contains_digit(self):
        return any(char.isdigit() for char in self.text)


class NutritionDetectionPipeline:
    def __init__(self):
        # initialise the models
        self.table_model = NutritionTableDetector()
        self.text_model = NutritionTextDetector()
        # specify directory for output when debug flag
        basedir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(basedir, 'data', 'result')

    def from_path(self, img_path, debug=False):
        """
        reads the nutritional table from a given image path
        :param img_path:
        :param debug:
        :return:
        """
        image = cv.imread(img_path)
        return self.from_cv_img(image, debug)

    def from_url(self, img_url, debug=False):
        image = url_to_image(img_url)
        return self.from_cv_img(image, debug)

    def from_cv_img(self, cv_image, debug=False):
        """
        extracts the nutritional table from a given cv_image
        :param cv_image: cv2 image
        :param debug: if True will save
        :return: dictionary of extracted nutritional information i.e {'energy': {'value': 1670, 'unit': 'kJ'}...}
        """
        checkpoints = []  # time checkpoints for performance analysis
        start = timeit.default_timer()
        if debug:
            cv.imwrite(os.path.join(self.output_dir, '0-input.jpg'), cv_image)
        image = process_image.threshold(image=cv_image)
        checkpoints.append(['process: threshold', timeit.default_timer() - start])
        # Get the bounding boxes from the nutritional table model
        dims = self._detect_table_box(image)
        checkpoints.append(['detect table area', timeit.default_timer() - start])
        # crop the image to the given bounding box
        # cropped_image should now only contain nutritional table
        cropped_image = process_image.crop(image=image, dims=dims,
                                           save=True, save_loc=os.path.join(self.output_dir, '1-preprocessed.jpg'))
        checkpoints.append(['crop table', timeit.default_timer() - start])
        # detect the text in the cropped image
        text_blob_list = self._detect_text(cropped_image)
        checkpoints.append(['detect text boxes', timeit.default_timer() - start])
        cells = []

        for blob_cord in text_blob_list:
            word_image = process_image.crop(image=cropped_image, dims=blob_cord, ext_ratio=0.005)
            if word_image.shape[1] > 0 and word_image.shape[0] > 0:
                # extract text from each image segment
                text = process_image.ocr(word_image, 1, 7)
                if text:
                    cells.append(Cell(bbox=blob_cord, text=text))

        checkpoints.append(['crop and OCR cells', timeit.default_timer() - start])

        if debug:
            with open(os.path.join(self.output_dir, '3-cell-output.txt'), 'w') as f:
                f.write('\n'.join(map(str, sorted(cells, key=lambda x: (x.y1, x.x1)))))
            for cell in cells:
                # draw a green box around the cells
                cropped_image = cv.rectangle(cropped_image, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 1)
            cv.imwrite(os.path.join(self.output_dir, '2-cells.jpg'), cropped_image)

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
                    if _in_bounds(cy, t_min, t_max) and target.contains_digit and target.x1 > cell.x1:
                        # label is on same row as the target value
                        row.append(target)
                if row:
                    # get the cell on the far right (quantity per 100g)
                    v = sorted(row, key=lambda c: c.x2, reverse=True)[0]
                    value, unit = extract_value_unit(v.text)
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
                                                     (0, 0, 255), 2)

                if debug:
                    cropped_image = cv.rectangle(cropped_image,
                                                 (max(cell.x1 - 2, 0),
                                                  max(cell.y1 - 2, 0)),
                                                 (min(cell.x2 + 2, mx),
                                                  min(cell.y2 + 2, my)),
                                                 (255, 0, 0), 2)
                    cv.imwrite(os.path.join(self.output_dir, '4-rows.jpg'), cropped_image)
        checkpoints.append(['extract labels, values, units', timeit.default_timer() - start])
        if debug:
            with open(os.path.join(self.output_dir, '5-output.txt'), 'w') as f:
                f.write(''.join([f'{k:18} {v["value"]:8} {v["unit"]:8}\n' for k, v in output.items()]))
            with open(os.path.join(self.output_dir, '6-times.txt'), 'w') as f:
                f.write(''.join([f'{section:50}{time:8}\n' for section, time in checkpoints]))
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

        blob = blobs['data']
        blobs['im_info'] = np.array([[blob.shape[1], blob.shape[2], scales[0]]], dtype=np.float32)
        cls_prob, box_pred = self.text_model.get_text_classification(blobs=blobs)
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'])

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


def extract_value_unit(string, debug=False):
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
    text = text.replace("not detected", "0")
    text = text.replace("nil detected", "0")
    text = text.replace("nil", "0")
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




