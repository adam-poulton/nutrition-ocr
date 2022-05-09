import unittest
import os
from ocr.detect import NutritionDetectionPipeline
from ocr.detect import extract_value_unit


class TestDetectionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = NutritionDetectionPipeline()
        cls.image_1_output = {'sodium': {'value': 848.0, 'unit': 'mg'}, 'carbohydrate': {'value': 62.6, 'unit': 'g'},
                              'protein': {'value': 7.9, 'unit': 'g'}, 'energy': {'value': 1970.0, 'unit': 'kJ'},
                              'fat-total': {'value': 20.0, 'unit': 'g'}, 'sugars': {'value': 1.3, 'unit': 'g'},
                              'fat-saturated': {'value': 3.9, 'unit': 'g'}}
        cls.image_8_output = {"energy": {"value": 1318.0, "unit": "kJ"}, "sodium": {"value": 14.0, "unit": "mg"},
                              "fat-saturated": {"value": 0.0, "unit": "g"}, "sugars": {"value": 82.1, "unit": "g"},
                              "carbohydrate": {"value": 82.1, "unit": "g"}, "fat-total": {"value": 0.0, "unit": "g"},
                              "protein": {"value": 0.3, "unit": "g"}}
        cls.image_10_output = {"gluten": {"value": 0.0, "unit": "g"}, "fat-total": {"value": 10.0, "unit": "g"},
                               "carbohydrate": {"value": 8.5, "unit": "g"}, "sodium": {"value": 70.0, "unit": "mg"},
                               "protein": {"value": 5.5, "unit": "g"}, "energy": {"value": 615.0, "unit": "kJ"},
                               "sugars": {"value": 8.5, "unit": "g"}, "fat-saturated": {"value": 6.5, "unit": "g"}}

    def test_label_1(self):
        image = os.path.join('..', 'data', 'images', 'label-1.jpg')
        result = self.pipeline.from_path(image)
        self.assertEqual(self.image_1_output, result)

    def test_label_8(self):
        image = os.path.join('..', 'data', 'images', 'label-8.jpg')
        result = self.pipeline.from_path(image)
        self.assertEqual(self.image_8_output, result)

    def test_label_10(self):
        image = os.path.join('..', 'data', 'images', 'label-10.jpg')
        result = self.pipeline.from_path(image)
        self.assertEqual(self.image_10_output, result)


class TestExtractValueUnit(unittest.TestCase):

    def test_extract_value_unit_1(self):
        # value only
        text = "25.7g"
        result = extract_value_unit(text)
        self.assertEqual((25.7, 'g'), result)

    def test_extract_value_unit_2(self):
        # value only
        text = "1,225.7mg"
        result = extract_value_unit(text)
        self.assertEqual((1225.7, 'mg'), result)

    def test_extract_value_unit_3(self):
        # combination of label and value
        text = "energy 25.7g"
        result = extract_value_unit(text)
        self.assertEqual((25.7, 'g'), result)

    def test_extract_value_unit_4(self):
        # combination of label and value
        text = "energy 1,560kJ 25.7g"
        result = extract_value_unit(text)
        self.assertEqual((25.7, 'g'), result)


if __name__ == '__main__':
    unittest.main()
