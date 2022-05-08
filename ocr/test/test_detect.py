import unittest
import os
from detect import DetectionPipeline


class TestDetectionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = DetectionPipeline()
        self.image_1_output = {'sodium': {'value': 848.0, 'unit': 'mg'}, 'carbohydrate': {'value': 62.6, 'unit': 'g'},
                               'protein': {'value': 7.9, 'unit': 'g'}, 'energy': {'value': 1970.0, 'unit': 'kJ'},
                               'fat-total': {'value': 20.0, 'unit': 'g'}, 'sugars': {'value': 1.3, 'unit': 'g'},
                               'fat-saturated': {'value': 3.9, 'unit': 'g'}}
        self.image_10_output = {"gluten": {"value": 0.0, "unit": "g"}, "fat-total": {"value": 10.0, "unit": "g"},
                                "carbohydrate": {"value": 8.5, "unit": "g"}, "sodium": {"value": 70.0, "unit": "mg"},
                                "protein": {"value": 5.5, "unit": "g"}, "energy": {"value": 615.0, "unit": "kJ"},
                                "sugars": {"value": 8.5, "unit": "g"}, "fat-saturated": {"value": 6.5, "unit": "g"}}

    def test_label_1(self):
        image = os.path.join('../images', 'label-1.jpg')
        result = self.pipeline.run_by_path(image)
        self.assertEqual(self.image_1_output, result)

    def test_label_10(self):
        image = os.path.join('../images', 'label-10.jpg')
        result = self.pipeline.run_by_path(image)
        self.assertEqual(self.image_10_output, result)


if __name__ == '__main__':
    unittest.main()
