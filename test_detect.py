import unittest
import os
from detect import DetectionPipeline


class TestDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = DetectionPipeline()
        self.image_1_output = {'sodium': {'value': 848.0, 'unit': 'mg'}, 'carbohydrate': {'value': 62.6, 'unit': 'g'},
                               'protein': {'value': 7.9, 'unit': 'g'}, 'energy': {'value': 1970.0, 'unit': 'kJ'},
                               'fat-total': {'value': 20.0, 'unit': 'g'}, 'sugars': {'value': 1.3, 'unit': 'g'},
                               'fat-saturated': {'value': 3.9, 'unit': 'g'}}

    def test_label_1(self):
        image = os.path.join('images', 'label-1.jpg')
        result = self.pipeline.detect(image)
        self.assertCountEqual(self.image_1_output, result)


if __name__ == '__main__':
    unittest.main()
