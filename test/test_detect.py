import unittest
from detect import separate_unit


class MyTestCase(unittest.TestCase):
    def test_separate_unit_1(self):
        result = separate_unit("25g")
        self.assertEqual((25.0, 'g'), result)

    def test_separate_unit_2(self):
        result = separate_unit("25.1g")
        self.assertEqual((25.1, 'g'), result)

    def test_separate_unit_3(self):
        result = separate_unit("25,1g")
        self.assertEqual((25.1, 'g'), result)

    def test_separate_unit_4(self):
        result = separate_unit("250,1 g")
        self.assertEqual((250.1, 'g'), result)

    def test_separate_unit_5(self):
        result = separate_unit("5.6% 250,1 g")
        self.assertEqual((250.1, 'g'), result)


if __name__ == '__main__':
    unittest.main()
