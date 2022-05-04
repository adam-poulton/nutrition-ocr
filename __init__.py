from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def main():
    print(pytesseract.image_to_string('images\label_2.jpg'))



if __name__ == '__main__':
    main()
