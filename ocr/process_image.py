import os
import cv2 as cv
import pytesseract
from PIL import Image


def threshold(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert to greyscale
    image = cv.GaussianBlur(image, (5, 5), 0)  # apply gaussian blur
    # apply threshold using binary inverted and otsu
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    image = cv.bitwise_not(image)  # re-invert after threshold
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # convert back to BGR
    return image


def ocr(image, oem=1, psm=3):
    # required for windows
    pytesseract.pytesseract.tesseract_cmd = os.path.join('C:\\', 'Program Files', 'Tesseract-OCR', 'tesseract.exe')
    config = '--oem {oem} --psm {psm}'.format(oem=oem, psm=psm)
    image = Image.fromarray(image)
    text = pytesseract.image_to_string(image, lang='eng', config=config)
    return text


def crop(image, dims, ext_ratio=0, save=False, save_loc=None):
    """
    Crop an image based on given coords and extention_ratio
    :param image: Image object to crop
    :param dims: Tuple of (x1, y1, x2, y2) to crop image t
    :param ext_ratio: The ratio to extend the 'crop' by to account for text cut-off
    :param save: Flag to save the cropped file
    :param save_loc: location to save the cropped file
    :return: the cropped image
    """
    nx = image.shape[1]
    ny = image.shape[0]
    # compute crop dimensions
    # - ensure crop dimensions inside original image bounds
    # - scale crop dimensions by ext_ratio to account for bounding box error
    # - extend boxes down and to the right to account for hanging letters such as 'g'
    c_dims = (max(0, int(dims[0] - ext_ratio * nx)),
              max(0, int(dims[1] - ext_ratio * ny)),
              min(nx, int(dims[2] + ext_ratio * nx) + 2),
              min(ny, int(dims[3] + ext_ratio * ny) + 2))
    cropped_image = image[c_dims[1]:c_dims[3], c_dims[0]:c_dims[2]]
    if save:
        cv.imwrite(save_loc, cropped_image)
    return cropped_image


def resize(image, scale=600, max_scale=1200):
    f = float(scale) / min(image.shape[0], image.shape[1])
    if max_scale is not None and f * max(image.shape[0], image.shape[1]) > max_scale:
        f = float(max_scale) / max(image.shape[0], image.shape[1])
    return cv.resize(image, None, None, fx=f, fy=f, interpolation=cv.INTER_LINEAR), f


def skew_correction(image, save=False, save_loc=None):
    # todo: implement skew correction preprocessing step
    pass
