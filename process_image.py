import os
import csv
import cv2
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np

from lib.text_connector.text_connect_cfg import Config as TextlineConfig



def pre_process(image, enhance=1):
    if enhance > 1:
        image = Image.fromarray(image)
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(enhance)
        image = np.asarray(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def ocr(image, oem=1, psm=3):
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    
    config = ('-l eng --oem {oem} --psm {psm}'.format(oem=oem, psm=psm))
    
    image = Image.fromarray(image)
    text = pytesseract.image_to_string(image, config)
    
    return text


def crop(image, dims, ext_ratio=0, save=False, save_loc=None):
    """
    Crop an image based on given coords and extention_raiot
    :param image: Image object to crop
    :param dims: Tuple of (x1, y1, x2, y2) to crop image to
    :param ext_ratio: The ratio to extend the 'crop' by to account for text cut-off
    :param save: Flag to save the cropped file
    :param save_loc: location to save the cropped file
    :return: the cropped image
    """
    nx = image.shape[1]
    ny = image.shape[0]
    
    c_dims = (
        int(dims[0] - ext_ratio * nx),
        int(dims[1] - ext_ratio * ny),
        int(dims[2] + ext_ratio * nx),
        int(dims[3] + ext_ratio * ny)
    )
    
    cropped_image = image[c_dims[1]:c_dims[3], c_dims[0]:c_dims[2]]
    
    if save:
        cv2.imwrite(save_loc, cropped_image)
        
    return cropped_image


def resize(image, scale=TextlineConfig.SCALE, max_scale=TextlineConfig.MAX_SCALE):
    f = float(scale) / min(image.shape[0], image.shape[1])
    if max_scale is not None and f * max(image.shape[0], image.shape[1]) > max_scale:
        f = float(max_scale) / max(image.shape[0], image.shape[1])
    return cv2.resize(image, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


