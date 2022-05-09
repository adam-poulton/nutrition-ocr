import os
from ocr.detect import NutritionDetectionPipeline


if __name__ == '__main__':
    detect_nutrition = NutritionDetectionPipeline()
    image_path = os.path.join('ocr/data/images', f'label-{5}.jpg')
    result = detect_nutrition.from_path(image_path, debug=True)
    # print(''.join([f'{k:18} {v["value"]:8} {v["unit"]:8}\n' for k, v in result.items()]))
