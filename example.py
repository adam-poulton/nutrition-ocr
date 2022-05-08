import json
import os
from ocr.detect import DetectionPipeline


if __name__ == '__main__':

    pipeline = DetectionPipeline()
    result = pipeline.run_by_path(os.path.join('images', f'label-{1}.jpg'), debug=True)
    print(json.dumps(result, indent=2))
