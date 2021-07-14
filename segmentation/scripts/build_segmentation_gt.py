import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(Path(__file__).parent.parent.parent)

from segmentation.evaluation.coco_gt import COCOGtCreator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("class_to_color_map")
    parser.add_argument("destination")

    args = parser.parse_args()
    image_files = list(Path(args.image_dir).glob('**/*.png'))

    with open(args.class_to_color_map) as f:
        class_to_color_map = json.load(f)
    gt_creator = COCOGtCreator(class_to_color_map)

    dataset = []
    for image_file in tqdm(image_files):
        with Image.open(str(image_file)) as image:
            data = {'file_name': str(image_file.relative_to(Path(args.destination).parent))}
            data.update(gt_creator.determine_classes_in_image(image))
            dataset.append(data)

    with open(args.destination, 'w') as f:
        json.dump(dataset, f)
