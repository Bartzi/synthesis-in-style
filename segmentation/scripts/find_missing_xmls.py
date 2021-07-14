import argparse
import shutil
import sys
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate our PAGE XML predictions with the DIVA tool")
    parser.add_argument("gt_xml_dir", help="Path to dir with all GT xmls to evaluate")
    parser.add_argument("gt_diva_image_dir", help="Path to dir with all images in required DIVA GT image format")
    parser.add_argument("pred_xml_dir", help="Path to dir with all predicted PAGE XML Files")
    parser.add_argument("--flat", action='store_true', default=False, help="flat xml dir")

    args = parser.parse_args()

    gt_xml_files = list(sorted(Path(args.gt_xml_dir).glob("**/*.xml")))
    predicted_xml_files = list(sorted(Path(args.pred_xml_dir).glob("**/*.xml")))
    gt_images = list(sorted(Path(args.gt_diva_image_dir).glob("**/*.png")))

    if len(gt_xml_files) == len(predicted_xml_files) == len(gt_images):
        if args.flat:
            xml_dir = Path(args.gt_xml_dir).parent / 'flat_xmls'
            xml_dir.mkdir(exist_ok=True)
            for path in tqdm(gt_xml_files, desc='flattening'):
                destination = xml_dir / path.name
                shutil.copy(path, destination)
        sys.exit(0)

    gt_image_map = {}

    for image_path in predicted_xml_files:
        same_xml_paths = [path for path in gt_xml_files if path.name == image_path.name]
        same_image_paths = [path for path in gt_images if path.stem == image_path.stem]
        if len(same_xml_paths) != 1:
            gt_image_map[image_path.name] = [same_xml_paths, same_image_paths]

    pprint(gt_image_map)
