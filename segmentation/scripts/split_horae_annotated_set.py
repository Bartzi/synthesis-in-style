import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

from pathlib import Path

from tqdm import tqdm

from page_to_label_image import get_text_lines


def load_split_urls(split_file: Path, split_name: str) -> dict:
    urls = {}
    with split_file.open() as f:
        reader = csv.DictReader(f)
        for line in reader:
            urls[line['Link']] = split_name
    return urls


def build_crop_dir_map(created_gt: Path) -> dict:
    with created_gt.open() as f:
        created_gt = json.load(f)

    crop_dir_map = defaultdict(list)
    for gt_item in created_gt:
        name = Path(gt_item['file_name'])
        crop_dir_map[name.parent.name].append(gt_item)
    return dict(crop_dir_map)


def save_split(dir: Path, split: dict, prefix: str):
    for split_name, data in split.items():
        file_name = dir / f"{prefix}_{split_name}.json"
        with file_name.open('w') as f:
            json.dump(data, f)


def main(args):
    page_root = Path(args.page_root)

    split_files = list(Path(args.split_dir).glob('**/*.csv'))
    urls = {}
    for split_file in split_files:
        urls.update(load_split_urls(split_file, split_file.stem))

    cropped_files = build_crop_dir_map(Path(args.download_json))
    save_root = Path(args.download_json).parent

    crop_splits = defaultdict(list)
    splits = defaultdict(list)
    for page_file in tqdm(list(page_root.glob('**/*.xml'))):
        page_tree = ET.parse(str(page_file))
        pages = get_text_lines(page_tree.getroot())

        for page in pages:
            url = page.file_name
            split = urls[url]

            splits[split].append({"file_name": str(Path('full_images') / f"{page_file.stem}.png"), "width": page.width, "height": page.height})
            crops_for_file = cropped_files[page_file.stem]
            crop_splits[split].extend(crops_for_file)

    save_split(save_root, splits, '')
    save_split(save_root, crop_splits, 'crops')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="json from page_to_label_image and a HORAE split csv and find images for corresponding set")
    parser.add_argument("download_json", help="path to json created after running `page_to_label_image.py`")
    parser.add_argument("page_root", help="Path to PAGE XML files")
    parser.add_argument("split_dir", help="path to dir with data about the splits")

    main(parser.parse_args())
