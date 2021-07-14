import argparse
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import cv2
import numpy
import requests
from PIL import ImageColor, Image
from dataclasses import dataclass

from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.contrib import tenumerate


@dataclass
class PAGEInfo:
    text_lines: List[numpy.ndarray]
    file_name: str
    width: int
    height: int


random.seed(666)


def get_text_lines(page_xml: ET.Element) -> List[PAGEInfo]:
    pages = []
    ns = {
        'page': 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
        'page_http': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
    }
    page_children = page_xml.findall('page:Page', ns)
    if len(page_children) == 0:
        page_children = page_xml.findall('page_http:Page', ns)
        ns_element = 'page_http'
    else:
        ns_element = 'page'
    for page in page_children:
        lines = []
        for text_region in page.findall(f'{ns_element}:TextRegion', ns):
            for text_line in text_region.findall(f'{ns_element}:TextLine', ns):
                coords = text_line.find(f'{ns_element}:Coords', ns).attrib['points']
                points = coords.split()
                points = [[int(c) for c in point.split(',')] for point in points]
                coords = numpy.array(points, dtype='int32')
                lines.append(coords)
        pages.append(
            PAGEInfo(
                lines, page.attrib['imageFilename'], int(page.attrib['imageWidth']), int(page.attrib['imageHeight'])
            )
        )
    return pages


def load_color_map(color_map_path: str) -> Dict[str, tuple]:
    with open(color_map_path) as f:
        color_map = json.load(f)
    color_map = {key: ImageColor.getrgb(color) for key, color in color_map.items()}
    return color_map


def crop_patches(image: numpy.ndarray, image_size: int) -> List[numpy.ndarray]:
    windows_in_width = math.ceil(image.shape[1] / image_size)
    total_width_overlap = windows_in_width * image_size - image.shape[1]
    windows_in_height = math.ceil(image.shape[0] / image_size)
    total_height_overlap = windows_in_height * image_size - image.shape[0]

    width_overlap_per_patch = total_width_overlap / windows_in_width
    height_overlap_per_patch = total_height_overlap / windows_in_height

    patches = []
    for y_idx in range(windows_in_height):
        start_y = int(y_idx * (image_size - height_overlap_per_patch))
        for x_idx in range(windows_in_width):
            start_x = int(x_idx * (image_size - width_overlap_per_patch))
            patches.append(numpy.copy(image[start_y:start_y+image_size,start_x:start_x+image_size, :]))

    return patches


def save_crops(rgb_patches: List[numpy.ndarray], label_patches: List[numpy.ndarray], save_root: Path):
    assert len(rgb_patches) == len(label_patches), "Number of label and rgb patches must be the same!"
    for idx, (rgb_patch, label_patch) in tenumerate(zip(rgb_patches, label_patches), leave=False, total=len(rgb_patches)):
        train_sample = numpy.concatenate([rgb_patch, label_patch], axis=1)
        train_sample = Image.fromarray(train_sample)
        file_name = save_root / f"{idx}.png"
        file_name.parent.mkdir(exist_ok=True, parents=True)
        train_sample.save(str(file_name))


def get_crops(rgb_image: numpy.ndarray, label_image: numpy.ndarray, patch_size: int) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
    rgb_patches = crop_patches(rgb_image, patch_size)
    label_patches = crop_patches(label_image, patch_size)
    return rgb_patches, label_patches


def download_image(url: str, destination: Path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Could not Download File from {url}")
    length = response.headers.get('content-length', None)
    if length is not None:
        length = int(length)

    image_buffer = BytesIO()
    for data in tqdm(response.iter_content(chunk_size=4096), total=length, leave=False, desc="Download Image"):
        image_buffer.write(data)

    with Image.open(image_buffer) as the_image:
        the_image.save(str(destination))


def thumbnail_image(image: numpy.ndarray, max_image_size: int) -> numpy.ndarray:
    image = Image.fromarray(image)
    image.thumbnail((max_image_size, max_image_size))
    return numpy.array(image)


def examine_page(page_file: Path, pages: List[PAGEInfo], color_map: dict, image_root: Path, save_root: Path, patch_size: int, max_image_size: int, do_not_overwrite: bool, class_name: str = 'handwritten_text') -> List[dict]:
    all_annotations = []
    for page in pages:
        label_image = numpy.zeros((page.height, page.width, 3), dtype='uint8')
        label_image[:, :] = color_map['background']

        label_image = cv2.fillPoly(label_image, page.text_lines, color_map[class_name])

        if page.file_name.startswith('http'):
            # we have a file that is available online
            image_download_dir = save_root / 'download'
            download_destination = image_download_dir / f"{page_file.stem}.png"
            if not download_destination.exists():
                download_destination.parent.mkdir(exist_ok=True, parents=True)
                download_image(page.file_name, download_destination)

            with Image.open(str(download_destination)) as rgb_image:
                rgb_image = numpy.array(rgb_image.convert('RGB'))
            page.file_name = str(download_destination)
        else:
            with Image.open(str(image_root / page.file_name)) as rgb_image:
                rgb_image = numpy.array(rgb_image.convert('RGB'))

        if max_image_size is not None:
            if any(side > max_image_size for side in rgb_image.shape[:2]):
                rgb_image = thumbnail_image(rgb_image, max_image_size)
                label_image = thumbnail_image(label_image, max_image_size)

        train_image = numpy.concatenate((rgb_image, label_image), axis=1)

        full_image_path = save_root / 'full_images' / f"{page_file.stem}.png"
        if full_image_path.exists() and do_not_overwrite:
            continue

        full_image_path.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(train_image).save(str(full_image_path))

        rgb_crops, label_crops = get_crops(rgb_image, label_image, patch_size)
        for _ in range(3):
            scale_factor = random.uniform(0.3, 0.95)
            new_size = (int(rgb_image.shape[0] * scale_factor), int(rgb_image.shape[1] * scale_factor))
            resized_rgb_image = cv2.resize(rgb_image, new_size, cv2.INTER_AREA)
            resized_label_image = cv2.resize(label_image, new_size, cv2.INTER_NEAREST)
            further_rgb_crops, further_label_crops = get_crops(resized_rgb_image, resized_label_image, patch_size)
            rgb_crops.extend(further_rgb_crops)
            label_crops.extend(further_label_crops)

        has_text = [(crop != 0).any() for crop in label_crops]
        save_crops(rgb_crops, label_crops, save_root / 'crops' / Path(page.file_name).stem)
        annotations = [{
            'file_name': str(Path('crops') / Path(page.file_name).stem / f'{idx}.png'),
            f'has_{class_name}': bool(text)
        } for idx, text in enumerate(has_text)]
        all_annotations.extend(annotations)
    return all_annotations


def create_diva_gt_image(page_file: Path, pages: List[PAGEInfo], save_root: Path):
    for page in pages:
        label_image = numpy.zeros((page.height, page.width, 3), dtype='uint8')
        label_image = cv2.fillPoly(label_image, page.text_lines, (0, 0, 8))

        file_name = f"{page_file.stem}.png"
        dest_name = save_root / 'diva_gt_images' / file_name
        dest_name.parent.mkdir(exist_ok=True, parents=True)
        label_image = Image.fromarray(label_image)
        label_image.save(str(dest_name))


def save_page_xml(page_file: Path, save_root: Path):
    dest_file_name = save_root / 'PAGE' / page_file.name
    dest_file_name.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(page_file, dest_file_name)


def main(args: argparse.Namespace):
    color_map = load_color_map(args.color_map)
    page_root = Path(args.page_root)
    default_image_root = Path(args.image_root)
    save_root = Path(args.image_destination)
    save_root.mkdir(exist_ok=True, parents=True)

    page_files = page_root.glob('**/*.xml')
    annotations = []
    for page_file in tqdm(list(page_files)):
        if str(page_file).endswith('TEST.xml'):
            continue
        image_extra_root = page_file.parent
        if image_extra_root.name == 'page':
            image_extra_root = image_extra_root.parent

        image_root = default_image_root / image_extra_root.relative_to(page_root)

        try:
            page_tree = ET.parse(str(page_file))
            pages = get_text_lines(page_tree.getroot())
            if not args.only_diva_gt_image:
                annotations.extend(examine_page(page_file, pages, color_map, image_root, save_root, args.patch_size, args.max_image_size, args.no_overwrite))
            create_diva_gt_image(page_file, pages, save_root)
            save_page_xml(page_file, save_root)
        except RuntimeError as e:
            print(e)
            continue

    if not args.only_diva_gt_image:
        with (save_root / 'train.json').open('w') as f:
            json.dump(annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take PAGE XML files and create image files to be used for patching and input to segmentation network")
    parser.add_argument("page_root", help="Path to PAGE XML files")
    parser.add_argument("image_root", help="Path where corresponding images are saved")
    parser.add_argument("image_destination", help="where to save resulting label images")
    parser.add_argument("color_map", help="Path to color map to use for creating label image")
    parser.add_argument("--class-name", default='handwritten_text', help="class to use for filling line boxes")
    parser.add_argument("--patch-size", type=int, default=256, help="size of patches to crop")
    parser.add_argument("-s", "--split-name", default='train', help='name of the split we create')
    parser.add_argument("--max-image-size", type=int, help="max size of input images")
    parser.add_argument("--no-overwrite", action='store_true', default=False, help="do not overwrite existing images")
    parser.add_argument("--only-diva-gt-image", action='store_true', default=False, help="only create the dive gt image and nothing else")

    main(parser.parse_args())
