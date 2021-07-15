import argparse
import json
import math
import os
import xml.etree.cElementTree as ET
from pathlib import Path
from typing import List, Dict, Type
from xml.dom import minidom

import cv2
import numpy
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as ImageClass
from torchvision import transforms
from tqdm import tqdm

from networks import load_weights
from networks.doc_ufcn import get_doc_ufcn
from utils.config import load_config
from utils.segmentation_utils import find_class_contours
from visualization.utils import network_output_to_color_image


def load_segmentation_network(args):
    config = load_config(args.model_checkpoint, None)
    segmentation_network_class = get_doc_ufcn(config.get('network', 'base'))
    segmentation_network = segmentation_network_class(3, 3)
    segmentation_network = load_weights(segmentation_network, args.model_checkpoint, key='segmentation_network')
    segmentation_network = segmentation_network.to(args.device)
    segmentation_network.eval()
    return segmentation_network


def crop_patches(image: ImageClass, image_size: int) -> List[Dict]:
    windows_in_width = math.ceil(image.width / image_size)
    total_width_overlap = windows_in_width * image_size - image.width
    windows_in_height = math.ceil(image.height / image_size)
    total_height_overlap = windows_in_height * image_size - image.height

    width_overlap_per_patch = total_width_overlap / windows_in_width
    height_overlap_per_patch = total_height_overlap / windows_in_height

    patches = []
    for y_idx in range(windows_in_height):
        start_y = int(y_idx * (image_size - height_overlap_per_patch))
        for x_idx in range(windows_in_width):
            start_x = int(x_idx * (image_size - width_overlap_per_patch))
            image_box = (start_x, start_y, start_x + image_size, start_y + image_size)
            patches.append({
                "image": image.crop(image_box),
                "bbox": image_box
            })

    return patches


def predict_patches(patches: List[Dict], segmentation_network, device: str) -> List[Dict]:
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform_list = transforms.Compose(transform_list)

    predicted_patches = []
    for patch in tqdm(patches, desc="Predicting patches...", leave=False):
        image = transform_list(patch["image"])
        batch = torch.unsqueeze(image, 0).to(device)
        with torch.no_grad():
            prediction = segmentation_network.predict(batch)

        predicted_patches.append({
            "prediction": torch.squeeze(torch.detach(prediction), dim=0).cpu(),
            "bbox": patch["bbox"]
        })

    return predicted_patches


def assemble_predictions(prediction_patches: List, image_size: tuple) -> torch.Tensor:
    # dimensions are height, width, class for easier access
    num_classes = prediction_patches[0]["prediction"].shape[0]
    max_width = image_size[0]
    max_height = image_size[1]
    assembled_predictions = torch.full((max_height, max_width, num_classes), float("-inf"))

    for patch in tqdm(prediction_patches, desc="Merging patches...", leave=False):
        reordered_patch = patch["prediction"].permute(1, 2, 0)
        bbox = patch["bbox"]
        x_start = bbox[0]
        y_start = bbox[1]
        x_end = min(bbox[2], max_width)
        y_end = min(bbox[3], max_height)
        window_height = y_end - y_start
        window_width = x_end - x_start

        assembled_window = assembled_predictions[y_start:y_end, x_start:x_end, :]
        patch_without_padding = reordered_patch[:window_height, :window_width, :]
        max_values = torch.maximum(assembled_window, patch_without_padding)
        assembled_predictions[y_start:y_end, x_start:x_end, :] = max_values

    return assembled_predictions.permute(2, 0, 1)


def fit_and_draw_line(contour: numpy.ndarray, bbox: List, new_out: numpy.ndarray):
    box_x_start = bbox[0]
    box_x_end = box_x_start + bbox[2]
    box_y_start = bbox[1]
    box_y_end = box_y_start + bbox[3]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

    if vx < 0:
        x0_border = box_x_start
        x1_border = box_x_end
    else:
        x0_border = box_x_end
        x1_border = box_x_start
    if vy < 0:
        y0_border = box_y_start
        y1_border = box_y_end
    else:
        y0_border = box_y_end
        y1_border = box_y_start
    ty0 = (y0_border - y) / vy
    tx0 = (x0_border - x) / vx
    if box_x_start <= x + ty0 * vx < box_x_end:
        t0 = ty0
    elif box_y_start <= y + tx0 * vy < box_y_end:
        t0 = tx0
    else:
        assert False
    p0 = ([x, y] + (t0 * [vx, vy])).astype(numpy.uint32)
    p0 = tuple(p0.ravel())

    ty1 = (y1_border - y) / vy
    tx1 = (x1_border - x) / vx
    if box_x_start <= x + ty1 * vx < box_x_end:
        t1 = ty1
    elif box_y_start <= y + tx1 * vy < box_y_end:
        t1 = tx1
    else:
        assert False
    p1 = ([x, y] + (t1 * [vx, vy])).astype(numpy.uint32)
    p1 = tuple(p1.ravel())
    cv2.line(new_out, p0, p1, (255, 255, 255), 2)


def get_overlapping_bboxes(reference_bbox: List, bboxes: List) -> List:
    x_end = reference_bbox[0] + reference_bbox[2]
    y_start = reference_bbox[1]
    y_end = y_start + reference_bbox[3]
    overlapping_bounding_boxes = []
    for bbox in bboxes:
        # TODO: may be helpful to relax the 0 <= x_diff constraint so that slight overlaps are possible
        # check for each bounding box that there is at least some vertical overlap to the reference box while it is to
        # the right of it
        x_diff = bbox[0] - x_end
        if 0 <= x_diff and not (y_start > bbox[1] + bbox[3] or y_end < bbox[1]):
            overlapping_bounding_boxes.append(bbox)

    return overlapping_bounding_boxes


def merge_bboxes(bboxes: List) -> List:
    x = bboxes[0][0]
    w = bboxes[-1][0] + bboxes[-1][2] - x
    y = min([b[1] for b in bboxes])
    h = max([b[1] + b[3] for b in bboxes]) - y

    return [x, y, w, h]


def create_lines_from_contours(bbox_contour_dict: Dict, max_x_diff: int, merge_mode: str = "convexHull") -> List:
    h_sorted_bboxes = list(bbox_contour_dict.keys()).copy()
    lines = []
    while len(h_sorted_bboxes) > 0:
        h_sorted_bboxes = sorted(h_sorted_bboxes, key=lambda x: x[0])
        leftmost_bbox = h_sorted_bboxes.pop(0)
        possible_bboxes = h_sorted_bboxes.copy()
        line = [leftmost_bbox]
        line_bbox = leftmost_bbox
        while True:
            overlapping_bounding_boxes = get_overlapping_bboxes(line_bbox, possible_bboxes)
            overlapping_bounding_boxes = sorted(overlapping_bounding_boxes, key=lambda x: x[0])

            # break if no bounding boxes are left or the remaining bounding boxes are to far away
            if len(overlapping_bounding_boxes) <= 0 \
                    or overlapping_bounding_boxes[0][0] - (line_bbox[0] + line_bbox[2]) > max_x_diff:
                break

            next_bbox = overlapping_bounding_boxes.pop(0)
            line.append(next_bbox)
            h_sorted_bboxes.remove(next_bbox)

            line_bbox = merge_bboxes(line)
            possible_bboxes = overlapping_bounding_boxes

        # merge the lines into one contour
        if merge_mode == "ConvexHull":
            merged_contours = []
            for bbox in line:
                contour = bbox_contour_dict[bbox]
                merged_contours.append(contour)
            merged_contours = numpy.concatenate(merged_contours)
            hull = cv2.convexHull(merged_contours)
            lines.append(numpy.squeeze(hull))
        elif merge_mode == "MinAreaBBox":
            hull_list = []
            for bbox in line:
                contour = bbox_contour_dict[bbox]
                hull = cv2.convexHull(contour)
                hull_list.append(hull)
            hull_list = numpy.concatenate(hull_list)

            bounding_rectangle = cv2.minAreaRect(hull_list)
            min_area_bbox = cv2.boxPoints(bounding_rectangle)
            min_area_bbox = numpy.int0(min_area_bbox)
            lines.append(min_area_bbox)
        elif merge_mode == "None":
            lines.extend([numpy.squeeze(bbox_contour_dict[bbox]) for bbox in line])
        else:
            raise NotImplementedError

    return lines


def find_line_contours(prediction: torch.Tensor, max_x_diff: int, merge_mode: str,
                       min_confidence: float = 0.7) -> (List, ImageClass):
    confidence_not_high_enough = prediction < min_confidence
    prediction[confidence_not_high_enough] = 0

    class_contours = find_class_contours(prediction, background_class_id=0)

    line_bboxes= []
    for class_id, contours in class_contours.items():
        bbox_contour_dict = {}
        for contour in contours:
            # Bounding Rect
            x, y, w, h = cv2.boundingRect(contour)
            bbox_contour_dict[(x, y, w, h)] = contour

        lines = create_lines_from_contours(bbox_contour_dict, max_x_diff, merge_mode)
        line_bboxes.extend(lines)

    return line_bboxes


def get_page_xml_from_contours(lines: List, image_filename: str, image_dimensions: tuple) -> str:
    root = ET.Element("PcGts")
    root.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd")
    page = ET.SubElement(root, "Page")
    page.set("imageFilename", image_filename)
    page.set("imageWidth", str(image_dimensions[0]))
    page.set("imageHeight", str(image_dimensions[1]))

    text_region = ET.SubElement(page, "TextRegion")
    text_region.set("id", "region_1")
    text_region.set("custom", "readingOrder {index:0;}")
    for i, line in enumerate(lines):
        xml_line = ET.SubElement(text_region, "TextLine")
        xml_line.set("id", f"line_{i}")
        coords = ET.SubElement(xml_line, "Coords")
        points_str = ""
        for point in line:
            points_str += f"{point[0]},{point[1]} "
        if len(points_str) > 0:
            points_str = points_str[:-1]
        coords.set("points", points_str)

    xml_str = minidom.parseString(ET.tostring(root, encoding="UTF-8", xml_declaration=True)).toprettyxml(indent="   ")
    return xml_str


def main(args):  # TODO: has to be tested again
    input_images = args.images
    if len(input_images) == 1 and input_images[0].endswith('.json'):
        # we are segmenting all images from the given json file
        image_json = Path(input_images[0])
        parent_dir = image_json.parent
        with image_json.open() as f:
            json_data = json.load(f)
        input_images = []
        for item in json_data:
            if isinstance(item, dict):
                input_images.append(image_json.parent / item['file_name'])
            else:
                input_images.append(image_json.parent / item)
    else:
        parent_dir = None

    # load the model
    segmentation_network = load_segmentation_network(args)

    for image_path in tqdm(input_images):
        image_path = Path(image_path)
        input_image = Image.open(str(image_path))

        # Get predictions for the whole image
        input_image = input_image.convert("RGB")

        original_size = input_image.size
        if any(side > args.max_image_size for side in original_size):
            input_image.thumbnail((args.max_image_size, args.max_image_size))

        patches = crop_patches(input_image, args.patch_size)
        predicted_patches = predict_patches(patches, segmentation_network, args.device)
        assembled_predictions = assemble_predictions(predicted_patches, input_image.size)

        with torch.no_grad():
            assembled_predictions = F.interpolate(assembled_predictions[None, ...], original_size[::-1])[0]

        # convert prediction to prediction image using the colors defined in the color map
        with open(args.class_to_color_map) as f:
            class_to_color_map = json.load(f)
        full_img_tensor = network_output_to_color_image(torch.unsqueeze(assembled_predictions, dim=0),
                                                        class_to_color_map)
        prediction_image = transforms.ToPILImage()(torch.squeeze(full_img_tensor, 0))

        # Saving
        model_root_dir = Path(args.model_checkpoint).parent.parent
        if parent_dir is None:
            parent = image_path.parent
        else:
            parent = parent_dir

        if args.output_dir is None:
            # we provide the output
            output_dir = model_root_dir / 'evaluation'
        else:
            output_dir = Path(args.output_dir)
        output_dir = output_dir / args.eval_name
        output_dir.mkdir(exist_ok=True, parents=True)
        image_dir = output_dir / 'images' / image_path.parent.relative_to(parent)
        image_dir.mkdir(exist_ok=True, parents=True)
        xml_dir = output_dir / 'xml' / image_path.parent.relative_to(parent)
        xml_dir.mkdir(exist_ok=True, parents=True)

        # Find contours in document and draw them
        max_x_diff = int(input_image.width * args.max_diff_percentage)
        contours = find_line_contours(assembled_predictions, max_x_diff, args.merge_mode)

        out_image = numpy.asarray(prediction_image)
        if not args.no_contours:
            for bbox in contours:
                cv2.drawContours(out_image, [bbox], 0, (255, 255, 255), 2)

        output_filename = image_dir / f"{image_path.stem}_contours{image_path.suffix}"
        Image.fromarray(out_image).save(str(output_filename))

        # Generate PAGE XML
        page_xml = get_page_xml_from_contours(contours, os.path.basename(image_path),
                                              (input_image.width, input_image.height))
        xml_filename = xml_dir / f"{image_path.stem}.xml"
        with xml_filename.open("w") as xml_file:
            xml_file.write(page_xml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="segment a given image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_checkpoint", help="Path to model checkpoint which shall be used for segmentation")
    parser.add_argument("images", nargs="+", help="path(s) to the images that should segmented")
    parser.add_argument("--patch-size", default=256,
                        help="the size of the patches that will be cropped out of the image")
    parser.add_argument("--device", default='cuda', help="which device to use (cuda, or cpu)")
    parser.add_argument("--output-dir")
    parser.add_argument("-e", "--eval-name", default='segmentation',
                        help='name you want to give to this evaluation run (might be helpful when evaluating on an eval'
                             ' dataset')
    parser.add_argument("--class-to-color-map", default="semantic_labeller/configs/handwriting_colors.json",
                        help="path to json file with class color map")
    parser.add_argument("--merge-mode", type=str, default="None", choices=["None", "MinAreaBBox", "ConvexHull"],
                        help="specifies how the contours of a line should be merged")
    parser.add_argument("--max-diff-percentage", type=float, default=0.05,
                        help="regulates the max distance between two contours that can be merged: "
                             "(image width * max-diff-percentage)")
    parser.add_argument("--max-image-size", type=int, default=3000,
                        help="max size of one side, each loaded image may have")
    parser.add_argument("--no-contours", action='store_true', default=False, help="do not draw merged contour in image")

    main(parser.parse_args())
