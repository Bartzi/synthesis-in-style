import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple, List, Union, NoReturn, Dict

import cv2
import numpy
import torch
from PIL import Image, UnidentifiedImageError
from PIL.Image import Image as ImageClass
from PIL.ImageDraw import ImageDraw
from tqdm import tqdm

from segmentation.segmenter import Segmenter
from utils.image_utils import opencv_image_to_pil, pil_image_to_opencv, resize_image
from utils.segmentation_utils import Color, find_class_contours, BBox


def get_bounding_boxes(prediction: torch.Tensor) -> Dict[int, List[BBox]]:
    class_contours = find_class_contours(prediction, background_class_id=0)

    bbox_dict = {}
    for class_id, contours in class_contours.items():
        bbox_dict[class_id] = []
        for contour in contours:
            bbox = BBox.from_opencv_bounding_rect(*cv2.boundingRect(contour))
            bbox_dict[class_id].append(bbox)

    return bbox_dict


def draw_bounding_boxes(image: ImageClass, bboxes: Tuple[BBox], outline_color: Color = (0, 255, 0),
                        stroke_width: int = 3) -> NoReturn:
    d = ImageDraw(image)
    for bbox in bboxes:
        d.rectangle(bbox, outline=outline_color, width=stroke_width)


def draw_segmentation(original_image: ImageClass, assembled_predictions: torch.Tensor, original_segmented_image: Image,
                      output_dir: Path, image_path: Path, copy_original: bool = False,
                      bboxes_for_patches: Union[Tuple[BBox], None] = None) -> Tuple[ImageClass, ImageClass]:
    if original_image.size != original_segmented_image.size:
        warnings.warn("Sizes of original_image and original_segmented_image do not match. It could be that there is "
                      "something wrong with the preprocessing of these images.")
    bbox_dict = get_bounding_boxes(assembled_predictions)
    bboxes = tuple([bbox for bboxes in list(bbox_dict.values()) for bbox in bboxes])

    segmented_image = original_segmented_image.copy()
    draw_bounding_boxes(segmented_image, bboxes)
    orig_image_output_path = output_dir / image_path.name

    image = original_image.copy()
    if copy_original and not orig_image_output_path.exists():
        print("Saving original image...")
        image.save(orig_image_output_path)

    draw_bounding_boxes(image, bboxes)

    if bboxes_for_patches is not None:
        draw_bounding_boxes(image, bboxes_for_patches, outline_color=(255, 0, 0), stroke_width=1)
        draw_bounding_boxes(segmented_image, bboxes_for_patches, outline_color=(255, 0, 0), stroke_width=1)

    return image, segmented_image


def save_contour_to_image(contour: numpy.ndarray, original_image: Union[ImageClass, numpy.ndarray], filename: Path,
                          background_color: Color = (255, 255, 255)) -> NoReturn:
    if isinstance(original_image, ImageClass):
        original_image = pil_image_to_opencv(original_image)

    mask = numpy.zeros_like(original_image)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    full_output_image = numpy.zeros_like(original_image)
    output_channels = []
    for i, (mask_channel, original_channel) in enumerate(zip(cv2.split(mask), cv2.split(original_image))):
        output_channel = numpy.where(mask_channel == 255, original_channel, background_color[i])
        output_channels.append(output_channel)
    cv2.merge(output_channels, full_output_image)

    x, y, w, h = cv2.boundingRect(contour)
    if len(full_output_image.shape) == 3:
        output_image = full_output_image[y:y + h, x:x + w, :]
    elif len(full_output_image.shape) == 2:
        output_image = full_output_image[y:y + h, x:x + w]
    else:
        raise NotImplementedError

    cv2.imwrite(str(filename), output_image)


def extract_and_save_contours(image: ImageClass, assembled_predictions: torch.Tensor, output_dir: Path,
                              image_path: Path) -> NoReturn:
    contour_dir = output_dir / "contours"
    contour_dir.mkdir(exist_ok=True)
    class_contours = find_class_contours(assembled_predictions, background_class_id=0)
    for class_id, contours in class_contours.items():
        for i, contour in enumerate(tqdm(contours, desc=f"Extracting contours for class {class_id}...", leave=False)):
            filename = contour_dir / f"{image_path.stem}_class_{class_id}_contour_{i}{image_path.suffix}"
            save_contour_to_image(contour, image, filename)


def save_bbox_to_image(bbox: BBox, original_image: Union[ImageClass, numpy.ndarray], filename: Path) -> NoReturn:
    if isinstance(original_image, numpy.ndarray):
        original_image = opencv_image_to_pil(original_image)
    cropped_area = original_image.crop(bbox)
    cropped_area.save(filename)


def extract_and_save_bounding_boxes(image: ImageClass, assembled_predictions: torch.Tensor, output_dir: Path,
                                    image_path: Path) -> NoReturn:
    bbox_dir = output_dir / "bboxes"
    bbox_dir.mkdir(exist_ok=True)
    bbox_dict = get_bounding_boxes(assembled_predictions)
    for class_id, bboxes in bbox_dict.items():
        for i, bbox in enumerate(tqdm(bboxes, desc="Cropping bboxes...", leave=False)):
            filename = bbox_dir / f"{image_path.stem}_class_{class_id}_bbox_{i}{image_path.suffix}"
            save_bbox_to_image(bbox, image, filename)


def main(args: argparse.Namespace) -> NoReturn:
    root_dir = Path(__file__).parent
    config_path = args.config_file
    output_dir = args.output_dir

    image_dir = args.image_dir
    images = [f for f in image_dir.iterdir() if f.is_file()]

    with config_path.open() as f:
        model_config = json.load(f)
    segmenter = Segmenter(
        model_config['checkpoint'],
        'cuda',
        root_dir / model_config['class_to_color_map'],
        max_image_size=int(model_config.get('max_image_size', 0)),
        print_progress=False,
        patch_overlap=args.absolute_patch_overlap,
        patch_overlap_factor=args.patch_overlap_factor
    )

    for i, image_path in enumerate(tqdm(images, desc="Processing images...", leave=False)):
        try:
            image = Image.open(image_path)
        except UnidentifiedImageError:
            print(f"File {image_path} is not an image.")
            continue
        if args.resize:
            image = resize_image(image, args.resize)

        # segmented image is an image that only displays the color-coded, predicted class for each pixel
        segmented_image, assembled_predictions = segmenter.segment_image(image)

        bboxes_for_patches = segmenter.calculate_bboxes_for_patches(*image.size) if args.draw_patches else None
        image_w_bboxes, segmented_image_w_bboxes = draw_segmentation(image, assembled_predictions, segmented_image,
                                                                     output_dir, image_path,
                                                                     copy_original=args.copy_original,
                                                                     bboxes_for_patches=bboxes_for_patches)
        image_w_bboxes.save(output_dir / f"{image_path.stem}_bboxes{image_path.suffix}")
        segmented_image.save(output_dir / f"{image_path.stem}_segmented_no_bboxes{image_path.suffix}")
        if args.draw_bboxes_on_segmentation:
            segmented_image_w_bboxes.save(output_dir / f"{image_path.stem}_segmented{image_path.suffix}")

        if args.save_bboxes:
            extract_and_save_bounding_boxes(image, assembled_predictions, output_dir, image_path)
        if args.save_contours:
            extract_and_save_contours(image, assembled_predictions, output_dir, image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes and contours from a given image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_dir", type=Path,
                        help="Path to a directory that contains the images that should be analyzed")
    parser.add_argument("-f", "--config-file", default="config.json", type=Path,
                        help="Path to the JSON file that contains the segmenter configuration")
    parser.add_argument("-o", "--output-dir", default="images", type=Path,
                        help="Path to the directory in which the results should be saved")
    parser.add_argument("-b", "--save-bboxes", action="store_true", default=False,
                        help="Crop bboxes and save them as separate images")
    parser.add_argument("-c", "--save-contours", action="store_true", default=False,
                        help="Crop contours and save them as separate images")
    parser.add_argument("--resize", nargs=2, type=int,
                        help="Resize input images to the given resolution [height width]. If one of the arguments is -1"
                             "the size of this dimension will be determined automatically, keeping the aspect ratio.")
    parser.add_argument("--absolute-patch-overlap", type=int, default=None,
                        help="Specify the overlap between patches in pixels. 0 < overlap < patch size")
    parser.add_argument("--patch-overlap-factor", type=float, default=None,
                        help="Specify the overlap between patches as a percentage. 0.0 < overlap < 1.0")
    parser.add_argument("--copy-original", action="store_true", default=False,
                        help="Copy the original image to the output dir as well (e.g. for easier comparison)")
    parser.add_argument("--draw-patches", action="store_true", default=False,
                        help="Show the borders of the patches, into which the image was")
    parser.add_argument("--draw-bboxes-on-segmentation", action="store_true", default=False,
                        help="Draw the determined bboxes not only on original image but on the segmented image as well")
    main(parser.parse_args())
