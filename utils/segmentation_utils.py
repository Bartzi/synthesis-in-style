from __future__ import annotations

from collections import defaultdict
from typing import Tuple, Dict, Union, List, NamedTuple

import cv2
import numpy
import torch

Color = Tuple[int, int, int]
# {class_name: {sub_image_id: list_of_batches[list_of_contours[numpy_array_of_points, ...]/None, ...]}}
ClassContoursForSubImages = Dict[str, Dict[str, List[List[numpy.ndarray]]]]
# {class_name: list_of_batches[list_of_contours[numpy_array_of_points, ...]/None, ...]}
ClassContours = Dict[str, List[Union[None, List[numpy.ndarray]]]]
# {sub_image_id: {class_name: predicted_tensor})}
PredictedClusters = Dict[str, Dict[str, torch.Tensor]]


class BBox(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_opencv_bounding_rect(cls, x, y, width, height):
        return cls(x, y, x + width, y + height)

    def top_left(self) -> Tuple:
        return self.left, self.top

    def bottom_right(self) -> Tuple:
        return self.right, self.bottom

    def as_points(self) -> Tuple:
        # top left, top right, bottom right, bottom left
        return self.top_left(), \
               (self.right, self.top), \
               self.bottom_right(), \
               (self.left, self.bottom)

    def is_overlapping_with(self, other_box: BBox):
        return self.left < other_box.right and \
               self.right > other_box.left and \
               self.top < other_box.bottom and \
               self.bottom > other_box.top

    def get_mutual_bbox(self, other_box):
        return BBox(
            min(self.left, other_box.left),
            min(self.top, other_box.top),
            max(self.right, other_box.right),
            max(self.bottom, other_box.bottom),
        )


def bounding_rect_from_contours(contours: List[numpy.ndarray]) -> numpy.ndarray:
    bounding_rects = numpy.concatenate([cv2.boundingRect(contour) for contour in contours])
    if bounding_rects.ndim == 1:
        bounding_rects = bounding_rects.reshape((1, len(bounding_rects)))
    return bounding_rects


def draw_contours_on_same_sized_canvases(contours: List[numpy.ndarray], minimal_canvas: bool = False) \
                                         -> List[numpy.ndarray]:
    combined_contours = numpy.concatenate(contours)
    x_max, y_max = combined_contours.max(axis=0)[0]
    x_min, y_min = 0, 0
    if minimal_canvas:
        # Determine how small a minimal canvas (numpy array) has to be to avoid creating an oversized numpy array that
        # contains a lot of empty space
        x_min, y_min = combined_contours.min(axis=0)[0]

    canvas = numpy.zeros((y_max - y_min + 1, x_max - x_min + 1))  # height x width
    return [cv2.drawContours(canvas.copy(), [contour - (x_min, y_min)], 0, 1, cv2.FILLED) for contour in contours]


def get_contours_from_prediction(predictions: torch.Tensor) -> Union[numpy.ndarray, None]:
    # TODO: maybe migrate from prediction to actual images (aka second function or change function call in doc_ufcn)
    scaled_predictions = predictions.cpu().numpy() * 255
    scaled_predictions = scaled_predictions.astype(numpy.uint8)
    class_predictions = cv2.morphologyEx(
        scaled_predictions,
        cv2.MORPH_CLOSE,
        numpy.ones((5, 5), numpy.uint8)
    )
    non_zero_predictions = class_predictions != 0
    if not non_zero_predictions.any():
        # nothing here, so no contours will be found
        return None
    contours, _ = cv2.findContours((non_zero_predictions * 255).astype('uint8'), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    return contours


def find_class_contours(class_predictions: torch.tensor, min_contour_area: int = 10,
                        background_class_id: int = 0) -> Dict[int, List[numpy.ndarray]]:
    all_contours = defaultdict(list)
    for class_id, predictions in enumerate(class_predictions):
        if class_id == background_class_id:
            continue

        contours = get_contours_from_prediction(predictions)
        if contours is None:
            continue

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                all_contours[class_id].append(contour)

    return all_contours
