from collections import defaultdict
from typing import Tuple, Dict, Union, List, NamedTuple

import cv2
import numpy
import torch

Color = Tuple[int, int, int]


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


def get_contours_from_prediction(predictions: torch.Tensor) -> Union[numpy.ndarray, None]:
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
