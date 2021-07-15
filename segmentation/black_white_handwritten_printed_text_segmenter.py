from functools import reduce
from typing import List, Dict, Tuple

import numpy
import torch

from segmentation.gan_segmenter import Segmenter, ClassContours
from utils.segmentation_utils import bounding_rect_from_contours, PredictedClusters


class BlackWhiteHandwrittenPrintedTextSegmenter(Segmenter):
    """
    This Segmenter is applied to black and white images that contain handwritten and printed text. Samples were
    originally taken from WPI auction catalogues.
    The model path is:
    black_white_handwritten_printed_text_wpi_auction_catalogues/2021-06-09T08:25:54.947435/checkpoints/040000.pt
    """

    def __init__(self, *args, keys_to_merge: Dict[str, List[str]] = None, only_keep_overlapping: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.keys_to_merge = keys_to_merge
        self.keys_for_generation = set(reduce(lambda x, y: x + y, self.keys_to_merge.values(),
                                              self.keys_for_class_determination +
                                              self.keys_for_finegrained_segmentation))
        self.only_keep_overlapping = only_keep_overlapping
        self.catalog = self.load_catalog()

    def merge_sub_images(self, predicted_clusters: PredictedClusters) -> PredictedClusters:
        for destination_key, keys_to_merge in self.keys_to_merge.items():
            sub_images_to_merge = [predicted_clusters[key] for key in keys_to_merge]
            merged_classes = {}
            for class_name in self.class_to_color_map:
                class_tensors = [sub_image_data[class_name] for sub_image_data in sub_images_to_merge]
                merged_tensor = reduce(torch.bitwise_or, class_tensors[1:], class_tensors[0])
                merged_classes[class_name] = merged_tensor
            predicted_clusters[destination_key] = merged_classes
        return predicted_clusters

    def extract_text_regions(self, predicted_clusters: PredictedClusters, batch_size: int) -> ClassContours:
        class_contours_for_sub_images = self.extract_contours(predicted_clusters, self.keys_for_class_determination)

        # merge all contours that have high overlap and are of the same class
        merged_contours = self.merge_contours_of_same_class_from_different_images(
            class_contours_for_sub_images,
            batch_size,
            only_keep_overlapping=self.only_keep_overlapping,
            drop_if_size_of_contours_zero=True,
        )
        if self.debug:
            self.render_debug_contours(merged_contours, "after_handwriting_merging")

        merged_contours = self.drop_too_small_contours(merged_contours)
        if self.debug:
            self.render_debug_contours(merged_contours, "after_small_dropping")
        return merged_contours

    def determine_images_to_drop(self, fine_grained_contours_per_image: ClassContours) -> List[int]:
        image_ids_to_drop = set()
        for class_name, batch_contours in fine_grained_contours_per_image.items():
            for image_id, contours in enumerate(batch_contours):
                if contours is None:
                    continue
                bounding_rects = bounding_rect_from_contours(contours)
                max_extent = int(self.image_size * 0.95)
                # each bounding rect is comprised of x, y, width, height
                if (bounding_rects[:, 3] > max_extent).any() and (bounding_rects[:, 2] > max_extent).any():
                    # box is too large, it is safer to drop this image than to save it, because it might contain
                    # false segmentation
                    image_ids_to_drop.add(image_id)
        return list(image_ids_to_drop)

    def create_segmentation_image(self, activations: Dict[int, torch.Tensor],
                                  class_label_map: Dict[str, Dict[str, list]]) -> Tuple[numpy.ndarray, List[int]]:
        if self.debug:
            self.debug_images.clear()

        batch_size = len(activations[0])
        predicted_clusters = self.predict_clusters(activations, class_label_map)
        predicted_clusters = self.resize_to_image_size(predicted_clusters)
        predicted_clusters = self.merge_sub_images(predicted_clusters)
        text_regions_per_sub_image = self.extract_text_regions(predicted_clusters, batch_size)
        fine_grained_contours_per_sub_image = self.merge_finegrained_segmentation(predicted_clusters, batch_size)

        if self.debug:
            self.render_debug_contours(fine_grained_contours_per_sub_image, "finegrained_contours")

        classified_contours = self.classify_fine_grained_contours(text_regions_per_sub_image,
                                                                  fine_grained_contours_per_sub_image,
                                                                  fine_grained_class_name='printed_text')

        classified_contours = self.drop_too_small_contours(classified_contours)
        image_ids_to_drop = self.determine_images_to_drop(classified_contours)

        segmentation_images = self.render_segmentation_image(
            predicted_clusters[self.keys_for_finegrained_segmentation[-1]],
            classified_contours,
            batch_size,
            cluster_class_name='printed_text'
        )
        return segmentation_images, image_ids_to_drop
