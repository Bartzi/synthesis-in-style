from typing import Dict, Tuple, List

import numpy
import torch

from segmentation.benchmark_segmenter import BenchmarkSegmenter


class HandwritingAndPrintedTextSegmenter(BenchmarkSegmenter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_segmentation_image(self, activations: Dict[int, torch.Tensor],
                                  class_label_map: Dict[str, Dict[str, list]]) -> Tuple[numpy.ndarray, List[int]]:
        if self.debug:
            self.debug_images.clear()

        batch_size = len(activations[0])
        predicted_clusters = self.predict_clusters(activations, class_label_map)
        predicted_clusters = self.resize_to_image_size(predicted_clusters)
        predicted_clusters = self.merge_sub_images(predicted_clusters)
        text_regions_per_feature_map = self.create_text_regions(predicted_clusters, batch_size)
        fine_grained_boxes_per_feature_map = self.extract_contours_and_boxes(predicted_clusters, self.keys_for_finegrained_segmentation)[self.keys_for_finegrained_segmentation[0]]
        if self.debug:
            self.render_debug_boxes(fine_grained_boxes_per_feature_map, "finegrained_boxes")

        classified_boxes = self.classify_fine_grained_boxes(text_regions_per_feature_map,
                                                            fine_grained_boxes_per_feature_map)
        classified_boxes = self.drop_too_small_boxes(classified_boxes)
        image_ids_to_drop = self.determine_images_to_drop(classified_boxes)

        segmentation_images = self.render_segmentation_image(
            predicted_clusters[self.keys_for_finegrained_segmentation[-1]],
            classified_boxes,
            batch_size,
        )
        return segmentation_images, image_ids_to_drop
