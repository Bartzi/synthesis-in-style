from collections import defaultdict
from functools import reduce
from typing import List, Dict, Tuple, Union

import numpy
import torch

from segmentation.gan_segmenter import Segmenter


class BenchmarkSegmenter(Segmenter):

    def __init__(self, *args, keys_to_merge: Dict[str, List[str]] = None, only_keep_overlapping_boxes: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.keys_to_merge = keys_to_merge
        self.keys_for_generation = set(reduce(lambda x, y: x + y, self.keys_to_merge.values(),
                                              self.keys_for_class_determination +
                                              self.keys_for_finegrained_segmentation))
        self.only_keep_overlapping_boxes = only_keep_overlapping_boxes
        self.catalog = self.load_catalog()

    def merge_sub_images(self, predicted_clusters: Dict[str, Dict[str, torch.Tensor]]) \
                         -> Dict[str, Dict[str, torch.Tensor]]:
        for destination_key, keys_to_merge in self.keys_to_merge.items():
            sub_images_to_merge = [predicted_clusters[key] for key in keys_to_merge]
            merged_classes = {}
            for class_name in self.class_to_color_map:
                class_tensors = [sub_image_data[class_name] for sub_image_data in sub_images_to_merge]
                merged_tensor = reduce(torch.bitwise_or, class_tensors[1:], class_tensors[0])
                merged_classes[class_name] = merged_tensor
            predicted_clusters[destination_key] = merged_classes
        return predicted_clusters

    def create_text_regions(self, predicted_clusters: Dict[str, Dict[str, torch.Tensor]], batch_size: int) \
                            -> Dict[str, List[Union[None, numpy.ndarray]]]:
        class_boxes_for_sub_images = self.extract_contours_and_boxes(predicted_clusters,
                                                                     self.keys_for_class_determination)

        # merge all boxes that have high overlap and are of the same class
        merged_boxes = self.merge_boxes_of_same_class_from_different_images(
            class_boxes_for_sub_images,
            batch_size,
            only_keep_overlapping_boxes=self.only_keep_overlapping_boxes,
            class_names_to_merge=('handwritten_text',),
            drop_if_size_of_boxes_zero=True,
        )
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after_handwriting_merging")

        merged_boxes = self.drop_too_small_boxes(merged_boxes)
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after_small_dropping")
        return merged_boxes

    def determine_images_to_drop(self, fine_grained_boxes_per_image: Dict[str, List[Union[None, numpy.ndarray]]]) \
                                 -> List[int]:
        image_ids_to_drop = set()
        for class_name, batch_boxes in fine_grained_boxes_per_image.items():
            for image_id, boxes in enumerate(batch_boxes):
                if boxes is None:
                    continue
                box_widths = boxes[:, 2] - boxes[:, 0]
                box_heights = boxes[:, 3] - boxes[:, 1]
                max_extent = int(self.image_size * 0.95)
                if (box_heights > max_extent).any() and (box_widths > max_extent).any():
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
        text_regions_per_feature_map = self.create_text_regions(predicted_clusters, batch_size)
        fine_grained_boxes_per_feature_map = self.extract_contours_and_boxes(predicted_clusters,self.keys_for_finegrained_segmentation)[self.keys_for_finegrained_segmentation[0]]
        if self.debug:
            self.render_debug_boxes(fine_grained_boxes_per_feature_map, "finegrained_boxes")

        classified_boxes = self.classify_fine_grained_boxes(text_regions_per_feature_map,
                                                            fine_grained_boxes_per_feature_map,
                                                            fine_grained_class_name='handwritten_text')
        classified_boxes = self.drop_too_small_boxes(classified_boxes)
        image_ids_to_drop = self.determine_images_to_drop(classified_boxes)

        segmentation_images = self.render_segmentation_image(
            predicted_clusters[self.keys_for_finegrained_segmentation[-1]],
            classified_boxes,
            batch_size,
            cluster_class_name='handwritten_text'
        )
        return segmentation_images, image_ids_to_drop


class TextLineBenchmarkSegmenter(Segmenter):

    def any_box_too_large(self, boxes: numpy.ndarray) -> bool:
        # we deem a box too large if it fills 3/4 of the image size
        image_area = self.image_size ** 2
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        return any(area > image_area * 0.75 for area in areas)

    def create_segmentation_image(self, activations: Dict[int, torch.Tensor],
                                  class_label_map: Dict[str, Dict[str, list]]) -> Tuple[numpy.ndarray, List[int]]:
        if self.debug:
            self.debug_images.clear()

        batch_size = len(activations[0])
        predicted_clusters = self.predict_clusters(activations, class_label_map)
        predicted_clusters = self.resize_to_image_size(predicted_clusters)
        boxes_image_1 = self.extract_contours_and_boxes(predicted_clusters, self.keys_for_class_determination)[self.keys_for_class_determination[0]]
        boxes_image_2 = self.extract_contours_and_boxes(predicted_clusters, self.keys_for_finegrained_segmentation)[self.keys_for_finegrained_segmentation[0]]

        boxes_to_render = defaultdict(list)
        key_for_rendering = None
        for class_name in boxes_image_1:
            image_1_boxes = boxes_image_1[class_name]
            image_2_boxes = boxes_image_2[class_name]

            for image_1_box, image_2_box in zip(image_1_boxes, image_2_boxes):
                if image_1_box is None or self.any_box_too_large(image_1_box):
                    boxes_to_render[class_name].append(image_2_box)
                    key_for_rendering = self.keys_for_finegrained_segmentation[0]
                else:
                    boxes_to_render[class_name].append(image_1_box)
                    key_for_rendering = self.keys_for_class_determination[0]

        image_ids_to_drop = self.determine_images_to_drop(boxes_to_render)
        segmentation_images = self.render_segmentation_image(
            predicted_clusters[key_for_rendering],
            boxes_to_render,
            batch_size,
            cluster_class_name='handwritten_text'
        )

        return segmentation_images, image_ids_to_drop
