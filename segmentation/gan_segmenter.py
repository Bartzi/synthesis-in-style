import copy
import pickle
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

import cv2
import numpy
import torch
import torch.nn.functional as F
from PIL import ImageColor

from utils.segmentation_utils import BBox


class Segmenter:

    def __init__(self, keys_for_class_determination: List[str], keys_for_finegrained_segmentation: List[str],
                 base_dir: Path, num_clusters: int, image_size: int, class_to_color_map: Dict,
                 debug: bool = False):
        # assert len(keys_for_class_determination) == 2, "Segmenter can only work with two keys for class determination " \
        #                                                "for now "
        self.keys_for_class_determination = keys_for_class_determination
        # assert len(keys_for_finegrained_segmentation) == 2, "Segmenter can only work with two keys for finegrained " \
        #                                                     "segmentation for now "
        self.keys_for_finegrained_segmentation = keys_for_finegrained_segmentation
        self.keys_for_generation = self.keys_for_class_determination + self.keys_for_finegrained_segmentation
        self.base_dir = base_dir
        self.num_clusters = num_clusters
        self.image_size = image_size
        self.debug = debug
        self.debug_images = {}
        self.max_debug_text_size = 20
        self.class_to_color_map = self.load_class_to_color_map(class_to_color_map)
        self.class_id_map = self.build_class_id_map(self.class_to_color_map)
        self.catalog = self.load_catalog()
        self.handwriting_overlap_threshold = 0.5
        self.min_class_box_area = 150

    def load_class_to_color_map(self, class_to_color_map: dict) -> dict:
        return {klass: ImageColor.getrgb(color) for klass, color in class_to_color_map.items()}

    def build_class_id_map(self, class_to_color_map: dict) -> dict:
        return {class_name: class_id for class_id, class_name in enumerate(class_to_color_map)}

    def load_catalog(self) -> dict:
        file_name = self.base_dir / 'catalogs' / f'{self.num_clusters}.pkl'
        with file_name.open('rb') as f:
            try:
                catalogs = pickle.load(f)
            except ModuleNotFoundError:
                # we are loading a legacy catalog -> we need to adjust the module paths
                import sys
                from segmentation import gan_local_edit
                sys.modules['gan_local_edit'] = gan_local_edit
                catalogs = pickle.load(f)
        return self.adjust_catalog(catalogs)

    def adjust_catalog(self, catalog: dict) -> dict:
        adjusted_catalog = {}
        for key in catalog.keys():
            if key in self.keys_for_generation:
                adjusted_catalog[key] = catalog[key]
        return adjusted_catalog

    def render_debug_boxes(self, boxes: Dict[str, List[Union[None, numpy.ndarray]]], name: str):
        debug_images = []
        batch_size = len(list(boxes.values())[0])
        for batch_id in range(batch_size):
            debug_image = numpy.zeros((self.image_size + self.max_debug_text_size, self.image_size, 3), dtype='uint8')
            # add white bounds around the image
            debug_image = cv2.rectangle(debug_image, (0, 0), debug_image.shape[:2], color=(255, 255, 255))
            for class_name, class_box_batch in boxes.items():
                if class_name == 'background':
                    continue

                image_boxes = class_box_batch[batch_id]
                color = self.class_to_color_map[class_name]

                if image_boxes is None:
                    # there is nothing in this image
                    continue

                for box in image_boxes:
                    box = BBox(*box.astype(numpy.int32))

                    # make boxes semi-transparent to see overlapping boxes
                    sub_img = debug_image[box.top:box.bottom, box.left:box.right]
                    colored_rect = numpy.ones(sub_img.shape, dtype=numpy.uint8) \
                                   * numpy.asarray(color, dtype=numpy.uint8)
                    transparent_sub_image = cv2.addWeighted(sub_img, 0.5, colored_rect, 0.5, 1.0)
                    debug_image[box.top:box.bottom, box.left:box.right] = transparent_sub_image

                    debug_image = cv2.rectangle(debug_image, box.top_left(), box.bottom_right(), color=(255, 255, 255))

            font_scale = 1
            font = cv2.FONT_HERSHEY_PLAIN
            text_size, baseline = cv2.getTextSize(name, font, font_scale, 1)
            assert text_size[1] < self.max_debug_text_size, "Debug text is too large!"
            debug_image = cv2.rectangle(
                debug_image,
                [0, self.image_size, self.image_size, self.image_size + self.max_debug_text_size],
                (255, 255, 255),
                cv2.FILLED
            )
            debug_image = cv2.putText(
                debug_image, name, (0, self.image_size + self.max_debug_text_size - baseline), font, font_scale,
                (0, 0, 0)
            )
            debug_images.append(debug_image.copy())
        self.debug_images[name] = copy.copy(debug_images)

    def predict_clusters(self, activations: Dict[int, torch.Tensor], class_label_map: Dict[str, Dict[str, list]]) \
                         -> Dict[str, Dict[str, torch.Tensor]]:
        predicted_classes = {}
        activations = {str(k): v for k, v in activations.items()}
        for layer_id, factor_catalog in self.catalog.items():
            activations_for_layer = activations[layer_id]
            # use k-means cluster centers to predict class membership of each pixel
            class_membership = factor_catalog.predict(activations_for_layer)

            # now merge all classes that belong to the same superclass together into one image with a channel for
            # each class maps from class name to array of shape [batch_size, height, width]
            images_per_class = {}
            for class_name, class_ids in class_label_map[layer_id].items():
                masks = [class_membership == class_id for class_id in class_ids]
                mask = reduce(torch.bitwise_or, masks, torch.zeros_like(masks[0], dtype=torch.bool))

                images_per_class[class_name] = mask

            predicted_classes[layer_id] = images_per_class
        return predicted_classes

    def resize_to_image_size(self, tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        resized = {}
        for key, class_tensors in tensors.items():
            resized_class_tensors = {}
            for class_name, tensor in class_tensors.items():
                if tensor.shape[-1] < self.image_size:
                    tensor = F.interpolate(tensor[:, None, ...].type(torch.uint8),
                                           (self.image_size, self.image_size)).type(tensor.dtype).squeeze(1)
                resized_class_tensors[class_name] = tensor
            resized[key] = resized_class_tensors
        return resized

    @staticmethod
    def open_cluster_images(predicted_clusters: numpy.ndarray, kernel: numpy.ndarray = None,
                            kernel_size: int = 3) -> numpy.ndarray:
        if kernel is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)).astype(numpy.uint8)

        batch_size, num_channels = predicted_clusters.shape[:2]
        opened_images = []
        for image_id in range(batch_size):
            opened = [
                cv2.morphologyEx(predicted_clusters[image_id, channel_id], cv2.MORPH_OPEN, kernel)
                for channel_id in range(num_channels)
            ]
            opened_images.append(numpy.stack(opened, axis=0))
        return numpy.stack(opened_images, axis=0)

    @staticmethod
    def dilate_image(image: numpy.ndarray, kernel: numpy.ndarray = None, kernel_size: int = 3) -> numpy.ndarray:
        if kernel is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)).astype(numpy.uint8)

        return cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        # kernel = torch.from_numpy(kernel[numpy.newaxis, numpy.newaxis, ...]).to(predicted_clusters.device)
        # dilated_slices = [
        #     torch.clamp(F.conv2d(class_map, kernel, padding=(1, 1)), 0, 1)
        #     for class_map in torch.split(predicted_clusters.type(torch.float32), 1, dim=1)
        # ]
        # dilated = torch.cat(dilated_slices, dim=1)
        # return dilated

    def cluster_image_to_boxes(self, cluster_arrays: numpy.ndarray) -> List[Union[numpy.ndarray, None]]:
        batch_rects = []
        for image in cluster_arrays:
            image = self.dilate_image(image)
            contours, hierachy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                rects = [numpy.array(cv2.boundingRect(contour), dtype=numpy.float32) for contour in contours]
                # shape: 4 -> top, left, bottom, right
                rects = numpy.stack(rects, axis=0)
                rects[:, 2] += rects[:, 0]
                rects[:, 3] += rects[:, 1]

            else:
                rects = None
            # shape: [batch_size, [num_boxes, 4/0]]
            batch_rects.append(rects)
        return batch_rects

    def box_iou(self, boxes1: numpy.ndarray, boxes2: numpy.ndarray) -> numpy.ndarray:
        l1, t1, r1, b1 = numpy.split(boxes1, 4, axis=1)
        l2, t2, r2, b2 = numpy.split(boxes2, 4, axis=1)

        t = numpy.maximum(l1, numpy.transpose(l2))
        l = numpy.maximum(t1, numpy.transpose(t2))
        b = numpy.minimum(r1, numpy.transpose(r2))
        r = numpy.minimum(b1, numpy.transpose(b2))

        inter_area = numpy.maximum(b - t + 1, 0) * numpy.maximum(r - l + 1, 0)
        box1_area = (r1 - l1 + 1) * (b1 - t1 + 1)
        box2_area = (r2 - l2 + 1) * (b2 - t2 + 1)

        # shape: [len(boxes1), len(boxes2)]
        iou = inter_area / (box1_area + numpy.transpose(box2_area) - inter_area)
        return iou

    def box_enclosed(self, boxes1: numpy.ndarray, boxes2: numpy.ndarray) -> numpy.ndarray:
        l1, t1, r1, b1 = numpy.split(boxes1, 4, axis=1)
        l2, t2, r2, b2 = numpy.split(boxes2, 4, axis=1)

        t_enclosed = l1 >= numpy.transpose(l2)
        l_enclosed = t1 >= numpy.transpose(t2)
        b_enclosed = r1 <= numpy.transpose(r2)
        r_enclosed = b1 <= numpy.transpose(b2)

        # box1 is enclosed in box2 iff all sides are enclosed
        box_enclosed = reduce(
            numpy.bitwise_and,
            [l_enclosed, b_enclosed, r_enclosed],
            t_enclosed
        )
        return box_enclosed

    def boxes_right_of(self, boxes1: numpy.ndarray, boxes2: numpy.ndarray, max_box_distance: int = 5) -> numpy.ndarray:
        # determine which boxes of boxes1 are direct right neighbours of boxes2
        l1, t1, r1, b1 = numpy.split(boxes1, 4, axis=1)
        l2, t2, r2, b2 = numpy.split(boxes2, 4, axis=1)

        shifted_right_of_box_2 = r2 - max_box_distance
        left_is_right_of_box_2 = l1 > numpy.transpose(shifted_right_of_box_2)
        distance_left_box_1_to_right_box_2 = l1 - numpy.transpose(r2)

        return numpy.bitwise_and(left_is_right_of_box_2, distance_left_box_1_to_right_box_2 < max_box_distance)

    def get_non_overlapping_of_first_sub_image(self, boxes_for_first_sub_image: numpy.ndarray,
                                               boxes_for_second_sub_image: numpy.ndarray) -> Optional[numpy.ndarray]:
        ious_of_same_class = self.box_iou(boxes_for_first_sub_image, boxes_for_second_sub_image)
        ious_greater_zero = ious_of_same_class > 0

        # determine all boxes of first image that do not overlap/intersect with any box in the second image
        has_no_overlap_with_any_box_in_second_image = reduce(
            numpy.bitwise_and,
            numpy.split(~ious_greater_zero, ious_of_same_class.shape[1], axis=1),
            numpy.ones((len(ious_of_same_class), 1), dtype='bool')
        ).squeeze()
        if has_no_overlap_with_any_box_in_second_image.any():
            return numpy.take(boxes_for_first_sub_image, has_no_overlap_with_any_box_in_second_image.nonzero()[0],
                              axis=0)
        return None

    def get_non_overlapping_boxes(self, boxes_for_current_batch: List[numpy.ndarray]) -> Optional[numpy.ndarray]:
        non_overlapping_boxes = []
        num_sub_images = len(boxes_for_current_batch)
        for first_sub_image_idx in range(num_sub_images):
            box_candidates_for_first_sub_image = []
            for second_sub_image_idx in range(num_sub_images):
                if first_sub_image_idx == second_sub_image_idx:
                    continue
                boxes_with_no_overlap = self.get_non_overlapping_of_first_sub_image(
                    boxes_for_current_batch[first_sub_image_idx],
                    boxes_for_current_batch[second_sub_image_idx]
                )
                if boxes_with_no_overlap is not None:
                    box_candidates_for_first_sub_image.append(boxes_with_no_overlap)

            if len(box_candidates_for_first_sub_image) == 0:
                # no boxes found for sub image
                continue

            # Only select those boxes that are non overlapping for all other sub images, i.e. rows that
            # appear in every list item of box_candidates_for_first_sub_image
            box_candidates_for_first_sub_image = numpy.concatenate(box_candidates_for_first_sub_image)
            unique_boxes, counts = numpy.unique(box_candidates_for_first_sub_image, return_counts=True, axis=0)
            non_overlapping_boxes_for_first_sub_image = numpy.take(unique_boxes,
                                                                   numpy.where(counts == (num_sub_images - 1))[0],
                                                                   axis=0)
            non_overlapping_boxes.append(non_overlapping_boxes_for_first_sub_image)

        return numpy.concatenate(non_overlapping_boxes) if len(non_overlapping_boxes) > 0 else None

    def merge_two_boxes(self, candidates: numpy.ndarray) -> numpy.ndarray:
        """
        merge two boxes without any checks, e.g. if the boxes are overlapping
        """
        l, t, r, b = numpy.split(candidates, 4, axis=1)
        merged_box = numpy.array([[numpy.min(l), numpy.min(t), numpy.max(r), numpy.max(b)]])
        return merged_box

    def merge_boxes_of_same_class_from_same_image(self, class_boxes: Dict[str, List[Union[numpy.ndarray, None]]]) \
                                                  -> Dict[str, List[Union[numpy.ndarray, None]]]:
        all_merged_boxes = {}
        for class_name, batch_boxes_for_class in class_boxes.items():
            merged_boxes = []
            for boxes_for_class in batch_boxes_for_class:
                if boxes_for_class is None:
                    merged_boxes.append(None)
                    continue
                merged = self.merge_boxes(boxes_for_class)
                merged_boxes.append(merged)
            all_merged_boxes[class_name] = merged_boxes
        return all_merged_boxes

    def _try_merge_boxes(self, merged_boxes: Dict[Tuple, numpy.ndarray]) -> bool:
        """
        Merges the first pair of overlapping boxes inplace. If a successful merge was conducted, returns True.
        If all boxes are non-overlapping the method returns False.
        """
        for first_box_key in merged_boxes.keys():
            for second_box_key in merged_boxes.keys():
                if first_box_key == second_box_key:
                    continue
                iou = self.box_iou(merged_boxes[first_box_key], merged_boxes[second_box_key])
                if iou[0][0] > 0:
                    merged_box = self.merge_two_boxes(numpy.concatenate((merged_boxes[first_box_key],
                                                                         merged_boxes[second_box_key])))
                    merged_boxes[first_box_key + second_box_key] = merged_box
                    merged_boxes.pop(first_box_key)
                    merged_boxes.pop(second_box_key)
                    return True
        return False

    def merge_boxes(self, boxes: numpy.ndarray, only_keep_overlapping_boxes: bool = False) \
                                             -> numpy.ndarray:
        """
        Iteratively try to find two boxes that are overlapping. If an overlap is found, these boxes will be merged.
        If there is no overlap anymore, merging is complete.
        """
        merged_boxes = {(i,): numpy.expand_dims(boxes[i], axis=0) for i in range(len(boxes))}
        new_merges = True
        while new_merges:
            new_merges = self._try_merge_boxes(merged_boxes)

        if only_keep_overlapping_boxes:
            overlapping_boxes = [box for ids, box in merged_boxes.items() if len(ids) > 1]
            return numpy.concatenate(overlapping_boxes if len(overlapping_boxes) > 0 else [[]])
        else:
            return numpy.concatenate(list(merged_boxes.values()))

    def merge_boxes_of_same_class_from_different_images(self,
                                                        class_boxes_for_sub_images: Dict[str, Dict[str, List[Union[numpy.ndarray, None]]]],
                                                        batch_size: int,
                                                        only_keep_overlapping_boxes: bool = False,
                                                        class_names_to_merge: Tuple[str] = (),
                                                        drop_if_size_of_boxes_zero: bool = False) -> Dict[str, List[Union[None, numpy.ndarray]]]:
        if len(class_names_to_merge) == 0:
            # we are merging all class ids we currently have in our class boxes
            class_names_to_merge = {class_name for sub_image_data in class_boxes_for_sub_images.values() for class_name
                                    in sub_image_data.keys()}

        # (Indirectly) maps the class names to bboxes. Each dict value is a list of m items, where m is the number of
        # sub images. These lists consists of a lists with n items, where n is the number of batches. The bboxes are
        # represented by ndarrays of of shape (number bboxes, 4)
        # class_name: [
        #       sub image 1 = [
        #           batch 1 = ndarray: (number bboxes, 4),
        #           ...,
        #           batch n
        #       ]
        #       ...,
        #       sub image m
        # ]
        class_to_boxes = defaultdict(list)
        for class_boxes in class_boxes_for_sub_images.values():
            for class_name, boxes in class_boxes.items():
                class_to_boxes[class_name].append(boxes)

        class_merged_boxes = defaultdict(list)
        for class_name, boxes_for_class in class_to_boxes.items():
            for batch_id in range(batch_size):
                boxes_for_current_batch = [box[batch_id] for box in boxes_for_class]

                size_of_box_is_zero = [box is None for box in boxes_for_current_batch]
                if all(size_of_box_is_zero):
                    # there is nothing in this image to merge
                    class_merged_boxes[class_name].append(None)
                    continue
                if drop_if_size_of_boxes_zero and class_name in class_names_to_merge and any(size_of_box_is_zero):
                    # one image does not contain anything, we are dropping!
                    class_merged_boxes[class_name].append(None)
                    continue
                if any(size_of_box_is_zero):
                    # no need to calculate IOUs we just keep the boxes with elements
                    for i, size_is_zeros in enumerate(size_of_box_is_zero):
                        if not size_is_zeros:
                            class_merged_boxes[class_name].append(boxes_for_class[i][batch_id])
                            break
                    continue
                num_sub_images = len(boxes_for_current_batch)
                if class_name not in class_names_to_merge or num_sub_images == 1:
                    # nothing to merge here, just keep the boxes
                    class_merged_boxes[class_name].append(numpy.concatenate(boxes_for_current_batch, axis=0))
                    continue

                assert num_sub_images >= 2, "The dict item 'keys_for_class_determination' in the config file must " \
                                            "contain at least one item."
                merged_boxes = self.merge_boxes(numpy.concatenate(boxes_for_current_batch), only_keep_overlapping_boxes)

                if len(merged_boxes) == 0:
                    class_merged_boxes[class_name].append(None)
                else:
                    class_merged_boxes[class_name].append(merged_boxes)

        # returns a dict of boxes for each class with shape {class_id: [num_boxes, 4]}
        return class_merged_boxes

    def determine_and_drop_boxes(self, class_boxes: Dict[str, List[Union[numpy.ndarray, None]]]) -> Dict[str, List[Union[numpy.ndarray, None]]]:
        # drop all printed text boxes that intersect with our handwriting boxes
        handwritten_name = 'handwritten_text'
        printed_name = 'printed_text'
        batch_handwriting_boxes = class_boxes[handwritten_name]
        batch_printed_boxes = class_boxes[printed_name]

        processed_boxes = {
            handwritten_name: [],
            printed_name: []
        }
        for handwriting_boxes, printed_boxes in zip(batch_handwriting_boxes, batch_printed_boxes):
            if handwriting_boxes is not None and printed_boxes is not None:
                ious = self.box_iou(handwriting_boxes, printed_boxes)
                # we delete a printed text box if it overlaps with any handwriting box
                ious_greater_zero = ious > 0
                printed_boxes = numpy.delete(printed_boxes, ious_greater_zero.nonzero()[1], axis=0)

            processed_boxes[handwritten_name].append(handwriting_boxes)
            processed_boxes[printed_name].append(printed_boxes)

        return processed_boxes

    def drop_too_small_boxes(self, class_boxes: Dict[str, List[Union[None, numpy.ndarray]]]) -> Dict[str, List[Union[numpy.ndarray]]]:
        adjusted_boxes = {}
        for class_name, batch_boxes in class_boxes.items():
            adjusted_batch_boxes = []
            for batch_id in range(len(batch_boxes)):
                boxes = batch_boxes[batch_id]
                if boxes is not None:
                    # a real box
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    boxes = numpy.delete(boxes, areas < self.min_class_box_area, axis=0)
                    if boxes.size == 0:
                        boxes = None
                adjusted_batch_boxes.append(boxes)
            adjusted_boxes[class_name] = adjusted_batch_boxes
        return adjusted_boxes

    def extract_contours_and_boxes(self, predicted_clusters: Dict[str, Dict[str, torch.Tensor]],
                                   image_ids_to_extract: List[str]) -> Dict[str, Dict[str, List[numpy.ndarray]]]:
        class_boxes_for_sub_images = {}
        for key_id in image_ids_to_extract:
            cluster_class_tensors = predicted_clusters[key_id]
            tensors_for_class = {}
            for class_name, class_tensor in cluster_class_tensors.items():
                if class_name == "background":
                    continue
                # expected shape: [batch_size, height, width]
                cluster_arrays = class_tensor.cpu().numpy().astype(numpy.uint8)
                # cluster_arrays = self.open_cluster_images(cluster_arrays)
                # then perform find contours and extract bounding boxes
                class_boxes = self.cluster_image_to_boxes(cluster_arrays)
                tensors_for_class[class_name] = class_boxes
            class_boxes_for_sub_images[key_id] = tensors_for_class
        return class_boxes_for_sub_images

    def classify_handwriting_boxes_directly_right_of_printed_text_as_printed_text(self, class_tensors: Dict[str, List[Union[numpy.ndarray, None]]]) -> Dict[str, List[Union[numpy.ndarray, None]]]:
        if any(all(box is None for box in class_boxes) for class_boxes in class_tensors.values()):
            # we can not drop anything
            return class_tensors
        printed_text_boxes = class_tensors['printed_text']
        handwriting_boxes = class_tensors['handwritten_text']
        for batch_id in range(len(printed_text_boxes)):
            handwriting_boxes_right_of_printed_boxes = self.boxes_right_of(handwriting_boxes[batch_id],
                                                                           printed_text_boxes[batch_id]).transpose()
            ious = self.box_iou(printed_text_boxes[batch_id], handwriting_boxes[batch_id])
            ious_greater_zero = ious > 0
            handwriting_boxes_directly_right_of_printed_boxes = numpy.bitwise_and(
                handwriting_boxes_right_of_printed_boxes,
                ious_greater_zero
            )
            handwriting_boxes_to_delete = numpy.add.reduce(handwriting_boxes_directly_right_of_printed_boxes,
                                                           axis=0).nonzero()
            new_printed_text_boxes = handwriting_boxes[batch_id][handwriting_boxes_to_delete].copy()
            handwriting_boxes[batch_id] = numpy.delete(handwriting_boxes[batch_id], handwriting_boxes_to_delete, axis=0)
            printed_text_boxes[batch_id] = numpy.concatenate([printed_text_boxes[batch_id], new_printed_text_boxes],
                                                             axis=0)

        return {
            "printed_text": printed_text_boxes,
            "handwritten_text": handwriting_boxes
        }

    def create_text_regions(self, predicted_clusters: Dict[str, Dict[str, torch.Tensor]],
                            batch_size: int) -> Dict[str, List[Union[None, numpy.ndarray]]]:
        class_boxes_for_sub_images = self.extract_contours_and_boxes(predicted_clusters,
                                                                     self.keys_for_class_determination)

        # now we handle each generated image individually
        class_boxes_for_sub_images_of_current_image = {
            sub_image_id: self.classify_handwriting_boxes_directly_right_of_printed_text_as_printed_text(class_tensors)
            for sub_image_id, class_tensors in class_boxes_for_sub_images.items()
        }
        if self.debug:
            for sub_image_id, boxes in class_boxes_for_sub_images_of_current_image.items():
                self.render_debug_boxes(boxes, f"after_classification_{sub_image_id}")

        # merge all boxes that have high overlap and are of the same class
        merged_boxes = self.merge_boxes_of_same_class_from_different_images(
            class_boxes_for_sub_images_of_current_image,
            batch_size,
            only_keep_overlapping_boxes=True,
            class_names_to_merge=('handwritten_text',),
            drop_if_size_of_boxes_zero=True,
        )
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after_handwriting_merging")

        merged_boxes = self.merge_boxes_of_same_class_from_same_image(merged_boxes)
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after_same_image_merging")
        merged_boxes = self.determine_and_drop_boxes(merged_boxes)
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after dropping of printed text")
        merged_boxes = self.drop_too_small_boxes(merged_boxes)
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after_small_dropping")
        return merged_boxes

    def merge_finegrained_segmentation(self, predicted_clusters: Dict[str, Dict[str, torch.Tensor]],
                                       batch_size: int) -> Dict[str, List[Union[numpy.ndarray, None]]]:
        boxes_for_sub_images = self.extract_contours_and_boxes(predicted_clusters,
                                                               self.keys_for_finegrained_segmentation)

        # for each image in our batch, we only keep those images where boxes of both steps overlap if one sub image
        # does not contain any boxes, but the other does, we do not keep these images -> they do not contain anything
        # useful
        merged_boxes = self.merge_boxes_of_same_class_from_different_images(
            boxes_for_sub_images,
            batch_size,
            only_keep_overlapping_boxes=True,
            drop_if_size_of_boxes_zero=True,
        )
        if self.debug:
            self.render_debug_boxes(merged_boxes, "after_finegrained_merging")

        return merged_boxes

    def classify_fine_grained_boxes(self,
                                    text_regions_per_class: Dict[str, List[Union[numpy.ndarray, None]]],
                                    fine_grained_boxes_per_class: Dict[str, List[Union[numpy.ndarray, None]]],
                                    fine_grained_class_name: str = 'printed_text') -> Dict[str, List[Union[None, numpy.ndarray]]]:
        assert len(text_regions_per_class) == len(fine_grained_boxes_per_class), "Num classes of text regions and " \
                                                                                 "fine grained boxes must be equal! "
        classified_boxes = defaultdict(list)
        # we always use fine-grained boxes of one class (default is printed text)
        fine_grained_boxes_batch = fine_grained_boxes_per_class[fine_grained_class_name]
        # sort the keys of text regions based on the order in color json file
        text_regions_per_class = dict(sorted(text_regions_per_class.items(), key=lambda x: self.class_id_map[x[0]]))
        for class_name, text_regions_batch in text_regions_per_class.items():
            for batch_id, (text_regions, fine_grained_boxes) in enumerate(zip(text_regions_batch,
                                                                              fine_grained_boxes_batch)):
                all_text_regions_empty = text_regions is None
                fine_grained_boxes_empty = fine_grained_boxes is None or len(fine_grained_boxes) == 0
                if all_text_regions_empty or fine_grained_boxes_empty:
                    # there is nothing in this image
                    classified_boxes[class_name].append(None)
                    continue

                ious_with_fine_grained_boxes = self.box_iou(text_regions, fine_grained_boxes)
                box_indices_to_keep = numpy.unique(ious_with_fine_grained_boxes.nonzero()[1])
                boxes_to_keep = numpy.take(fine_grained_boxes, box_indices_to_keep, axis=0)
                fine_grained_boxes_batch[batch_id] = numpy.delete(fine_grained_boxes_batch[batch_id],
                                                                  box_indices_to_keep, axis=0)
                classified_boxes[class_name].append(boxes_to_keep)
        if self.debug:
            self.render_debug_boxes(classified_boxes, "classified_boxes")
        return classified_boxes

    def render_segmentation_image(self,
                                  fine_grained_clusters: Dict[str, torch.Tensor],
                                  classified_boxes: Dict[str, List[Union[None, numpy.ndarray]]],
                                  batch_size: int,
                                  cluster_class_name: str = 'printed_text') -> numpy.ndarray:
        segmentation_images = []
        fine_grained_clusters = {class_name: clusters.cpu().numpy() for class_name, clusters in
                                 fine_grained_clusters.items()}
        for batch_id in range(batch_size):
            segmentation_image = numpy.zeros((self.image_size, self.image_size, 3), dtype=numpy.uint8)
            segmentation_image[:, :] = self.class_to_color_map['background']

            for class_name in fine_grained_clusters.keys():
                if class_name == 'background':
                    continue

                boxes_for_class = classified_boxes[class_name]
                image_boxes = boxes_for_class[batch_id]

                if image_boxes is None:
                    # there is nothing in this image
                    continue

                for box in image_boxes:
                    box = box.astype(numpy.int32)
                    box_to_fill = segmentation_image[box[1]:box[3], box[0]:box[2]]
                    mask_for_box = fine_grained_clusters[cluster_class_name][batch_id, box[1]:box[3], box[0]:box[2]]
                    box_to_fill[mask_for_box] = self.class_to_color_map[class_name]
                    segmentation_image[box[1]:box[3], box[0]:box[2]] = box_to_fill
            segmentation_images.append(segmentation_image)

        segmentation_images = numpy.stack(segmentation_images, axis=0)
        if self.debug:
            batch_size, height, width, num_channels = segmentation_images.shape
            extra = numpy.full((batch_size, self.max_debug_text_size, width, num_channels), 255, dtype='uint8')
            rgb_images = numpy.concatenate([segmentation_images, extra], axis=1)
            self.debug_images['result'] = rgb_images
        return segmentation_images

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
                if (box_heights > max_extent).any() or (box_widths > max_extent).any():
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
        text_regions_per_feature_map = self.create_text_regions(predicted_clusters, batch_size)
        fine_grained_boxes_per_feature_map = self.merge_finegrained_segmentation(predicted_clusters, batch_size)
        classified_boxes = self.classify_fine_grained_boxes(text_regions_per_feature_map,
                                                            fine_grained_boxes_per_feature_map)
        image_ids_to_drop = self.determine_images_to_drop(classified_boxes)

        segmentation_images = self.render_segmentation_image(
            predicted_clusters[self.keys_for_finegrained_segmentation[-1]],
            classified_boxes,
            batch_size
        )
        return segmentation_images, image_ids_to_drop


if __name__ == "__main__":
    # test the dilate implementation
    im = numpy.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=numpy.float32)
    nd_kernel = numpy.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=numpy.uint8)
    # nd_kernel = numpy.array([[1, 1, 0 ]], dtype=numpy.uint8)
    print(cv2.dilate(im, nd_kernel))
    # [[1. 1. 1. 0. 0.]
    #  [1. 1. 1. 1. 0.]
    #  [1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1.]
    #  [0. 0. 1. 1. 1.]]
    im_tensor = torch.Tensor(numpy.expand_dims(numpy.expand_dims(im, 0), 0))  # size:(1, 1, 5, 5)
    nd_kernel_tensor = torch.Tensor(numpy.expand_dims(numpy.expand_dims(nd_kernel, 0), 0))  # size: (1, 1, 3, 3)
    torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, nd_kernel_tensor, padding=(1, 1)), 0, 1)
    assert numpy.allclose(cv2.dilate(im, nd_kernel),
                          Segmenter.open_cluster_images(im_tensor, kernel=nd_kernel.astype(numpy.float32)).numpy())
    print(torch_result)
    # tensor([[[[1., 1., 1., 0., 0.],
    #           [1., 1., 1., 1., 0.],
    #           [1., 1., 1., 1., 1.],
    #           [1., 1., 1., 1., 1.],
    #           [0., 0., 1., 1., 1.]]]])
