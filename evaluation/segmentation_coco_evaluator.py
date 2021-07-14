import os
import re
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from textwrap import dedent
from typing import List

import cv2
import numpy
import pycocotools.mask
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from networks.doc_ufcn.doc_ufcn import DocUFCN
from pytorch_training import Reporter
from pytorch_training.extensions import Evaluator
from pytorch_training.extensions.evaluator import eval_mode
from segmentation.evaluation.coco_gt import COCOGtCreator


class DocUFCNEvalFunc:

    def __init__(self, network: DocUFCN, device: int, background_class_id: int = 0):
        self.network = network
        self.device = device
        self.background_class_id = background_class_id

    def segmentation_image_to_coco_annotations(self, segmentation_image: torch.Tensor, image_start_id: int, annotation_id: int) -> List[dict]:
        batch_size, num_classes, height, width = segmentation_image.shape
        annotations = []
        for image_id in range(batch_size):
            for class_id in range(num_classes):
                if class_id == self.background_class_id:
                    # no need to examine background
                    continue
                non_zero_pixels = segmentation_image[image_id, class_id] != 0
                if not non_zero_pixels.any():
                    continue

                non_zero_pixels = non_zero_pixels.cpu().numpy()
                rles = COCOGtCreator.extract_rle(non_zero_pixels)

                for rle in rles:
                    annotation = {
                        "image_id": image_start_id + image_id,
                        "category_id": class_id,
                        "segmentation": rle,
                        "score": float(segmentation_image[image_id, class_id, non_zero_pixels].mean())
                    }
                    annotations.append(annotation)
                    annotation_id += 1
        return annotations

    def __call__(self, batch: torch.Tensor, image_start_id: int, annotation_start_id: int) -> List[dict]:
        with torch.no_grad(), eval_mode(self.network):
            segmentation_result = self.network.predict(batch['images'])

        annotations = self.segmentation_image_to_coco_annotations(segmentation_result, image_start_id, annotation_start_id)
        return annotations


class SegmentationCocoEvaluator(Evaluator):

    def __init__(self, *args, coco_gt_file: Path = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert coco_gt_file is not None, "COCO GT file must not be none if segmentation evaluator is to be used"

        with open(os.devnull, 'w') as dev_null:
            with redirect_stdout(dev_null):
                self.coco_gt = COCO(coco_gt_file)

    def gather_coco_results(self, predictions: List[dict]) -> str:
        if len(predictions) == 0:
            return self.empty_output()

        with open(os.devnull, 'w') as dev_null:
            with redirect_stdout(dev_null):
                coco_detections = self.coco_gt.loadRes(predictions)
                coco_eval = COCOeval(self.coco_gt, coco_detections, 'segm')
                coco_eval.evaluate()
                coco_eval.accumulate()

        coco_output = StringIO()
        with redirect_stdout(coco_output):
            coco_eval.summarize()
        return coco_output.getvalue()

    @staticmethod
    def empty_output() -> str:
        return dedent(
            """
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.0
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.0
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.0
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.0
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.0
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.0
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.0
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.0
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.0
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.0
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.0
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.0
            """
        )

    def parse_coco_output(self, coco_result: str):
        """
        Example:
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.032257
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.085206
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.018478
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006291
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.023219
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.061838
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.026431
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.078792
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.100950
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.024902
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.064485
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.176208
        """
        results = []
        line_re = re.compile(r'.+\((?P<metric_name>[APR]{2})\) @\[ IoU=(?P<iou_range>[\d.: ]+) \| area=(?P<area>[\w ]+) \| maxDets=(?P<max_detections>[\d ]+) ] = (?P<result>[-\d.]+)$')
        for line in coco_result.split('\n'):
            if len(line) == 0:
                continue
            match = line_re.match(line)
            assert match is not None, f"Could not parse the following line during COCO evaluation result parsing: {line}"
            results.append(
                {key: match.group(key).strip() for key in ['metric_name', 'iou_range', 'area', 'max_detections', 'result']}
            )
        return results

    def evaluate(self, reporter: Reporter):
        predictions = []
        current_annotation_id = 0
        for batch_id, batch in enumerate(self.progress_bar()):
            images = batch['images'].to(self.device)
            batch_size = len(images)
            new_predictions = self.eval_func({'images': images}, batch_id * batch_size, current_annotation_id)
            current_annotation_id += len(new_predictions)
            predictions.extend(new_predictions)

        coco_eval_result = self.gather_coco_results(predictions)
        loggable_results = self.parse_coco_output(coco_eval_result)

        with reporter:
            for r in loggable_results:
                log_key = f"{r['metric_name']}@{r['iou_range']},area:{r['area']},@{r['max_detections']}dets"
                reporter.add_observation({log_key: r['result']}, prefix='evaluation')

        torch.cuda.empty_cache()

