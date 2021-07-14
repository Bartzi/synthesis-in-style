import numpy
import pytest
from typing import Dict, List, Union

from segmentation.gan_segmenter import Segmenter


class MergeTestSegmenter(Segmenter):
    def __init__(self):
        # The functionality we want to test is independent of Segmenter state so we can override the initialization
        # function to avoid unnecessary overhead.
        pass


class TestBoxMerging:

    INPUT_BOXES_TWO_SUB_IMAGES = [
        # first sub-image
        [
            [0, 0, 10, 10],
            [20, 20, 30, 25]
        ],
        # second sub-image
        [
            [5, 5, 15, 15],
            [0, 40, 40, 50]
        ]
    ]
    RESULTING_BOXES_TWO_SUB_IMAGES = [
        # merged overlapping boxes
        [
            [0, 0, 15, 15],
        ],
        # non-overlapping boxes
        [
            [20, 20, 30, 25],
            [0, 40, 40, 50]
        ]
    ]

    INPUT_BOXES_THREE_SUB_IMAGES = [
        [
            [20, 20, 40, 40],
            [130, 145, 140, 160],
            [200, 200, 210, 220],
            [300, 310, 315, 315],
            [500, 500, 505, 505],
            [750, 740, 770, 760],
        ],
        [
            [10, 33, 25, 45],
            [100, 100, 120, 140],
            [138, 110, 150, 163],
            [205, 207, 215, 221],
            [410, 444, 418, 477],
            [500, 500, 505, 505],
            [600, 600, 605, 605],
        ],
        [
            [15, 25, 55, 30],
            [115, 130, 135, 150],
            [306, 312, 317, 318],
            [404, 420, 414, 469],
            [808, 888, 888, 898],
        ]
    ]
    RESULTING_BOXES_THREE_SUB_IMAGES = [
        [
            [10, 20, 55, 45],
            [100, 100, 150, 163],
            [200, 200, 215, 221],
            [300, 310, 317, 318],
            [404, 420, 418, 477],
            [500, 500, 505, 505]
        ],
        [
            [600, 600, 605, 605],
            [750, 740, 770, 760],
            [808, 888, 888, 898]
        ]
    ]

    INPUT_BOXES_NO_OVERLAP = [
        [
            [20, 20, 40, 40],
        ],
        [
            [100, 100, 120, 140],
        ],
        [
            [600, 600, 605, 605],
        ]
    ]

    INPUT_BOXES_ONE_SUB_IMAGE_NONE = [
        None,
        [
            [5, 5, 15, 15],
            [0, 40, 40, 50]
        ]
    ]
    RESULTING_BOXES_ONE_SUB_IMAGE_NONE = [
        [
            [5, 5, 15, 15],
            [0, 40, 40, 50]
        ],
        []
    ]

    INPUT_BOXES_ALL_SUB_IMAGES_NONE = [None, None]
    RESULTING_BOXES_ALL_SUB_IMAGES_NONE = [None, None]

    def _format_input(self, input_boxes: List[List[List[int]]]) -> Dict[str, Dict[str, List[numpy.ndarray]]]:
        box_dict = {}
        for i in range(len(input_boxes)):
            box_dict[str(i)] = {
                "printed_text": [
                    numpy.asarray(input_boxes[i]) if input_boxes[i] is not None else None
                ]
            }
        return box_dict

    def _format_result(self, result):
        if result[0] is None and result[1] is None:
            return {False: [None], True: [None]}
        resulting_boxes = {
            False: [numpy.asarray(result[0] + result[1])],
            True: [numpy.asarray(result[0])]
        }
        return resulting_boxes

    def _sort_boxes(self, boxes:numpy.ndarray) -> List[List[int]]:
        return sorted([[el for el in row] for row in boxes])

    def _results_equal(self, boxes_a: Union[numpy.ndarray, None], boxes_b: Union[numpy.ndarray, None]) -> bool:
        if boxes_a is None and boxes_b is None:
            return True
        return self._sort_boxes(boxes_a) == self._sort_boxes(boxes_b)

    @pytest.fixture(params=[
        (INPUT_BOXES_TWO_SUB_IMAGES, RESULTING_BOXES_TWO_SUB_IMAGES),
        (INPUT_BOXES_THREE_SUB_IMAGES, RESULTING_BOXES_THREE_SUB_IMAGES),
        (INPUT_BOXES_ONE_SUB_IMAGE_NONE, RESULTING_BOXES_ONE_SUB_IMAGE_NONE),
        (INPUT_BOXES_ALL_SUB_IMAGES_NONE, RESULTING_BOXES_ALL_SUB_IMAGES_NONE),
    ], ids=["two sub-images", "three sub-images", "one sub-image none", "all sub-images none"])
    def boxes(self, request):
        input_boxes = self._format_input(request.param[0])
        resulting_boxes = self._format_result(request.param[1])
        return input_boxes, resulting_boxes

    @pytest.fixture(params=[
        (INPUT_BOXES_TWO_SUB_IMAGES, RESULTING_BOXES_TWO_SUB_IMAGES),
        (INPUT_BOXES_THREE_SUB_IMAGES, RESULTING_BOXES_THREE_SUB_IMAGES),
    ])
    def single_sub_image_boxes(self, request):
        input_boxes = self._format_input(request.param[0])
        merged_input_boxes = {"printed_text": [numpy.concatenate(([v["printed_text"][0] for v in input_boxes.values()]))]}
        resulting_boxes = self._format_result(request.param[1])
        return merged_input_boxes, resulting_boxes

    @pytest.fixture(params=[
        (INPUT_BOXES_TWO_SUB_IMAGES, RESULTING_BOXES_TWO_SUB_IMAGES),
        (INPUT_BOXES_THREE_SUB_IMAGES, RESULTING_BOXES_THREE_SUB_IMAGES),
    ], ids=["two sub-images", "three sub-images"])
    def boxes_with_multiple_batches(self, request):
        input_boxes = self._format_input(request.param[0])
        for v in input_boxes.values():
            v["printed_text"].append(v["printed_text"][0] + 10)
        resulting_boxes = self._format_result(request.param[1])
        for k, v in resulting_boxes.items():
            resulting_boxes[k].append(v[0] + 10)

        return input_boxes, resulting_boxes

    @pytest.fixture
    def segmenter(self):
        return MergeTestSegmenter()

    @pytest.mark.parametrize("only_keep_overlapping_boxes", [True, False], ids=["only keep overlapping", "keep all"])
    def test_merging_multiple_sub_images(self, only_keep_overlapping_boxes, boxes, segmenter):
        input_boxes, resulting_boxes = boxes
        result = resulting_boxes[only_keep_overlapping_boxes][0]
        merged_boxes = segmenter.merge_boxes_of_same_class_from_different_images(input_boxes, 1,
                                                                                 only_keep_overlapping_boxes,
                                                                                 ("printed_text",))["printed_text"][0]
        assert self._results_equal(result, merged_boxes)

    @pytest.mark.parametrize("box_slice", [2, 3], ids=["two sub-images", "three sub-images"])
    def test_no_overlap(self, box_slice, segmenter):
        input_boxes = self._format_input(self.INPUT_BOXES_NO_OVERLAP[:box_slice])
        merged_boxes = segmenter.merge_boxes_of_same_class_from_different_images(input_boxes, 1, True,
                                                                                 ("printed_text",))["printed_text"][0]
        assert merged_boxes is None

    def test_merging_single_sub_image(self, single_sub_image_boxes, segmenter):
        input_boxes, resulting_boxes = single_sub_image_boxes
        merged_boxes = segmenter.merge_boxes_of_same_class_from_same_image(input_boxes)
        assert self._results_equal(merged_boxes["printed_text"][0], resulting_boxes[False][0])

    @pytest.mark.parametrize("only_keep_overlapping_boxes", [True, False], ids=["only keep overlapping", "keep all"])
    def test_merging_with_multiple_batches(self, only_keep_overlapping_boxes, boxes_with_multiple_batches, segmenter):
        input_boxes, resulting_boxes = boxes_with_multiple_batches
        result = resulting_boxes[only_keep_overlapping_boxes]
        batch_size = len(result)
        merged_boxes = segmenter.merge_boxes_of_same_class_from_different_images(input_boxes, batch_size,
                                                                                 only_keep_overlapping_boxes,
                                                                                 ("printed_text",))["printed_text"]
        for i in range(batch_size):
            assert self._results_equal(result[i], merged_boxes[i])
