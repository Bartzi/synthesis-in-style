import numpy as np
import pytest
from PIL import Image

from evaluation.segmentation_diva_evaluator import DivaLayoutAnalysisEvaluator


def split_img(img):
    assert img.width % 2 == 0
    new_width = int(img.width / 2)
    cropped_img = img.crop((new_width, 0, img.width, img.height))

    return cropped_img


class TestDivaLayoutAnalysisEvaluator:

    @pytest.fixture(params=[
        "testdata/diva_layout_analysis/printed.png",
        "testdata/diva_layout_analysis/hw.png",
        "testdata/diva_layout_analysis/both.png",
        "testdata/diva_layout_analysis/no.png"
    ])
    def img(self, request):
        img = split_img(Image.open(request.param))
        return img

    def test_diva_layout_analysis_for_img(self, img):
        diva_img = DivaLayoutAnalysisEvaluator.segmented_image_to_diva(img)
        diva_img_array = np.asarray(diva_img)

        # Check that only blue channel is used for DIVA labels
        assert np.count_nonzero(diva_img_array[:, :, 0]) == 0 and np.count_nonzero(diva_img_array[:, :, 1]) == 0

        # Check that the DIVA image contains only zeroes or labels for "main text body"
        unique_values = list(np.unique(diva_img_array))
        num_unique_values = len(unique_values)
        if num_unique_values == 2:
            assert unique_values == [0, 8]
        elif num_unique_values == 1:
            assert unique_values == [0] or unique_values == [8]
        else:
            assert False

        # Check that labels are only at positions where the original image has colored pixels/where text was
        # detected
        diva_img_nonzero = np.nonzero(np.sum(diva_img_array, axis=2))
        img_array = np.asarray(img)
        img_nonzero = np.nonzero(np.sum(img_array, axis=2))

        assert np.array_equal(diva_img_nonzero[0], img_nonzero[0]) and np.array_equal(diva_img_nonzero[1],
                                                                                      img_nonzero[1])
