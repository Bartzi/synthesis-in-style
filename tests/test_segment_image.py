import pytest
import torch

from segment_image import assemble_predictions


def get_image_size(predictions_non_overwriting):
    image_width = predictions_non_overwriting[-1]["bbox"][2]
    image_height = predictions_non_overwriting[-1]["bbox"][3]
    image_size = (image_width, image_height)

    return image_size


class TestSegmentImage:

    @pytest.fixture(params=[
        [5, 0, 10, 10],
        [0, 5, 10, 10]
    ])
    def predictions_different_classes(self, request):
        prediction_class_one = {
            "prediction": torch.cat((torch.ones((1, 10, 10)), torch.zeros((2, 10, 10)),), dim=0),
            "bbox": [0, 0, 10, 10]
        }
        prediction_class_two = {
            "prediction": torch.cat((torch.zeros((1, 10, 10)), torch.ones((1, 10, 10)), torch.zeros((1, 10, 10)),),
                                    dim=0),
            "bbox": request.param
        }
        return [prediction_class_one, prediction_class_two]

    @pytest.fixture(params=[
        [5, 0, 10, 10],
        [0, 5, 10, 10]
    ])
    def predictions_same_class(self, request):
        prediction_class_one = {
            "prediction": torch.cat((torch.full((1, 10, 10), 2.0), torch.zeros((2, 10, 10)),), dim=0),
            "bbox": [0, 0, 10, 10]
        }
        prediction_class_two = {
            "prediction": torch.cat((torch.ones((1, 10, 10)), torch.zeros((2, 10, 10)),), dim=0),
            "bbox": request.param
        }
        return [prediction_class_one, prediction_class_two]

    def test_assemble_predictions_different_classes(self, predictions_different_classes):
        image_size = get_image_size(predictions_different_classes)

        ap_accurate = assemble_predictions(predictions_different_classes, image_size)
        assert torch.all(ap_accurate != float("-inf"))
        assert torch.all(ap_accurate[0, :, :] == 1.0)
        assert torch.count_nonzero(ap_accurate[1, :, :]) == image_size[0] * image_size[1] // 2
        assert torch.all(ap_accurate[2, :, :] == 0.0)

    def test_assemble_predictions_same_class(self, predictions_same_class):
        image_size = get_image_size(predictions_same_class)

        ap_accurate = assemble_predictions(predictions_same_class, image_size)
        assert torch.all(ap_accurate != float("-inf"))
        assert torch.all(ap_accurate[0, :, :] == 2.0)
        assert torch.all(ap_accurate[1, :, :] == 0.0)
        assert torch.all(ap_accurate[2, :, :] == 0.0)

