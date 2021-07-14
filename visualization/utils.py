from typing import Dict

import numpy
import torch
from PIL import ImageColor


def network_output_to_color_image(network_outputs: torch.Tensor, class_to_color_map: Dict) -> torch.Tensor:
    batch_size, num_classes, height, width = network_outputs.shape
    assert num_classes == len(
        class_to_color_map), f"Number of predicted classes and expected classes does not match {num_classes} vs {len(class_to_color_map)}"
    segmentation_images = numpy.zeros((batch_size, height, width, 3), dtype=numpy.uint8)
    segmentation_images[:, :] = ImageColor.getrgb(class_to_color_map['background'])
    segmentation_images = torch.from_numpy(segmentation_images).to(network_outputs.device)

    for class_id, color in enumerate(class_to_color_map.values()):
        colorization_mask = network_outputs[:, class_id, ...] > 0
        segmentation_images[colorization_mask] = torch.from_numpy(
            numpy.asarray(ImageColor.getrgb(color), numpy.uint8)).to(network_outputs.device)

    segmentation_images = (segmentation_images / 255) * 2 - 1
    return segmentation_images.permute(0, 3, 1, 2)
