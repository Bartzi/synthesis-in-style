import json
import os
from pathlib import Path
from typing import Dict, Callable

import numpy
import torch
import torch.nn.functional as F
from PIL import ImageColor
from torchvision.transforms import ToTensor

from pytorch_training.data.json_dataset import JSONDataset
from pytorch_training.data.utils import default_loader
from pytorch_training.images import is_image


class SegmentationDataset(JSONDataset):

    def __init__(self, json_file: str, root: str = None, transforms: Callable = None, loader: Callable = default_loader,
                 class_to_color_map_path: Path = None, background_class_name: str = 'background', image_size: int = None):

        with open(json_file) as f:
            self.image_data = json.load(f)
            self.image_data = [path for file_path in self.image_data if is_image(path:=file_path['file_name'])]

        self.root = root
        self.transforms = transforms
        self.loader = loader
        self.background_class_name = background_class_name
        self.image_size = image_size

        assert class_to_color_map_path is not None, "Segmentation Dataset requires a class_to_color_map"
        with class_to_color_map_path.open() as f:
            self.class_to_color_map = json.load(f)
            self.reversed_class_to_color_map = {v: k for k,v in self.class_to_color_map.items()}
        assert self.background_class_name in self.class_to_color_map, f"Background class name: {self.background_class_name} not found in class to color map"

    def segmentation_image_to_class_image(self, segmentation_image: numpy.ndarray) -> numpy.ndarray:
        class_id_map = {class_name: i for i, class_name in enumerate(self.class_to_color_map.keys())}
        class_image = numpy.full(segmentation_image.shape[:2], class_id_map[self.background_class_name], dtype=segmentation_image.dtype)

        for class_name, color in self.class_to_color_map.items():
            if class_name == self.background_class_name:
                continue
            class_id_image = numpy.full_like(class_image, class_id_map[class_name])
            class_image = numpy.where(
                numpy.multiply.reduce(segmentation_image[:, :] == ImageColor.getrgb(color), axis=2).astype('bool'),
                class_id_image,
                class_image
            )
        return class_image

    def class_image_to_tensor(self, class_image: numpy.ndarray) -> torch.Tensor:
        class_image = class_image[numpy.newaxis, ...]
        class_image = torch.from_numpy(class_image)
        if self.image_size is not None:
            with torch.no_grad():
                class_image = F.interpolate(class_image[None, ...], (self.image_size, self.image_size))[0]
        return class_image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.image_data[index]
        if self.root is not None:
            path = os.path.join(self.root, path)

        image = self.loader(path)
        input_image = image.crop((0, 0, image.width // 2, image.height))
        segmentation_image = image.crop((image.width // 2, 0, image.width, image.height))

        input_image = self.transforms(input_image)
        segmentation_image = self.segmentation_image_to_class_image(numpy.array(segmentation_image))
        segmentation_image = self.class_image_to_tensor(segmentation_image).type(torch.LongTensor)
        assert input_image.shape[-2:] == segmentation_image.shape[-2:], "Input image and segmentation shape should be the same!"

        return {
            "images": input_image,
            "segmented": segmentation_image
        }
