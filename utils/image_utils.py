from typing import List

import cv2
import numpy
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageClass


def render_text_on_image(text: str, image: Image) -> Image:
    draw = ImageDraw.Draw(image)

    font = draw.getfont()
    text_size = draw.textsize(text, font=font)
    text_location = (image.width - text_size[0], image.height - text_size[1], image.width, image.height)
    draw.rectangle(text_location, fill=(255, 255, 255, 128))
    draw.text(text_location[:2], text, font=font, fill=(0, 255, 0))

    return image


def pil_image_to_opencv(pil_image: ImageClass) -> numpy.ndarray:
    if pil_image.mode == "RGB":
        return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
    elif pil_image.mode == "L":
        return numpy.array(pil_image)
    else:
        raise NotImplementedError


def opencv_image_to_pil(opencv_image: numpy.ndarray) -> ImageClass:
    return Image.fromarray(opencv_image)


def resize_image(image: ImageClass, new_dimensions: List[int]) -> ImageClass:
    assert any([size > 0 for size in new_dimensions]), "One of the given resize dimensions has to be greater than 0."
    if new_dimensions[0] == -1:
        resize_factor = new_dimensions[1] / image.width
        new_dimensions[0] = int(image.height * resize_factor)
    elif new_dimensions[1] == -1:
        resize_factor = new_dimensions[0] / image.height
        new_dimensions[1] = int(image.width * resize_factor)
    return image.resize((new_dimensions[1], new_dimensions[0]))
