import torch
from PIL import Image
from torchvision import transforms


class DivaLayoutAnalysisEvaluator:
    @staticmethod
    def segmented_image_to_diva(img: Image) -> Image:
        # Colors in segmented input images
        # printed text: 255, 0, 0
        # handwritten (hw) text: 0, 0, 255
        # background: 0, 0, 0

        img_tensor = transforms.ToTensor()(img)
        assert img_tensor.ndim == 3, "The input has to be a single RGB-image"
        diva_array = torch.zeros((3, *img_tensor.shape[1:]), dtype=torch.uint8)

        for height_idx in range(img.height):
            for width_idx in range(img.width):
                pixel_value = img_tensor[:, width_idx, height_idx]
                if pixel_value[0] == 1.0 or pixel_value[2] == 1.0:
                    diva_array[2, width_idx, height_idx] = 0x000008

        diva_img = transforms.ToPILImage()(diva_array)
        return diva_img
