import os
from pathlib import Path
from typing import Union, Dict, Iterable, Type, Callable, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from data.autoencoder_dataset import AutoencoderDataset
from latent_projecting import Latents
from networks import StyleganAutoencoder
from pytorch_training.data.json_dataset import JSONDataset
from pytorch_training.data.utils import default_loader
from pytorch_training.distributed import get_world_size, get_rank


def resilient_loader(path):
    try:
        return default_loader(path)
    except Exception:
        print(f"Could not load {path}")
        return Image.new('RGB', (256, 256))


def build_data_loader(image_path: Union[str, Path], config: dict, uses_absolute_paths: bool, shuffle_off: bool = False,
                      dataset_class: Type[JSONDataset] = AutoencoderDataset, loader_func: Callable = resilient_loader,
                      drop_last: bool = True, collate_func: Callable = None) -> DataLoader:
    transform_list = [
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * config['input_dim'], (0.5,) * config['input_dim'])
    ]
    transform_list = transforms.Compose(transform_list)

    dataset = dataset_class(
        image_path,
        root=os.path.dirname(image_path) if not uses_absolute_paths else None,
        transforms=transform_list,
        loader=loader_func,
    )

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=not shuffle_off)
        sampler.set_epoch(get_rank())

    if shuffle_off:
        shuffle = False
    else:
        shuffle = sampler is None

    loader = DataLoader(
        dataset,
        config['batch_size'],
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_func,
    )
    return loader


def build_latent_and_noise_generator(autoencoder: StyleganAutoencoder, config: Dict, seed=1) -> Iterable:
    torch.random.manual_seed(seed)
    while True:
        latent_code = torch.randn(config['batch_size'], config['latent_size'])
        noise = autoencoder.decoder.make_noise()
        yield Latents(latent_code, noise)


def fill_plot_images(data_loader: Iterable, num_desired_images: int = 16, image_key: str = 'images') -> List[torch.Tensor]:
    """
        Gathers images to be used with ImagePloter
    """
    image_list = []
    num_images = 0
    for batch in data_loader:
        for image in batch[image_key]:
            image_list.append(image)
            num_images += 1
            if num_images >= num_desired_images:
                return image_list
    raise RuntimeError(f"Could not gather enough plot images for display size {num_desired_images}.")
