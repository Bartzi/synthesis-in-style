import argparse
import functools
import json
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Tuple, List

import numpy
import torch
from PIL import Image
from pytorch_training.images import make_image
from tqdm import tqdm

from latent_projecting import Latents
from networks import StyleganAutoencoder, TwoStemStyleganAutoencoder, load_autoencoder_or_generator
from segmentation.benchmark_segmenter import BenchmarkSegmenter, TextLineBenchmarkSegmenter
from segmentation.evaluation.coco_gt import iter_through_images_in, COCOGtCreator
from segmentation.full_image_horae_segmenter import FullImageSegmenter
from segmentation.gan_segmenter import Segmenter
from segmentation.handwriting_printed_text_segmenter import HandwritingAndPrintedTextSegmenter
from utils.config import load_config, get_root_dir_of_checkpoint
from utils.data_loading import build_latent_and_noise_generator


def generate_images(batch: Union[Latents, dict], autoencoder: Union[StyleganAutoencoder, TwoStemStyleganAutoencoder],
                    device: str = 'cuda', mean_latent: torch.Tensor = None) -> Tuple[
    Dict[int, torch.Tensor], torch.Tensor]:
    if not isinstance(batch, Latents):
        batch = {k: v.to(device) for k, v in batch.items()}
        latents = autoencoder.encode(batch['input_image'])
    else:
        latents = batch.to(device)

    with torch.no_grad():
        generated_image, intermediate_activations = autoencoder.decoder(
            [latents.latent],
            input_is_latent=False if isinstance(batch, Latents) else autoencoder.is_wplus(latents),
            noise=latents.noise,
            return_intermediate_activations=True,
            truncation=0.7 if mean_latent is not None else 1,
            truncation_latent=mean_latent
        )
        return intermediate_activations, generated_image


def load_class_label_map(base_dir: Path, num_clusters: int) -> Dict[str, Dict[str, list]]:
    map_file_name = base_dir / f"merged_classes_{num_clusters}.json"
    with map_file_name.open() as f:
        class_label_map = json.load(f)

    inverted_class_label_map = {}
    for key, sub_label_map in class_label_map.items():
        inverted_sub_label_map = defaultdict(list)
        for sub_key, label_name in sub_label_map.items():
            inverted_sub_label_map[label_name].append(int(sub_key))
        inverted_class_label_map[key] = inverted_sub_label_map
    return inverted_class_label_map


def check_sanity_of_class_label_map(class_label_map: Dict[str, Dict[str, list]], creation_config: Dict) -> Dict:
    relevant_keys = creation_config["keys_for_class_determination"] \
                    + creation_config["keys_for_finegrained_segmentation"] \
                    + [key for key_list in creation_config["keys_to_merge"].values() for key in key_list]
    relevant_keys = set(relevant_keys)
    color_keys = list(creation_config["class_to_color_map"].keys())
    unlabelled_clusters = {}
    for relevant_key in relevant_keys:
        for class_label in class_label_map[relevant_key]:
            if class_label not in color_keys:
                if relevant_key not in unlabelled_clusters:
                    unlabelled_clusters[relevant_key] = []
                unlabelled_clusters[relevant_key].append(class_label)
    return unlabelled_clusters


def save_image(image: numpy.ndarray, image_id: int, base_dir: Path, name_format: str = "{id}.png"):
    save_sub_folder_1 = str(image_id // 1000)
    save_sub_folder_2 = str(image_id // 100000)
    dest_file_name = base_dir / save_sub_folder_2 / save_sub_folder_1 / name_format.format(id=image_id)
    dest_file_name.parent.mkdir(exist_ok=True, parents=True)
    image = Image.fromarray(image)
    image.save(str(dest_file_name))


def save_generated_images(generated_images: numpy.ndarray, semantic_segmentation_images: numpy.ndarray, batch_id: int,
                          base_dir: Path, num_images: int):
    images = numpy.concatenate([generated_images, semantic_segmentation_images], axis=2)

    for idx, image in enumerate(images):
        image_id = batch_id + idx
        save_image(image, image_id, base_dir, name_format=f"{{id:0{max(4, len(str(num_images)))}d}}.png")


def save_debug_images(debug_images: Dict[str, List[numpy.ndarray]], iteration: int, base_dir: Path):
    for batch_id in range(len(list(debug_images.values())[0])):
        image = numpy.concatenate([images[batch_id] for images in debug_images.values()], axis=1)
        image_id = iteration + batch_id
        save_image(image, image_id, base_dir, name_format=f"{{id:04d}}_debug.png")


def build_dataset(args: argparse.Namespace, creation_config: Dict):
    config = load_config(args.checkpoint, None)
    config['batch_size'] = args.batch_size
    autoencoder = load_autoencoder_or_generator(args, config)

    data_loader = build_latent_and_noise_generator(autoencoder, config, seed=creation_config['seed'])

    image_save_base_dir, semantic_segmentation_base_dir = get_base_dirs(args)

    if creation_config['segmenter_type'] == 'benchmark':
        segmenter_class = functools.partial(BenchmarkSegmenter, keys_to_merge=creation_config['keys_to_merge'])
    elif creation_config['segmenter_type'] == 'full_image':
        segmenter_class = functools.partial(FullImageSegmenter, keys_to_merge=creation_config['keys_to_merge'])
    elif creation_config['segmenter_type'] == 'hw_printed':
        assert 'only_keep_overlapping_boxes' in creation_config, 'The key "only_keep_overlapping_boxes" must be ' \
                                                                 'specified in the config file.'
        segmenter_class = functools.partial(HandwritingAndPrintedTextSegmenter,
                                            keys_to_merge=creation_config['keys_to_merge'],
                                            only_keep_overlapping_boxes=creation_config['only_keep_overlapping_boxes'])
    elif creation_config['segmenter_type'] == 'textline_benchmark':
        segmenter_class = TextLineBenchmarkSegmenter
    else:
        segmenter_class = Segmenter

    segmenter = segmenter_class(
        creation_config['keys_for_class_determination'],
        creation_config['keys_for_finegrained_segmentation'],
        semantic_segmentation_base_dir,
        args.num_clusters,
        config['image_size'],
        creation_config['class_to_color_map'],
        debug=args.debug
    )
    class_label_map = load_class_label_map(semantic_segmentation_base_dir, args.num_clusters)
    unlabelled_clusters = check_sanity_of_class_label_map(class_label_map, creation_config)
    assert not unlabelled_clusters, f"Some of the activation maps were not labelled completely (map_id: cluster_id):\n" \
                                    f"{unlabelled_clusters}"
    data_iter = iter(data_loader)

    if args.truncate:
        mean_latent = autoencoder.decoder.mean_latent(4096)
    else:
        mean_latent = None

    with tqdm(total=args.num_images) as pbar:
        while pbar.n < args.num_images:
            batch = next(data_iter)
            activations, generated_images = generate_images(batch, autoencoder, mean_latent=mean_latent)
            semantic_label_images, image_ids_to_drop = segmenter.create_segmentation_image(activations, class_label_map)

            generated_images = make_image(generated_images)
            if not args.debug:
                generated_images = numpy.delete(generated_images, image_ids_to_drop, axis=0)
                semantic_label_images = numpy.delete(semantic_label_images, image_ids_to_drop, axis=0)

            num_generated_images = len(semantic_label_images)
            if num_generated_images > 0:
                save_generated_images(generated_images, semantic_label_images, pbar.n, image_save_base_dir,
                                      args.num_images)
            if args.debug:
                save_debug_images(segmenter.debug_images, pbar.n, image_save_base_dir)
                pbar.update(args.batch_size)
            else:
                pbar.update(num_generated_images)


def get_base_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    base_dir = get_root_dir_of_checkpoint(args.checkpoint)
    semantic_segmentation_base_dir = base_dir / 'semantic_segmentation'
    if args.save_to is None:
        image_save_base_dir = base_dir / "generated_images"
    else:
        image_save_base_dir = Path(args.save_to)
    image_save_base_dir.mkdir(parents=True, exist_ok=True)
    return image_save_base_dir, semantic_segmentation_base_dir


def create_dataset_json_data(image_paths: List[Path], image_root: Path, gt_creator: COCOGtCreator) \
                             -> Tuple[List[dict], bool]:
    dataset_data = []
    try:
        for image_path in tqdm(image_paths, desc='create dataset json data', leave=False, unit='img'):
            with Image.open(str(image_path)) as the_image:
                data = {
                    "file_name": str(image_path.relative_to(image_root)),
                }
                data.update(gt_creator.determine_classes_in_image(the_image))
            dataset_data.append(data)
    except:
        print(traceback.format_exc())
        return dataset_data, False

    return dataset_data, True


def main(args: argparse.Namespace):
    with open(args.config) as f:
        config = json.load(f)

    if not args.only_create_train_val_split:
        build_dataset(args, config)

    if args.debug:
        # no need for gt if only creating debug images
        return

    image_save_base_dir, _ = get_base_dirs(args)
    generated_images = list(iter_through_images_in(image_save_base_dir))
    random.seed(config['seed'])
    random.shuffle(generated_images)

    coco_creator = COCOGtCreator(config['class_to_color_map'], image_root=image_save_base_dir)

    # 10% validation data
    split_index = int(len(generated_images) * 0.9)
    training_images = generated_images[:split_index]
    validation_images = generated_images[split_index:]

    training_gt, success = create_dataset_json_data(training_images, image_save_base_dir, coco_creator)
    train_filename = image_save_base_dir / ('train.json' if success else 'train.json.part')
    with train_filename.open('w') as f:
        json.dump(training_gt, f)
    del training_gt

    validation_gt, success = create_dataset_json_data(validation_images, image_save_base_dir, coco_creator)
    val_filename = image_save_base_dir / ('val.json' if success else 'val.json.part')
    with val_filename.open('w') as f:
        json.dump(validation_gt, f)
    del validation_gt

    coco_gt = coco_creator.create_coco_gt_from_image_paths(validation_images)
    with (image_save_base_dir / 'coco_gt.json').open('w') as f:
        json.dump(coco_gt, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to trained autoencoder/generator for dataset creation")
    parser.add_argument("num_clusters", type=int, help="number of classes labelled with semantic labeller")
    parser.add_argument("config", help="path to json file containing config for generation")
    parser.add_argument("-n", "--num-images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("-s", "--save-to",
                        help="path where to save generated images (default is save in dir of run of used checkpoint)")
    parser.add_argument("-b", "--batch-size", default=10, type=int, help="batch size for generation of images on GPU")
    parser.add_argument("-d", "--device", default='cuda',
                        help="CUDA device to use, either any (cuda) or the id of the device")
    parser.add_argument("--only-create-train-val-split", action='store_true', default=False,
                        help="do not create an entire dataset, rather use the save_path and build a train validation "
                             "split with according COCO GT")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="render debug output during image generation")
    parser.add_argument("--truncate", action='store_true', default=False, help="Use truncation trick during generation")

    main(parser.parse_args())
