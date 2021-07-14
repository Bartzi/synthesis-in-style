import argparse
import datetime
import functools
import multiprocessing
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.segmentation_dataset import SegmentationDataset
from networks import load_weights
from networks.doc_ufcn import get_doc_ufcn
from pytorch_training.data.caching_loader import CachingLoader
from pytorch_training.distributed import synchronize, get_rank, get_world_size
from pytorch_training.distributed.utils import strip_parallel_module
from pytorch_training.extensions import Snapshotter
from pytorch_training.extensions.logger import WandBLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.optimizer import GradientClipAdam
from pytorch_training.trainer import DistributedTrainer
from pytorch_training.triggers import get_trigger
from updater.segmentation_updater import SegmentationUpdater
from utils.config import load_yaml_config, merge_config_and_args
from utils.data_loading import build_data_loader, fill_plot_images, resilient_loader
from visualization.segmentation_plotter import SegmentationPlotter


def main(args: argparse.Namespace, rank: int, world_size: int):
    config = load_yaml_config(args.config)
    config = merge_config_and_args(config, args)

    class_to_color_map_path = Path(args.class_to_color_map)
    if args.cache_root is not None:
        loader_func = CachingLoader(Path(args.train_json).parent, args.cache_root, base_loader=resilient_loader)
    else:
        loader_func = resilient_loader

    dataset_class = functools.partial(SegmentationDataset, class_to_color_map_path=class_to_color_map_path,
                                      image_size=config['image_size'])
    train_data_loader = build_data_loader(args.train_json, config, False, dataset_class=dataset_class,
                                          loader_func=loader_func)

    segmentation_network_class = get_doc_ufcn(args.network)
    segmentation_network = segmentation_network_class(3, 3)
    if args.fine_tune is not None:
        load_weights(segmentation_network, args.fine_tune, key='segmentation_network')

    optimizer_opts = {
        'betas': (config['beta1'], config['beta2']),
        'weight_decay': config['weight_decay'],
        'lr': float(config['lr']),
    }
    optimizer = GradientClipAdam(segmentation_network.parameters(), **optimizer_opts)

    segmentation_network = segmentation_network.to('cuda')
    if world_size > 1:
        distributed = functools.partial(DDP, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False,
                                        output_device=rank)
        segmentation_network = distributed(segmentation_network.to('cuda'))

    updater = SegmentationUpdater(
        iterators={'images': train_data_loader},
        networks={'segmentation': segmentation_network},
        optimizers={'main': optimizer},
        device='cuda',
        copy_to_device=world_size == 1
    )

    if 'max_iter' in config:
        stop_trigger = (config['max_iter'], 'iteration')
    else:
        stop_trigger = (config['epochs'], 'epoch')

    trainer = DistributedTrainer(
        updater,
        stop_trigger=get_trigger(stop_trigger)
    )

    logger = WandBLogger(
        args.log_dir,
        args,
        config,
        os.path.dirname(os.path.realpath(__file__)),
        trigger=get_trigger((config['log_iter'], 'iteration')),
        master=rank == 0,
        project_name=args.wandb_project_name,
        entity=args.wandb_entity,
        run_name=args.log_name,
    )

    if args.validation_json is not None:
        val_data_loader = build_data_loader(args.validation_json, config, False, shuffle_off=True,
                                            dataset_class=dataset_class, loader_func=loader_func, drop_last=False)

        # evaluator = SegmentationCocoEvaluator(
        #     val_data_loader,
        #     logger,
        #     DocUFCNEvalFunc(
        #         strip_parallel_module(segmentation_network),
        #         rank
        #     ),
        #     rank,
        #     trigger=get_trigger((1, 'epoch')),
        #     coco_gt_file=args.coco_gt,
        # )
        # trainer.extend(evaluator)

    if rank == 0:
        snapshotter = Snapshotter(
            {
                'segmentation_network': strip_parallel_module(segmentation_network),
                'optimizer': optimizer,
            },
            args.log_dir,
            trigger=get_trigger((config['snapshot_save_iter'], 'iteration'))
        )
        trainer.extend(snapshotter)

        plot_data_loader = val_data_loader if args.validation_json is not None else train_data_loader
        plot_images = fill_plot_images(plot_data_loader, image_key='images', num_desired_images=config['display_size'])
        label_images = fill_plot_images(plot_data_loader, image_key='segmented',
                                        num_desired_images=config['display_size'])
        image_plotter = SegmentationPlotter(
            plot_images,
            [strip_parallel_module(segmentation_network)],
            args.log_dir,
            trigger=get_trigger((config['image_save_iter'], 'iteration')),
            plot_to_logger=True,
            class_to_color_map=class_to_color_map_path,
            label_images=label_images
        )
        trainer.extend(image_plotter)

    schedulers = {
        "encoder": CosineAnnealingLR(optimizer, trainer.num_iterations, eta_min=1e-8)
    }
    lr_scheduler = LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))
    trainer.extend(lr_scheduler)

    trainer.extend(logger)

    synchronize()
    print("Starting training")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a network for semantic segmentation of documents")
    parser.add_argument("config", help="path to config with common train settings, such as LR")
    parser.add_argument("--images", dest="train_json", required=True, help="Path to json file with train images")
    parser.add_argument("--val-images", dest="validation_json", help="path to json file with validation images")
    parser.add_argument("--coco-gt", help="PAth to COCO GT required, if you set validation images")
    parser.add_argument('-l', '--log-dir', default='training', help="outputs path")
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mpi-backend', default='gloo', choices=['nccl', 'gloo'],
                        help="MPI backend to use for interprocess communication")
    parser.add_argument("--class-to-color-map", default="handwriting_colors.json",
                        help="path to json file with class color map")
    parser.add_argument("-c", "--cache-root",
                        help="path to a folder where you want to cache images on the local file system")
    parser.add_argument("--network", default='base', help="type of network to use")
    parser.add_argument("--fine-tune", help="Path to model to finetune from")
    parser.add_argument("--wandb-project-name", default="Semantic Segmentation for Document Images",
                        help="The project name of the WandB project")
    parser.add_argument("--wandb-entity", help="The name of the WandB entity")

    parsed_args = parser.parse_args()
    parsed_args.log_dir = os.path.join('logs', parsed_args.log_dir, parsed_args.log_name, datetime.datetime.now().isoformat())

    # if parsed_args.validation_json is not None:
    #     assert parsed_args.coco_gt is not None, "If you want to validate,you also have to supply a COCO gt file"

    if torch.cuda.device_count() > 1:
        multiprocessing.set_start_method('forkserver')
        torch.cuda.set_device(parsed_args.local_rank)
        torch.distributed.init_process_group(backend=parsed_args.mpi_backend, init_method='env://')
        synchronize()

    main(parsed_args, get_rank(), get_world_size())
