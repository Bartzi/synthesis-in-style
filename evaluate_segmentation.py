import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

from data.coco_dataset import COCODataset
from evaluation.segmentation_coco_evaluator import SegmentationCocoEvaluator, DocUFCNEvalFunc
from networks import load_weights
from networks.doc_ufcn.doc_ufcn import DocUFCN
from pytorch_training.reporter import Reporter
from utils.config import load_config
from utils.data_loading import build_data_loader, fill_plot_images
from visualization.segmentation_plotter import SegmentationPlotter


def adapt_config_for_debug(config: dict) -> dict:
    # config['batch_size'] = 1
    config['num_workers'] = 0
    return config


def main(args: argparse.Namespace):
    trained_model_file = Path(args.trained_model)
    root_dir = trained_model_file.parent.parent
    output_dir = root_dir / 'evaluation_results'
    output_dir.mkdir(exist_ok=True, parents=True)

    config = load_config(args.trained_model, None)
    if args.debug:
        config = adapt_config_for_debug(config)

    segmentation_network = DocUFCN(3, 3)
    segmentation_network = load_weights(segmentation_network, args.trained_model, key='segmentation_network')
    segmentation_network = segmentation_network.to(args.device)

    data_loader = build_data_loader(args.gt, config, False, shuffle_off=True, dataset_class=COCODataset, drop_last=False, collate_func=COCODataset.collate_func)

    evaluator = SegmentationCocoEvaluator(
        data_loader,
        None,
        DocUFCNEvalFunc(segmentation_network, args.device),
        args.device,
        coco_gt_file=args.gt,
    )
    reporter = Reporter()
    evaluator.evaluate(reporter)
    observation = reporter.get_observations()

    with (output_dir / f"results_{trained_model_file.stem}.json").open('w') as f:
        json.dump(observation, f, indent='\t')

    if args.debug:
        image_plot_dir = output_dir / 'debug_images'
        image_plot_dir.mkdir(exist_ok=True)

        image_plotter = SegmentationPlotter(
            fill_plot_images(data_loader, num_desired_images=len(data_loader.dataset), image_key='images'),
            [segmentation_network],
            image_plot_dir,
            class_to_color_map=Path(config['class_to_color_map']),
        )
        attrs = {'updater.iteration': 1}
        mocked_trainer = MagicMock(**attrs)
        image_plotter.run(mocked_trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a given segmentation model using COCO Metrics")
    parser.add_argument("trained_model", help="path to trained model")
    parser.add_argument("gt", help="path to gt to evaluate on")
    parser.add_argument("--image-root", help="path to image root, if root is not the same as location of the gt file")
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'], help="device to use")
    parser.add_argument("--debug", default=False, action='store_true', help="run in debug mode")

    main(parser.parse_args())
