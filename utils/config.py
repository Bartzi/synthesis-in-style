import argparse
import json
from pathlib import Path
from typing import Union

import yaml


def load_config(checkpoint_path: str = None, config: str = None) -> dict:
    if checkpoint_path is None and config is None:
        raise RuntimeError("You have to supply either checkpoint path or path to a config file!")

    if config is not None:
        with open(config) as f:
            config = json.load(f)
            if checkpoint_path is not None:
                config['stylegan_checkpoint'] = checkpoint_path
            assert config.get('stylegan_checkpoint', None) is not None

            return config

    config_dir = Path(checkpoint_path).parent.parent / 'config'
    original_config = config_dir / 'config.json'
    with open(original_config) as f:
        original_config = json.load(f)

    original_args = config_dir / 'args.json'
    with open(original_args) as f:
        original_args = json.load(f)

    original_config.update(original_args)

    return original_config


def load_yaml_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    for key in dir(args):
        if key.startswith("_"):
            continue
        config[key] = getattr(args, key)
    return config


def get_root_dir_of_checkpoint(checkpoint_file: Union[str, Path]) -> Path:
    if isinstance(checkpoint_file, str):
        checkpoint_file = Path(checkpoint_file)
    return checkpoint_file.parent.parent
