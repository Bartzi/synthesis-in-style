import argparse
import json
from pathlib import Path

from pytorch_training.images import is_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")

    args = parser.parse_args()

    dir = Path(args.dir)
    all_images = [str(f.relative_to(dir)) for f in dir.glob('**/*') if is_image(f)]
    with (dir / 'test.json').open('w') as f:
        json.dump(all_images, f)
