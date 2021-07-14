import argparse
import json
import random
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple jsons together")
    parser.add_argument("destination", help="where to saved merged gt")
    parser.add_argument("jsons", nargs='+', help="path to jsons to merge")
    parser.add_argument("--balance", action='store_true', default=False, help="balance resulting dataset")
    parser.add_argument("--balance-factor", type=int, default=3, help="Allow number of items from file to be this times larger than the smallest items from a afile")

    args = parser.parse_args()

    data = []
    for json_file in args.jsons:
        with open(json_file) as f:
            data.append(json.load(f))

    lengths = [len(x) for x in data]
    if args.balance:
        min_length = min(lengths)
        max_length = min_length * args.balance_factor
    else:
        max_length = max(lengths)

    merged_data = []
    destination_path = Path(args.destination)
    for i, items in enumerate(data):
        base_path = Path(args.jsons[i]).parent
        for j in range(len(items[:max_length])):
            if isinstance(items[j], str):
                original_path = base_path / items[j]
                new_path = original_path.relative_to(destination_path.parent)
                merged_data.append(str(new_path))
            else:
                item = items[j]
                original_path = base_path / item['file_name']
                new_path = original_path.relative_to(destination_path.parent)
                item['file_name'] = str(new_path)
                merged_data.append(item)

    random.shuffle(merged_data)
    with destination_path.open('w') as f:
        json.dump(merged_data, f)
