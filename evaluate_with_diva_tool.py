import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import List

from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class DIVAMetrics:
    ious: list
    recalls: list
    precisions: list
    f1_scores: list

    def mean(self):
        return {
            "iou": statistics.mean(self.ious),
            "recall": statistics.mean(self.recalls),
            "precision": statistics.mean(self.precisions),
            "f1_score": statistics.mean(self.f1_scores)
        }

    def save(self, destination: Path):
        destination.parent.mkdir(exist_ok=True, parents=True)
        with destination.open('w') as f:
            json.dump(self.mean(), f, indent='\t')


def main(args: argparse.Namespace):
    gt_xml_files = list(sorted(Path(args.gt_xml_dir).resolve().glob("**/*.xml")))
    predicted_xml_files = list(sorted(Path(args.pred_xml_dir).resolve().glob("**/*.xml")))
    gt_images = list(sorted(Path(args.gt_diva_image_dir).resolve().glob("**/*.png")))

    assert len(gt_xml_files) == len(predicted_xml_files) == len(gt_images), f"Number of files in each dir is not equal! {len(gt_xml_files)} vs. {len(predicted_xml_files)} vs. {len(gt_images)}"

    results = []
    for gt_xml, predicted_xml, gt_image in tqdm(zip(gt_xml_files, predicted_xml_files, gt_images), total=len(gt_images)):
        assert gt_xml.stem == predicted_xml.stem == gt_image.stem, f"Filenames are not equal. Does sorting work? {gt_xml} vs {predicted_xml} vs {gt_image}"

        run_diva_tool(args, gt_image, gt_xml, predicted_xml)
        try:
            results.extend(parse_diva_csv(predicted_xml))
        except FileNotFoundError:
            print(f"could not find result for {predicted_xml}")

    metrics = calculate_metrics(results)
    result_save_location = Path(args.pred_xml_dir).parent / 'eval_results.json'
    metrics.save(result_save_location)
    print(metrics.mean())


def calculate_metrics(results: List[dict]) -> DIVAMetrics:
    metrics = DIVAMetrics([], [], [], [])

    for result in results:
        iou = float(result['LinesIU'])
        recall = float(result['LinesRecall'])
        precision = float(result['LinesPrecision'])
        f1_score = float(result['LinesFMeasure'])
        if any(math.isnan(metric) for metric in [iou, recall, precision, f1_score]):
            continue

        metrics.ious.append(iou)
        metrics.recalls.append(recall)
        metrics.precisions.append(precision)
        metrics.f1_scores.append(f1_score)

    return metrics


def parse_diva_csv(xml: Path) -> List[dict]:
    csv_file_name = xml.parent / 'eval_CSVs' / f"{xml.stem}-results.csv"
    results = []
    with csv_file_name.open() as f:
        reader = csv.DictReader(f)
        for line in reader:
            line['file_name'] = str(xml)
            results.append(line)
    return results


def run_diva_tool(args, gt_image, gt_xml, predicted_xml):
    eval_dir = predicted_xml.parent / 'eval_CSVs'
    eval_dir.mkdir(exist_ok=True)

    if (eval_dir / f"{predicted_xml.stem}-results.csv").exists():
        return

    diva_args = [
        "java",
        "-jar",
        "LineSegmentationEvaluator.jar",
        "-csv",
        "-igt",
        str(gt_image),
        "-xgt",
        str(gt_xml),
        "-xp",
        str(predicted_xml),
        "-out",
        str(eval_dir.relative_to(predicted_xml.parent)),
        "-mt",
        str(args.matching_threshold)
    ]
    subprocess.run(diva_args, cwd=args.diva_tool_location, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate our PAGE XML predictions with the DIVA tool")
    parser.add_argument("gt_xml_dir", help="Path to dir with all GT xmls to evaluate")
    parser.add_argument("gt_diva_image_dir", help="Path to dir with all images in required DIVA GT image format")
    parser.add_argument("pred_xml_dir", help="Path to dir with all predicted PAGE XML Files")
    parser.add_argument("diva_tool_location", help="Path to location of diva line segmentation tool")
    parser.add_argument("-mt", "--matching-threshold", type=float, default=0.5, help="matching threshold for DIVA tool")

    main(parser.parse_args())
