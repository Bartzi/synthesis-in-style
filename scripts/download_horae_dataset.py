import argparse
import csv
import re
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path
from time import sleep
from typing import Tuple, Union, List, Dict

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.contrib import tenumerate


def download_file(url: str) -> requests.Response:
    with requests.Session() as session:
        session.mount(url, HTTPAdapter(max_retries=5))

        response = session.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Could not Download File from {url}")
    return response


class IIIFDownloader:

    BASE_URL_REGEX = re.compile(r'(?P<base_url>https://[-\w.]+)/(?P<location>[-\w./=%?&\\:()]+)$')

    def __init__(self, iiif_manifests: List[Dict[str, str]], max_worker: int = 3, max_image_size: int = 3000):
        self.num_max_workers = max_worker
        self.manifest_urls = [manifest['iiif manifest'] for manifest in iiif_manifests]
        self.max_image_size = max_image_size

    def download_and_save_image(self, url: str, dest_file_name: Path):
        match = self.BASE_URL_REGEX.fullmatch(url)
        assert match is not None, f"URL regex does not match on {url}"
        base_url = match.group('base_url')
        location = match.group('location')
        session = requests.Session()
        session.mount(base_url, HTTPAdapter(max_retries=5))

        response = session.get(f'{base_url}/{location}')

        if response.status_code != 200:
            raise RuntimeError(f"Response code {response.status_code} not 200")

        dest_file_name.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(BytesIO(response.content)) as the_image:
            if any(size > self.max_image_size for size in the_image.size):
                the_image.thumbnail((self.max_image_size, self.max_image_size))
            the_image.save(str(dest_file_name))

    def extract_zip(self, zip_file: Path, dest_dir: Path):
        with zipfile.ZipFile(zip_file, 'r') as zip_handle:
            files = zip_handle.infolist()
            for file_name in tqdm(files, desc="Extracting Zip Files", leave=False, unit='files'):
                zip_handle.extract(file_name, dest_dir)

    def download_and_extract(self, manifest_data: dict, dest_dir: Path, object_id: str) -> List[str]:
        errors = []
        try:
            canvas_data = self.get_canvases(manifest_data)
        except Exception as e:
            return [f"Could not parse manifest data of {object_id}. Reason: {repr(e)}"]

        for canvas_id, canvas in tenumerate(canvas_data, desc=f"download canvases for {object_id}", unit='image', leave=False):
            try:
                dest_path = dest_dir / f"{Path(canvas['label']).stem}.png"
                if not dest_path.exists():
                    self.download_and_save_image(canvas['url'], dest_path)
            except Exception as e:
                errors.append(f"Could not download/extract {object_id}/{canvas_id}. Reason: {repr(e)}")
        return errors

    def download_manifest(self, url: str) -> dict:
        response = download_file(url)

        json_data = response.json()
        return json_data

    def get_canvases(self, manifest_data: dict) -> List[dict]:
        all_canvases = []
        canvas_lists = manifest_data['sequences']
        image_id = 0
        for canvases in canvas_lists:
            for canvas_data in canvases['canvases']:
                canvas_images = []
                for image_data in canvas_data['images']:
                    image_resource = image_data['resource']
                    canvas = {
                        'label': image_resource.get('label', str(image_id)),
                        'url': image_resource['@id']
                    }
                    canvas_images.append(canvas)
                    image_id += 1
                all_canvases.extend(canvas_images)
        return all_canvases

    def get_object_id(self, metadata: Union[None,List[dict]]) -> Union[str, None]:
        if metadata is None:
            return None
        try:
            for meta_info in metadata:
                if meta_info['label'] == "Object ID":
                    return meta_info['value']
        except Exception:
            pass
        return None

    def download_loop(self, destination: Path, failure_file):
        failure_writer = csv.DictWriter(failure_file, fieldnames=['reason'])
        failure_writer.writeheader()
        def handle_finished_task(future):
            errors = future.result()

            for error in errors:
                failure_writer.writerow({"reason": error})
            failure_file.flush()

        with ThreadPoolExecutor(max_workers=self.num_max_workers) as executor:
            current_jobs = []
            for i, manifest_url in tenumerate(self.manifest_urls, unit='manifest'):
                try:
                    manifest_data = self.download_manifest(manifest_url)
                except Exception as e:
                    failure_writer.writerow({"reason": f"Could not download manifest from {manifest_url} reason {repr(e)}"})
                    continue

                object_id = self.get_object_id(manifest_data.get('metadata', None))
                if object_id is None:
                   object_id = str(i)
                dest_dir = destination / object_id

                submitted_job = executor.submit(self.download_and_extract, manifest_data, dest_dir, object_id)
                current_jobs.append(submitted_job)

                while len(current_jobs) >= self.num_max_workers:
                    done_jobs = []
                    for idx, future in enumerate(current_jobs):
                        done = future.done()
                        if done:
                            done_jobs.append(idx)
                            handle_finished_task(future)

                    if len(done_jobs) > 0:
                        for done_id in sorted(done_jobs, reverse=True):
                            current_jobs.pop(done_id)
                    else:
                        sleep(60)

            for future in current_jobs:
                handle_finished_task(future)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to download the images of the HORAE dataset")
    parser.add_argument("destination", help="Path to dir where downloaded images shall be saved")
    parser.add_argument("--csv", default="https://raw.githubusercontent.com/oriflamms/HORAE/master/Corpus/BooksOfHours_MSS_Manifests.csv", help="URL of csv with data for download")
    parser.add_argument("--num-worker", type=int, default=3, help="number of parallel download threads")
    parser.add_argument("--max-image-size", type=int, default=3000, help="max size of largest image side in pixels")

    args = parser.parse_args()

    csv_response = download_file(args.csv)
    with StringIO(csv_response.text) as f:
        reader = csv.DictReader(f, delimiter=';')
        manifests_for_download = [l for l in reader]

    downloader = IIIFDownloader(manifests_for_download, max_worker=args.num_worker, max_image_size=args.max_image_size)

    destination = Path(args.destination)
    destination.mkdir(exist_ok=True, parents=True)

    with open(destination / 'failures.csv', 'w', newline='') as failure_csv:
        downloader.download_loop(destination, failure_csv)
