"""Extract compressed video packet sizes, in bits, without re-encoding."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

from tqdm import tqdm


def grab_all_videos(input_dir):
    """Find videos recursively, including subject/sequence/view/video datasets."""
    extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
    return sorted(p for p in Path(input_dir).rglob('*') if p.suffix.lower() in extensions and p.is_file())


def extract_cover(input_file_path, output_dir, out_filename, ffprobe='ffprobe'):
    """Write headerless source/frame-index/size-bits CSV in packet (coding) order."""
    source = Path(input_file_path)
    result = subprocess.run([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=size', '-of', 'json', str(source),
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'ffprobe failed for {source}: {result.stderr.strip()}')
    packets = json.loads(result.stdout).get('packets', [])
    if not packets:
        raise ValueError(f'No video packets in {source}')
    destination = Path(output_dir) / f'{out_filename}.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.csv.tmp')
    try:
        with temporary.open('w', newline='') as handle:
            writer = csv.writer(handle)
            for index, packet in enumerate(packets):
                writer.writerow([str(source), index, int(packet['size']) * 8])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_covers(input_dir, output_dir, *, workers=8, ffprobe='ffprobe', overwrite=False):
    """Extract covers while retaining every directory below ``input_dir``."""
    source, output = Path(input_dir), Path(output_dir)
    videos = grab_all_videos(source)
    if not videos:
        raise ValueError(f'No videos found in {source}')
    destinations = [output / video.relative_to(source).with_suffix('.csv') for video in videos]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Multiple videos map to the same cover CSV')

    def process(pair):
        video, destination = pair
        if overwrite or not destination.exists():
            extract_cover(video, destination.parent, destination.stem, ffprobe)
        return destination

    with ThreadPoolExecutor(max_workers=workers) as executor:
        covers = list(tqdm(executor.map(process, zip(videos, destinations)),
                           total=len(videos), desc='Extracting covers', unit='video'))
    return covers
