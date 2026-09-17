"""Collate per-video covers into aligned examples, subject labels and provenance."""

import csv
import json
from pathlib import Path

import numpy as np


def cover_labels(relative: Path) -> tuple[str, str, str]:
    """Return subject, condition sequence, and view from a cover path.

    Covers mirror the video tree. CASIA video exports historically used
    ``subject/sequence/view.csv``; supplied video datasets use
    ``subject/condition-sequence/view-angle/video.csv``. A leading dataset
    directory is also accepted when collating multiple dataset roots together.
    """
    parts = relative.parts
    if relative.suffix.lower() != '.csv':
        raise ValueError(f'Expected a cover CSV path: {relative}')
    if len(parts) == 3:
        subject, sequence, filename = parts
        return subject, sequence, Path(filename).stem
    if len(parts) == 4:
        dataset, subject, sequence, filename = parts
        # The supplied CASIA roots may themselves contain legacy exports. This
        # otherwise ambiguous four-part form is a leading dataset directory
        # followed by subject/sequence/view.csv.
        if dataset in {'casiasilhouette', 'casiabrgb'}:
            return subject, sequence, Path(filename).stem
        subject, sequence, view, _filename = parts
        return subject, sequence, view
    if len(parts) == 5:
        dataset, subject, sequence, view, _filename = parts
        # A numeric first component is a CASIA subject ID, not a dataset root;
        # accepting it here would silently treat an extra directory in a direct
        # modern layout as a dataset prefix.
        if dataset.isdigit():
            raise ValueError(
                'Expected subject/condition-sequence/view-angle/video.csv; '
                f'unexpected directory below subject {dataset}: {relative}'
            )
        return subject, sequence, view
    raise ValueError(
        'Expected subject/sequence/view.csv, '
        'subject/condition-sequence/view-angle/video.csv, or a leading dataset '
        f'directory followed by that layout: {relative}'
    )


def collate_into_matrix(covers_path, design_matrix_path, *, frame_num=3000,
                        short_policy='pad', cover_files=None):
    root, output = Path(covers_path), Path(design_matrix_path)
    files = sorted(root.rglob('*.csv') if cover_files is None else cover_files)
    if not files:
        raise ValueError(f'No cover CSVs found in {root}')
    rows, metadata, short = [], [], []
    for path in files:
        relative = path.relative_to(root)
        subject, sequence, view = cover_labels(relative)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            records = list(csv.reader(handle))
        sizes = np.asarray([int(record[2]) for record in records], dtype=np.int64)
        if not len(sizes) or np.any(sizes <= 0):
            raise ValueError(f'Empty or invalid cover: {path}')
        length = len(sizes)
        if length < frame_num:
            short.append(str(relative))
            if short_policy in ('error', 'drop'):
                continue
            if short_policy == 'pad':
                sizes = np.pad(sizes, (0, frame_num - length))
            elif short_policy == 'repeat':
                sizes = np.resize(sizes, frame_num)
            else:
                raise ValueError(f'Unknown short policy: {short_policy}')
        rows.append(sizes[:frame_num])
        metadata.append([len(rows) - 1, subject, sequence, view,
                         str(relative), records[0][0], length, min(length, frame_num)])
    if short and short_policy == 'error':
        raise ValueError(f'{len(short)}/{len(files)} videos have fewer than {frame_num} frames. '
                         'Choose --short-policy pad, repeat, or drop explicitly. Covers were retained.')
    if not rows:
        raise ValueError('No examples remain after filtering; covers were retained.')
    matrix = np.stack(rows)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'X.npy', matrix)
    np.save(output / 'Y.npy', np.asarray([row[1] for row in metadata]))
    np.savetxt(output / 'X.csv', matrix, delimiter=',', fmt='%d',
               header=','.join(f'size{i}' for i in range(frame_num)), comments='')
    with (output / 'Y.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['subject'])
        writer.writerows([row[1]] for row in metadata)
    with (output / 'examples.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row', 'subject', 'sequence', 'view', 'cover', 'video',
                         'original_frames', 'valid_frames'])
        writer.writerows(metadata)
    summary = dict(shape=list(matrix.shape), units='bits', order='compressed packet order',
                   short_policy=short_policy, short_videos=len(short), input_videos=len(files),
                   truncated_videos=sum(row[6] > frame_num for row in metadata))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved {matrix.shape[0]} examples × {matrix.shape[1]} frames to {output}')
    return matrix
