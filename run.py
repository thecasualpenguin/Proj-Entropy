"""Prepare CASIA-B videos, or extract and collate their compressed frame sizes."""

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = [] if argv is None else list(argv)
    directory = Path(__file__).resolve().parent
    if '--extract' not in argv and '--collate' not in argv:
        if '--help' in argv or '-h' in argv:
            print('Additional stages: --extract (extract + collate), --collate (existing covers).\n'
                  'Run --extract --help for extraction options.\n')
        runpy.run_path(str(directory / 'data.py'))['main'](argv)
        return
    datasets = Path(os.environ.get('SANDBOX_ROOT', directory.parents[1])).expanduser() / 'datasets'
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--extract', action='store_true', help='Extract covers, then collate')
    stage.add_argument('--collate', action='store_true', help='Collate existing covers only')
    parser.add_argument(
        '--source', type=Path, default=datasets / 'casia-b-video',
        help=('Video dataset root. Supports subject/sequence/view.mp4 and '
              'subject/condition-sequence/view-angle/video.mp4 layouts; a '
              'leading dataset directory is also accepted'),
    )
    parser.add_argument('--covers', type=Path, default=directory / 'outputs/covers')
    parser.add_argument('--output', type=Path, default=directory / 'outputs/matrix')
    parser.add_argument('--frames', type=int, default=3000)
    parser.add_argument('--short-policy', choices=['error', 'pad', 'repeat', 'drop'], default='pad')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--overwrite', action='store_true', help='Re-extract existing covers')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.workers <= 0:
        parser.error('--frames and --workers must be positive')
    source, covers, output = [p.expanduser().resolve() for p in (args.source, args.covers, args.output)]
    for a, b in [(source, covers), (source, output), (covers, output)]:
        if a == b or a in b.parents or b in a.parents:
            parser.error('Source, covers and output must be separate, non-nested directories')
    try:
        extractor = runpy.run_path(str(directory / 'extract.py'))
        if args.extract:
            videos = extractor['grab_all_videos'](source)
            if not videos:
                raise ValueError(f'No videos found in {source}')
            print(f'{len(videos)} videos; covers: {covers}; matrix: {output}; '
                  f'frames: {args.frames}; short policy: {args.short_policy}', flush=True)
        if args.dry_run:
            return
        if args.collate and source.exists() and covers.exists():
            videos = extractor['grab_all_videos'](source)
            cover_count = sum(1 for _ in covers.rglob('*.csv'))
            if videos and cover_count < len(videos):
                print(f'Warning: {cover_count} cover CSVs in {covers} but {len(videos)} '
                      f'videos in {source}. Run with --extract to generate missing covers.',
                      flush=True)
        cover_files = None
        if args.extract:
            probe = shutil.which(args.ffprobe)
            if probe is None:
                raise ValueError('ffprobe is required for extraction')
            cover_files = extractor['extract_covers'](source, covers, workers=args.workers,
                                                     ffprobe=probe, overwrite=args.overwrite)
        runpy.run_path(str(directory / 'collate.py'))['collate_into_matrix'](
            covers, output, frame_num=args.frames, short_policy=args.short_policy,
            cover_files=cover_files)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main(sys.argv[1:])
