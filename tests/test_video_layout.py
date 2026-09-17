"""Video dataset hierarchy handling for extraction and collation."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import collate
import extract


class VideoLayoutTests(unittest.TestCase):
    def test_extract_mirrors_nested_video_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, covers = root / 'videos', root / 'covers'
            videos = [
                source / 'casiasilhouette/004/nm-02/090/090.mp4',
                source / 'casiabrgb/011/nm-02/162/162.mp4',
            ]
            for video in videos:
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()

            def fake_extract(video, output_dir, stem, ffprobe):
                destination = Path(output_dir) / f'{stem}.csv'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f'{video},0,8\n')
                return destination

            with patch('extract.extract_cover', side_effect=fake_extract):
                extracted = extract.extract_covers(source, covers, workers=1)

            self.assertEqual(extracted, [
                covers / 'casiabrgb/011/nm-02/162/162.csv',
                covers / 'casiasilhouette/004/nm-02/090/090.csv',
            ])

    def test_collate_recognizes_dataset_roots_for_legacy_layout(self):
        self.assertEqual(
            collate.cover_labels(Path('casiasilhouette/004/nm-02/090.csv')),
            ('004', 'nm-02', '090'),
        )
        self.assertEqual(
            collate.cover_labels(Path('casiabrgb/011/bg-01/162.csv')),
            ('011', 'bg-01', '162'),
        )

    def test_collate_rejects_malformed_cover_paths(self):
        for relative in (Path('004/nm-02.csv'), Path('004/nm-02/090/090/extra.csv'),
                         Path('004/nm-02/090.txt')):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    collate.cover_labels(relative)

    def test_collate_uses_directory_sequence_and_view_for_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            covers, output = root / 'covers', root / 'matrix'
            entries = {
                'casiasilhouette/004/nm-02/090/090.csv': '/videos/004/nm-02/090/090.mp4',
                'casiabrgb/011/nm-02/162/162.csv': '/videos/011/nm-02/162/162.mp4',
                '012/bg-01/000.csv': '/videos/012/bg-01/000.mp4',
                '014/nm-03/108/108.csv': '/videos/014/nm-03/108/108.mp4',
            }
            for relative, source in entries.items():
                cover = covers / relative
                cover.parent.mkdir(parents=True, exist_ok=True)
                cover.write_text(f'{source},0,8\n{source},1,16\n')

            matrix = collate.collate_into_matrix(covers, output, frame_num=3)

            np.testing.assert_array_equal(matrix, np.array([[8, 16, 0]] * 4))
            np.testing.assert_array_equal(np.load(output / 'Y.npy'), ['012', '014', '011', '004'])
            with (output / 'examples.csv').open(newline='') as handle:
                examples = list(csv.DictReader(handle))
            self.assertEqual(
                [(row['subject'], row['sequence'], row['view']) for row in examples],
                [('012', 'bg-01', '000'), ('014', 'nm-03', '108'),
                 ('011', 'nm-02', '162'), ('004', 'nm-02', '090')],
            )
            self.assertEqual(examples[2]['cover'], 'casiabrgb/011/nm-02/162/162.csv')
            self.assertEqual(examples[2]['video'], '/videos/011/nm-02/162/162.mp4')


if __name__ == '__main__':
    unittest.main()
