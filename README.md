# A Zero Decoding Approach to Video Classification
*Chen Ye Gan, Jiangtao Wen, Yuxing Han*

This repository is the official implementation of **A Zero Decoding Approach to Video Classification**, accepted at ICME 2025.

[![Paper](https://img.shields.io/badge/Paper-ICME_2025-blue)](link-to-camera-ready)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

## ✨ Highlights
- **Pixel-free**: learns directly from the H.264/AVC bitstream—no decode, protects data privacy.
- **15,000×** real-time throughput on 30 fps video.
- ~90 % accuracy on a 6,000hr YouTube dataset (11 classes).
- **7 orders-of-magnitude** faster than DTW, **3 orders** faster than CNN baselines.

<p align="center" width="100%">
<img src="overview.png" width="100%" height="100%">
</p>

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)
6. [Contributing](#contributing)

<a name="overview"></a>
## Overview

*Abstract* — Classifying videos into distinct categories, such as Sport and Music Video, is crucial for multimedia understanding and retrieval, especially with growing content volume. Traditional methods require video decompression to extract pixel level features like color, texture, and motion, thereby increasing computational and storage demands. We present a novel approach that examines only the compressed bitstream of a video to perform classification, eliminating the need for bitstream decoding. To validate our approach, we built a comprehensive data set comprising over 29,000 YouTube video clips, totaling 6,000 hours and spanning 11 distinct categories. Our evaluations indicate precision, accuracy, and recall rates consistently above 80%, many exceeding 90%, and some reaching 99%. The algorithm operates approximately 15,000 times faster than real-time for 30fps videos, outperforming traditional Dynamic Time Warping (DTW) algorithm by seven orders of magnitude and state-of-the-art video classification model by three orders of magnitude.

<a name="installation"></a>
## Installation

The maintained supervised pipeline is in `Video_Classification_Model`. From the repository root, install its Python dependencies with pip:

```sh
python -m pip install -r Video_Classification_Model/requirements.txt
```

<a name="quick-start"></a>
## Quick Start

Before running the pipeline, prepare a **well-formed packet-size data CSV and matching label CSV**. Video extraction is not part of the current pipeline. The data CSV must already contain the required `size0` through `size2999` columns, and the label CSV must already be aligned row-for-row with it.

```sh
cd Video_Classification_Model
python run_pipeline.py train --config configs/default.yaml --run-name my_run
```

Set the data paths and training options in a copy of `configs/default.yaml` before training. The command creates a run, trains, and evaluates its saved test split after normal completion or early stopping. To use the interactive equivalent instead, run `python run_pipeline.py` with no arguments.

See [`Video_Classification_Model/README.md`](Video_Classification_Model/README.md) for the exact CSV schema and preprocessing, configuration, training/resume and evaluation commands, outputs, tests, and linear probes.

<a name="citation"></a>
## Citation
```bibtex
@inproceedings{gan2025zerodecode,
  title     = {A Zero Decoding Approach to Video Classification},
  author    = {Gan, Chen Ye and Wen, Jiangtao and Han, Yuxing},
  booktitle = {Proc. IEEE Int. Conf. Multimedia \& Expo (ICME)},
  year      = {2025}
}
```

<a name="acknowledgements"></a>
## Acknowledgements

The authors thank Yuchen Deng and Fengpu Pan for their assistance in data collection and Haoyue Han for paper review.

This work is supported by Shenzhen Startup Funding No.QD2023014C.

<a name="contributing"></a>
## Contributing

We would like to keep this version archival to match our paper's results, but freel free to fork or clone the repository—just don’t forget to give us a shout-out!
