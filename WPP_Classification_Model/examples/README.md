# Illustrative CSV pairs

These CSVs are contract examples only. The referenced `features/*.npz` files are intentionally **not included**, so `configs/example.yaml` cannot run unchanged. Their relative paths demonstrate that NPZ resolution begins in this `examples/` directory, not next to the YAML file.

Each split has one data row per whole video and one matching label row per `video_id`. Label rows intentionally use a different order, proving that the reader joins by ID rather than row number.

The example configuration omits a class list. Its three training labels are read as text and define the inferred vocabulary in sorted canonical order (`0`, `1`, `2`). In a real identity-labelled export, values such as `016` remain text and retain their leading zeroes. Every class occurs in `training_labels.csv`; validation and test also show every class. These are not source, generator, prompt, or DeepAction assumptions.

On Linux, an absolute `features_path` can be `/data/features/video.npz`. On native Windows it can be `C:\data\features\video.npz`; CSV quoting may be needed for paths containing commas. Forward-slash relative paths are portable where supported. See `../INPUT_CONTRACT.md` for required NPZ arrays and exact timing semantics.
