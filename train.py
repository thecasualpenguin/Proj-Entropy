"""Train CASIA-B subject classification with the entropy-with-transformer recipe."""
import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

try:
    from sandbox.plotting import plot_loss_accuracy
except ModuleNotFoundError:
    from plotting import plot_loss_accuracy

from resnet_tsc import ResNet
from transformer_tsc import TransformerTSC

EXPERIMENT_DIR = Path(__file__).resolve().parent


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


@contextmanager
def log_to_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = _Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("transformer", "resnet"), default="resnet")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "outputs/matrix",
                        help="Directory containing collated X.npy and Y.npy")
    parser.add_argument("--runs-dir", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument("--length-aware", action=argparse.BooleanOptionalAction, default=False,
                        help="Use original frame lengths; exclude padding from normalization and models")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--seed", type=int, default=42, help="Model initialization and batch shuffle seed")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout probability in [0, 1); defaults: transformer 0.2, ResNet 0")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    if args.patch_size < 1:
        parser.error("--patch-size must be a positive integer")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    run_dir = new_run_directory(args.runs_dir.expanduser().resolve())
    with log_to_file(run_dir / "run.log"):
        print(f"Run directory: {run_dir}")
        print(f"Model: {args.model} | Epochs: {args.epochs} | Patch size: {args.patch_size}")
        _run(args.model, args.patch_size, args.epochs, run_dir=run_dir,
             data_dir=args.data_dir.expanduser().resolve(), length_aware=args.length_aware,
             weight_decay=args.weight_decay, optimizer_name=args.optimizer, seed=args.seed, dropout=args.dropout)


def load_training_data(data_dir: Path):
    """Read aligned numeric covers and string subject IDs from the collator."""
    try:
        X_all = np.load(data_dir / "X.npy", allow_pickle=False).astype(np.float32)
        labels = np.load(data_dir / "Y.npy", allow_pickle=False)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing training matrix in {data_dir}; run python run.py --extract first") from exc
    if X_all.ndim != 2 or not all(X_all.shape):
        raise ValueError("X.npy must be a nonempty examples-by-frames matrix")
    if labels.ndim != 1 or len(labels) != len(X_all):
        raise ValueError("Y.npy must contain exactly one subject label per matrix row")
    if not np.isfinite(X_all).all() or np.any(X_all < 0):
        raise ValueError("Frame sizes must be finite and nonnegative")
    y_all_raw = labels.astype(str).tolist()
    if any(not label for label in y_all_raw):
        raise ValueError("Subject labels must not be empty")
    classes, counts = np.unique(y_all_raw, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("Training needs at least two subjects and two videos per subject for the 90/10 split")
    return X_all, y_all_raw


def load_lengths(data_dir, X, labels):
    """Validate manifest order and distinguish real frames from stored right padding."""
    with (data_dir / "examples.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != len(X):
        raise ValueError("examples.csv must have one row per example")
    lengths = []
    for i, record in enumerate(records):
        if int(record["row"]) != i or record["subject"] != labels[i]:
            raise ValueError("examples.csv order/subjects do not match the matrix")
        original = int(record["original_frames"])
        valid = min(original, X.shape[1])
        if original < 1 or int(record["valid_frames"]) != valid:
            raise ValueError(f"Invalid frame length in manifest row {i}")
        if np.any(X[i, valid:] != 0):
            raise ValueError(f"Expected zero padding after real frames in row {i}")
        lengths.append(valid)
    return np.asarray(lengths, dtype=np.int64)


def fit_normalization(X_train, lengths=None):
    if lengths is None:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
    else:
        valid = np.arange(X_train.shape[1])[None, :] < lengths[:, None]
        values = X_train[valid]
        # One scale across real frames avoids sparse tail-column statistics.
        mean = np.full(X_train.shape[1], values.mean(dtype=np.float64), dtype=np.float32)
        std = np.full(X_train.shape[1], values.std(dtype=np.float64), dtype=np.float32)
    std[std == 0] = 1.0
    return mean, std


def normalize_inputs(X, mean, std, lengths=None):
    normalized = (X - mean) / std
    if lengths is not None:
        valid = np.arange(X.shape[1])[None, :] < lengths[:, None]
        normalized = np.where(valid, normalized, 0)
    return normalized.astype(np.float32)


def _run(model_name="resnet", patch_size=16, epochs=50, *, run_dir: Path,
         data_dir: Path = EXPERIMENT_DIR / "outputs/matrix", length_aware=False,
         weight_decay=0.01, optimizer_name="adamw", seed=42, dropout=None) -> None:
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("optimizer_name must be adam or adamw")
    dropout = (0.2 if model_name == "transformer" else 0.0) if dropout is None else dropout
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Length-aware: {length_aware} | Optimizer: {optimizer_name} | "
          f"Weight decay: {weight_decay} | Dropout: {dropout} | Seed: {seed}")
    X_all, y_all_raw = load_training_data(data_dir)
    lengths = load_lengths(data_dir, X_all, y_all_raw) if length_aware else None

    classes = sorted(set(y_all_raw))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_all = np.array([label_to_idx[y] for y in y_all_raw], dtype=np.int64)

    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for label_idx in range(len(classes)):
        indices = np.where(y_all == label_idx)[0]
        rng.shuffle(indices)
        n_test = max(1, round(0.1 * len(indices)))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    # Indices refer directly to rows in the collated examples.csv manifest.
    np.savez(run_dir / "split.npz", train_indices=np.asarray(train_idx),
             validation_indices=np.asarray(test_idx))
    (run_dir / "data.json").write_text(json.dumps({
        "data_dir": str(data_dir), "shape": list(X_all.shape),
        "classes": classes, "split_seed": 42,
        "split": "90/10 within each subject, rounded with at least one validation video",
        "normalization": "global real training values" if length_aware else "per-column including padding",
        "length_aware": length_aware, "optimizer": optimizer_name,
        "weight_decay": weight_decay, "seed": seed, "dropout": dropout,
    }, indent=2) + "\n")
    manifest = data_dir / "examples.csv"
    if manifest.exists():
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, records = reader.fieldnames, list(reader)
        if len(records) != len(X_all):
            raise ValueError("examples.csv row count does not match X.npy")
        validation_rows = set(test_idx)
        with (run_dir / "examples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fields, "split"])
            writer.writeheader()
            for i, record in enumerate(records):
                if int(record["row"]) != i or record["subject"] != y_all_raw[i]:
                    raise ValueError("examples.csv row order or subjects do not match Y.npy")
                writer.writerow({**record, "split": "validation" if i in validation_rows else "train"})

    X_train = X_all[np.array(train_idx)]
    y_train = y_all[np.array(train_idx)]
    X_test = X_all[np.array(test_idx)]
    y_test = y_all[np.array(test_idx)]

    train_lengths = lengths[train_idx] if length_aware else None
    test_lengths = lengths[test_idx] if length_aware else None
    mean, std = fit_normalization(X_train, train_lengths)
    X_train = normalize_inputs(X_train, mean, std, train_lengths)
    X_test = normalize_inputs(X_test, mean, std, test_lengths)

    n_total = len(X_all)
    del X_all, y_all, y_all_raw

    print(f"Total samples: {n_total}")
    print(f"Train: {len(X_train)} ({100 * len(X_train) / n_total:.1f}%)")
    print(f"Test:  {len(X_test)} ({100 * len(X_test) / n_total:.1f}%)")
    print(f"Features: {X_train.shape[1]} | Classes: {len(classes)}")

    n_feature_maps = 64
    batch_size = 64
    lr = 1e-3

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if model_name == "transformer":
        model = TransformerTSC(
            (1, X_train.shape[1]), len(classes), patch_size=patch_size, length_aware=length_aware, dropout=dropout
        ).to(device)
        print(f"Transformer maximum tokens: {model.num_tokens} | Patch size: {patch_size}")
    else:
        model = ResNet((1, X_train.shape[1]), len(classes), n_feature_maps, length_aware=length_aware, dropout=dropout).to(device)
    print(f"Model: {model_name} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    summary_buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = summary_buffer
    try:
        if length_aware:
            summary(model, input_data=(torch.from_numpy(X_train[:batch_size]).to(device),
                                       torch.from_numpy(train_lengths[:batch_size]).to(device)),
                    device=device, mode="eval")
        else:
            summary(model, input_size=(batch_size, X_train.shape[1]), device=device, mode="eval")
    finally:
        sys.stdout = old_stdout
    print(summary_buffer.getvalue(), end="")

    # Summary forwards must not change initialization or training randomness.
    torch.manual_seed(seed)
    optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_x = torch.from_numpy(X_train)
    train_y = torch.from_numpy(y_train)
    val_x = torch.from_numpy(X_test)
    val_y = torch.from_numpy(y_test)
    del X_train, y_train, X_test, y_test

    train_dataset = TensorDataset(train_x, train_y, *(
        [torch.from_numpy(train_lengths)] if length_aware else []))
    val_dataset = TensorDataset(val_x, val_y, *(
        [torch.from_numpy(test_lengths)] if length_aware else []))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_time": [],
    }

    best_train = best_validation = -math.inf
    checkpoint_options = dict(model=model, optimizer=optimizer, mean=mean, std=std,
                              classes=classes, label_to_idx=label_to_idx,
                              n_feature_maps=n_feature_maps, lr=lr, batch_size=batch_size,
                              length_aware=length_aware, weight_decay=weight_decay,
                              optimizer_name=optimizer_name, seed=seed)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for batch in train_loader:
            xb, yb, *batch_lengths = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(xb, *batch_lengths), yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if not all(math.isfinite(v) for v in (train_loss, val_loss, train_acc, val_acc)):
            raise RuntimeError("Non-finite epoch metrics; previous checkpoints retained")
        metrics = dict(train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)
        if train_acc > best_train:
            best_train = train_acc
            save_checkpoint(run_dir / "best_train.pt", epoch=epoch, metrics=metrics,
                            selection="train_acc", **checkpoint_options)
        if val_acc > best_validation:
            best_validation = val_acc
            save_checkpoint(run_dir / "best_validation.pt", epoch=epoch, metrics=metrics,
                            selection="val_acc", **checkpoint_options)
        save_checkpoint(run_dir / "last.pt", epoch=epoch, metrics=metrics,
                        selection="last", **checkpoint_options)
        release_device_memory(device)

        epoch_time = time.perf_counter() - epoch_start
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        (run_dir / "loss.json").write_text(json.dumps(history, indent=2) + "\n")
        avg_epoch_time = sum(history["epoch_time"]) / len(history["epoch_time"])
        eta = avg_epoch_time * (epochs - epoch)
        print(
            f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"train acc {train_acc:.2f} | val acc {val_acc:.2f} | {epoch_time:.2f}s | "
            f"ETA {format_duration(eta)}"
        )

    loss_json = run_dir / "loss.json"
    with loss_json.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Wrote metrics to {loss_json}")

    plot_path = run_dir / "loss-accuracy.png"
    plot_loss_accuracy(history, title_prefix=model_name, out_path=plot_path)
    print(f"Wrote plot to {plot_path}")

    print(f"Checkpoints: {run_dir / 'best_train.pt'}, {run_dir / 'best_validation.pt'}, {run_dir / 'last.pt'}")


def new_run_directory(directory: Path) -> Path:
    """Create a unique local-date run directory, safe for concurrent runs."""
    directory.mkdir(parents=True, exist_ok=True)
    prefix = date.today().isoformat()
    indices = [
        int(path.name[len(prefix) + 1:])
        for path in directory.glob(f"{prefix}-*")
        if path.name[len(prefix) + 1:].isdigit()
    ]
    index = max(indices, default=0) + 1
    while True:
        path = directory / f"{prefix}-{index:03d}"
        try:
            # Atomic creation prevents simultaneous runs claiming the same folder.
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    mean: np.ndarray,
    std: np.ndarray,
    classes: list[str],
    label_to_idx: dict[str, int],
    n_feature_maps: int,
    lr: float,
    batch_size: int,
    length_aware=False,
    weight_decay=0.01,
    optimizer_name="adamw",
    seed=42,
    metrics=None,
    selection="last",
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "selection": selection,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean": mean,
            "std": std,
            "classes": classes,
            "label_to_idx": label_to_idx,
            "config": {
                "model": "transformer" if isinstance(model, TransformerTSC) else "resnet",
                "seq_len": int(mean.shape[0]),
                "n_classes": len(classes),
                **(model.model_config if isinstance(model, TransformerTSC) else {"n_feature_maps": n_feature_maps, "dropout": model.dropout.p}),
                "lr": lr,
                "batch_size": batch_size,
                "length_aware": length_aware,
                "normalization": "global_real" if length_aware else "per_column_padded",
                "optimizer": optimizer_name, "weight_decay": weight_decay, "seed": seed,
            },
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device | None = None):
    """Restore model (and optionally optimizer) for inference or continued training."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg["model"] == "transformer":
        model = TransformerTSC(
            (1, cfg["seq_len"]), cfg["n_classes"],
            length_aware=cfg.get("length_aware", False),
            **{key: cfg[key] for key in (
                "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "patch_size"
            )},
        ).to(device)
    elif cfg["model"] == "resnet":
        model = ResNet(
            (1, cfg["seq_len"]), cfg["n_classes"], cfg["n_feature_maps"],
            length_aware=cfg.get("length_aware", False), dropout=cfg.get("dropout", 0.0)
        ).to(device)
    else:
        raise ValueError(f"Unknown checkpoint model: {cfg['model']}")
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb, *batch_lengths = [t.to(device) for t in batch]
            logits = model(xb, *batch_lengths)
            total_loss += criterion(logits, yb).item() * len(xb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)
    return total_loss / total, total_correct / total


def release_device_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


if __name__ == "__main__":
    main()
