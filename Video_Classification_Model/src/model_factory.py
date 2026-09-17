"""Centralized architecture validation and construction."""
from typing import Mapping
from model_resnet_tsc import ResNet
from transformer_tsc import TransformerTSC

ARCHITECTURES = ("resnet", "transformer")

def validate_model_config(config: Mapping):
    architecture = config.get("architecture")
    if architecture not in ARCHITECTURES: raise ValueError("model.architecture must be resnet or transformer")
    allowed = {"architecture", "initial_feature_maps"} if architecture == "resnet" else {"architecture", "patch_size", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "length_aware"}
    extra = set(config) - allowed
    if extra: raise ValueError(f"unsupported {architecture} model settings: {sorted(extra)}")
    if architecture == "resnet":
        if isinstance(config.get("initial_feature_maps"), bool) or not isinstance(config.get("initial_feature_maps"), int) or config["initial_feature_maps"] <= 0: raise ValueError("model.initial_feature_maps must be a positive integer")
    else:
        for key in ("patch_size", "d_model", "nhead", "num_layers", "dim_feedforward"):
            if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] <= 0: raise ValueError(f"model.{key} must be a positive integer")
        if config["d_model"] % config["nhead"]: raise ValueError("model.d_model must be divisible by model.nhead")
        if not isinstance(config.get("dropout"), (int, float)) or not 0 <= config["dropout"] < 1: raise ValueError("model.dropout must be in [0, 1)")
        if not isinstance(config.get("length_aware"), bool): raise ValueError("model.length_aware must be boolean")

def make_model(config: Mapping, num_classes: int):
    validate_model_config(config)
    if config["architecture"] == "resnet": return ResNet((1, 3000), num_classes, config["initial_feature_maps"])
    return TransformerTSC((1, 3000), num_classes, **{k: config[k] for k in config if k != "architecture"})
