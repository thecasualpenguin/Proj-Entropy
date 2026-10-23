"""Native WPP model validation and construction."""
from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

from .wpp_transformer import WPPTransformer

MODEL_KIND = "wpp_transformer"
_MODEL_KEYS = {"kind", "patch_size", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout"}


def validate_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("model configuration must be a mapping")
    if set(config) != _MODEL_KEYS:
        raise ValueError(f"native WPP model keys must be exactly {sorted(_MODEL_KEYS)}")
    if config.get("kind") != MODEL_KIND:
        raise ValueError("model.kind must be wpp_transformer; legacy architectures are unsupported")
    result = dict(config)
    for key in ("patch_size", "d_model", "nhead", "num_layers", "dim_feedforward"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if result["d_model"] % result["nhead"] or result["d_model"] % 4:
        raise ValueError("model.d_model must be divisible by nhead and four")
    value = result["dropout"]
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= float(value) < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    result["dropout"] = float(value)
    return result


def make_model(config: Mapping[str, Any], num_classes: int) -> WPPTransformer:
    model = validate_model_config(config)
    return WPPTransformer(num_classes, **{key: model[key] for key in _MODEL_KEYS - {"kind"}})
