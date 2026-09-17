"""Validate lengths and remove batch-wide trailing padding."""

import torch


def trim_to_lengths(x, lengths):
    if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
        raise ValueError('lengths must contain one integer per example')
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError('lengths must be integer tensors')
    lengths = lengths.to(x.device)
    if torch.any(lengths < 1) or torch.any(lengths > x.shape[-1]):
        raise ValueError('lengths must be between 1 and the input width')
    x = x[..., :int(lengths.max().item())]
    valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
    return x.masked_fill(~valid[:, None, :], 0), valid
