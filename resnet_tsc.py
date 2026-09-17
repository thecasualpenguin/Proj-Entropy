"""1D residual CNN for time-series classification."""

import torch
from torch import nn

from sequence_lengths import trim_to_lengths


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=stride, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size[1], stride=stride, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size[2], stride=stride, padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn4 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x, valid=None):
        if valid is not None:
            # Each convolution sees zero outside the real sequence. BatchNorm
            # estimates its statistics using only valid batch/time positions.
            out = self.relu1(masked_batch_norm(self.conv1(x), self.bn1, valid))
            out = self.relu2(masked_batch_norm(self.conv2(out), self.bn2, valid))
            out = self.relu3(masked_batch_norm(self.conv3(out), self.bn3, valid)
                             + masked_batch_norm(self.shortcut(x), self.bn4, valid))
            return out.masked_fill(~valid[:, None, :], 0)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)) + self.bn4(self.shortcut(x)))
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, num_classes, n_feature_maps=64, length_aware=False, dropout=0.0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = nn.Dropout(dropout)
        self.length_aware = length_aware
        self.layer1 = ResidualBlock(input_shape[0], n_feature_maps, kernel_size=[9, 5, 3])
        self.layer2 = ResidualBlock(n_feature_maps, n_feature_maps * 2, kernel_size=[9, 5, 3])
        self.layer3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, kernel_size=[9, 5, 3])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.length_aware and lengths is None:
            raise ValueError("This model requires original frame lengths")
        if lengths is not None:
            x, valid = trim_to_lengths(x, lengths)
            out = self.layer3(self.layer2(self.layer1(x, valid), valid), valid)
            return self.fc(self.dropout(out.sum(dim=-1) / valid.sum(dim=-1, keepdim=True)))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        return self.fc(self.dropout(out))


def masked_batch_norm(x, norm, valid):
    """Apply ordinary BatchNorm to real positions only, then restore zero padding."""
    values = x.transpose(1, 2)[valid]
    if norm.training and values.shape[0] == 1:
        # A batch containing one single-frame clip cannot estimate variance.
        values = nn.functional.batch_norm(values, norm.running_mean, norm.running_var,
                                          norm.weight, norm.bias, training=False, eps=norm.eps)
    else:
        values = norm(values)
    output = torch.zeros_like(x.transpose(1, 2))
    output[valid] = values
    return output.transpose(1, 2)
