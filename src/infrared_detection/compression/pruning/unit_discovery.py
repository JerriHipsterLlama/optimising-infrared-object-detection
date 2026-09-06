"""Discover and validate independent structural pruning units."""

from __future__ import annotations


def requested_prune_count(channels: int, ratio: float) -> int:
    """Convert a requested ratio into a valid dense-model channel count."""

    if channels < 2:
        raise ValueError("A pruning unit must have at least two output channels.")
    if not 0.0 < ratio < 1.0:
        raise ValueError("Pruning ratio must be between zero and one.")
    count = round(channels * ratio)
    if not 1 <= count < channels:
        raise ValueError("Requested ratio does not remove a valid channel count.")
    return count
