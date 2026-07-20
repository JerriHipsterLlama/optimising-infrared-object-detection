"""Portable model size and parameter statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def collect_model_stats(model_or_path: nn.Module | str | Path) -> dict[str, Any]:
    """Collect statistics that are comparable before deployment."""

    file_size = None
    if isinstance(model_or_path, (str, Path)):
        path = Path(model_or_path)
        if not path.exists():
            raise FileNotFoundError(path)
        file_size = path.stat().st_size
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, nn.Module):
            model = payload
        elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            model = None
            tensors = payload["state_dict"].values()
        elif isinstance(payload, dict):
            model = None
            tensors = payload.values()
        else:
            raise TypeError("Checkpoint must contain a model or state dictionary.")
    else:
        model = model_or_path

    if model is not None:
        tensors = (parameter for parameter in model.parameters())
    tensor_list = [tensor for tensor in tensors if torch.is_tensor(tensor)]
    parameter_count = sum(tensor.numel() for tensor in tensor_list)
    parameter_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensor_list)
    nonzero_count = sum(torch.count_nonzero(tensor).item() for tensor in tensor_list)
    return {
        "parameter_count": parameter_count,
        "nonzero_parameter_count": nonzero_count,
        "parameter_bytes": parameter_bytes,
        "serialized_bytes": file_size,
        "sparsity": 1.0 - (nonzero_count / parameter_count if parameter_count else 0.0),
    }

