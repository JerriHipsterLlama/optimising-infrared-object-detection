import torch
from torch import nn

from apps.compress import compress


def test_quantize_dispatch_returns_model_and_records_payload():
    model = nn.Linear(4, 2)

    result = compress(model, {"method": "quantize", "bit_width": 8})

    assert result is model
    assert model._compression_quantization["bit_width"] == 8
    assert "weight" in model._compression_quantization["quantized_state_dict"]
