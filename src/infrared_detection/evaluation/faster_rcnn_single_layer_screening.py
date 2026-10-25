"""Faster R-CNN adapters for the shared one-filter performance screen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from infrared_detection.compression.pruning.dependency_graph import (
    build_faster_rcnn_dependency_graph,
)
from infrared_detection.compression.pruning.unit_discovery import (
    _dependency_operations,
    _has_coupled_output,
    _has_unresolved_dependency_target,
)
from infrared_detection.compression.pruning.yolo_pruner import (
    synchronize_module_channel_metadata,
)
from infrared_detection.data.dataset import CAMELDataset
from infrared_detection.evaluation.model_stats import collect_model_stats
from infrared_detection.models.faster_rcnn.training import (
    build_faster_rcnn_model,
    collate_fn,
    compute_detection_metrics,
)


@dataclass
class FasterRCNNWrapper:
    model: nn.Module
    device: torch.device
    model_config: Mapping[str, Any]


def _device(settings: Mapping[str, Any]) -> torch.device:
    value = str(settings.get("runtime", {}).get("device", "0")).lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        device = torch.device(value)
    else:
        device = torch.device(f"cuda:{value}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Faster R-CNN performance screening requires CUDA.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device.index} is unavailable.")
    return device


def _precision(settings: Mapping[str, Any]) -> str:
    precision = str(settings.get("runtime", {}).get("precision", "fp32")).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported Faster R-CNN screening precision: {precision!r}.")
    return precision


def make_adapters(config: Mapping[str, Any]):
    """Bind Faster R-CNN model, CAMEL validation, pruning, and profiling operations."""

    from infrared_detection.evaluation.single_layer_performance_screening import (
        ScreeningAdapters,
        UnscorableLayerError,
        _production_dependency,
    )

    model_settings = config.get("model", {})
    data_settings = config.get("data", {})
    config_path = model_settings.get("config")
    dataset_root = data_settings.get("dataset_root")
    if not config_path or not dataset_root:
        raise ValueError("Faster R-CNN screening requires model.config and data.dataset_root.")
    model_payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(model_payload, dict) or not isinstance(model_payload.get("model"), dict):
        raise ValueError("Faster R-CNN model config must contain a model mapping.")
    model_payload["model"]["pretrained"] = False
    number_of_classes = int(
        model_payload.get("data", {}).get(
            "num_classes", model_payload["model"].get("num_classes", 0)
        )
    )
    if number_of_classes < 2:
        raise ValueError("Faster R-CNN model config must define at least two classes.")
    device = _device(config)
    _precision(config)

    def load_model(checkpoint_path: str | Path) -> FasterRCNNWrapper:
        model = build_faster_rcnn_model(number_of_classes, model_payload).to(device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, Mapping):
            raise ValueError(f"Faster R-CNN checkpoint has no state dictionary: {checkpoint_path}")
        model.load_state_dict(state)
        shadow = payload.get("ema_state_dict", {}).get("shadow", {}) if isinstance(payload, dict) else {}
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in shadow.items():
                if name in parameters and torch.is_tensor(value) and parameters[name].shape == value.shape:
                    parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
        model.eval()
        return FasterRCNNWrapper(model, device, model_payload)

    def evaluate(wrapper: FasterRCNNWrapper, settings: Mapping[str, Any]) -> dict[str, Any]:
        runtime = settings.get("runtime", {})
        experiment = settings.get("experiment", {})
        data = settings.get("data", {})
        dataset = CAMELDataset(
            str(dataset_root), split=str(data.get("split", "val")), augment=False
        )
        loader = DataLoader(
            dataset,
            batch_size=int(experiment.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(runtime.get("workers", data.get("workers", 0))),
            collate_fn=collate_fn,
        )
        predictions, targets = [], []
        wrapper.model.to(wrapper.device).eval()
        if _precision(settings) == "fp16":
            wrapper.model.half()
        with torch.inference_mode():
            with tqdm(
                total=len(dataset),
                desc=f"Faster R-CNN validation ({data.get('split', 'val')})",
                unit="image",
                leave=False,
            ) as progress:
                for images, batch_targets in loader:
                    image_dtype = (
                        torch.float16
                        if _precision(settings) == "fp16"
                        else torch.float32
                    )
                    device_images = [
                        image.to(device=wrapper.device, dtype=image_dtype)
                        for image in images
                    ]
                    outputs = wrapper.model(device_images)
                    predictions.extend(
                        {key: value.detach().cpu() for key, value in output.items()}
                        for output in outputs
                    )
                    targets.extend(batch_targets)
                    progress.update(len(images))
        metrics = compute_detection_metrics(
            predictions,
            targets,
            iou_threshold=float(runtime.get("iou", 0.5)),
            conf_threshold=float(runtime.get("conf", 0.25)),
            num_classes=number_of_classes,
        )
        return {
            "map50_95": float(metrics["mAP50_95"]),
            "map50": float(metrics["mAP50"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }

    def prune_filter(
        wrapper: FasterRCNNWrapper,
        layer: str,
        physical_filter_index: int,
        settings: Mapping[str, Any],
    ) -> FasterRCNNWrapper:
        import torch_pruning as tp

        target = dict(wrapper.model.named_modules()).get(layer)
        if not isinstance(target, nn.Conv2d):
            raise ValueError(f"Layer {layer!r} is not a Faster R-CNN Conv2d module.")
        if not 0 <= physical_filter_index < target.out_channels:
            raise IndexError(f"Filter index {physical_filter_index} is outside layer {layer!r}.")
        size = int(
            settings.get("experiment", {}).get(
                "image_size", model_payload["model"].get("img_size", 336)
            )
        )
        parameter = next(wrapper.model.parameters())
        example_input = torch.zeros(
            3, size, size, device=parameter.device, dtype=parameter.dtype
        )
        dependency = build_faster_rcnn_dependency_graph(wrapper.model, example_input)
        group = dependency.graph.get_pruning_group(
            target, tp.prune_conv_out_channels, idxs=[physical_filter_index]
        )
        if not dependency.graph.check_pruning_group(group):
            raise RuntimeError(f"Dependency graph rejected pruning group for {layer}.")
        operations = _dependency_operations(wrapper.model, group)
        if _has_unresolved_dependency_target(operations) or _has_coupled_output(
            wrapper.model, layer, operations
        ):
            raise UnscorableLayerError(
                f"Pruning {layer}[{physical_filter_index}] also changes another module's output channels."
            )
        group.prune()
        synchronize_module_channel_metadata(wrapper.model)
        wrapper.model.eval()
        return wrapper

    class DetectorForward(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(self, batch: torch.Tensor):
            return self.model([batch.squeeze(0)])

    def profile(wrapper: FasterRCNNWrapper, settings: Mapping[str, Any]) -> Mapping[str, Any]:
        device = _device(settings)
        use_half = _precision(settings) == "fp16"
        model = wrapper.model.to(device).eval()
        if use_half:
            model.half()
        size = int(
            settings.get("experiment", {}).get(
                "image_size", model_payload["model"].get("img_size", 336)
            )
        )
        example = torch.zeros(
            1,
            3,
            size,
            size,
            device=device,
            dtype=torch.float16 if use_half else torch.float32,
        )
        return _production_dependency("benchmark_pytorch_cuda_forward")(
            DetectorForward(model),
            example,
            warmup=int(settings["screening"]["warmup"]),
            iterations=int(settings["screening"]["iterations"]),
        )

    def save_checkpoint(wrapper: FasterRCNNWrapper, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": wrapper.model.state_dict()}, path)

    return ScreeningAdapters(
        load_model=load_model,
        evaluate=evaluate,
        prune_filter=prune_filter,
        profile=profile,
        stats=lambda wrapper: collect_model_stats(wrapper.model),
        save_checkpoint=save_checkpoint,
    )
