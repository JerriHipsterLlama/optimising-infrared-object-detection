"""Faster R-CNN backend for the shared accuracy-sensitivity workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from infrared_detection.compression.pruning.unit_discovery import (
    capture_faster_rcnn_contract,
    discover_faster_rcnn_pruning_units,
    validate_and_prune_faster_rcnn_unit,
)
from infrared_detection.data.dataset import CAMELDataset
from infrared_detection.models.faster_rcnn.training import (
    build_faster_rcnn_model,
    collate_fn,
    compute_detection_metrics,
)


@dataclass
class FasterRCNNWrapper:
    model: torch.nn.Module
    device: torch.device
    model_config: Mapping[str, Any]


def make_adapters(config):
    """Bind checkpoint loading and CAMEL Pascal-VOC validation to shared orchestration."""

    from infrared_detection.evaluation.accuracy_sensitivity import (
        SensitivityAdapters,
        ValidationMetrics,
    )

    if config.model_config is None or config.dataset_root is None:
        raise ValueError("Faster R-CNN screening requires model.config and data.dataset_root.")
    if not torch.cuda.is_available():
        raise RuntimeError("Faster R-CNN accuracy sensitivity screening requires CUDA.")
    device_index = int(config.validation["device"])
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device_index} is unavailable.")
    device = torch.device(f"cuda:{device_index}")
    model_payload = yaml.safe_load(config.model_config.read_text(encoding="utf-8"))
    model_settings = model_payload["model"]
    data_settings = model_payload.get("data", {})
    model_settings["pretrained"] = False
    number_of_classes = int(data_settings.get("num_classes", model_settings["num_classes"]))

    def load_model(checkpoint_path: Path) -> FasterRCNNWrapper:
        model = build_faster_rcnn_model(number_of_classes, model_payload).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        shadow = checkpoint.get("ema_state_dict", {}).get("shadow", {})
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in shadow.items():
                if name in parameters and parameters[name].shape == value.shape:
                    parameters[name].copy_(value.to(device=parameters[name].device,
                                                    dtype=parameters[name].dtype))
        model.eval()
        return FasterRCNNWrapper(model, device, model_payload)

    def evaluate(wrapper: FasterRCNNWrapper, arguments: Mapping[str, Any]) -> ValidationMetrics:
        dataset = CAMELDataset(
            str(config.dataset_root), split=str(arguments.get("split", config.split)),
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(arguments.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(arguments.get("workers", 0)),
            collate_fn=collate_fn,
        )
        predictions = []
        targets = []
        inference_events = []
        wrapper.model.eval()
        with torch.inference_mode():
            with tqdm(
                total=len(dataset),
                desc=f"Validation ({arguments.get('split', config.split)})",
                unit="image",
                leave=False,
            ) as progress:
                for images, batch_targets in loader:
                    device_images = [image.to(wrapper.device) for image in images]
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    outputs = wrapper.model(device_images)
                    end.record()
                    inference_events.append((start, end))
                    progress.update(len(images))
                    predictions.extend(
                        {key: value.detach().cpu() for key, value in output.items()}
                        for output in outputs
                    )
                    targets.extend(batch_targets)
        torch.cuda.synchronize(wrapper.device)
        inference_ms_per_image = sum(
            start.elapsed_time(end) for start, end in inference_events
        ) / len(dataset)
        metrics = compute_detection_metrics(
            predictions,
            targets,
            iou_threshold=float(arguments.get("iou_threshold", 0.5)),
            conf_threshold=float(arguments.get("conf_threshold", 0.25)),
            num_classes=number_of_classes,
        )
        return ValidationMetrics(
            map50_95=float(metrics["mAP50_95"]),
            map50=float(metrics["mAP50"]),
            precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),
            inference_ms_per_image=inference_ms_per_image,
        )

    def make_example(model: torch.nn.Module, size: int) -> torch.Tensor:
        parameter = next(model.parameters())
        return torch.zeros(3, size, size, device=parameter.device, dtype=parameter.dtype)

    def forward(model: torch.nn.Module, image: torch.Tensor):
        model.eval()
        with torch.inference_mode():
            return model([image])

    def versions() -> Mapping[str, str]:
        import torchvision
        import torch_pruning

        return {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torch_pruning": getattr(torch_pruning, "__version__", "unknown"),
        }

    return SensitivityAdapters(
        load_model=load_model,
        unwrap_model=lambda wrapper: wrapper.model,
        replace_model=lambda wrapper, model: setattr(wrapper, "model", model),
        evaluate=evaluate,
        make_example_input=make_example,
        discover_units=discover_faster_rcnn_pruning_units,
        capture_contract=capture_faster_rcnn_contract,
        forward=forward,
        probe=validate_and_prune_faster_rcnn_unit,
        versions=versions,
    )
