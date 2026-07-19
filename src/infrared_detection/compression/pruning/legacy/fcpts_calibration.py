"""FCPTS calibration utilities for detection model sparsification."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.func import functional_call
from tqdm import tqdm


OutputAdapter = Callable[..., List[Tensor]]


def _safe_key(name: str) -> str:
    return name.replace(".", "__dot__")


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        values = [_to_device(v, device) for v in obj]
        return type(obj)(values)
    return obj


def _flatten_tensors(output: Any) -> List[Tensor]:
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, dict):
        flattened: List[Tensor] = []
        for value in output.values():
            flattened.extend(_flatten_tensors(value))
        return flattened
    if isinstance(output, (list, tuple)):
        flattened = []
        for value in output:
            flattened.extend(_flatten_tensors(value))
        return flattened
    return []


def estimate_density_at_threshold(weights: Tensor, threshold: Tensor, bandwidth: float = 0.1, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
    """Estimate p(t), p(-t), and dr/dt using Gaussian KDE over weights."""

    if weights.numel() == 0:
        zero = threshold.new_tensor(0.0)
        return zero, zero, zero

    flat = weights.detach().reshape(-1).to(dtype=torch.float32)
    bw = max(float(bandwidth), float(eps))
    denom = bw * math.sqrt(2.0 * math.pi)

    t = threshold.detach().to(dtype=torch.float32)
    p_t = torch.exp(-0.5 * ((flat - t) / bw) ** 2).mean() / denom
    p_neg_t = torch.exp(-0.5 * ((flat + t) / bw) ** 2).mean() / denom
    dr_dt = p_t + p_neg_t
    return p_t.to(threshold.dtype), p_neg_t.to(threshold.dtype), dr_dt.to(threshold.dtype)


class DifferentiablePruningMaskFn(Function):
    """Custom autograd for hard pruning masks with KDE threshold gradients."""

    @staticmethod
    def forward(ctx, weights: Tensor, threshold: Tensor, bandwidth: float = 0.1) -> Tensor:
        ctx.save_for_backward(weights, threshold)
        ctx.bandwidth = float(bandwidth)
        mask = 0.5 * torch.sign(torch.abs(weights) - threshold) + 0.5
        return mask

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        weights, threshold = ctx.saved_tensors
        bandwidth = float(ctx.bandwidth)
        bw = max(bandwidth, 1e-8)

        # STE-style gradient wrt weights centered around |w| ~= t.
        local = torch.exp(-0.5 * ((torch.abs(weights) - threshold) / bw) ** 2)
        grad_weights = grad_output * local * torch.sign(weights)

        # KDE estimate for dr/dt = p(t) + p(-t), used as threshold gradient surrogate.
        _, _, dr_dt = estimate_density_at_threshold(weights, threshold, bandwidth=bw)
        dmask_dt = -dr_dt

        if threshold.numel() == 1:
            grad_threshold = grad_output.sum() * dmask_dt
        else:
            grad_threshold = grad_output * dmask_dt
            grad_threshold = grad_threshold.sum_to_size(threshold.shape)

        return grad_weights, grad_threshold, None


def _default_feature_map_adapter(
    model: nn.Module,
    batch: Any,
    targets: Any = None,
    params_override: Mapping[str, Tensor] | None = None,
) -> List[Tensor]:
    if params_override is None:
        outputs = model(batch) if targets is None else model(batch, targets)
    else:
        args = (batch,) if targets is None else (batch, targets)
        outputs = functional_call(model, params_override, args)
    return _flatten_tensors(outputs)


class CalibrationRunner:
    """Calibration engine for FCPTS sparse threshold learning."""

    def __init__(
        self,
        dense_model: nn.Module,
        sparse_model: nn.Module,
        output_adapter: OutputAdapter | None = None,
        target_sparsity: float = 0.5,
        lambda_control: float = 1.0,
        bandwidth: float = 0.1,
        prunable_param_names: Iterable[str] | None = None,
    ):
        self.dense_model = dense_model.eval()
        self.sparse_model = sparse_model.train()
        self.output_adapter = output_adapter or _default_feature_map_adapter
        self.target_sparsity = float(target_sparsity)
        self.lambda_control = float(lambda_control)
        self.bandwidth = float(bandwidth)

        for param in self.dense_model.parameters():
            param.requires_grad_(False)

        all_names = [name for name, _ in self.sparse_model.named_parameters()]
        selected_names = set(prunable_param_names) if prunable_param_names else {
            name for name, param in self.sparse_model.named_parameters() if param.requires_grad and param.ndim > 1
        }
        self.prunable_param_names = [name for name in all_names if name in selected_names]

        self._name_to_key = {name: _safe_key(name) for name in self.prunable_param_names}
        thresholds: Dict[str, nn.Parameter] = {}
        for name, param in self.sparse_model.named_parameters():
            if name not in self._name_to_key:
                continue
            q = min(max(self.target_sparsity, 0.0), 1.0)
            init = torch.quantile(param.detach().abs().reshape(-1), q) if param.numel() > 0 else param.new_tensor(0.0)
            thresholds[self._name_to_key[name]] = nn.Parameter(init.to(param.dtype))
        self.thresholds = nn.ParameterDict(thresholds)

    def _masked_parameter_map(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        masked_params: Dict[str, Tensor] = {}
        masks: Dict[str, Tensor] = {}
        for name, param in self.sparse_model.named_parameters():
            if name in self._name_to_key:
                threshold = self.thresholds[self._name_to_key[name]]
                mask = DifferentiablePruningMaskFn.apply(param, threshold, self.bandwidth)
                masked_params[name] = param * mask
                masks[name] = mask
            else:
                masked_params[name] = param
        return masked_params, masks

    def _global_sparsity(self, masks: Mapping[str, Tensor]) -> Tensor:
        weighted_sum = None
        total_params = 0
        for name, mask in masks.items():
            n_i = mask.numel()
            r_i = 1.0 - mask.mean()
            contribution = r_i * float(n_i)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
            total_params += n_i
        if weighted_sum is None or total_params == 0:
            return torch.tensor(0.0, device=next(self.sparse_model.parameters()).device)
        return weighted_sum / float(total_params)

    def _adapter_call(
        self,
        model: nn.Module,
        batch: Any,
        targets: Any = None,
        params_override: Mapping[str, Tensor] | None = None,
    ) -> List[Tensor]:
        try:
            outputs = self.output_adapter(model, batch, targets=targets, params_override=params_override)
        except TypeError as exc:
            raise TypeError(
                "output_adapter must accept (model, batch, targets=None, params_override=None)."
            ) from exc
        if not isinstance(outputs, list) or any(not torch.is_tensor(t) for t in outputs):
            raise TypeError("output_adapter must return List[torch.Tensor].")
        return outputs

    def compute_losses(self, batch: Any, targets: Any = None) -> Dict[str, Tensor]:
        with torch.no_grad():
            dense_outputs = self._adapter_call(self.dense_model, batch, targets=targets, params_override=None)

        masked_params, masks = self._masked_parameter_map()
        sparse_outputs = self._adapter_call(
            self.sparse_model,
            batch,
            targets=targets,
            params_override=masked_params,
        )

        if len(dense_outputs) != len(sparse_outputs):
            raise ValueError("Adapter outputs mismatch: dense and sparse output counts are different.")

        mse_terms: List[Tensor] = []
        for dense_tensor, sparse_tensor in zip(dense_outputs, sparse_outputs):
            if dense_tensor.shape != sparse_tensor.shape:
                raise ValueError("Adapter outputs mismatch: dense and sparse tensor shapes differ.")
            mse_terms.append(F.mse_loss(sparse_tensor, dense_tensor))
        l_rec = torch.stack(mse_terms).mean() if mse_terms else torch.tensor(0.0, device=next(self.sparse_model.parameters()).device)

        global_sparsity = self._global_sparsity(masks)
        target = global_sparsity.new_tensor(self.target_sparsity)
        l_c = torch.abs(global_sparsity - target)
        loss = l_rec + (self.lambda_control * l_c)
        return {
            "loss": loss,
            "l_rec": l_rec,
            "l_c": l_c,
            "global_sparsity": global_sparsity,
        }

    def optimizable_parameters(self) -> List[nn.Parameter]:
        params = [param for name, param in self.sparse_model.named_parameters() if name in self.prunable_param_names]
        params.extend(self.thresholds.parameters())
        return params

    def threshold_registry(self) -> Dict[str, Tensor]:
        return {name: self.thresholds[key] for name, key in self._name_to_key.items()}


def _extract_batch(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, None


def _create_csv_logger(csv_path: str | Path) -> logging.Logger:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"fcpts.calibration.{abs(hash(str(csv_path.resolve())))}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(csv_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    if csv_path.stat().st_size == 0:
        logger.info("iteration,epoch,loss,l_rec,l_c,global_sparsity")
    return logger


def calibrate_model(
    runner: CalibrationRunner,
    dataloader: Iterable[Any],
    epochs: int = 1,
    lr: float = 1e-3,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    metrics_csv_path: str | Path = "logs/fcpts_calibration.csv",
) -> CalibrationRunner:
    """Run FCPTS calibration loop with tqdm progress and CSV logging."""

    resolved_device = device or next(runner.sparse_model.parameters()).device
    runner.dense_model.to(resolved_device)
    runner.sparse_model.to(resolved_device)

    opt = optimizer or torch.optim.Adam(runner.optimizable_parameters(), lr=lr)
    csv_logger = _create_csv_logger(metrics_csv_path)

    handlers = list(csv_logger.handlers)
    try:
        iteration = 0
        for epoch in range(epochs):
            progress = tqdm(dataloader, desc=f"Calibration {epoch + 1}/{epochs}")
            for raw_batch in progress:
                iteration += 1
                batch_inputs, batch_targets = _extract_batch(raw_batch)
                batch_inputs = _to_device(batch_inputs, resolved_device)
                batch_targets = _to_device(batch_targets, resolved_device) if batch_targets is not None else None

                opt.zero_grad(set_to_none=True)
                losses = runner.compute_losses(batch_inputs, targets=batch_targets)
                losses["loss"].backward()
                opt.step()

                loss_value = float(losses["loss"].detach().item())
                l_rec_value = float(losses["l_rec"].detach().item())
                l_c_value = float(losses["l_c"].detach().item())
                sparsity_value = float(losses["global_sparsity"].detach().item())

                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    l_rec=f"{l_rec_value:.4f}",
                    l_c=f"{l_c_value:.4f}",
                    sparsity=f"{sparsity_value:.4f}",
                )
                csv_logger.info(
                    f"{iteration},{epoch + 1},{loss_value:.8f},{l_rec_value:.8f},{l_c_value:.8f},{sparsity_value:.8f}"
                )
    finally:
        for handler in handlers:
            handler.flush()
            handler.close()
            csv_logger.removeHandler(handler)
    return runner


def finalize_and_export(
    sparse_model: nn.Module,
    threshold_registry: Mapping[str, Tensor],
    bandwidth: float = 0.1,
) -> nn.Module:
    """Apply learned masks permanently to sparse model weights."""

    del bandwidth  # interface compatibility for future variants

    with torch.no_grad():
        for name, param in sparse_model.named_parameters():
            threshold = threshold_registry.get(name)
            if threshold is None or param.ndim <= 1:
                continue
            threshold_tensor = threshold if torch.is_tensor(threshold) else torch.tensor(threshold, device=param.device, dtype=param.dtype)
            threshold_tensor = threshold_tensor.to(device=param.device, dtype=param.dtype)
            mask = 0.5 * torch.sign(torch.abs(param) - threshold_tensor) + 0.5
            param.mul_(mask)
    return sparse_model
