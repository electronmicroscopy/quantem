"""Inference normalization strategies for polymer peak-detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NormalizationStrategy(Protocol):
    """A fittable, reusable normalization operation."""

    def fit(self, sample_batch: Any) -> Any:
        """Fit parameters from a representative batch."""

    def transform(self, batch: Any, parameters: Any) -> Any:
        """Normalize a batch using previously fitted parameters."""


def _is_torch(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _percentiles(batch: Any, lower: float, upper: float) -> tuple[float, float]:
    if _is_torch(batch):
        import torch

        flat = batch.detach().flatten().float()
        if flat.numel() == 0:
            raise ValueError("Cannot fit normalization from an empty batch.")
        values = torch.quantile(
            flat,
            torch.tensor(
                [lower / 100.0, upper / 100.0],
                device=flat.device,
                dtype=flat.dtype,
            ),
        )
        return float(values[0].item()), float(values[1].item())

    array = np.asarray(batch)
    if array.size == 0:
        raise ValueError("Cannot fit normalization from an empty batch.")
    values = np.percentile(array, [lower, upper])
    return float(values[0]), float(values[1])


def _percentile_transform(batch: Any, parameters: tuple[float, float]) -> Any:
    lower, upper = parameters
    if _is_torch(batch):
        import torch

        if not (batch.dtype.is_floating_point or batch.dtype.is_complex):
            batch = batch.float()
        lo = torch.as_tensor(lower, device=batch.device, dtype=batch.dtype)
        hi = torch.as_tensor(upper, device=batch.device, dtype=batch.dtype)
        return (torch.clamp(batch, lo, hi) - lo) / (hi - lo + 1e-8)
    array = np.asarray(batch)
    return (np.clip(array, lower, upper) - lower) / (upper - lower + 1e-8)


def _per_image_minmax(batch: Any) -> Any:
    """Min-max each image over its final two dimensions."""

    if getattr(batch, "ndim", None) is None or batch.ndim < 2:
        raise ValueError("Normalization expects an image or a batch of images.")
    axes = (-2, -1)
    if _is_torch(batch):
        import torch

        if not (batch.dtype.is_floating_point or batch.dtype.is_complex):
            batch = batch.float()
        minimum = torch.amin(batch, dim=axes, keepdim=True)
        maximum = torch.amax(batch, dim=axes, keepdim=True)
        span = maximum - minimum
        return torch.where(span > 0, (batch - minimum) / span, torch.zeros_like(batch))
    array = np.asarray(batch)
    minimum = np.min(array, axis=axes, keepdims=True)
    maximum = np.max(array, axis=axes, keepdims=True)
    span = maximum - minimum
    return np.divide(
        array - minimum,
        span,
        out=np.zeros_like(array, dtype=np.result_type(array.dtype, np.float32)),
        where=span > 0,
    )


@dataclass(frozen=True)
class GlobalPercentileNormalization:
    """Clip and scale using percentiles fitted across the entire sample batch."""

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    def fit(self, sample_batch: Any) -> tuple[float, float]:
        return _percentiles(
            sample_batch, self.lower_percentile, self.upper_percentile
        )

    def transform(self, batch: Any, parameters: Any) -> Any:
        return _percentile_transform(batch, parameters)


@dataclass(frozen=True)
class PerImageMinMaxPercentileNormalization:
    """Min-max each image, then clip and scale by fitted global percentiles."""

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    def fit(self, sample_batch: Any) -> tuple[float, float]:
        normalized = _per_image_minmax(sample_batch)
        return _percentiles(
            normalized, self.lower_percentile, self.upper_percentile
        )

    def transform(self, batch: Any, parameters: Any) -> Any:
        return _percentile_transform(_per_image_minmax(batch), parameters)


@dataclass(frozen=True)
class LegacyNormalizationAdapter:
    """Adapt the historical compute/normalize callback pair to the strategy API."""

    compute_parameters: Callable[..., Any]
    normalize_data: Callable[..., Any]
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    def fit(self, sample_batch: Any) -> Any:
        return self.compute_parameters(
            sample_batch,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )

    def transform(self, batch: Any, parameters: Any) -> Any:
        if isinstance(parameters, tuple):
            return self.normalize_data(batch, *parameters)
        return self.normalize_data(batch, parameters)


_STRATEGIES: dict[str, type[NormalizationStrategy]] = {
    "global_percentile": GlobalPercentileNormalization,
    "v1_global_percentile": GlobalPercentileNormalization,
    "per_scan_percentile": GlobalPercentileNormalization,
    "per_image_minmax_percentile": PerImageMinMaxPercentileNormalization,
    "v2_per_image_minmax_percentile": PerImageMinMaxPercentileNormalization,
}


def resolve_normalization_strategy(
    specification: str | Mapping[str, Any] | NormalizationStrategy,
) -> NormalizationStrategy:
    """Resolve a registered strategy name/configuration or return a strategy instance."""

    if isinstance(specification, str):
        mode, parameters = specification, {}
    elif isinstance(specification, Mapping):
        config = dict(specification)
        try:
            mode = str(config.pop("mode"))
        except KeyError as exc:
            raise ValueError("Normalization configuration requires a 'mode'.") from exc
        parameters = config
    elif isinstance(specification, NormalizationStrategy):
        return specification
    else:
        raise TypeError(
            "normalization_strategy must be a registered name, configuration mapping, "
            "or object implementing fit() and transform()."
        )

    strategy_type = _STRATEGIES.get(mode)
    if strategy_type is None:
        raise ValueError(
            f"Unknown normalization strategy {mode!r}; registered strategies are "
            f"{sorted(_STRATEGIES)}."
        )
    if "p_lower" in parameters:
        parameters.setdefault("lower_percentile", parameters.pop("p_lower"))
    if "p_upper" in parameters:
        parameters.setdefault("upper_percentile", parameters.pop("p_upper"))
    return strategy_type(**parameters)


# Concise aliases for callers that prefer strategy-oriented names.
GlobalPercentileStrategy = GlobalPercentileNormalization
PerImageMinMaxPercentileStrategy = PerImageMinMaxPercentileNormalization


__all__ = [
    "GlobalPercentileNormalization",
    "GlobalPercentileStrategy",
    "LegacyNormalizationAdapter",
    "NormalizationStrategy",
    "PerImageMinMaxPercentileNormalization",
    "PerImageMinMaxPercentileStrategy",
    "resolve_normalization_strategy",
]
