from typing import Any, Self

import numpy as np
import torch


class Dataset4dstemTorch:
    """Minimal torch-backed 4D-STEM container.

    Holds a 4D ``torch.Tensor`` (any device) plus standard metadata fields
    (``array``, ``name``, ``origin``, ``sampling``, ``units``,
    ``signal_units``).

    Goal: keep 4D-STEM data on the GPU end-to-end and skip VRAM <-> RAM
    round-trips. Each hop is expensive on multi-GB datasets and doubles
    peak memory. Wrapping a cupy or torch GPU array here is effectively
    free and holds the original VRAM allocation.
    """

    def __init__(
        self,
        array: torch.Tensor,
        name: str = "4dstem (torch)",
        origin: tuple[float, ...] | list[float] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        signal_units: str = "arb. units",
    ):
        if not (isinstance(array, torch.Tensor) and array.ndim == 4):
            raise TypeError("Dataset4dstemTorch requires a 4D torch tensor")
        self.array = array
        self.name = name
        self.origin = np.asarray(origin if origin is not None else (0, 0, 0, 0), dtype=float)
        self.sampling = np.asarray(sampling if sampling is not None else (1, 1, 1, 1), dtype=float)
        self.units = list(units) if units is not None else ["pixels"] * 4
        self.signal_units = signal_units

    @classmethod
    def from_array(
        cls,
        array: Any,
        name: str | None = None,
        origin: tuple[float, ...] | list[float] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create from a torch tensor or any dlpack-compatible array.

        torch tensors pass through. cupy / jax / other GPU arrays wrap
        zero-copy via ``torch.from_dlpack``.
        """
        if not isinstance(array, torch.Tensor):
            array = torch.from_dlpack(array)
        return cls(
            array=array,
            name=name if name is not None else "4dstem (torch)",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )
