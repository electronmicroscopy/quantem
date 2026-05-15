from typing import Any, Self

import numpy as np
import torch


class Dataset4dstemTorch:
    """Torch-backed 4D-STEM container. GPU counterpart to ``Dataset4dstem``.

    Same metadata surface (``array``, ``name``, ``origin``, ``sampling``,
    ``units``, ``signal_units``) but ``array`` is a 4D ``torch.Tensor``
    on any device. Use when the raw data already lives on the GPU (CUDA
    pipelines, live detector frames, GPU file readers) so it stays in
    VRAM end-to-end. Wrapping with ``Dataset4dstem`` instead requires a
    device-to-host transfer on wrap and a host-to-device transfer on
    consume, doubling peak memory.
    """

    _token = object()

    def __init__(
        self,
        array: torch.Tensor,
        name: str = "4dstem (torch)",
        origin: tuple[float, ...] | list[float] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset4dstemTorch.from_array() to instantiate this class.")
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
            _token=cls._token,
        )
