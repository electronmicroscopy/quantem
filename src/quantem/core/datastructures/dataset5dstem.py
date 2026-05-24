from typing import Iterator, Self

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.validators import validate_ndinfo, validate_units


_SERIES_TYPES = ("time", "tilt", "energy", "dose", "focus", "generic")


class Dataset5dstem(Dataset):
    """**EXPERIMENTAL.** Torch-backed 5D-STEM series ``(N, scan_row, scan_col, k_row, k_col)``.

    Stack of 4D-STEM acquisitions sharing identical scan + k calibration. Axis 0
    represents ONE monotonically varying experimental parameter (time, tilt,
    focus, dose, energy, generic).

    ``sampling`` / ``units`` / ``origin`` are 4-length (scan + k only) - the
    series axis is described separately by ``series_type`` + ``series``. This
    diverges from base Dataset's ``len(sampling) == ndim`` convention but keeps
    the user-facing API clean (no axis-0 placeholders).

    Single-tensor / single-device only. Sharding deferred. API is experimental.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        name: str = "",
        sampling: NDArray | tuple | list | None = None,
        units: list[str] | tuple | list | None = None,
        origin: NDArray | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        series_type: str = "generic",
        series: NDArray | list | tuple | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use Dataset5dstem.from_tensor() or Dataset5dstem.from_4dstem() to instantiate."
            )
        if series_type not in _SERIES_TYPES:
            raise ValueError(f"series_type must be one of {_SERIES_TYPES}, got {series_type!r}.")
        super().__init__(
            tensor=tensor, name=name,
            sampling=sampling, units=units, origin=origin,
            signal_units=signal_units, metadata=metadata, _token=_token,
        )
        self.series_type = series_type
        self.series = series

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        name: str | None = None,
        sampling: NDArray | tuple | list | None = None,
        units: list[str] | tuple | list | None = None,
        origin: NDArray | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        series_type: str = "generic",
        series: NDArray | list | tuple | None = None,
    ) -> Self:
        """Wrap a 5D torch tensor. ``sampling`` / ``units`` / ``origin`` are 4-length
        (scan_row, scan_col, k_row, k_col); axis 0 lives in ``series_type`` + ``series``.
        """
        if tensor.ndim != 5:
            raise ValueError(
                f"from_tensor requires 5D tensor (N, scan_row, scan_col, k_row, k_col), "
                f"got shape {tuple(tensor.shape)}."
            )
        return cls(
            tensor=tensor,
            name=name if name is not None else "5D-STEM dataset (torch)",
            sampling=sampling if sampling is not None else np.ones(4),
            units=units if units is not None else ["pixels"] * 4,
            origin=origin if origin is not None else np.zeros(4),
            signal_units=signal_units, metadata=metadata,
            series_type=series_type, series=series,
            _token=cls._token,
        )

    @classmethod
    def from_4dstem(
        cls,
        datasets: list[Dataset4dstem],
        name: str | None = None,
        series_type: str = "generic",
        series: NDArray | list | tuple | None = None,
    ) -> Self:
        """Stack tensor-backed Dataset4dstem (same device). Spatial cal inherits from first."""
        member_tensors = [d.tensor for d in datasets]
        devices = {str(t.device) for t in member_tensors}
        if len(devices) > 1:
            raise ValueError(
                f"All Dataset4dstem must share device; got {sorted(devices)}. "
                f"Sharding not yet supported - move to one device first via ds.to('cuda:N')."
            )
        first = datasets[0]
        return cls.from_tensor(
            tensor=torch.stack(member_tensors, dim=0),
            name=name if name is not None else f"{len(datasets)}x {first.name}",
            sampling=first.sampling, units=first.units, origin=first.origin,
            series_type=series_type, series=series,
        )

    # --- Override base sampling/units/origin: 4-length (scan + k), not ndim-length ---
    @property
    def sampling(self) -> NDArray: return self._sampling

    @sampling.setter
    def sampling(self, value) -> None:
        self._sampling = validate_ndinfo(value, 4, "sampling")

    @property
    def origin(self) -> NDArray: return self._origin

    @origin.setter
    def origin(self, value) -> None:
        self._origin = validate_ndinfo(value, 4, "origin")

    @property
    def units(self) -> list[str]: return self._units

    @units.setter
    def units(self, value) -> None:
        self._units = validate_units(value, 4)

    # --- Series metadata ---
    @property
    def series(self) -> NDArray | None:
        return self._series

    @series.setter
    def series(self, value) -> None:
        if value is None:
            self._series = None
            return
        arr = np.asarray(value, dtype=float)
        n = len(self)
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"series must be 1D length {n}, got shape {arr.shape}.")
        self._series = arr

    # --- Frame access ---
    def __len__(self) -> int:
        return int(self._tensor.shape[0])

    def __getitem__(self, index: int | slice) -> Dataset4dstem | Self:
        if isinstance(index, int):
            return Dataset4dstem.from_tensor(
                self._tensor[index],
                name=f"{self.name}[{index}]",
                sampling=self.sampling, units=self.units,
            )
        sub_series = None if self._series is None else self._series[index]
        return Dataset5dstem.from_tensor(
            tensor=self._tensor[index],
            name=self.name,
            sampling=self.sampling, units=self.units, origin=self.origin,
            signal_units=self.signal_units, metadata=self._metadata,
            series_type=self.series_type, series=sub_series,
        )

    def __iter__(self) -> Iterator[Dataset4dstem]:
        for i in range(len(self)):
            yield self[i]
