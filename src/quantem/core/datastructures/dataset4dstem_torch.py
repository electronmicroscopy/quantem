from typing import Self

import numpy as np
import torch


class Dataset4dstemTorch:
    """A torch-backed 4D-STEM dataset class. GPU counterpart to ``Dataset4dstem``.

    Same metadata surface as ``Dataset4dstem`` (``array``, ``name``,
    ``origin``, ``sampling``, ``units``, ``signal_units``) but ``array``
    is a 4D ``torch.Tensor`` on any device instead of a numpy array. Use
    when the raw data already lives on the GPU (CUDA pipelines, live
    detector frames, GPU file readers) so it stays in VRAM end-to-end.
    Wrapping with ``Dataset4dstem`` instead requires a device-to-host
    transfer on wrap and a host-to-device transfer on consume, doubling
    peak memory.

    Notes
    -----
    Construction is gated by a token: use ``Dataset4dstemTorch.from_array``
    rather than calling the constructor directly.
    """

    _token = object()

    def __init__(
        self,
        array: torch.Tensor,
        name: str = "4D-STEM dataset (torch)",
        origin: tuple[float, ...] | list[float] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a torch-backed 4D-STEM dataset.

        Parameters
        ----------
        array : torch.Tensor
            The underlying 4D tensor data on any device.
        name : str, optional
            A descriptive name for the dataset, by default "4D-STEM dataset (torch)".
        origin : tuple[float, ...] | list[float] | None, optional
            The origin coordinates for each dimension in calibrated units.
            If None, defaults to zeros.
        sampling : tuple[float, ...] | list[float] | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones.
        units : list[str] | tuple[str, ...] | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4.
        signal_units : str, optional
            Units for the array values, by default "arb. units".
        _token : object | None, optional
            Token to prevent direct instantiation, by default None.
        """
        if _token is not self._token:
            raise RuntimeError(
                "Use Dataset4dstemTorch.from_array() to instantiate this class."
            )
        if not isinstance(array, torch.Tensor):
            raise TypeError(
                f"Dataset4dstemTorch requires a torch.Tensor, got {type(array).__name__}. "
                f"Convert with torch.from_dlpack(array) for cupy / jax arrays, or use "
                f"Dataset4dstem for numpy."
            )
        if array.ndim != 4:
            raise ValueError(
                f"Dataset4dstemTorch requires a 4D tensor (scan_rows, scan_cols, "
                f"det_rows, det_cols), got shape {tuple(array.shape)} "
                f"({array.ndim}D)."
            )
        self.array = array
        self.name = name
        self.origin = np.asarray(origin if origin is not None else (0, 0, 0, 0), dtype=float)
        self.sampling = np.asarray(sampling if sampling is not None else (1, 1, 1, 1), dtype=float)
        self.units = list(units) if units is not None else ["pixels"] * 4
        self.signal_units = signal_units

    @classmethod
    def from_array(
        cls,
        array: object,
        name: str | None = None,
        origin: tuple[float, ...] | list[float] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a new Dataset4dstemTorch from a torch tensor or dlpack-compatible array.

        torch tensors pass through. Cupy / jax / other GPU arrays exposing
        the dlpack protocol wrap zero-copy via ``torch.from_dlpack``.

        Parameters
        ----------
        array : object
            A 4D ``torch.Tensor`` or any object exposing the dlpack
            protocol (e.g. ``cupy.ndarray``, ``jax.Array``). Non-tensor
            inputs wrap zero-copy via ``torch.from_dlpack``.
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to
            "4D-STEM dataset (torch)".
        origin : tuple[float, ...] | list[float] | None, optional
            The origin coordinates for each dimension in calibrated units.
            If None, defaults to zeros.
        sampling : tuple[float, ...] | list[float] | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones.
        units : list[str] | tuple[str, ...] | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4.
        signal_units : str, optional
            Units for the array values, by default "arb. units".

        Returns
        -------
        Dataset4dstemTorch
            A new Dataset4dstemTorch instance wrapping ``array`` in place.

        Examples
        --------
        >>> import torch
        >>> ds = Dataset4dstemTorch.from_array(torch.rand(64, 64, 128, 128, device="cuda"))
        >>> ds.array.device
        device(type='cuda', index=0)

        >>> import cupy as cp
        >>> ds = Dataset4dstemTorch.from_array(cp.zeros((64, 64, 128, 128), dtype=cp.uint16))
        """
        if not isinstance(array, torch.Tensor):
            array = torch.from_dlpack(array)
        return cls(
            array=array,
            name=name if name is not None else "4D-STEM dataset (torch)",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=cls._token,
        )
