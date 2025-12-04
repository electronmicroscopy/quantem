from __future__ import annotations

from collections.abc import Sequence
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array


class RDF(AutoSerialize):
    """
    Radial distribution / fluctuation electron microscopy analysis helper.

    This class wraps a 4D-STEM (or 2D diffraction) dataset and stores a
    polar-transformed representation as a Polar4dstem instance in `self.polar`.
    Analysis methods (radial statistics, PDF, FEM, clustering, etc.) are
    provided as stubs for now and will be implemented in future revisions.
    """

    _token = object()

    def __init__(
        self,
        polar: Polar4dstem,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use RadialDistributionFunction.from_data() to instantiate this class."
            )

        super().__init__()
        self.polar = polar
        self.input_data = input_data

        # Placeholders for analysis results (to be populated by future methods)
        self.radial_mean: NDArray | None = None
        self.radial_var: NDArray | None = None
        self.radial_var_norm: NDArray | None = None

        self.pdf_r: NDArray | None = None
        self.pdf_reduced: NDArray | None = None
        self.pdf: NDArray | None = None

        self.Sk: NDArray | None = None
        self.fk: NDArray | None = None
        self.bg: NDArray | None = None
        self.offset: float | None = None
        self.Sk_mask: NDArray | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: Union[NDArray, Dataset2d, Dataset3d, Dataset4dstem, Polar4dstem],
        *,
        origin_row: float | None = None,
        origin_col: float | None = None,
        ellipse_params: tuple[float, float, float] | None = None,
        num_annular_bins: int = 180,
        radial_min: float = 0.0,
        radial_max: float | None = None,
        radial_step: float = 1.0,
        two_fold_rotation_symmetry: bool = False,
    ) -> "RadialDistributionFunction":
        """
        Create a RadialDistributionFunction object from various input types.

        Parameters
        ----------
        data
            Supported inputs:
            - 2D numpy array (single diffraction pattern)
            - 4D numpy array (scan_y, scan_x, ky, kx)
            - Dataset2d
            - Dataset4dstem
            - Polar4dstem
        origin_row, origin_col
            Diffraction-space origin (in pixels). If None, defaults to the
            central pixel of the diffraction pattern.
        Other parameters
            Passed through to Dataset4dstem.polar_transform when needed.
        """
        # Polar input: use directly
        if isinstance(data, Polar4dstem):
            polar = data
            return cls(polar=polar, input_data=data, _token=cls._token)

        # Dataset4dstem input: polar-transform it
        if isinstance(data, Dataset4dstem):
            scan_y, scan_x, ny, nx = data.array.shape
            if origin_row is None:
                origin_row = (ny - 1) / 2.0
            if origin_col is None:
                origin_col = (nx - 1) / 2.0

            polar = data.polar_transform(
                origin_row=origin_row,
                origin_col=origin_col,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            )
            return cls(polar=polar, input_data=data, _token=cls._token)

        # Dataset2d input: wrap as a trivial 4D-STEM (1x1 scan) then polar-transform
        if isinstance(data, Dataset2d):
            arr2d = data.array
            if arr2d.ndim != 2:
                raise ValueError("Dataset2d for RDF must be 2D.")
            arr4 = arr2d[None, None, ...]  # (1, 1, ky, kx)

            ds4 = Dataset4dstem.from_array(
                array=arr4,
                name=f"{data.name}_as4dstem" if getattr(data, "name", None) else "rdf_4dstem_from_2d",
                origin=np.concatenate(
                    [np.zeros(2, dtype=float), np.asarray(data.origin, dtype=float)]
                ),
                sampling=np.concatenate(
                    [np.ones(2, dtype=float), np.asarray(data.sampling, dtype=float)]
                ),
                units=["pixels", "pixels"] + list(data.units),
                signal_units=data.signal_units,
            )
            ny, nx = ds4.array.shape[-2:]
            if origin_row is None:
                origin_row = (ny - 1) / 2.0
            if origin_col is None:
                origin_col = (nx - 1) / 2.0

            polar = ds4.polar_transform(
                origin_row=origin_row,
                origin_col=origin_col,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            )
            return cls(polar=polar, input_data=data, _token=cls._token)

        # Dataset3d input: not yet specified how to interpret
        if isinstance(data, Dataset3d):
            raise NotImplementedError(
                "RadialDistributionFunction.from_data does not yet support Dataset3d inputs."
            )

        # Numpy array input
        arr = ensure_valid_array(data)
        if arr.ndim == 2:
            ds2 = Dataset2d.from_array(arr, name="rdf_input_2d")
            return cls.from_data(
                ds2,
                origin_row=origin_row,
                origin_col=origin_col,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            )
        elif arr.ndim == 4:
            ds4 = Dataset4dstem.from_array(arr, name="rdf_input_4dstem")
            return cls.from_data(
                ds4,
                origin_row=origin_row,
                origin_col=origin_col,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            )
        else:
            raise ValueError(
                "RadialDistributionFunction.from_data only supports 2D or 4D arrays."
            )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def qq(self) -> Any:
        """
        Scattering vector coordinate array along the radial dimension of `self.polar`,
        in physical units (using Polar4dstem.sampling and origin).
        """
        # Polar4dstem dims: (scan_y, scan_x, phi, r)
        # radial axis is 3
        return self.polar.coords_units(3)

    @property
    def radial_bins(self) -> Any:
        """
        Radial bin centers in pixel units (convenience alias).
        """
        return self.polar.coords(3)

    # ------------------------------------------------------------------
    # Analysis method stubs (py4DSTEM-style API)
    # ------------------------------------------------------------------
    def calculate_radial_statistics(
        self,
        mask_realspace: NDArray | None = None,
        plot_results_mean: bool = False,
        plot_results_var: bool = False,
        figsize: tuple[float, float] = (8, 4),
        returnval: bool = False,
        returnfig: bool = False,
        progress_bar: bool = True,
    ):
        """
        Stub for radial statistics (FEM-style) calculation on the polar data.

        Intended to compute radial mean, variance, and normalized variance
        from self.polar. Not implemented yet.
        """
        raise NotImplementedError("calculate_radial_statistics is not implemented yet.")

    def plot_radial_mean(
        self,
        log_x: bool = False,
        log_y: bool = False,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting radial mean intensity vs scattering vector.
        """
        raise NotImplementedError("plot_radial_mean is not implemented yet.")

    def plot_radial_var_norm(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting normalized radial variance vs scattering vector.
        """
        raise NotImplementedError("plot_radial_var_norm is not implemented yet.")

    def calculate_pair_dist_function(
        self,
        k_min: float = 0.05,
        k_max: float | None = None,
        k_width: float = 0.25,
        k_lowpass: float | None = None,
        k_highpass: float | None = None,
        r_min: float = 0.0,
        r_max: float = 20.0,
        r_step: float = 0.02,
        damp_origin_fluctuations: bool = True,
        enforce_positivity: bool = True,
        density: float | None = None,
        plot_background_fits: bool = False,
        plot_sf_estimate: bool = False,
        plot_reduced_pdf: bool = True,
        plot_pdf: bool = False,
        figsize: tuple[float, float] = (8, 4),
        maxfev: int | None = None,
        returnval: bool = False,
        returnfig: bool = False,
    ):
        """
        Stub for pair distribution function (PDF) calculation from radial statistics.

        Intended to estimate S(k), background, and transform to real-space g(r)/G(r).
        """
        raise NotImplementedError("calculate_pair_dist_function is not implemented yet.")

    def plot_background_fits(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting background fit vs radial mean intensity.
        """
        raise NotImplementedError("plot_background_fits is not implemented yet.")

    def plot_sf_estimate(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting reduced structure factor S(k).
        """
        raise NotImplementedError("plot_sf_estimate is not implemented yet.")

    def plot_reduced_pdf(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting reduced PDF g(r).
        """
        raise NotImplementedError("plot_reduced_pdf is not implemented yet.")

    def plot_pdf(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting full PDF G(r).
        """
        raise NotImplementedError("plot_pdf is not implemented yet.")
