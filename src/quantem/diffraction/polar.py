from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array

KIRKLAND_PARAMS_PATH = Path(__file__).with_name("kirkland_params.json")


class PairDistributionFunction(AutoSerialize):
    """
    Pair distribution function (PDF) utilities for diffraction / 4D-STEM data.

    This class wraps a 4D-STEM (or 2D diffraction) dataset and stores a
    polar-transformed representation as a Polar4dstem instance in `self.polar`.
    The PDF pipeline provides methods to compute:

    - azimuthal integration to obtain I(k)
    - background fitting using a parametric model in k^2 / k^4
    - formation of F(k) and a windowed sine transform to obtain G(r)
    - optional density estimation and origin correction (Yoshimoto & Omote-style iteration)
    - basic plotting helpers for I(k), background, F(k), G(r), and g(r)
    Some analysis methods (FEM, clustering, etc.) will be implemented in future revisions.
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
                "Use PairDistributionFunction.from_data() to instantiate this class."
            )

        super().__init__()
        self.polar = polar
        self.input_data = input_data

        # Placeholders for analysis results (to be populated by future methods)
        self.radial_mean: NDArray | None = None
        self.radial_var: NDArray | None = None
        self.radial_var_norm: NDArray | None = None

        self.pdf_r: NDArray | None = None
        self.reduced_pdf: NDArray | None = None
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
        find_origin: bool = True,
        origin_row: float | None = None,
        origin_col: float | None = None,
        ellipse_params: tuple[float, float, float] | None = None,
        num_annular_bins: int = 180,
        radial_min: float = 0.0,
        radial_max: float | None = None,
        radial_step: float = 1.0,
        two_fold_rotation_symmetry: bool = False,
    ):
        """
         -> "PairDistributionFunction"
        Create a PairDistributionFunction object from various input types.

        Parameters
        ----------
        data
            Supported inputs:
            - 2D numpy array (single diffraction pattern)
            - 4D numpy array (scan_y, scan_x, ky, kx)
            - Dataset2d
            - Dataset4dstem
            - Polar4dstem

            If a :class:`Polar4dstem` is provided, it is used directly and no origin finding
            or polar transform is performed.
        find_origin
            If True, finds the origin for each scan position by calling
            :meth:`find_origin`. If False, `origin_row`/`origin_col` are used (or default
            to the image center).
        origin_row, origin_col
            Diffraction-space origin (in pixels), used only if `find_origin=False`. If None,
            defaults to the central pixel of the diffraction pattern.
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
            if find_origin:
                origin_array = cls.find_origin(
                    data,
                    ellipse_params=ellipse_params,
                    num_annular_bins=num_annular_bins,
                    radial_min=radial_min,
                    radial_max=radial_max,
                    radial_step=radial_step,
                    two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                )
            else:
                if origin_row is None:
                    origin_row = (ny - 1) / 2.0
                if origin_col is None:
                    origin_col = (nx - 1) / 2.0
                origin_array = np.zeros((scan_y, scan_x, 2), dtype=float)
                origin_array[..., 0] = origin_row
                origin_array[..., 1] = origin_col

            polar = data.polar_transform(
                origin_array=origin_array,
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
                raise ValueError("Dataset2d for PairDistributionFunction must be 2D.")
            arr4 = arr2d[None, None, ...]  # (1, 1, ky, kx)

            ds4 = Dataset4dstem.from_array(
                array=arr4,
                name=f"{data.name}_as4dstem"
                if getattr(data, "name", None)
                else "rdf_4dstem_from_2d",
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

            if find_origin:
                origin_array = cls.find_origin(
                    ds4,
                    ellipse_params=ellipse_params,
                    num_annular_bins=num_annular_bins,
                    radial_min=radial_min,
                    radial_max=radial_max,
                    radial_step=radial_step,
                    two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                )
            else:
                if origin_row is None:
                    origin_row = (ny - 1) / 2.0
                if origin_col is None:
                    origin_col = (nx - 1) / 2.0
                origin_array = np.zeros((1, 1, 2), dtype=float)
                origin_array[0, 0] = [origin_row, origin_col]

            polar = ds4.polar_transform(
                origin_array=origin_array,
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
                "PairDistributionFunction.from_data does not yet support Dataset3d inputs."
            )

        # Numpy array input
        arr = ensure_valid_array(data)
        if arr.ndim == 2:
            ds2 = Dataset2d.from_array(arr, name="rdf_input_2d")

            return cls.from_data(
                ds2,
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
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            )
        else:
            raise ValueError("PairDistributionFunction.from_data only supports 2D or 4D arrays.")

    @staticmethod
    def find_origin(
        data,
        *,
        ellipse_params=None,
        num_annular_bins=180,
        radial_min=0.0,
        radial_max=None,
        radial_step=1.0,
        two_fold_rotation_symmetry=False,
    ):
        """
        Automatic diffraction center finding by minmizing the standard deviation along the annular direction.

        For each scan position, this routine:
        1) Computes a polar transform at an initial origin (image center).
        2) Evaluates the sum of the standard deviation across angle (phi) over a mid-radius band.
        3) Performs a local search over neighboring pixel origins until the
        objective no longer improves.

        Parameters
        ----------
        data
            A :class:`Dataset4dstem` object
        ellipse_params, num_annular_bins, radial_min, radial_max, radial_step, two_fold_rotation_symmetry
            Forwarded to the polar transform call.

        Returns
        -------
        origin_array : np.ndarray
            Array of shape (scan_y, scan_x, 2) containing (row, col) origin estimates in pixels.

        """
        if len(data.array.shape) == 2:
            ny, nx = data.array.shape
            scan_y, scan_x = 1, 1
        elif len(data.array.shape) == 4:
            scan_y, scan_x, ny, nx = data.array.shape
        else:
            raise ValueError("find_origin only supports 2D or 4D-STEM datasets for now.")

        origin_array = np.zeros((scan_y, scan_x, 2), dtype=float)

        max_steps = 1000  # prevent infinite loops

        # start with center of image for now
        estimated_origin_row = (ny - 1) / 2.0
        estimated_origin_col = (nx - 1) / 2.0
        test_origin = np.array([[[estimated_origin_row, estimated_origin_col]]], dtype=float)

        for y_pos in range(scan_y):
            for x_pos in range(scan_x):
                # print(f"Finding origin for scan pos ({y_pos}, {x_pos})")

                coords_cache = {}

                polar = data.polar_transform(
                    origin_array=test_origin,
                    ellipse_params=ellipse_params,
                    num_annular_bins=num_annular_bins,
                    radial_min=radial_min,
                    radial_max=radial_max,
                    radial_step=radial_step,
                    two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                    scan_pos=(y_pos, x_pos),
                )

                min_r = int(np.floor(0.1 * polar.shape[1]))
                max_r = int(np.ceil(0.9 * polar.shape[1]))
                std_est_origin = polar[:, min_r:max_r].std(axis=0)
                std_est_origin_sum = std_est_origin.sum()

                origin_row = int(round(estimated_origin_row))
                origin_col = int(round(estimated_origin_col))
                coords_cache[(origin_row, origin_col)] = std_est_origin_sum

                if y_pos == 0 and x_pos == 0:
                    print(f"Initial std sum at estimated origin: {std_est_origin_sum}")

                converged = False
                best = std_est_origin_sum
                steps = 0
                while not converged and steps < max_steps:
                    steps += 1
                    moved = False

                    neighbors = [
                        (origin_row + dr, origin_col + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if not (dr == 0 and dc == 0)
                    ]
                    neighbors = [(r, c) for (r, c) in neighbors if 0 <= r < ny and 0 <= c < nx]

                    for origin_r, origin_c in neighbors:
                        if (origin_r, origin_c) not in coords_cache:
                            test_origin = np.array([[[origin_r, origin_c]]], dtype=float)
                            polar = data.polar_transform(
                                origin_array=test_origin,
                                ellipse_params=ellipse_params,
                                num_annular_bins=num_annular_bins,
                                radial_min=radial_min,
                                radial_max=radial_max,
                                radial_step=radial_step,
                                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                                scan_pos=(y_pos, x_pos),
                            )
                            std_test = polar[:, min_r:max_r].std(axis=0)
                            coords_cache[(origin_r, origin_c)] = std_test.sum()

                        if coords_cache[(origin_r, origin_c)] < best:
                            origin_row = origin_r
                            origin_col = origin_c
                            best = coords_cache[(origin_r, origin_c)]
                            moved = True
                            print(f"Moved to ({origin_row}, {origin_col}) with std sum {best}")

                    if not moved:
                        converged = True

                if y_pos == 0 and x_pos == 0:
                    print(f"Final std sum at found origin ({origin_row}, {origin_col}): {best}")
                origin_array[y_pos, x_pos, 0] = origin_row
                origin_array[y_pos, x_pos, 1] = origin_col

        return origin_array

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
    # Helper functions
    # ------------------------------------------------------------------
    def _get_mask_bool(self, mask_realspace):
        """
        Normalize a real-space mask specification to a boolean (rx, ry) mask.

        Parameters
        ----------
        mask_realspace
            - None: no masking
            - bool ndarray of shape (rx, ry): True indicates included probe positions
            - array-like of shape (2, 2): two opposite (rx, ry) corners defining a rectangle

        Returns
        -------
        mask_bool : np.ndarray or None
            Boolean mask of shape (rx, ry), or None if `mask_realspace` is None.
        """
        mask_bool = None
        if mask_realspace is not None:
            rx, ry = self.polar.array.shape[:2]
            mask_realspace = np.asarray(mask_realspace)

            # mask given as boolean array
            if mask_realspace.dtype == bool and mask_realspace.shape == (rx, ry):
                mask_bool = mask_realspace

            # mask given as list of corners
            elif mask_realspace.shape == (2, 2):
                (rx1, ry1), (rx2, ry2) = mask_realspace.astype(int)
                rx_min, rx_max = sorted((rx1, rx2))
                ry_min, ry_max = sorted((ry1, ry2))

                # vectorized bounds check
                bad = (rx_min < 0) | (rx_max >= rx) | (ry_min < 0) | (ry_max >= ry)
                if bad:
                    raise ValueError(f"Mask points outside valid range {(rx, ry)}")

                mask_bool = np.zeros((rx, ry), dtype=bool)
                mask_bool[rx_min : rx_max + 1, ry_min : ry_max + 1] = True
            else:
                raise ValueError(
                    "mask_realspace must be boolean array or two opposite (rx, ry) corner points."
                )
        return mask_bool

    @staticmethod
    def _scattering_model(k2, c, i0, s0, i1, s1):
        """
        Background model used for fitting I(k).
        Model form (using k^2 as input):
            c + i0 * exp(-k^2 / (2 s0^2)) + i1 * exp(-k^4 / (2 s1^4))

        Parameters
        ----------
        k2
            Array of k^2 values.
        c, i0, s0, i1, s1
            Model parameters.
        """
        return (
            c
            + i0 * np.exp(k2 / (-2.0 * s0**2))
            + i1 * np.exp((k2**2) / (-2.0 * s1**4))  # k2**2 = k^4
        )

    @staticmethod
    def _lorch_window(k, kmin, kmax):
        """
        Construct a combined low-q taper and high-q Lorch window.

        The returned window is:
        - zero outside [kmin, kmax]
        - smoothly rises from 0→1 near kmin using a sin^2 ramp over 10% of the band
        - applies a Lorch-style sinc factor over the full in-band region:
            sin(pi * k/kmax) / (pi * k/kmax)
        """
        # low q taper
        edge_frac_low = 0.1  # 10% of range at low-q
        edge_width_low = edge_frac_low * (kmax - kmin)

        wk = np.ones_like(k, dtype=float)
        low = (k >= kmin) & (k < kmin + edge_width_low)
        t = (k[low] - kmin) / edge_width_low
        wk[low] = np.sin(0.5 * np.pi * t) ** 2
        wk[k < kmin] = 0.0
        wk[k > kmax] = 0.0

        # high q taper with Lorch window: w(k) = sin(pi*k/kmax)/(pi*k/kmax)
        lorch = np.zeros_like(k, dtype=float)
        inband = (k >= kmin) & (k <= kmax)
        x = k[inband] / kmax
        lorch[inband] = np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))

        wk *= lorch

        return wk

    @staticmethod
    def _compute_alpha_beta(Q2d, r2d, G_beta, r_1d):
        """
        Compute Yoshimoto-Omote alpha(Q) and beta(Q) integrals used for density estimation.
        This is an internal helper that performs the r-integrals via trapezoidal integration.
        """
        Qsafe = np.where(Q2d == 0.0, 1e-12, Q2d)
        alpha_int = -4 * np.pi * r2d * np.sin(Qsafe * r2d) / Qsafe
        beta_int = G_beta[None, :] * np.sin(Qsafe * r2d) / Qsafe
        alpha = np.trapz(alpha_int, x=r_1d, axis=1)
        beta = np.trapz(beta_int, x=r_1d, axis=1)
        return alpha, beta

    # ------------------------------------------------------------------
    # Analysis method stubs (py4DSTEM-style API)
    # ------------------------------------------------------------------

    # TODO: linting and docstrings
    def calculate_radial_mean(
        self,
        mask_realspace: NDArray | None = None,
        returnval: bool = False,
    ):
        """
        Calculate the radial mean intensity from the Polar4dSTEM dataset.

        The polar array is assumed to have shape (scan_y, scan_x, phi, k).
        This method computes, for each scan position, the mean over the azimuthal
        axis (phi), then averages across scan positions to produce a single 1D
        radial curve. This result is stored in ``self.radial_mean``.

        If a real-space mask is provided, only the selected scan positions are
        used in the scan-position average.

        Parameters
        ----------
        mask_realspace : NDArray or None, optional
            Boolean mask in real space used to select probe positions.
            If ``None``, all probe positions are used.
            Must have shape (scan_y, scan_x) where True means "include".
            (If using rectangle-corner inputs, pass them through
            `_get_mask_bool` before calling this method.)
        returnval : bool, optional
            If True, return the computed 1D radial mean array.

        Returns
        -------
        radial_mean : np.ndarray or None
            If `returnval=True`, returns the 1D radial mean intensity (Nk,).
            Otherwise returns None unless `returnfig=True`.
        """

        # init radial data array
        if mask_realspace is None:
            # calculate intensity over q-range for each probe position
            radial_probe = self.polar.array.mean(axis=2)  # axis 0: ry, 1: rx, 2: theta, 3: q
            # average over all probe positions
            self.radial_mean = np.mean(radial_probe, axis=(0, 1))

        elif mask_realspace is not None:
            masked_polar = self.polar.array[mask_realspace]  # (N_valid, N_theta, N_k)
            radial_probe = masked_polar.mean(axis=1)
            # average over all probe positions, only those unmasked
            self.radial_mean = radial_probe.mean(axis=0)

        if returnval:
            return self.radial_mean
        else:
            return

    def fit_bg(self, Ik, kmin, kmax):
        """
        Fit a smooth background B(k) to a radial intensity curve I(k) using
        non-linear least squares (SciPy `curve_fit`), with a weighting that
        downweights the low-k region and emphasizes higher k.

        The fitted function uses the following form:
            B(k) = c
                + i0 * exp(-k^2 / (2 s0^2))
                + i1 * exp(-k^4 / (2 s1^4))

        Parameters
        ----------
        Ik
            1D radial intensity array (Nk,). Typically produced by
            :meth:`calculate_radial_mean`.
        kmin, kmax
            k-range (in the same units as the internally constructed `k` grid)
            used to build the low-k weighting mask. (Currently k is derived from
            `self.qq` with a calibration factor.)

        Returns
        -------
        bg : np.ndarray
            Fitted background curve B(k), shape (Nk,).
        f : np.ndarray
            Background minus the constant offset, f(k) = B(k) - c, or functionally
            similar to ⟨f⟩²(k)
        """

        k = self.qq

        int_mean = np.mean(Ik)
        k2 = k**2

        # initial guesses
        const_bg = np.min(Ik) / int_mean
        int0 = np.median(Ik) / int_mean - const_bg
        sigma0 = np.mean(k)
        p0 = [const_bg, int0, sigma0, int0, sigma0]

        dk = k[1] - k[0]
        k_width = kmax - kmin
        mask_low = (
            np.sin(
                np.clip(
                    (k - kmin) / k_width,
                    0,
                    1,
                )
                * np.pi
                / 2.0,
            )
            ** 2
        )
        # weighting function for fitting atomic scattering factors
        weights_fit = np.divide(
            1,
            mask_low,
            where=mask_low > 1e-4,
        )
        weights_fit[mask_low <= 1e-4] = np.inf
        # Scale weighting to favour high k values
        weights_fit *= k[-1] - 0.9 * k + dk

        # bounds
        lb = [0, 0, 0, 0, 0]
        ub = [np.inf, np.inf, np.inf, np.inf, np.inf]

        # fit normalized data
        kwargs = dict(sigma=weights_fit, p0=p0, bounds=(lb, ub), xtol=1e-8, maxfev=10000)

        coefs, pcov = curve_fit(self._scattering_model, k2, Ik / int_mean, **kwargs)

        # rescale back to original intensity units (same as script)
        coefs = np.array(coefs, float)
        coefs[0] *= int_mean
        coefs[1] *= int_mean
        coefs[3] *= int_mean

        bg = self._scattering_model(k2, *coefs)
        f = bg - coefs[0]  # "form factor" without constant offset, like the script

        return bg, f

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
        mask_realspace: NDArray | None = None,
        calculate_pdf: bool = False,
        density: float | None = None,
        damp_origin_oscillations: bool = False,
        set_pdf_positive: bool = False,
        returnval: bool = False,
    ):
        """
        Calculate the (reduced) pair distribution function from a 4D-STEM dataset.

        This routine:
        * Computes the radial mean intensity I(k) from self.polar (optionally
            restricted to a real-space mask).
        * Fit a smooth background B(k) and associated f(k) using :meth:`fit_bg`.
        * Estimates and subtracts a background from I(k).
        * Constructs the reduced structure factor F(k) with optional low/highpass filtering.
        * Apply a window in k (low-k sin^2 ramp × Lorch high-k taper)
        * Compute the reduced PDF using a discrete sine transform:
           G(r) = sum_k sin(2π k r) * F_windowed(k)
        * If `calculate_pdf=True`, g(r) is computed from G(r) using:
           g(r) = 1 + G(r) / (4π r ρ0)
           with ρ0 either provided by the user (`density`) or estimated via
           :meth:`estimate_density`.

        The computed quantities are also stored on the instance as:
        * self.radial_mean     – radial mean intensity I(k)  (via calculate_radial_mean)
        * self.bg              – background bg(k)
        * self.Sk              – structure factor (computed as 1 + (Ik - bg)/f)
        * self.Fk              – unwindowed reduced structure function F(k)
        * self.Fk_masked       – windowed reduced structure function F(k)
        * self.r               – r grid (in angstroms)
        * self.reduced_pdf     – reduced PDF G(r)
        * self.pdf             – PDF g(r)  (if computed)

        Parameters
        ----------
        k_min : float, optional
            Minimum k (Å⁻¹) to use when building masks and transforms. If None,
            `self.kmin` is set to `k.min()`.
        k_max : float or None, optional
            Maximum k (Å⁻¹) to use when building masks and transforms. If None,
            `self.kmax` is set to `k.max()`.
        k_width : float, optional
            Width parameter (in Å⁻¹) intended for edge masks. Note: in the current implementation
            this parameter is not yet used as a true "width"; the code uses `k_width = kmax-kmin`.
        k_lowpass : float or None, optional
            If provided and > 0, applies a low-pass Gaussian filter to F(k) with
            sigma = k_lowpass / dk, where dk is the k-grid spacing.
        k_highpass : float or None, optional
            If provided and > 0, constructs a low-pass filtered copy of F(k) with
            sigma = k_highpass / dk and subtracts it from F(k), effectively
            applying a high-pass filter.
        r_min : float, optional
            Minimum r (Å) for the real-space grid used to compute G(r).
        r_max : float, optional
            Maximum r (Å) for the real-space grid used to compute G(r).
        r_step : float, optional
            Step size in r (Å) for the real-space grid.
        mask_realspace : NDArray or None, optional
            Real-space mask specifying which probe positions (rx, ry) to include.
            Either:
            * A boolean array of shape (rx, ry) where True means “include this
                probe position”, or
            * An array-like of shape (2, 2) giving two opposite (rx, ry) corner
                points that define a rectangular region of interest.
            If None, all probe positions are used.
        calculate_pdf
            If True, compute g(r) and store it to `self.pdf`.
        density
            If provided, use this number density (atoms/Å^3) when computing g(r).
            If None and `calculate_pdf=True`, density is estimated using :meth:`estimate_density`.
        damp_origin_oscillations
            If True, compute a density correction and replace the stored F(k)/G(r) with the
            corrected versions returned by :meth:`estimate_density`.
        set_pdf_positive
            If True, sets negative values to 0.
        returnval : bool, optional
            If True, the function returns (r, G(r), g(r)). If
            False, no numerical results are returned (but attributes on `self`
            are still updated).


        Returns
        -------
        results : list[np.ndarray] or None
            If `returnval=True`, returns [r, reduced_pdf, pdf] where:
            - r is the real-space grid (Nr,)
            - reduced_pdf is G(r) (Nr,)
            - pdf is g(r) (Nr,) or None if `calculate_pdf=False`
            Otherwise returns None.
        """
        k_width = np.array(k_width)
        if k_width.size == 1:
            k_width = k_width * np.ones(2)

        k = self.qq
        dk = k[1] - k[0]

        self.kmax = k_max if k_max is not None else k.max()
        self.kmin = k_min if k_min is not None else k.min()
        # BUG: implement k_width properly
        k_width = self.kmax - self.kmin

        mask_bool = self._get_mask_bool(mask_realspace)

        Ik = self.calculate_radial_mean(mask_realspace=mask_bool, returnval=True)

        bg, f = self.fit_bg(Ik, self.kmin, self.kmax)

        Fk = (Ik - bg) * k / f

        # band pass filtering
        if (
            k_lowpass is not None
            and k_lowpass > 0.0
            and k_highpass is not None
            and k_highpass > 0.0
        ):
            if k_highpass > k_lowpass:
                raise ValueError(
                    "Invalid band-pass parameters: k_highpass > k_lowpass. "
                    "Gaussian band-pass filtering requires k_highpass < k_lowpass "
                    "because these parameters are smoothing widths."
                )
            Fk_low = gaussian_filter1d(Fk, sigma=k_lowpass / dk, mode="nearest")
            Fk_high = gaussian_filter1d(Fk, sigma=k_highpass / dk, mode="nearest")
            Fk = Fk_high - Fk_low
        elif k_lowpass is not None and k_lowpass > 0.0:
            Fk = gaussian_filter1d(Fk, sigma=k_lowpass / dk, mode="nearest")
        elif k_highpass is not None and k_highpass > 0.0:
            Fk_low = gaussian_filter1d(Fk, sigma=k_highpass / dk, mode="nearest")
            Fk = Fk - Fk_low

        # Apply wk to F(Q) and rescale
        wk = self._lorch_window(k, self.kmin, self.kmax)
        Fk_win = Fk * wk * 2 * np.pi

        r = np.arange(r_min, r_max, r_step)
        ra, ka = np.meshgrid(r, k)
        # incorrectly scaled in py4dstem , should include 2pi factor in dk and Fk like below
        reduced_pdf = (
            (2 / np.pi)
            * dk
            * 2
            * np.pi
            * np.sum(
                np.sin(2 * np.pi * ra * ka) * Fk_win[:, None],
                axis=0,
            )
        )
        reduced_pdf[0] = 0  # physically must be at 0 when r = 0

        self.Ik = Ik
        self.bg = bg
        self.Fk = Fk * 2 * np.pi
        self.Fk_masked = Fk_win
        self.r = r
        self.reduced_pdf = reduced_pdf

        denscorr = None
        if damp_origin_oscillations or (calculate_pdf and density is None):
            self.Sk = np.ones_like(k, dtype=float)
            mask = k > 0
            self.Sk[mask] = 1.0 + (Fk[mask] / k[mask])
            self.Sk[~mask] = 1.0  # or np.nan, depending on preference

            denscorr = self.estimate_density(
                max_iter=20,
                tol_percent=1e-1,
            )

        if damp_origin_oscillations:
            self.Fk_damped = denscorr[1]
            self.reduced_pdf_damped = denscorr[2]
        else:
            self.reduced_pdf_damped = self.reduced_pdf

        if returnval:
            Gr = getattr(self, "reduced_pdf_damped", None)
            if Gr is None:
                Gr = self.reduced_pdf
            results = [self.r, Gr]

        # option to return pdf also using the density calculation method
        # from Yoshimoto and Omote, 2022.
        if calculate_pdf:
            if density is None:
                rho0 = denscorr[0]
                # print(f"Estimated density: rho0 = {rho0:.4f} atoms / Å³")
            else:
                print(f"Using provided density rho0 = {density:.4f} atoms / Angstrom^3")
                rho0 = density

            mask = r > 0
            pdf = np.ones_like(self.reduced_pdf_damped)

            pdf[mask] = 1 + self.reduced_pdf_damped[mask] / (4 * np.pi * r[mask] * rho0)
            pdf[~mask] = 0.0

            if set_pdf_positive:
                pdf = np.maximum(pdf, 0.0)

            self.pdf = pdf

            if returnval:
                results.append(self.pdf)

        if returnval:
            return results
        else:
            return

    def estimate_density(
        self,
        max_iter: int = 20,
        tol_percent: float = 1e-4,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate number density rho0 (atoms/Å^3) and compute a corrected G(r).

        This method implements an iterative Q-space density estimation by
        Yoshimoto & Omote (2022). It uses the structure factor `self.Sk` and
        the reduced PDF `self.reduced_pdf` to iteratively update rho0 and a
        corrected S(k) so that the implied G(r) is more physically consistent
        at low r.

        This method requires that :meth:`calculate_pair_dist_function` has already
        been run, because it depends on `self.Sk`, `self.reduced_pdf`, `self.r`,
        and the k-window bounds (`self.kmin`, `self.kmax`).

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of Q-space iterations.
        tol_percent : float, optional
            Convergence threshold on the relative change in rho0 (in %),
            as defined in Eq. (12) of Yoshimoto & Omote (2022).

        Returns
        -------
        rho0 : float
            Estimated microscopic number density (atoms/Å^3).
        Fk_win_damped : np.ndarray
            Windowed corrected reduced structure function used for the transform.
        G_cor : np.ndarray
            Reduced PDF G(r) with dampened oscillations near origin.
        """
        if self.Sk is None or self.reduced_pdf is None or self.r is None:
            raise RuntimeError("Run calculate_pair_dist_function() before estimate_density().")

        k = self.qq
        dk = k[1] - k[0]
        k_fit_mask = k >= self.kmin
        k_fit = k[k_fit_mask]
        ra, ka = np.meshgrid(self.r, k)

        r_cut = 0.8  # Angstrom
        mask_search = self.r >= r_cut
        r_search = self.r[mask_search]
        G_search = self.reduced_pdf[mask_search]

        # find primary peak
        ind_max = np.argmax(G_search)
        r_max = r_search[ind_max]

        # find first local minimum to the left of r_peak
        left = self.r < r_max
        if not np.any(left):
            # fallback: if peak is immediately at cutoff, just use cutoff as rmin
            rmin = r_cut
        else:
            r_left = self.r[left]
            G_left = self.reduced_pdf[left]

            mins = np.where((G_left[1:-1] < G_left[:-2]) & (G_left[1:-1] < G_left[2:]))[0] + 1
            # minimum closest to the peak, else global min in left interval
            rmin = r_left[mins[-1]] if mins.size else r_left[np.argmin(G_left)]

        # restrict r to [0, rmin] for alpha/beta integrals
        r_mask = (self.r >= 0.0) & (self.r <= rmin)
        r_short = self.r[r_mask]
        G_short = self.reduced_pdf[r_mask]

        # iterative refinement of rho0 and S(k)
        rho0_prev = None
        Sk_cor = self.Sk.copy()
        G_cor = self.reduced_pdf.copy()

        # use current G(r) (from Sk_cor) in beta(Q)
        G_beta = G_short
        k_fit = k_fit * 2 * np.pi
        for j in range(max_iter):
            if j > 0:
                G_beta = G_cor[r_mask]

            k2d_fit, r2d_fit = np.meshgrid(k_fit, r_short, indexing="ij")
            alpha, beta = self._compute_alpha_beta(k2d_fit, r2d_fit, G_beta, r_short)
            rho0 = np.sum(alpha * beta) / np.sum(alpha**2)

            if rho0_prev is not None:
                Rj = np.sqrt(((rho0_prev - rho0) ** 2) / (rho0**2)) * 100.0
                if Rj < tol_percent:
                    # print(
                    #     f"Converged after {j} iterations: rho0 = {rho0:.4f} atoms / Å³, Rj = {Rj:.4f}%"
                    # )
                    break

            # update S_cor(Q)
            Sk_cor[k_fit_mask] = Sk_cor[k_fit_mask] - beta + rho0 * alpha
            Fk_cor = k * (Sk_cor - 1.0)

            wk = self._lorch_window(k, self.kmin, self.kmax)

            Fk_win_damped = Fk_cor * wk * 2 * np.pi

            G_cor = (
                (2.0 / np.pi)
                * dk
                * 2
                * np.pi
                * np.sum(np.sin(2 * np.pi * ka * ra) * Fk_win_damped[:, None], axis=0)
            )
            G_cor[0] = 0.0

            rho0_prev = rho0

        return rho0, Fk_win_damped, G_cor

    # ------------------------------------------------------------------
    # Plotting functions
    # ------------------------------------------------------------------

    PlotName = Literal[
        "radial_mean",
        "background",
        "reduced_sf",
        "reduced_pdf",
        "pdf",
    ]

    from typing import Optional, Tuple

    def _apply_xrange(
        self,
        x: NDArray,
        y: NDArray,
        xmin: float | None,
        xmax: float | None,
    ) -> tuple[NDArray, NDArray]:
        if xmin is None and xmax is None:
            return x, y
        xmin_eff = x.min() if xmin is None else xmin
        xmax_eff = x.max() if xmax is None else xmax
        if xmax_eff <= xmin_eff:
            raise ValueError(f"xmax must be > xmin (got xmin={xmin_eff}, xmax={xmax_eff}).")
        m = (x >= xmin_eff) & (x <= xmax_eff)
        # avoid empty plots
        if not np.any(m):
            raise ValueError("Requested plot range contains no data.")
        return x[m], y[m]

    def plot_pdf_results(
        self,
        which: Iterable[PlotName] = ("reduced_pdf",),
        *,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,
        rmax: float | None = None,
        figsize: tuple[float, float] = (8, 4),
        returnfigs: bool = False,
    ):
        """
        Convenience plotting dispatcher.

        Examples
        --------
        pdfc.calculate_pair_dist_function(...)
        pdfc.plot(["radial_mean", "background", "reduced_pdf"])
        """
        mapping = {
            "radial_mean": self.plot_radial_mean,
            "background_fits": self.plot_background_fits,
            "reduced_sf": self.plot_reduced_sf,
            "reduced_pdf": self.plot_reduced_pdf,
            "pdf": self.plot_pdf,
            "oscillation_damping": self.plot_oscillation_damping,
        }

        figs = []
        for name in which:
            if name not in mapping:
                raise ValueError(f"Unknown plot '{name}'. Options: {tuple(mapping)}")
            fig = mapping[name](
                qmin=qmin, qmax=qmax, rmin=rmin, rmax=rmax, figsize=figsize, returnfig=returnfigs
            )
            if returnfigs:
                figs.append(fig)

        return figs if returnfigs else None

    def _auto_ylim_after_direct_beam_trough(self, y, *, scale=2.0, smooth_sigma=2.0):
        y = np.asarray(y, dtype=float)
        if y.size < 10:
            return None

        # direct beam peak is usually the first big max; assume it's at/near index 0
        # find first local minimum after index 0
        dy = np.diff(y)
        mins = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1

        if mins.size == 0:
            # fallback: ignore first 5% if we can't find a trough
            start = max(1, int(0.05 * y.size))
        else:
            start = int(mins[0])

        y_use = y[start:]
        y_use = y_use[np.isfinite(y_use)]
        if y_use.size == 0:
            return None

        ymax = np.max(y_use)
        if not np.isfinite(ymax) or ymax <= 0:
            return None

        return (0.0, scale * ymax)

    def plot_radial_mean(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting radial mean intensity vs scattering vector.
        """

        if self.radial_mean is None:
            raise RuntimeError("Radial mean intensity has not been calculated yet.")

        x = np.asarray(self.qq)
        y = np.asarray(self.radial_mean)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Radial Mean Intensity I(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity vs Scattering Vector")
        ax.legend()
        ax.set_yscale("log")
        # ylim = self._auto_ylim_after_direct_beam_trough(self.radial_mean, scale=2.0)
        # if ylim is not None:
        #     ax.set_ylim(*ylim)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_radial_var_norm(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Stub for plotting normalized radial variance vs scattering vector.
        """
        raise NotImplementedError("plot_radial_var_norm is not implemented yet.")

    def plot_background_fits(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting background fit vs radial mean intensity.
        """
        if self.Ik is None or self.bg is None:
            raise RuntimeError("Radial mean intensity or background has not been calculated yet.")

        x = np.asarray(self.qq)
        y1 = np.asarray(self.radial_mean)
        x, y1 = self._apply_xrange(x, y1, qmin, qmax)
        x = np.asarray(self.qq)
        y2 = np.asarray(self.bg)
        x, y2 = self._apply_xrange(x, y2, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y1, label="Radial Mean Intensity I(k)")
        ax.plot(x, y2, label="Background B(k)", linestyle="--")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity and Background Fit")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        # ylim = self._auto_ylim_after_direct_beam_trough(self.radial_mean, scale=2.0)
        # if ylim is not None:
        #     ax.set_ylim(*ylim)

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_sf(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced structure factor F(k).
        """
        if self.Fk_masked is None:
            raise RuntimeError("Reduced structure factor F(k) has not been calculated yet.")

        Fk = getattr(self, "Fk_damped", None)
        if Fk is None:
            Fk = self.Fk_masked

        x = np.asarray(self.qq)
        y = np.asarray(Fk)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Reduced Structure Factor F(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Reduced Structure Factor F(k)")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_pdf(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced PDF g(r).
        """
        if self.reduced_pdf is None:
            raise RuntimeError("Reduced PDF has not been calculated yet.")
        Gr = getattr(self, "reduced_pdf_damped", None)
        if Gr is None:
            Gr = self.reduced_pdf

        x = np.asarray(self.r)
        y = np.asarray(Gr)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        # Find radial value of primary peak and trough for y-limits
        ind_max = np.argmax(y)
        y_max = y[ind_max]

        ind_min = np.argmin(y)
        y_min = y[ind_min]
        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Reduced Pair Distribution Function G(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Reduced Pair Distribution Function G(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_pdf(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting pair distribution function g(r).
        """
        if self.reduced_pdf is None or self.pdf is None:
            raise RuntimeError("Reduced PDF or PDF has not been calculated yet.")

        x = np.asarray(self.r)
        y = np.asarray(self.pdf)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        # Find radial value of primary peak
        ind_max = np.argmax(y)
        y_max = y[ind_max]

        ind_min = np.argmin(y)
        y_min = y[ind_min]

        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Pair Distribution Function g(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Pair Distribution Function g(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_oscillation_damping(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        k = np.asarray(self.qq)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # F(k)
        axS_top = axes[0, 0]
        axS_res = axes[1, 0]
        axS_top.plot(k, self.Fk_masked, label="F_obs(k)", color="gray")
        axS_top.plot(k, self.Fk_damped, label="F_cor(k)", color="red")
        axS_top.set_xlabel("k (Å$^{-1}$)")
        axS_top.set_ylabel("F(k)")
        axS_top.legend()

        axS_res.plot(k, self.Fk_damped - self.Fk_masked, color="blue")
        axS_res.set_xlabel("k (Å$^{-1}$)")
        axS_res.set_ylabel("F_cor - F_obs")

        # G(r)
        axG_top = axes[0, 1]
        axG_res = axes[1, 1]
        axG_top.plot(self.r, self.reduced_pdf, label="G_obs(r)", color="gray")
        axG_top.plot(self.r, self.reduced_pdf_damped, label="G_cor(r)", color="red")
        axG_top.set_xlabel("r (Å)")
        axG_top.set_ylabel("G(r)")
        axG_top.legend()

        axG_res.plot(self.r, self.reduced_pdf_damped - self.reduced_pdf, color="blue")
        axG_res.set_xlabel("r (Å)")
        axG_res.set_ylabel("G_cor - G_obs")

        fig.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()
