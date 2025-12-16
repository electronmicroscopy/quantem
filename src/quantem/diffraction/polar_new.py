from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array

KIRKLAND_PARAMS_PATH = Path(__file__).with_name("kirkland_params.json")


class RDF_new(AutoSerialize):
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
    ):
        """
         -> "RadialDistributionFunction"
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
            raise ValueError("RadialDistributionFunction.from_data only supports 2D or 4D arrays.")

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

    # TODO: linting and docstrings
    def calculate_radial_mean(
        self,
        mask_realspace: NDArray | None = None,
        figsize: tuple[float, float] = (8, 4),
        returnval: bool = False,
        returnfig: bool = False,
    ):
        """
        Calculate the radial mean intensity from the Polar4dSTEM dataset.

        This performs an azimuthal integration over all angles at each k value.
        The result is stored in ``self.radial_mean`` and can optionally be
        returned, along with a figure of the radial mean intensity.

        Parameters
        ----------
        mask_realspace : NDArray or None, optional
            Boolean mask in real space used to select probe positions.
            If ``None``, all probe positions are used.
        figsize : tuple of float, optional
            Figure size passed to ``plot_radial_mean`` when ``returnfig`` is True.
        returnval : bool, optional
            If True, return the computed radial mean array.
        returnfig : bool, optional
            If True, also return a figure object from ``plot_radial_mean``.

        Returns
        -------
        NDArray or list or None
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
            results = self.radial_mean
        else:
            results = None if not returnfig else []

        if returnfig:
            fig = self.plot_radial_mean(
                figsize=figsize,
                returnfig=returnfig,
            )
            results.append(fig)

        return results

    def compute_bg_constant_offset(self, Ik: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """
        Compute the background intensity B(k) as:
            B(k) = N * f²(k) + C

        where:
        - N is a scaling factor for inelastic + multiple scattering background
        - C is a constant offset term
        """
        # fit background parameters N and C
        Ik_region = Ik[-50:]  # high-k region for fitting, hardcoded for now
        f2_region = f2[-50:]

        # least squares fitting to find best parameters
        A = np.column_stack((f2_region, np.ones_like(f2_region)))
        N, C = np.linalg.lstsq(A, Ik_region, rcond=None)[0]

        # this is monotonic background + constant offset
        bg = N * f2 + C

        return bg

    def compute_bg_snip(self, Ik, k, m=25):
        """Compute the background intensity B(k) using the SNIP algorithm,
        as described in Liu et al. (2023, EDP2PDF), following Morháč et al. (1997).

        Parameters
        ----------
        k : array_like
            1D array of scattering vector values (Q, q, or channel positions).
            Only the length is used here; SNIP itself works in channel space.
        intensity : array_like
            1D array y(i) of diffraction intensities (must be non-negative or
            at least > -1 so that y+1 is positive).
        m : int
            Number of SNIP iterations. The paper recommends setting this to
            about half of the major peak FWHM in *channels*.
        smooth_window : int, optional
            Window size for the pre-smoothing step. The paper uses 2n+1=7.
            Set to None or 1 to disable smoothing.
        use_savgol_like : bool, optional
            If True and smooth_window==7, emulate the Savitzky–Golay-like
            convolution used in the paper with hard-coded weights
            (2, 3, 6, 7, 6, 3, 2) / sum.
            If False, no external scipy dependency is assumed (still uses
            that hard-coded kernel when smooth_window==7).

        Returns
        -------
        baseline : ndarray
            Estimated background b(i) from SNIP.
        net_intensity : ndarray
            Background-subtracted intensity y(i) - b(i)."""

        # add de-noising step?

        # twice log operators plus square-root operator
        v = np.log(np.log(np.sqrt(Ik + 1.0) + 1.0) + 1.0)

        # set m to FWHM of the major peak in the future
        # for now, clamp m so the window doesn't exceed half the spectrum
        m = max(1, min(m, len(Ik) // 2 - 1))

        # snip iterations
        for p in range(1, m + 1):
            # get channels shifted by p
            left = np.empty_like(v)
            left[p:] = v[:-p]
            right = np.empty_like(v)
            right[:-p] = v[p:]
            # leave boundary edge cases uunshifted
            left[:p] = v[:p]
            right[-p:] = v[-p:]

            vp = (left + right) / 2
            v = np.minimum(v, vp)

        # inverse lsr
        t = np.exp(v)
        bg = (np.exp(t - 1.0) - 1.0) ** 2 - 1.0
        bg = np.clip(bg, 0.0, None)

        # #show bg fit
        # #TODO: make plotting optional
        # plt.figure()
        # plt.plot(Ik, label="Original Intensity")
        # plt.plot(bg, label=f"SNIP Background, m={m}")
        # plt.ylim(0, 0.0003)
        # plt.legend()
        # plt.xlabel("k")
        # plt.ylabel("Intensity")
        # plt.title("SNIP Background Estimation")
        # plt.show()

        return bg

    def get_atomic_scattering_factors(
        self,
        elements: Sequence[str],
        atomic_frac: Sequence[float],
        k2_values: Iterable[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve atomic scattering factors for specified elements.

        Parameters
        ----------
        elements : Sequence[str]
            List of element symbols (e.g., ["Si", "O"]).
        atomic_frac : Sequence[float]
            Atomic fractions for each element in `elements`. Must have the same
            length as `elements`. The values will be converted to a NumPy array
            of dtype float.
        k2_values : Iterable[float]
            Squared scattering vector magnitude values (k^2) at which the
            scattering factors are evaluated.

        Returns
        -------
        f2 : np.ndarray
            Weighted sum of squared atomic scattering factors:
            f2(k^2) = Σ_i x_i * f_i(k^2)^2
            where x_i is the atomic fraction of element i.
        f_2 : np.ndarray
            Square of the weighted sum of atomic scattering factors:
            f_2(k^2) = (Σ_i x_i * f_i(k^2))^2
        """
        # initialize array to hold f values
        atomic_frac = np.asarray(atomic_frac, dtype=float)
        n_elements = len(elements)
        k2_array = np.asarray(k2_values, dtype=float)
        len_k = len(k2_array)

        f = np.zeros((n_elements, len_k), dtype=float)

        # load Kirkland parameters from JSON
        with KIRKLAND_PARAMS_PATH.open(encoding="utf-8") as file:
            kirkland_params: dict[str, dict[str, list[float]]] = json.load(file)

        for i, element in enumerate(elements):
            try:
                params = kirkland_params[element]
            except KeyError:
                raise ValueError(f"Element {element} not found in Kirkland parameters table.")

            a = np.asarray(params["a"], float)
            b = np.asarray(params["b"], float)
            c = np.asarray(params["c"], float)
            d = np.asarray(params["d"], float)

            # Lorentzian and Gaussian terms
            l_term = (a[:, None] / (k2_array[None, :] + b[:, None])).sum(
                axis=0
            )  # a[:, None] and b[:, None] → shape (3, 1)
            g_term = (c[:, None] * np.exp(-d[:, None] * k2_array[None, :])).sum(
                axis=0
            )  # k2_array[None, :] → shape (1, len_k)

            f[i, :] = l_term + g_term

        f2 = (f**2 * atomic_frac[:, None]).sum(axis=0)
        f_weighted = (f * atomic_frac[:, None]).sum(axis=0)
        f_2 = f_weighted**2

        return f2, f_2

    def calculate_pair_dist_function(
        self,
        el: List[str],
        atomic_frac: List[float],
        k_min: float = 0.05,
        k_max: float | None = None,
        k_width: float = 0.25,
        k_lowpass: float | None = None,
        k_highpass: float | None = None,
        r_min: float = 0.0,
        r_max: float = 20.0,
        r_step: float = 0.02,
        mask_realspace: NDArray | None = None,
        damp_origin_fluctuations: bool = True,
        calculate_pdf: bool = True,
        density: float | None = None,
        plot_options: dict[str, bool] = {
            "plot_radial_mean": False,
            "plot_background_fits": False,
            "plot_sf_estimate": False,
            "plot_reduced_pdf": True,
            "plot_pdf": False,
        },
        figsize: tuple[float, float] = (8, 4),
        returnval: bool = False,
        returnfig: bool = False,
    ):
        """
        Calculate the (reduced) pair distribution function from a 4D-STEM dataset.

        This routine:
        * Computes the radial mean intensity I(k) from self.polar (optionally
            restricted to a real-space mask).
        * Computes element-weighted elastic scattering factors ⟨f²⟩(k) and
            ⟨f⟩²(k).
        * Estimates and subtracts a background from I(k).
        * Constructs the structure factor S(k) and reduced structure function
            F(k) = k [S(k) - 1], with optional low-/high-pass filtering.
        * Applies smooth edge masking in k-space.
        * Performs a sine transform to obtain the reduced pair distribution
            function G(r).

        The computed quantities are also stored on the instance as:
        * self.Ik        – radial mean intensity I(k)
        * self.bg        – background bg(k)
        * self.Fk        – reduced structure function F(k)
        * self.pdf_r     – r grid
        * self.reduced_pdf – reduced PDF G(r)

        Parameters
        ----------
        el : list of str
            List of element symbols (e.g. ["Ta", "O"]) in the sample.
        atomic_frac : list of float
            Atomic fractions for each element in `el`. Must be the same length as
            `el` and typically sum to 1.0.
        k_min : float, optional
            Minimum k (Å⁻¹) to use when building masks and transforms. If not
            None, this value overrides the minimum of the k-grid derived from
            `self.qq`. If None, `self.kmin` is set to `k.min()`.
        k_max : float or None, optional
            Maximum k (Å⁻¹) to use when building masks and transforms. If not
            None, this value overrides the maximum of the k-grid derived from
            `self.qq`. If None, `self.kmax` is set to `k.max()`.
        k_width : float, optional
            Width parameter (in Å⁻¹) controlling the smooth edge mask in k-space.
            It enters the construction of `mask_low` and `mask_high`.
        k_lowpass : float or None, optional
            If provided and > 0, applies a low-pass Gaussian filter to S(k) with
            sigma = k_lowpass / dk, where dk is the k-grid spacing.
        k_highpass : float or None, optional
            If provided and > 0, constructs a low-pass filtered copy of S(k) with
            sigma = k_highpass / dk and subtracts it from S(k), effectively
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
        plot_options : dict[str, bool] or None, optional
            Dictionary of plotting flags:
            - "plot_radial_mean"
            - "plot_background_fits"
            - "plot_sf_estimate"
            - "plot_reduced_pdf"
            - "plot_pdf"
            In this method it is currently used only to decide whether to request
            a figure from `calculate_radial_mean` via
            `returnfig=plot_options["plot_radial_mean"]`. If None, a default
            dictionary is created internally.
        figsize : tuple[float, float], optional
            Figure size passed to `calculate_radial_mean` (and potentially to
            future plotting routines).
        maxfev : int or None, optional
            Maximum number of function evaluations for any internal fit routines.
            Currently reserved for future use.
        returnval : bool, optional
            If True, the function returns a tuple `(pdf_r, reduced_pdf)`. If
            False, no numerical results are returned (but attributes on `self`
            are still updated).
        returnfig : bool, optional
            If True, this method may in future also return figure objects (e.g.
            appended to the `results` list). At present, figure-generation code
            is commented out and this flag has no effect beyond shaping the
            structure of the returned `results` object.

        Returns
        -------
        pdf_r : np.ndarray
            Real-space r grid on which the reduced PDF is evaluated.
        reduced_pdf : np.ndarray
            Reduced pair distribution function G(r).
        pdf : np.ndarray
            Pair distribution function g(r).

        #TODO: add notes of density calculation for pdf and background calculation method
        """
        k_width = np.array(k_width)
        if k_width.size == 1:
            k_width = k_width * np.ones(2)

        # BUG: make calibration automatic
        k = self.qq * 0.01488
        dk = k[1] - k[0]
        k2 = k**2

        self.kmax = k_max if k_max is not None else k.max()
        self.kmin = k_min if k_min is not None else k.min()

        # TODO: test
        # this should be from avg, not sum!
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

        self.calculate_radial_mean(
            mask_realspace=mask_bool, figsize=figsize, returnfig=plot_options["plot_radial_mean"]
        )
        Ik = self.radial_mean

        # get <f^2> and <f>^2 for elements and atomic frac
        f2, f_2 = self.get_atomic_scattering_factors(el, atomic_frac, k2)

        # BUG: implement k_width properly
        k_width = self.kmax - self.kmin
        # Calculate structure factor mask
        # mask_low = (
        #     np.sin(
        #         np.clip(
        #             (k - self.kmin) / k_width,
        #             0,
        #             1,
        #         )
        #         * np.pi
        #         / 2.0,
        #     )
        #     ** 2
        # )
        # mask_high = (
        #     np.sin(
        #         np.clip(
        #             (self.kmax - k) / k_width,
        #             0,
        #             1,
        #         )
        #         * np.pi
        #         / 2.0,
        #     )
        #     ** 2
        # )
        # mask = mask_low * mask_high

        bg = self.compute_bg_snip(Ik, k, m=25)
        Ik_net = Ik - bg

        # scaling region (avoid direct beam, etc.)
        k_scale_min = max(self.kmin, 1.25)
        k_scale_max = self.kmax
        mask_int = (k >= k_scale_min) & (k <= k_scale_max)

        k_int = k[mask_int]
        Ik_int = np.clip(Ik_net[mask_int], 0.0, None)
        f2_int = f2[mask_int]

        integral_Ik = np.trapz(Ik_int, k_int)
        integral_f2 = np.trapz(f2_int, k_int)

        eta = integral_Ik / integral_f2
        print(f"Scaling factor eta = {eta:.4f}")

        f2_scaled = eta * f2
        f_2_scaled = eta * f_2

        Sk = 1.0 + (Ik_net - f2_scaled) / f_2_scaled  # back to intensity scaling

        # high and lowpass filtering
        if k_lowpass is not None and k_lowpass > 0.0:
            Sk = gaussian_filter(Sk, sigma=k_lowpass / dk, mode="nearest")
        if k_highpass is not None and k_highpass > 0.0:
            Sk_lowpass = gaussian_filter(Sk, sigma=k_highpass / dk, mode="nearest")
            Sk -= Sk_lowpass
            self.Sk_lowpass = Sk_lowpass

        Fk = 2 * np.pi * k * (Sk - 1)

        # high q taper
        Q = 2.0 * np.pi * k  # Q in 1/Å
        Qmin = 2.0 * np.pi * self.kmin
        Qmax = 2.0 * np.pi * self.kmax

        # Build Lorch window: w(Q) = sin(pi*Q/Qmax)/(pi*Q/Qmax)
        window = np.zeros_like(Q)
        inband = (Q >= Qmin) & (Q <= Qmax)

        x = Q[inband] / Qmax
        # handle Q=0 safely (though inband excludes it if Qmin>0)
        window[inband] = np.sin(np.pi * x) / (np.pi * x)

        # Apply window to F(Q)
        FQ_win = Fk * window

        r = np.arange(r_min, r_max, r_step)
        ra, ka = np.meshgrid(r, k)
        # i think the np.sin kernel in py4dstem should not include the 2pi?
        # or depending on k, need to ALSO add 2pi factor to dk
        reduced_pdf = (
            (2 / np.pi)
            * dk
            * 2
            * np.pi
            * np.sum(
                np.sin(2 * np.pi * ra * ka) * FQ_win[:, None],
                axis=0,
            )
        )
        reduced_pdf[0] = 0  # physically must be at 0 when r = 0

        self.Ik = Ik
        self.bg = bg
        self.Sk = Sk
        self.Fk = Fk
        self.Fk_masked = FQ_win
        self.r = r
        self.reduced_pdf = reduced_pdf

        # add option to return pdf also using the density calculation method
        # from Yoshimoto and Omote, 2022.

        # BUG: for now
        calculate_pdf = True
        # rho0 = 0.05284
        if calculate_pdf:
            if density is None:
                rho0, Fk_cor, G_cor = self.estimate_density(
                    max_iter=20, tol_percent=1e-1, make_plots=True, Fk_masked=FQ_win
                )
                print(f"Estimated density rho0 = {rho0:.4f} atoms / Angstrom^3")
            else:
                print(f"Using provided density rho0 = {density:.4f} atoms / Angstrom^3")
                rho0 = density
            pdf = 1 + (1 / (4 * np.pi * r * rho0)) * G_cor
            pdf[0] = 0.0  # avoid singularity at r=0
            self.pdf = pdf

            self.Fk_masked = Fk_cor
            self.r = r
            self.reduced_pdf = G_cor

        # if returnfig and self.plot_options != {}:
        #     self.plot_functions()

        # if returnval:
        #     results = (self.r, self.reduced_pdf)
        # else:
        #     results = None if not returnfig else []

        # # handle mutable default for plot_options
        # if plot_options is None:
        #     plot_options = {
        #         "plot_radial_mean": False,
        #         "plot_background_fits": False,
        #         "plot_sf_estimate": False,
        #         "plot_reduced_pdf": True,
        #         "plot_pdf": False,
        #     }

        # if returnfig and plot_options != {}:
        #     # fig = self.plot_radial_mean(
        #     #     figsize=figsize,
        #     #     returnfig=returnfig,
        #     # )
        #     # results.append(fig)

        # return results

        # if returnval:
        #     results = (self.r, self.reduced_pdf)
        return self.r, self.reduced_pdf, Fk, Sk, Ik, bg, k, Ik_net, FQ_win, self.pdf

    def compute_alpha_beta(self, Q2d, r2d, G_beta, r_1d):
        Qsafe = np.where(Q2d == 0.0, 1e-12, Q2d)
        alpha_int = -4 * np.pi * r2d * np.sin(Qsafe * r2d) / Qsafe
        beta_int = G_beta[None, :] * np.sin(Qsafe * r2d) / Qsafe
        alpha = np.trapz(alpha_int, x=r_1d, axis=1)
        beta = np.trapz(beta_int, x=r_1d, axis=1)
        return alpha, beta

    def estimate_density(
        self,
        max_iter: int = 1000,
        tol_percent: float = 1e-4,
        make_plots: bool = True,
        Fk_masked: np.ndarray | None = None,
        figsize: Tuple[float, float] = (8.0, 6.0),
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate microscopic number density rho0 from S(k) using the
        Yoshimoto & Omote (2022) Q-space iteration method, and return
        corrected S(k) and G(r).

        Parameters
        ----------
        k : ndarray, shape (Nk,)
            Scattering vector k (Å⁻1).
        Sk_obs : ndarray, shape (Nk,)
            Observed structure factor S_obs(k).
        r : ndarray, shape (Nr,)
            Real-space grid for G(r) (Å).
        max_iter : int, optional
            Maximum number of Q-space iterations.
        tol_percent : float, optional
            Convergence threshold on the relative change in rho0 (in %),
            as defined in Eq. (12) of Yoshimoto & Omote (2022).
        make_plots : bool, optional
            If True, plot S_obs vs S_cor and G_obs vs G_cor with residuals.
        figsize : tuple, optional
            Figure size for the plots.

        Returns
        -------
        rho0 : float
            Estimated microscopic number density (in the same units implied
            by your S(k) normalization; typically atoms / Å^3).
        Sk_cor : ndarray, shape (Nk,)
            Corrected structure factor S_cor(k).
        G_obs : ndarray, shape (Nr,)
            Observed G_obs(r) computed from S_obs(k).
        G_cor : ndarray, shape (Nr,)
            Corrected G_cor(r) computed from S_cor(k).
        """
        # BUG: make calibration automatic
        k = self.qq * 0.01488  # convert from 1/Angstrom to Angstrom^-1
        dk = k[1] - k[0]
        k_fit_mask = (k >= self.kmin) & (
            k <= self.kmax
        )  # or drop <= self.kmax if you want full high-k
        k_fit = k[k_fit_mask]
        ra, ka = np.meshgrid(self.r, k)

        # ---- choose r1st ignoring r < r_art_cut ----
        r_art_cut = 1.5  # Angstrom, adjust as needed
        mask_search = self.r >= r_art_cut
        r_search = self.r[mask_search]
        G_search = self.reduced_pdf[mask_search]  # or whatever G array you're using in the loop

        # indices of local maxima in the search region (no smoothing)
        peaks = np.where((G_search[1:-1] > G_search[:-2]) & (G_search[1:-1] > G_search[2:]))[0] + 1
        print(f"Found local maxima at r = {r_search[peaks]} Å")
        if len(peaks) == 0:
            raise RuntimeError(
                f"No local maxima found for r >= r_art_cut={r_art_cut}. "
                "Increase r_max, decrease r_art_cut, or check G(r)."
            )

        idx_peak = peaks[0]
        r1st = r_search[idx_peak]

        # ---- find a local minimum to the left of r1st but still >= r_art_cut ----
        left = (self.r >= r_art_cut) & (self.r < r1st)
        if not np.any(left):
            # fallback: if peak is immediately at cutoff, just use cutoff as rmin
            rmin = r_art_cut
        else:
            r_left = self.r[left]
            G_left = self.reduced_pdf[left]

            mins = np.where((G_left[1:-1] < G_left[:-2]) & (G_left[1:-1] < G_left[2:]))[0] + 1
            if len(mins) == 0:
                # fallback: use the global minimum on the left interval
                rmin = r_left[np.argmin(G_left)]
            else:
                rmin = r_left[mins[-1]]  # minimum closest to the peak

            print(f"Using rmin = {rmin:.3f} Å for density estimation.")

        # restrict r to [0, rmin] for alpha/beta integrals
        r_mask = (self.r >= 0.0) & (self.r <= rmin)
        r_short = self.r[r_mask]
        G_short = self.reduced_pdf[r_mask]

        ra_short, ka_short = np.meshgrid(r_short, k)  # shape (Nr_short, Nk)

        # iterative refinement of rho0 and S(k)
        rho0_prev = None
        Sk_cor = self.Sk.copy()
        G_cor = self.reduced_pdf.copy()

        # use current G(r) (from Sk_cor) in beta(Q)
        G_beta = G_short
        # r_short = r_short * 2 * np.pi
        k_fit = k_fit * 2 * np.pi
        for j in range(max_iter):
            if j > 0:
                G_beta = G_cor[r_mask]

            # Q2d, r2d = np.meshgrid(k, r_short, indexing="ij")  # (Nk, Nr_short)
            # alpha, beta = self.compute_alpha_beta(Q2d, r2d, G_beta, r_short)
            # # least-squares estimate of rho0
            # rho0 = np.sum(alpha * beta) / np.sum(alpha**2)

            # k-range used only for alpha/beta fit
            k2d_fit, r2d_fit = np.meshgrid(k_fit, r_short, indexing="ij")  # (Nk_fit, Nr_short)

            # IMPORTANT: do NOT rescale r_short in-place (remove r_short = r_short * 2*np.pi)
            alpha, beta = self.compute_alpha_beta(k2d_fit, r2d_fit, G_beta, r_short)

            # rho0 fit only over the masked k-range
            rho0 = np.sum(alpha * beta) / np.sum(alpha**2)

            if rho0_prev is not None:
                Rj = np.sqrt(((rho0_prev - rho0) ** 2) / (rho0**2)) * 100.0
                if Rj < tol_percent:
                    print(
                        f"Converged after {j} iterations: rho0 = {rho0:.4f} atoms / Å³, Rj = {Rj:.4f}%"
                    )
                    break

            # update S_cor(Q) according to Eq. (8)
            # Sk_cor = Sk_cor - beta + rho0 * alpha
            Sk_cor[k_fit_mask] = Sk_cor[k_fit_mask] - beta + rho0 * alpha
            Fk_cor = 2 * np.pi * k * (Sk_cor - 1.0)

            # low q taper
            edge_frac_low = 0.1  # 10% of range at low-q
            edge_width_low = edge_frac_low * (self.kmax - self.kmin)

            window = np.ones_like(k)

            # low-q edge (same as before)
            low = (k >= self.kmin) & (k < self.kmin + edge_width_low)
            t = (k[low] - self.kmin) / edge_width_low
            window[low] = np.sin(0.5 * np.pi * t) ** 2

            # outside [kmin, kmax] -> 0
            window[k < self.kmin] = 0.0
            window[k > self.kmax] = 0.0

            # high q taper
            Q = 2.0 * np.pi * k  # Q in 1/Å
            Qmin = 2.0 * np.pi * self.kmin
            Qmax = 2.0 * np.pi * self.kmax

            # Build Lorch window: w(Q) = sin(pi*Q/Qmax)/(pi*Q/Qmax)
            window = np.zeros_like(Q)
            inband = (Q >= Qmin) & (Q <= Qmax)

            x = Q[inband] / Qmax
            # handle Q=0 safely (though inband excludes it if Qmin>0)
            window[inband] = np.sin(np.pi * x) / (np.pi * x)

            # Apply window to F(Q)
            FQ_win = Fk_cor * window

            G_cor = (
                (2.0 / np.pi)
                * dk
                * 2
                * np.pi
                * np.sum(np.sin(2 * np.pi * ka * ra) * FQ_win[:, None], axis=0)
            )
            G_cor[0] = 0.0  # enforce G(0) = 0

            rho0_prev = rho0

            if make_plots and (j == 0):
                fig, ax = plt.subplots(figsize=(7, 4))
                # ax.plot(k, alpha, label="alpha(k)")
                ax.plot(k_fit, beta, label="beta(k)")
                ax.plot(k_fit, rho0 * alpha, "--", label="rho0 * alpha(k)")
                ax.set_xlabel("k (1/Å)")
                ax.set_title(f"iter {j} (rho0={rho0:.4g})")
                ax.legend()
                plt.show()

        fig, ax = plt.subplots(figsize=(7, 4))
        # ax.plot(k, alpha, label="alpha(k)")
        ax.plot(k_fit, beta, label="beta(k)")
        ax.plot(k_fit, rho0 * alpha, "--", label="rho0 * alpha(k)")
        ax.set_xlabel("k (1/Å)")
        ax.set_title(f"iter {j} (rho0={rho0:.4g})")
        ax.legend()
        plt.show()
        print(f"Total iterations: {j + 1}, Final rho0 = {rho0:.4f} atoms / Å³")

        # --- Step 4: plotting (optional) ---
        if make_plots:
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # S(Q)
            axS_top = axes[0, 0]
            axS_res = axes[1, 0]
            axS_top.plot(k, self.Fk_masked, label="F_obs(k)", color="gray")
            axS_top.plot(k, FQ_win, label="F_cor(k)", color="red")
            axS_top.set_xlabel("k (Å$^{-1}$)")
            axS_top.set_ylabel("F(k)")
            axS_top.legend()

            axS_res.plot(k, FQ_win - self.Fk_masked, color="blue")
            axS_res.set_xlabel("k (Å$^{-1}$)")
            axS_res.set_ylabel("F_cor - F_obs")

            # G(r)
            axG_top = axes[0, 1]
            axG_res = axes[1, 1]
            axG_top.plot(self.r, self.reduced_pdf, label="G_obs(r)", color="gray")
            axG_top.plot(self.r, G_cor, label="G_cor(r)", color="red")
            # axG_top.plot(r_short, G_short, label="G_obs(r)", color="gray")
            # axG_top.plot(r_short, G_beta, label="G_cor(r)", color="red")
            axG_top.set_xlabel("r (Å)")
            axG_top.set_ylabel("G(r)")
            axG_top.legend()

            axG_res.plot(self.r, G_cor - self.reduced_pdf, color="blue")
            axG_res.set_xlabel("r (Å)")
            axG_res.set_ylabel("G_cor - G_obs")

            fig.tight_layout()
            plt.show()

        return rho0, FQ_win, G_cor

    def plot_radial_mean(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting radial mean intensity vs scattering vector.
        """

        if self.radial_mean is None:
            raise RuntimeError("Radial mean intensity has not been calculated yet.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.qq, self.radial_mean, label="Radial Mean Intensity I(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity vs Scattering Vector")
        ax.legend()
        ax.tight_layout()

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
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting background fit vs radial mean intensity.
        """
        if self.Ik is None or self.bg is None:
            raise RuntimeError("Radial mean intensity or background has not been calculated yet.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.qq, self.Ik, label="Radial Mean Intensity I(k)")
        ax.plot(self.qq, self.bg, label="Background B(k)", linestyle="--")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity and Background Fit")
        ax.legend()
        ax.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_sf_estimate(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting structure factor S(k).
        """

        if self.Sk is None:
            raise RuntimeError("Structure factor S(k) has not been calculated yet.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.qq, self.Sk, label="Structure Factor S(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Structure Factor S(k)")
        ax.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_sf_estimate(
        self,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced structure factor F(k).
        """
        if self.Fk is None:
            raise RuntimeError("Reduced structure factor F(k) has not been calculated yet.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.qq, self.Fk, label="Reduced Structure Factor F(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Reduced Structure Factor F(k)")
        ax.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_pdf(
        self,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced PDF g(r).
        """
        if self.reduced_pdf is None:
            raise RuntimeError("Reduced PDF has not been calculated yet.")

        # Find radial value of primary peak and trough for y-limits
        ind_max = np.argmax(self.reduced_pdf)
        y_max = self.reduced_pdf[ind_max]

        ind_min = np.argmin(self.reduced_pdf)
        y_min = self.reduced_pdf[ind_min]
        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.r, self.reduced_pdf, label="Reduced Pair Distribution Function G(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Reduced Pair Distribution Function G(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_pdf(
        self,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting pair distribution function g(r).
        """
        if self.reduced_pdf is None or self.pdf is None:
            raise RuntimeError("Reduced PDF or PDF has not been calculated yet.")

        # Find radial value of primary peak
        ind_max = np.argmax(self.reduced_pdf)
        y_max = self.pdf[ind_max]

        # look to right of primary peak for minimum
        reduced_pdf_region = self.pdf[ind_max + 1 :]
        ind_min = np.argmin(reduced_pdf_region) + (ind_max + 1)
        y_min = self.pdf[ind_min]

        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.r, self.pdf, label="Pair Distribution Function g(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Pair Distribution Function g(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()
