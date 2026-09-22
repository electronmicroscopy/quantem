import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from quantem.core.visualization import show_2d
from quantem.spectroscopy.dataset3dspectroscopy import Dataset3dspectroscopy
from quantem.spectroscopy.spectroscopy_visualzitions import (
    interpret_thickness_quality as _visualize_thickness_quality,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_absolute_thickness as _visualize_absolute_thickness,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_absolute_zlp_shift as _visualize_absolute_zlp_shift,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_dual_eels_picker as _visualize_dual_eels_picker,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_quantem_diagnostic as _visualize_quantem_diagnostic,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_zlp_drift_diagnostics as _visualize_zlp_drift_diagnostics,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    visualize_thickness_windows as _visualize_thickness_windows,
)


def _estimate_zlp_fwhm(energy_axis, mean_spectrum):
    """Estimate the ZLP FWHM (eV) from a mean spectrum via half-max crossings
    walked outward from the peak channel. Used to resolve fit_window="auto"."""
    peak_idx = int(np.nanargmax(mean_spectrum))
    peak_val = mean_spectrum[peak_idx]
    half = peak_val / 2.0

    left_idx = peak_idx
    while left_idx > 0 and mean_spectrum[left_idx] > half:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(mean_spectrum) - 1 and mean_spectrum[right_idx] > half:
        right_idx += 1

    fwhm = float(energy_axis[right_idx] - energy_axis[left_idx])
    if fwhm <= 0:
        raise ValueError(
            "Could not estimate a positive ZLP FWHM from the mean spectrum; "
            "pass fit_window explicitly instead of 'auto'."
        )
    return fwhm


def _resolve_auto_fit_window(fit_window, energy_axis, mean_spectrum, multiplier):
    """Resolve fit_window="auto" to `multiplier * FWHM`; pass through numeric values."""
    if not isinstance(fit_window, str):
        return fit_window
    if fit_window != "auto":
        raise ValueError(f"fit_window must be a positive number or 'auto', got {fit_window!r}")
    fwhm = _estimate_zlp_fwhm(energy_axis, mean_spectrum)
    return multiplier * fwhm


@dataclass
class _NearZlpComponentSpec:
    """One component (the ZLP tail, or a peak) for fit_near_zlp_transitions().

    Every parameter carries explicit (lo, hi) bounds -- amplitude, center,
    fwhm, and (for pseudo_voigt) eta -- so the fit is never unbounded.
    """

    name: str
    shape: str  # "gaussian" | "lorentzian" | "pseudo_voigt"
    amplitude_guess: float
    amplitude_bounds: tuple
    center_guess: float
    center_bounds: tuple
    fwhm_guess: float
    fwhm_bounds: tuple
    eta_guess: float = 0.5
    eta_bounds: tuple = (0.0, 1.0)

    def __post_init__(self):
        if self.shape not in {"gaussian", "lorentzian", "pseudo_voigt"}:
            raise ValueError(
                f"Unknown lineshape {self.shape!r} for component {self.name!r}; "
                f"choose 'gaussian', 'lorentzian', or 'pseudo_voigt'."
            )
        for bound_name, bounds in (
            ("amplitude_bounds", self.amplitude_bounds),
            ("center_bounds", self.center_bounds),
            ("fwhm_bounds", self.fwhm_bounds),
            ("eta_bounds", self.eta_bounds),
        ):
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    f"{bound_name} for component {self.name!r} must be (lo, hi) with "
                    f"lo < hi, got {bounds}"
                )

    @property
    def n_params(self) -> int:
        return 4 if self.shape == "pseudo_voigt" else 3


def _eval_near_zlp_component(shape, x, amplitude, center, fwhm, eta=0.5):
    """Evaluate one lineshape, normalized to peak height 1 at x=center."""
    if shape == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    if shape == "lorentzian":
        gamma = fwhm / 2.0
        return amplitude * (gamma**2) / ((x - center) ** 2 + gamma**2)
    # pseudo_voigt: linear mix of normalized Gaussian and Lorentzian at the
    # same center/fwhm; eta is the Lorentzian fraction.
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lor = (gamma**2) / ((x - center) ** 2 + gamma**2)
    return amplitude * (eta * lor + (1.0 - eta) * gauss)


def _eval_near_zlp_model(x, params, specs):
    total = np.zeros_like(x, dtype=float)
    idx = 0
    for spec in specs:
        n = spec.n_params
        amplitude, center, fwhm = params[idx], params[idx + 1], params[idx + 2]
        eta = params[idx + 3] if spec.shape == "pseudo_voigt" else 0.5
        total += _eval_near_zlp_component(spec.shape, x, amplitude, center, fwhm, eta)
        idx += n
    return total


def _fit_powerlaw_background(E_window, I_window, eval_energy, maxfev=5000):
    """
    Fit a power-law background ``A * E**(-r)`` to a pre-edge window and
    evaluate it on a (possibly wider) energy axis.

    Shared by ``subtract_background_limited_preedge`` (``method='powerlaw'``)
    and ``powerlaw_backgroundfit_eels``.

    Parameters
    ----------
    E_window, I_window : ndarray
        Energy (eV) and intensity of the pre-edge fitting window.
        ``E_window`` must be strictly positive -- ``E**(-r)`` is undefined
        for E <= 0.
    eval_energy : ndarray
        Energy axis to evaluate the fitted background on. May extend to
        E <= 0 (e.g. the ZLP side of a low-loss spectrum); those channels
        are returned as 0 rather than NaN/inf.
    maxfev : int, optional
        Max function evaluations passed to ``curve_fit``. Default 5000.

    Returns
    -------
    background : ndarray
        Fitted background evaluated on ``eval_energy``, same shape.
    popt : (float, float)
        Fitted (A, r).

    Raises
    ------
    ValueError
        If ``E_window`` is not strictly positive.
    RuntimeError
        If the fit fails to converge.
    """
    E_window = np.asarray(E_window, dtype=float)
    I_window = np.asarray(I_window, dtype=float)
    eval_energy = np.asarray(eval_energy, dtype=float)

    if np.any(E_window <= 0):
        raise ValueError(
            f"Power-law background fitting requires a strictly positive fit "
            f"window (E > 0 eV); got E_window=[{E_window.min():.3f}, "
            f"{E_window.max():.3f}] eV. Use a positive-energy fit window, or "
            f"a different background method."
        )

    def powerlaw(E, A, r):
        return A * (E ** (-r))

    A0 = I_window[0] * (E_window[0] ** 3)
    r0 = 3.0

    try:
        popt, _ = curve_fit(
            powerlaw,
            E_window,
            I_window,
            p0=[A0, r0],
            bounds=([0, 0], [np.inf, 10]),
            maxfev=maxfev,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Power-law fit failed to converge on a "
            f"{E_window[-1] - E_window[0]:.1f} eV fit window ({len(E_window)} "
            f"points). Error: {e}"
        )

    # eval_energy can legitimately include E <= 0 channels (e.g. the
    # ZLP/negative-energy region of a low-loss spectrum) even when the fit
    # window itself is positive. E**(-r) is undefined there, so evaluate
    # the fit only on the positive part (substituting a safe dummy value
    # elsewhere to avoid a spurious 0**negative warning) and leave E <= 0
    # channels at 0 background rather than poisoning them with NaN.
    positive_energy = eval_energy > 0
    energy_safe = np.where(positive_energy, eval_energy, 1.0)
    background = np.where(positive_energy, powerlaw(energy_safe, popt[0], popt[1]), 0.0)

    return background, tuple(popt)


class Dataset3deels(Dataset3dspectroscopy):
    """An EELS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (scan_row, scan_col, energy).
    The first two dimensions represent real space sampling, while the last dimension
    represents the energy axis.

    """

    element_info = None
    element_info_path = "eels_edges.csv"
    dataset_type = "EELS"

    plot_absolute_zlp_shift = _visualize_absolute_zlp_shift
    visualize_thickness_windows = _visualize_thickness_windows
    interpret_thickness_quality = _visualize_thickness_quality
    plot_absolute_thickness = _visualize_absolute_thickness
    plot_dual_eels_picker = _visualize_dual_eels_picker
    plot_quantem_diagnostic = _visualize_quantem_diagnostic
    plot_zlp_drift_diagnostics = _visualize_zlp_drift_diagnostics

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a 3D EELS dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )
        self._virtual_images = {}
        self.dataset_type = "eels"

    # ========== NEW METHOD: Background subtraction for limited pre-edge data ==========

    def subtract_background_limited_preedge(
        self,
        target_edge,
        pre_edge_range=None,
        method="auto",
        polynomial_degree=2,
        show=True,
        return_dataset=True,
        display_energy_range=None,
        display_intensity_range=None,
        display_subtracted_intensity_range=None,
    ):
        """
        Background subtraction optimized for limited pre-edge data.

        This method bypasses the 10-30% window_size constraint in the standard
        subtract_background() method, allowing background fitting when only a
        small pre-edge region is available (common in high-loss only acquisitions).

        Parameters
        ----------
        target_edge : float
            Energy of the edge onset (eV)
            Examples: 285 for C K-edge, 532 for O K-edge, 284 for C K-edge
        pre_edge_range : tuple of float, optional
            Explicit (start, end) energies in eV for pre-edge fitting window.
            If None, automatically uses all available data before edge.
            Example: (519, 527) for O K-edge when data starts at 518 eV
        method : str, optional
            Background fitting method:
            - 'auto': Pick a method from the available pre-edge width (default),
              following the "Recommended methods by pre-edge size" table below.
            - 'polynomial': Polynomial fit (most stable for short ranges)
            - 'linear': Linear fit (equivalent to polynomial degree=1)
            - 'powerlaw': Power-law A*E^(-r) (needs longer pre-edge, may fail)
        polynomial_degree : int, optional
            Degree of polynomial (1=linear, 2=quadratic, 3=cubic). Default is 2.
            Only used when method='polynomial'.
        show : bool, optional
            Display before/after visualization. Default True.
        return_dataset : bool, optional
            If True, return Dataset3deels. If False, return numpy array. Default True.
        display_energy_range : (float, float), optional
            (lo, hi) eV to zoom the x-axis of both before/after panels into.
            Purely a display crop -- the fit itself still uses the full
            energy axis; this only changes what's visible. Defaults to the
            full energy range (no zoom).
        display_intensity_range : (float, float), optional
            (lo, hi) to zoom the "before" panel's y-axis (raw spectrum +
            fitted background) into. Display-only, same caveat as
            ``display_energy_range``.
        display_subtracted_intensity_range : (float, float), optional
            (lo, hi) to zoom the "after" panel's y-axis (background-
            subtracted spectrum) into. Separate from
            ``display_intensity_range`` since the subtracted spectrum is
            typically a much smaller scale than the raw intensity.
            Display-only, same caveat as ``display_energy_range``.

        Returns
        -------
        Dataset3deels or ndarray
            Background-subtracted data

        Raises
        ------
        ValueError
            If pre-edge region is insufficient or target_edge is out of range
        RuntimeError
            If fitting fails (typically with powerlaw on limited data)

        Notes
        -----
        **When to use this method:**
        - Data starts close to the edge (limited pre-edge region)
        - Standard subtract_background() fails with window_size error
        - High-loss only acquisitions (no low-loss data)
        - Cropped energy ranges

        **Recommended methods by pre-edge size (used by method='auto'):**
        - < 10 eV: method='linear' (most stable)
        - 10-20 eV: method='polynomial', degree=2
        - > 20 eV: method='powerlaw' (falls back to method='polynomial', degree=3
          if the pre-edge window isn't strictly positive, since powerlaw requires
          E > 0)

        There is no fixed eV-width minimum -- the real constraint is having
        enough data points to fit (at least 3), so a narrow window is fine on
        finely-dispersed (e.g. monochromated) data as long as it contains
        enough channels. E.g. a 0.4 eV window at 0.009 eV/channel dispersion
        has ~44 points, comfortably enough for a linear fit.

        **Comparison to GMS background subtraction:**
        This mimics the GMS "Fit Background" function but without the
        window percentage constraint, using direct energy range specification.

        Examples
        --------
        >>> # O K-edge at 532 eV, data starts at 518 eV (only 14 eV pre-edge)
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=532,
        ...     method='polynomial',
        ...     polynomial_degree=2
        ... )

        >>> # Specify exact pre-edge window
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=532,
        ...     pre_edge_range=(519, 527),  # 8 eV window
        ...     method='linear',
        ...     show=True
        ... )

        >>> # C K-edge with enough pre-edge for power-law
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=285,
        ...     pre_edge_range=(200, 280),  # 80 eV window
        ...     method='powerlaw'
        ... )

        See Also
        --------
        subtract_background : Standard method with window_size percentage
        powerlaw_backgroundfit_eels : Direct power-law fitting function
        """

        energy = self.energy_axis
        mean_spec = self.calculate_mean_spectrum()

        if pre_edge_range is None:
            pre_edge_start = float(energy[0])
            pre_edge_end = float(target_edge - 5)
            print(f"Auto-detected pre-edge: {pre_edge_start:.1f} - {pre_edge_end:.1f} eV")
        else:
            pre_edge_start, pre_edge_end = float(pre_edge_range[0]), float(pre_edge_range[1])
            print(f"Using specified pre-edge: {pre_edge_start:.1f} - {pre_edge_end:.1f} eV")

        if target_edge < energy[0] or target_edge > energy[-1]:
            raise ValueError(
                f"Target edge {target_edge} eV is outside data range "
                f"[{energy[0]:.1f}, {energy[-1]:.1f}] eV"
            )

        if pre_edge_start < energy[0]:
            raise ValueError(
                f"Pre-edge start {pre_edge_start:.1f} eV is before data start {energy[0]:.1f} eV"
            )

        if pre_edge_end >= target_edge:
            raise ValueError(
                f"Pre-edge end {pre_edge_end:.1f} eV must be before target edge {target_edge:.1f} eV"
            )

        available_preedge = pre_edge_end - pre_edge_start

        if available_preedge <= 0:
            raise ValueError(
                f"Insufficient pre-edge region: only {available_preedge:.3f} eV available."
            )

        if method == "auto":
            if available_preedge < 10:
                method = "linear"
            elif available_preedge <= 20:
                method = "polynomial"
                polynomial_degree = 2
            elif pre_edge_start > 0:
                method = "powerlaw"
            else:
                # powerlaw requires a strictly positive fitting window; fall
                # back to a stable polynomial fit when it isn't available.
                method = "polynomial"
                polynomial_degree = 3
            print(f"Auto-selected method: '{method}' ({available_preedge:.1f} eV pre-edge)")

        # Warn if pre-edge is very limited
        if available_preedge < 10:
            warnings.warn(
                f"Limited pre-edge region ({available_preedge:.1f} eV). "
                f"Background fit may be unreliable. Consider method='linear' for stability.",
                UserWarning,
            )

        mask = (energy >= pre_edge_start) & (energy <= pre_edge_end)
        E_window = energy[mask]
        I_window = mean_spec[mask]

        n_points = len(E_window)
        print(f"Pre-edge region: {available_preedge:.1f} eV ({n_points} data points)")

        if n_points < 3:
            raise ValueError(
                f"Insufficient data points in pre-edge window: only {n_points} points. "
                f"Need at least 3 for fitting."
            )

        if method == "linear" or (method == "polynomial" and polynomial_degree == 1):
            coeffs = np.polyfit(E_window, I_window, deg=1)
            background = np.polyval(coeffs, energy)
            fit_info = f"Linear: y = {coeffs[0]:.2e}*E + {coeffs[1]:.2e}"

        elif method == "polynomial":
            if polynomial_degree > n_points - 1:
                warnings.warn(
                    f"Polynomial degree {polynomial_degree} too high for {n_points} points. "
                    f"Using degree {n_points - 1} instead.",
                    UserWarning,
                )
                polynomial_degree = n_points - 1

            coeffs = np.polyfit(E_window, I_window, deg=polynomial_degree)
            background = np.polyval(coeffs, energy)
            fit_info = f"Polynomial (degree {polynomial_degree})"

        elif method == "powerlaw":
            if pre_edge_start <= 0:
                raise ValueError(
                    f"Power-law background fitting requires a strictly positive "
                    f"pre-edge fitting window (E > 0 eV); got pre_edge_range="
                    f"({pre_edge_start:.3f}, {pre_edge_end:.3f}) eV. Choose a "
                    f"pre_edge_range with both bounds > 0 eV, or use "
                    f"method='polynomial' or 'linear' instead."
                )
            try:
                background, popt = _fit_powerlaw_background(E_window, I_window, energy)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Power-law fit failed to converge with {available_preedge:.1f} eV pre-edge. "
                    f"Try method='polynomial' or 'linear' instead. Error: {e}"
                )
            fit_info = f"Power-law: A={popt[0]:.2e}, r={popt[1]:.2f}"
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'auto', 'linear', 'polynomial', or 'powerlaw'."
            )

        print(f"✓ Fit method: {fit_info}")

        data_sub = np.maximum(self.array - background[None, None, :], 0)

        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Background Subtraction: {self.name}\nEdge at {target_edge} eV",
                fontsize=14,
                fontweight="bold",
            )

            # Before subtraction
            ax1.plot(energy, mean_spec, "k-", lw=1.5, label="Raw spectrum")
            ax1.plot(energy, background, "r--", lw=2, label=f"Background ({fit_info})")
            ax1.axvspan(
                pre_edge_start,
                pre_edge_end,
                alpha=0.2,
                color="green",
                label=f"Fit region ({available_preedge:.1f} eV)",
            )
            ax1.axvline(target_edge, color="orange", ls=":", lw=2, label="Edge onset")
            ax1.set_xlabel("Energy (eV)", fontsize=12)
            ax1.set_ylabel("Intensity", fontsize=12)
            ax1.set_title("Before Background Subtraction")
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # After subtraction
            subtracted_spec = mean_spec - background
            ax2.plot(energy, subtracted_spec, "b-", lw=1.5, label="Background-subtracted")
            ax2.axvline(target_edge, color="orange", ls=":", lw=2, label="Edge onset")
            ax2.axhline(0, color="gray", ls="--", alpha=0.5)
            ax2.set_xlabel("Energy (eV)", fontsize=12)
            ax2.set_ylabel("Intensity", fontsize=12)
            ax2.set_title("After Background Subtraction")
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            if display_energy_range is not None:
                e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
                if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
                    raise ValueError(
                        f"display_energy_range must be (lo, hi) with lo < hi, "
                        f"got {display_energy_range!r}"
                    )
                ax1.set_xlim(e_lo, e_hi)
                ax2.set_xlim(e_lo, e_hi)

            if display_intensity_range is not None:
                i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
                if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
                    raise ValueError(
                        f"display_intensity_range must be (lo, hi) with lo < hi, "
                        f"got {display_intensity_range!r}"
                    )
                ax1.set_ylim(i_lo, i_hi)

            if display_subtracted_intensity_range is not None:
                s_lo, s_hi = (
                    float(display_subtracted_intensity_range[0]),
                    float(display_subtracted_intensity_range[1]),
                )
                if not (np.isfinite(s_lo) and np.isfinite(s_hi) and s_lo < s_hi):
                    raise ValueError(
                        f"display_subtracted_intensity_range must be (lo, hi) with lo < hi, "
                        f"got {display_subtracted_intensity_range!r}"
                    )
                ax2.set_ylim(s_lo, s_hi)

            plt.tight_layout()
            plt.show()

        if return_dataset:
            result = Dataset3deels.from_array(
                data_sub,
                sampling=self.sampling,
                origin=self.origin,
                units=self.units,
                name=f"{self.name} (background subtracted)",
            )
            print(f"✓ Created background-subtracted dataset: {result.shape}")
            return result
        else:
            return data_sub

    def fit_near_zlp_transitions(
        self,
        peaks,
        spectrum=None,
        energy_range=(0.0, 2.5),
        zlp_shape="lorentzian",
        zlp_center_guess=0.0,
        zlp_center_bounds=(-0.1, 0.1),
        zlp_width_guess=0.1,
        zlp_width_bounds=(0.01, 1.0),
        zlp_eta_guess=0.5,
        zlp_eta_bounds=(0.0, 1.0),
        maxfev=20000,
    ):
        """
        Fit a spectrum near the ZLP as a ZLP-tail component plus N caller-
        specified peaks, instead of subtracting a background first.

        Intended for transitions close enough to the ZLP (a few tenths to a
        few eV) that background-subtract-then-look-for-peaks is unreliable:
        pre-edge power-law/polynomial background models assume a
        featureless continuum and a fit window well clear of the ZLP tail,
        neither of which holds this close in. Here the ZLP's own decaying
        tail is modeled directly as one component (so it isn't
        misattributed to a peak) alongside the peaks of interest, all fit
        simultaneously to the raw (un-subtracted) spectrum.

        Parameters
        ----------
        peaks : list of dict
            One dict per peak, each with required keys:

            - ``"name"`` : str
            - ``"center_guess"`` : float (eV)
            - ``"center_bounds"`` : (lo, hi) eV
            - ``"width_guess"`` : float (eV, FWHM)
            - ``"width_bounds"`` : (lo, hi) eV FWHM

            and optional keys:

            - ``"shape"`` : ``"gaussian"`` (default), ``"lorentzian"``, or
              ``"pseudo_voigt"``
            - ``"amplitude_guess"`` : float, default the spectrum's value
              at the energy sample closest to ``center_guess``
            - ``"amplitude_bounds"`` : (lo, hi), default ``(0.0, inf)``
            - ``"eta_guess"`` / ``"eta_bounds"`` : only used when
              ``shape="pseudo_voigt"`` (Lorentzian fraction, 0-1)

            No parameter is left unbounded: every component's amplitude,
            center, and width (and eta, for pseudo_voigt) gets an explicit
            ``(lo, hi)`` passed to ``curve_fit``.
        spectrum : ndarray, optional
            1D spectrum to fit, same length as ``self.energy_axis``.
            Defaults to ``self.calculate_mean_spectrum()`` if not given.
        energy_range : (float, float), optional
            ``(lo, hi)`` eV window to fit over. Default ``(0.0, 2.5)`` --
            wide enough to cover the ZLP's descending tail and typical
            near-ZLP transitions, but should be adjusted to the data at
            hand.
        zlp_shape : {"lorentzian", "pseudo_voigt"}, optional
            Lineshape for the ZLP-tail component. Default ``"lorentzian"``
            (a heavier tail than Gaussian -- the usual choice for modeling
            the ZLP's decay).
        zlp_center_guess, zlp_center_bounds : float, (float, float)
            Initial guess and bounds (eV) for the ZLP-tail center. Default
            ``0.0 +/- 0.1`` eV -- the ZLP is expected to sit at (or very
            near) the calibrated energy origin.
        zlp_width_guess, zlp_width_bounds : float, (float, float)
            Initial guess and bounds (eV, FWHM) for the ZLP-tail width.
        zlp_eta_guess, zlp_eta_bounds : float, (float, float)
            Only used when ``zlp_shape="pseudo_voigt"``.
        maxfev : int, optional
            Forwarded to ``scipy.optimize.curve_fit``.

        Returns
        -------
        dict
            - ``"energy"`` : ndarray, the windowed energy axis used for the fit
            - ``"spectrum"`` : ndarray, the windowed input spectrum
            - ``"total_fit"`` : ndarray, the summed model evaluated on ``"energy"``
            - ``"residual"`` : ndarray, ``"spectrum" - "total_fit"``
            - ``"components"`` : dict keyed by component name (``"zlp_tail"``,
              plus each peak's ``"name"``), each a dict with ``"curve"``
              (ndarray), ``"shape"``, ``"amplitude"``, ``"center"``,
              ``"fwhm"``, ``"area"`` (numerically integrated over
              ``"energy"`` via the trapezoid rule), and
              ``"amplitude_stderr"``/``"center_stderr"``/``"fwhm_stderr"``
              (1-sigma, from the fit covariance; ``NaN`` if the covariance
              is singular)
            - ``"popt"`` : ndarray, the full best-fit parameter vector
            - ``"pcov"`` : ndarray, its covariance matrix

        Raises
        ------
        ValueError
            If ``peaks`` is empty, a peak dict is missing a required key,
            a bound is inverted, ``energy_range`` is invalid or doesn't
            overlap the data, or too few points fall in the window for the
            requested number of free parameters.
        RuntimeError
            If ``curve_fit`` fails to converge.
        """
        if not peaks:
            raise ValueError("peaks must be a non-empty list of peak specification dicts")

        energy_full = np.asarray(self.energy_axis, dtype=float)
        if spectrum is None:
            spectrum_full = np.asarray(self.calculate_mean_spectrum(), dtype=float)
        else:
            spectrum_full = np.asarray(spectrum, dtype=float)
            if spectrum_full.shape != energy_full.shape:
                raise ValueError(
                    f"spectrum shape {spectrum_full.shape} does not match "
                    f"energy_axis shape {energy_full.shape}"
                )

        e_lo, e_hi = float(energy_range[0]), float(energy_range[1])
        if e_lo >= e_hi:
            raise ValueError("energy_range must be (lo, hi) with lo < hi")
        if e_hi < energy_full[0] or e_lo > energy_full[-1]:
            raise ValueError(
                f"energy_range {energy_range} does not overlap the data's energy axis "
                f"[{energy_full[0]:.3f}, {energy_full[-1]:.3f}] eV"
            )

        window = (energy_full >= e_lo) & (energy_full <= e_hi)
        energy = energy_full[window]
        y = spectrum_full[window]

        specs = [
            _NearZlpComponentSpec(
                name="zlp_tail",
                shape=zlp_shape,
                amplitude_guess=float(y[np.argmin(np.abs(energy - zlp_center_guess))]),
                amplitude_bounds=(0.0, np.inf),
                center_guess=float(zlp_center_guess),
                center_bounds=(float(zlp_center_bounds[0]), float(zlp_center_bounds[1])),
                fwhm_guess=float(zlp_width_guess),
                fwhm_bounds=(float(zlp_width_bounds[0]), float(zlp_width_bounds[1])),
                eta_guess=float(zlp_eta_guess),
                eta_bounds=(float(zlp_eta_bounds[0]), float(zlp_eta_bounds[1])),
            )
        ]

        required_keys = {"name", "center_guess", "center_bounds", "width_guess", "width_bounds"}
        for i, peak in enumerate(peaks):
            missing = required_keys - set(peak)
            if missing:
                raise ValueError(f"peaks[{i}] is missing required key(s): {sorted(missing)}")

            center_guess = float(peak["center_guess"])
            center_bounds = (float(peak["center_bounds"][0]), float(peak["center_bounds"][1]))
            width_guess = float(peak["width_guess"])
            width_bounds = (float(peak["width_bounds"][0]), float(peak["width_bounds"][1]))
            shape = peak.get("shape", "gaussian")

            if "amplitude_guess" in peak:
                amplitude_guess = float(peak["amplitude_guess"])
            else:
                amplitude_guess = float(y[np.argmin(np.abs(energy - center_guess))])
            amplitude_bounds = peak.get("amplitude_bounds", (0.0, np.inf))
            amplitude_bounds = (float(amplitude_bounds[0]), float(amplitude_bounds[1]))
            eta_bounds = peak.get("eta_bounds", (0.0, 1.0))

            specs.append(
                _NearZlpComponentSpec(
                    name=str(peak["name"]),
                    shape=shape,
                    amplitude_guess=amplitude_guess,
                    amplitude_bounds=amplitude_bounds,
                    center_guess=center_guess,
                    center_bounds=center_bounds,
                    fwhm_guess=width_guess,
                    fwhm_bounds=width_bounds,
                    eta_guess=float(peak.get("eta_guess", 0.5)),
                    eta_bounds=(float(eta_bounds[0]), float(eta_bounds[1])),
                )
            )

        n_params = sum(spec.n_params for spec in specs)
        if len(energy) < n_params:
            raise ValueError(
                f"Only {len(energy)} points in energy_range {energy_range}, but the "
                f"model has {n_params} free parameters across {len(specs)} components. "
                f"Widen energy_range or reduce the number of components."
            )

        p0, lo_bounds, hi_bounds = [], [], []
        for spec in specs:
            p0.extend([spec.amplitude_guess, spec.center_guess, spec.fwhm_guess])
            lo_bounds.extend(
                [spec.amplitude_bounds[0], spec.center_bounds[0], spec.fwhm_bounds[0]]
            )
            hi_bounds.extend(
                [spec.amplitude_bounds[1], spec.center_bounds[1], spec.fwhm_bounds[1]]
            )
            if spec.shape == "pseudo_voigt":
                p0.append(spec.eta_guess)
                lo_bounds.append(spec.eta_bounds[0])
                hi_bounds.append(spec.eta_bounds[1])

        def _model(x, *params):
            return _eval_near_zlp_model(x, params, specs)

        try:
            popt, pcov = curve_fit(
                _model, energy, y, p0=p0, bounds=(lo_bounds, hi_bounds), maxfev=maxfev
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"fit_near_zlp_transitions: curve_fit failed to converge with "
                f"{len(specs)} components over {energy_range} eV. Error: {exc}"
            ) from exc

        if np.all(np.isfinite(pcov)):
            perr = np.sqrt(np.diag(pcov))
        else:
            perr = np.full(len(popt), np.nan)

        total_fit = _model(energy, *popt)
        components = {}
        idx = 0
        for spec in specs:
            n = spec.n_params
            params = popt[idx : idx + n]
            errs = perr[idx : idx + n]
            amplitude, center, fwhm = params[0], params[1], params[2]
            eta = params[3] if spec.shape == "pseudo_voigt" else 0.5
            curve = _eval_near_zlp_component(spec.shape, energy, amplitude, center, fwhm, eta)
            components[spec.name] = {
                "curve": curve,
                "shape": spec.shape,
                "amplitude": float(amplitude),
                "center": float(center),
                "fwhm": float(fwhm),
                "area": float(np.trapezoid(curve, energy)),
                "amplitude_stderr": float(errs[0]),
                "center_stderr": float(errs[1]),
                "fwhm_stderr": float(errs[2]),
            }
            idx += n

        return {
            "energy": energy,
            "spectrum": y,
            "total_fit": total_fit,
            "residual": y - total_fit,
            "components": components,
            "popt": popt,
            "pcov": pcov,
        }

    def powerlaw_backgroundfit_eels(
        self, spectrum, energy_range, target_edge, window_size, show: bool = True
    ):
        """
        Fit a power-law background ``A * E^(-r)`` to a pre-edge window and
        return it evaluated over ``energy_range`` (or the full energy axis).

        Uses a window of the energy axis preceding the target edge, ending
        5 eV before it; the window width is ``window_size`` percent of
        ``target_edge``. Unlike ``subtract_background_limited_preedge``,
        this does not subtract the background from the dataset -- it only
        fits and returns the background curve for a single 1D spectrum.

        Parameters
        ----------
        spectrum : ndarray
            1D spectrum to fit, indexed the same way as the (possibly
            ``energy_range``-cropped) energy axis.
        energy_range : (float, float) or None
            (lo, hi) eV to restrict fitting/display to. Clipped to the
            dataset's own energy range. ``None`` uses the full axis.
        target_edge : float
            Energy of the edge onset (eV); the fit window ends 5 eV before it.
        window_size : float
            Pre-edge fit window width, as a percentage of ``target_edge``
            (must be between 10 and 30).
        show : bool, optional
            Display the spectrum/background/window-limits plot. Default True.

        Returns
        -------
        ndarray
            Fitted background, evaluated on the (possibly cropped) energy axis.

        See Also
        --------
        subtract_background_limited_preedge : Also subtracts and returns a Dataset3deels.
        """

        energy_axis = self.energy_axis

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[2])

        # Check that input window size is between 10% and 30%

        if window_size < 10 or window_size > 30:
            raise ValueError("Invalid window size. Please input a value of between 10 and 30.")

        # Check that the target edge is within the energy range of the spectrum
        # and that a pre-edge region of size at least 10% of the target edge, ending 5 eV before the target edge
        # exists for pre-edge fitting.

        if target_edge < energy_axis[0] or target_edge > energy_axis[-1]:
            raise ValueError("Target edge is outside of energy range.")
        elif ((target_edge - 5) - target_edge * (window_size / 100)) < energy_axis[0]:
            raise ValueError(
                "Insufficient pre-edge background fitting region for this target edge and window size within given energy range."
            )

        # Fit power law function to spectrum within window region of the energy exis

        window_min_E = (target_edge - 5) - target_edge * (window_size / 100)
        window_max_E = target_edge - 5

        window_indices = np.where((energy_axis >= window_min_E) & (energy_axis <= window_max_E))[0]

        window_E = energy_axis[window_indices]
        window_I = spectrum[window_indices]

        background_fit, _popt = _fit_powerlaw_background(
            window_E, window_I, energy_axis, maxfev=2000
        )

        if show:
            # Plot the region of the spectrum between user-specified energy range, overlaid with
            # the background fit curve, with background estimation window boundaries indicated
            fig, ax = plt.subplots()
            ax.plot(energy_axis, spectrum, label="spectrum", color="b")
            ax.plot(energy_axis, background_fit, label="background", color="r")
            ax.vlines(
                x=[window_min_E, window_max_E],
                ymin=0,
                ymax=np.max(spectrum),
                label="window limits",
                color="k",
                linestyle="dashed",
            )
            ax.legend()

        return background_fit

    def smooth_eels_rolling_average(self, roi=None, energy_range=None, mask=None, kernel_size=10):
        energy_axis = self.energy_axis

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[2])

        array3d_subrange = self.array[:, :, indices]

        kernel = np.ones(kernel_size) / kernel_size

        # For each probe position, convolve spectral data with smoothing kernel

        array3d_smoothed = np.zeros(array3d_subrange.shape)

        scan_row, scan_col, _n_energy = array3d_subrange.shape
        for i_row in range(scan_row):
            for i_col in range(scan_col):
                probe_spectrum = array3d_subrange[i_row, i_col, :]
                spectrum_smoothed = np.convolve(probe_spectrum, kernel, mode="same")
                array3d_smoothed[i_row, i_col, :] = spectrum_smoothed

        output_origin = np.array(self.origin, dtype=float, copy=True)
        output_origin[2] = energy_axis[0]
        smoothed_data3d = Dataset3deels.from_array(
            array=array3d_smoothed,
            sampling=self.sampling,
            origin=output_origin,
            units=self.units,
        )

        # Plot raw and smoothed mean spectra on the same set of axes

        mean_spectrum_raw = self.calculate_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            mask=mask,
        )
        mean_spectrum_smoothed = smoothed_data3d.calculate_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            mask=mask,
        )

        fig, ax = plt.subplots()
        ax.plot(energy_axis, mean_spectrum_raw, label="raw spectrum", color="b")
        ax.plot(energy_axis, mean_spectrum_smoothed, label="kernel-smoothed spectrum", color="r")
        ax.legend()

        return smoothed_data3d

    def measure_zlp_offset(
        self,
        zlp_guess_x=None,
        search_window=10,
        fit_window=0.8,
        median_filter_pixels=3,
        polynomial_order_rows=3,
        polynomial_order_columns=3,
        fit_to_plane=False,
        fit_to_polynomial=False,
        fit_zlp=True,
        mask=None,
        fit_window_fwhm_multiplier=3.5,
    ):
        """
        Measure ZLP offset at each pixel position by using a guess of ZLP posfitting each spectrum to a Gaussian

        Finds the difference between the maximum of the ZLP Gaussian fit and 0 eV at every pixel,
        and fits a 2D plane to measured ZLP offsets if fit_to_plane=True.

        Parameters
        ----------
        zlp_guess_x : float or None
            Expected energy position of the ZLP in eV. If None, uses the
            tallest peak in each spectrum as the ZLP. If provided, searches
            for the tallest peak within the search window around that energy.
        search_window : int
            Number of channels to search on either side of center_guess.
            Only used when center_guess is not None. Default is 10.
        fit_window : float or "auto"
            Half-width (eV) of the window the per-pixel Gaussian is fit
            over. Default is ``0.8`` (unchanged, for backward compatibility).
            Pass ``"auto"`` to size it from the data instead:
            ``fit_window_fwhm_multiplier * FWHM``, where FWHM is estimated
            from the dataset's mean spectrum. Useful for monochromated data
            where the ZLP is much narrower than 0.8 eV -- an oversized
            window biases (or, in the extreme, fails to converge for) the
            per-pixel fit.
        fit_window_fwhm_multiplier : float
            Multiplier applied to the estimated FWHM when
            ``fit_window="auto"``. Ignored otherwise. Default 3.5.
        mask : ndarray of bool, shape (scan_row, scan_col), optional
            ``True`` marks a pixel to exclude from fitting entirely (e.g. a
            known vacuum/dead-detector region). Excluded pixels are treated
            the same as fit failures: their ZLP shift is left as NaN in the
            raw per-pixel output, and interpolated from the fitted surface
            when ``fit_to_plane`` or ``fit_to_polynomial`` is requested.

        Notes
        -----
        A per-pixel Gaussian fit (``fit_zlp=True``) can fail outright
        (``RuntimeError``/``ValueError`` from ``curve_fit``, or an all-NaN
        window), or it can "succeed" with ``mu`` pinned at the edge of its
        ``[mu0 - fit_window, mu0 + fit_window]`` bound -- a silent bad fit.
        Both cases are treated as failures: the pixel is left as ``NaN``
        rather than filled with the channel-resolution argmax position
        (which would inject imprecise, unflagged values into a downstream
        plane/polynomial fit). If ``fit_to_plane`` or ``fit_to_polynomial``
        is set, the surface is fit with ``np.linalg.lstsq`` over only the
        finite, unmasked pixels, and used to fill in the missing ones --
        this assumes the ZLP shift is smooth across the scan (spectrometer
        drift geometry), the same assumption ``fit_to_plane=True`` already
        makes. If fewer than ``max(6, 5%)`` of pixels remain usable, a
        ``ValueError`` is raised naming the counts rather than fitting a
        surface to noise. A summary ``UserWarning`` reports how many pixels
        were masked, failed (by exception type), and boundary-pinned.

        With ``fit_to_plane=False`` and ``fit_to_polynomial=False``, the
        raw per-pixel array is returned and can still contain ``NaN`` at
        failed/masked pixels -- this is intentional (the caller asked for
        raw values, not a smoothed surface). Passing such an array on to
        ``apply_zlp_correction`` via ``zlp_shifts_array`` will raise its
        existing "ZLP shifts must contain only finite values" error.

        Returns
        -------
        Dataset3deels
            New dataset with corrected energy calibration.

        """

        # Define Gaussian constraint to fit ZLP to
        def _gaussian_fit(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        def _plane_fit_2d(M, a, b, c):
            row, col = M
            return (a * row) + (b * col) + c

        def _polynomial_fit_2d(M, c00, c10, c01, c20, c11, c02):
            row, col = M
            return (
                c00
                + (c10 * row)
                + (c01 * col)
                + (c20 * row**2)
                + (c11 * row * col)
                + (c02 * col**2)
            )

        scan_row, scan_col, n_energy = self.array.shape
        energy_axis = self.energy_axis

        fit_window = _resolve_auto_fit_window(
            fit_window, energy_axis, self.calculate_mean_spectrum(), fit_window_fwhm_multiplier
        )

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (scan_row, scan_col):
                raise ValueError(
                    f"mask shape {mask.shape} does not match scan shape {(scan_row, scan_col)}"
                )
        else:
            mask = np.zeros((scan_row, scan_col), dtype=bool)
        n_masked = int(mask.sum())

        # For each pixel, measure the zlp position by fitting a Gaussian to the measured zero-loss signal and taking its center as the zlp position.

        zlp_measured = np.full((scan_row, scan_col), np.nan)
        failure_counts: dict[str, int] = {}
        n_boundary_pinned = 0

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                if mask[i_row, i_col]:
                    continue

                # Apply median filter to discount hot pixels that might spuriously produce the maximum intensity of the spectrum
                if median_filter_pixels > 0:
                    spec_filt = median_filter(self.array[i_row, i_col, :], median_filter_pixels)
                else:
                    spec_filt = self.array[i_row, i_col, :]

                if not fit_zlp:
                    try:
                        zlp_crude_idx = int(np.nanargmax(spec_filt))
                        zlp_measured[i_row, i_col] = energy_axis[zlp_crude_idx]
                    except ValueError as exc:
                        failure_counts[type(exc).__name__] = (
                            failure_counts.get(type(exc).__name__, 0) + 1
                        )
                    continue

                # Use initial guess for ZLP to define window for Gaussian fitting. If zlp_guess_x=None (default) use the maximum value of the spectrum
                try:
                    if zlp_guess_x is not None:
                        zlp_crude_idx = int(np.argmin(np.abs(energy_axis - zlp_guess_x)))
                    else:
                        zlp_crude_idx = int(np.nanargmax(spec_filt))
                except ValueError as exc:
                    failure_counts[type(exc).__name__] = (
                        failure_counts.get(type(exc).__name__, 0) + 1
                    )
                    continue

                mu0 = energy_axis[zlp_crude_idx]

                lo = mu0 - fit_window
                hi = mu0 + fit_window

                x_mask = (energy_axis >= lo) & (energy_axis <= hi)

                xw = energy_axis[x_mask]
                yw = spec_filt[x_mask]

                A0 = spec_filt[zlp_crude_idx]
                sigma0 = fit_window / 2

                p0 = (A0, mu0, sigma0)

                bounds = (
                    (
                        0.0,
                        lo,
                        1e-12,
                    ),
                    (
                        np.inf,
                        hi,
                        np.inf,
                    ),
                )

                try:
                    popt, _ = curve_fit(_gaussian_fit, xw, yw, p0=p0, bounds=bounds)
                except (RuntimeError, ValueError) as exc:
                    failure_counts[type(exc).__name__] = (
                        failure_counts.get(type(exc).__name__, 0) + 1
                    )
                    continue

                mu = popt[1]
                tol = max(1e-3, 0.01 * (hi - lo))
                if abs(mu - lo) < tol or abs(mu - hi) < tol:
                    n_boundary_pinned += 1
                    continue

                zlp_measured[i_row, i_col] = mu

        n_total = scan_row * scan_col
        n_used = int(np.sum(np.isfinite(zlp_measured)))
        n_failed = sum(failure_counts.values())
        min_required = max(6, int(np.ceil(0.05 * n_total)))

        if n_used < min_required:
            raise ValueError(
                f"Only {n_used}/{n_total} pixels produced a usable ZLP fit "
                f"(masked={n_masked}, failed={n_failed} {failure_counts}, "
                f"boundary_pinned={n_boundary_pinned}); need at least "
                f"{min_required}. Check fit_window (try fit_window='auto') "
                f"and/or mask."
            )

        if n_masked or n_failed or n_boundary_pinned:
            warnings.warn(
                f"measure_zlp_offset: {n_used}/{n_total} pixels used "
                f"(fit_window={fit_window:.4g} eV); {n_masked} masked, "
                f"{n_failed} failed {failure_counts}, {n_boundary_pinned} "
                f"boundary-pinned (rejected).",
                UserWarning,
            )

        if fit_to_plane:
            # Fit a 2D plane to the array of measured ZLPs, using only the
            # finite, unmasked pixels (np.linalg.lstsq: exact for a linear
            # model, no convergence failure mode).
            row_data, col_data = np.meshgrid(
                np.arange(scan_row), np.arange(scan_col), indexing="ij"
            )
            valid = np.isfinite(zlp_measured)
            rows_valid = row_data[valid].astype(float)
            cols_valid = col_data[valid].astype(float)
            y_valid = zlp_measured[valid]
            design = np.column_stack([rows_valid, cols_valid, np.ones_like(rows_valid)])
            coeffs, *_ = np.linalg.lstsq(design, y_valid, rcond=None)

            zlp_plane_2d = _plane_fit_2d((row_data, col_data), *coeffs)

            fig, _ = show_2d(
                [zlp_measured, zlp_plane_2d],
                cmap="magma",
                cbar=True,
                title=["Measured ZLP\n(mean of Gaussian fit)", "ZLP plane fit"],
                tight_layout=False,
            )
            fig.subplots_adjust(top=0.88, wspace=0.35)
            return zlp_plane_2d
        elif fit_to_polynomial:
            # Fit a 2D polynomial to the array of measured ZLPs, using only
            # the finite, unmasked pixels.
            row_data, col_data = np.meshgrid(
                np.arange(scan_row), np.arange(scan_col), indexing="ij"
            )
            valid = np.isfinite(zlp_measured)
            rows_valid = row_data[valid].astype(float)
            cols_valid = col_data[valid].astype(float)
            y_valid = zlp_measured[valid]
            design = np.column_stack(
                [
                    np.ones_like(rows_valid),
                    rows_valid,
                    cols_valid,
                    rows_valid**2,
                    rows_valid * cols_valid,
                    cols_valid**2,
                ]
            )
            coeffs, *_ = np.linalg.lstsq(design, y_valid, rcond=None)

            zlp_plane_2d = _polynomial_fit_2d((row_data, col_data), *coeffs)

            fig, _ = show_2d(
                [zlp_measured, zlp_plane_2d],
                cmap="magma",
                cbar=True,
                title=["Measured ZLP\n(mean of Gaussian fit)", "ZLP polynomial fit"],
                tight_layout=False,
            )
            fig.subplots_adjust(top=0.88, wspace=0.35)
            return zlp_plane_2d

        else:
            fig, _ = show_2d(
                [zlp_measured],
                cmap="magma",
                cbar=True,
                title=["Measured ZLP\n(mean of Gaussian fit)"],
                tight_layout=False,
            )
            fig.subplots_adjust(top=0.88)
            return zlp_measured

    def apply_zlp_correction(
        self,
        zlp_guess_x=None,
        zlp_shifts_array=None,
        fit_window=0.8,
        measure_offset=True,
        fit_to_plane=True,
        fit_to_polynomial=False,
        fit_zlp=True,
        return_3d_dataset=True,
        return_shifts=False,
        in_place=False,
        mask=None,
        fit_window_fwhm_multiplier=3.5,
        display_energy_range=None,
        display_intensity_range=None,
    ):
        # display_energy_range / display_intensity_range: (lo, hi) to zoom
        # the raw/corrected mean-spectrum preview plot's x/y axes into.
        # Display-only -- the ZLP measurement and correction always use the
        # full energy axis; this only changes what's visible. Default None
        # shows the full range.
        #
        # Default behavior is to automatically call measure_zlp_offset to generate an array of ZLP shifts for each scan position.
        # Alternatively, a 2D array matching the scan_row and scan_col dimensions of the 3D dataset can be supplied as the value of zlp_shifts_array to skip this step.
        # If measure_offset is False and no 2D ZLP shifts array is provided, a scalar input for zlp_guess_x can be used to shift the energy axis at every scan position by that amount.
        # fit_window="auto" and mask are forwarded straight through to
        # measure_zlp_offset(); see its docstring.
        if measure_offset:
            zlp_array = self.measure_zlp_offset(
                zlp_guess_x=zlp_guess_x,
                fit_window=fit_window,
                fit_to_plane=fit_to_plane,
                fit_to_polynomial=fit_to_polynomial,
                fit_zlp=fit_zlp,
                mask=mask,
                fit_window_fwhm_multiplier=fit_window_fwhm_multiplier,
            )
        elif zlp_shifts_array is not None:
            zlp_array = np.asarray(zlp_shifts_array, dtype=float)
            if zlp_array.shape != self.array.shape[0:2]:
                raise ValueError(
                    "Dimensions of input array for ZLP shifts do not match scan_row and scan_col dimensions of 3D spectroscopy dataset."
                )
        elif zlp_guess_x is not None:
            zlp_array = np.ones(self.array.shape[0:2], dtype=float) * zlp_guess_x
        else:
            raise ValueError(
                "measure_offset was set to False and no input argument for ZLP shifts was provided."
            )

        zlp_array = np.asarray(zlp_array, dtype=float)
        if not np.all(np.isfinite(zlp_array)):
            raise ValueError("ZLP shifts must contain only finite values.")

        # Initialize 3D array to populate with spectra aligned along the energy axis
        corrected_array = np.empty(self.array.shape, dtype=np.result_type(self.array.dtype, float))

        scan_row, scan_col, n_energy = self.array.shape

        energy_axis = self.energy_axis

        # Apply sub-channel ZLP shifts using 1D linear interpolation along the energy axis.
        for i_row in range(scan_row):
            for i_col in range(scan_col):
                spec = self.array[i_row, i_col, :]
                corrected_array[i_row, i_col, :] = np.interp(
                    energy_axis + zlp_array[i_row, i_col],
                    energy_axis,
                    spec,
                    left=np.nan,
                    right=np.nan,
                )

        # Remove all planes along energy axis containing NaN, to equalize spectra lengths across all scan positions
        mask = np.isnan(corrected_array).any(axis=(0, 1))
        aligned_data_3d = corrected_array[:, :, ~mask]
        new_Eaxis = energy_axis[~mask]

        if aligned_data_3d.shape[2] == 0:
            raise ValueError(
                "ZLP shifts leave no shared energy range after alignment. "
                "Check that zlp_shifts_array is in energy units, not channel indices."
            )

        new_origin = new_Eaxis[0]

        # Calculate mean spectra before and after correction for plotting
        mean_spectrum_raw = self.array.mean(axis=(0, 1))
        mean_spectrum_corrected = aligned_data_3d.mean(axis=(0, 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(energy_axis, mean_spectrum_raw, label="Raw mean spectrum", color="r")
        ax2.plot(new_Eaxis, mean_spectrum_corrected, label="ZLP-corrected spectrum", color="b")
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Intensity")
        ax1.grid(True, alpha=0.1)
        ax1.legend()
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Intensity")
        ax2.grid(True, alpha=0.1)
        ax2.legend()

        if display_energy_range is not None:
            e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
            if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
                raise ValueError(
                    f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
                )
            ax1.set_xlim(e_lo, e_hi)
            ax2.set_xlim(e_lo, e_hi)

        if display_intensity_range is not None:
            i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
            if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
                raise ValueError(
                    f"display_intensity_range must be (lo, hi) with lo < hi, "
                    f"got {display_intensity_range!r}"
                )
            ax1.set_ylim(i_lo, i_hi)
            ax2.set_ylim(i_lo, i_hi)

        fig.tight_layout()

        if return_3d_dataset:
            corrected_dataset = Dataset3deels.from_array(
                array=aligned_data_3d,
                name=self.name,
                sampling=self.sampling,
                origin=new_origin,
                units=self.units,
            )
            if return_shifts:
                return corrected_dataset, zlp_array
            else:
                return corrected_dataset
        elif in_place:
            self.array = aligned_data_3d
            if return_shifts:
                return aligned_data_3d, zlp_array
            else:
                return aligned_data_3d
        else:
            if return_shifts:
                return aligned_data_3d, zlp_array
            else:
                return aligned_data_3d

    def correct_high_loss_energy_axis(
        self,
        ll_3d_dataset=None,
        zlp_guess_x=None,
        zlp_shifts_array=None,
        fit_window=0.8,
        measure_offset=True,
        fit_to_plane=True,
        fit_to_polynomial=False,
        fit_zlp=True,
        return_3d_dataset=True,
        return_shifts=False,
        in_place=False,
    ):
        """
        Applies ZLP correction to low-loss 3D EELS dataset and extends the computed shift at each
        pixel position to correct the corresponding high-loss 3D EELS dataset
        """
        if ll_3d_dataset is None:
            raise ValueError("No ll_3d_dataset provided for ZLP alignment")
        elif ll_3d_dataset.__class__ != Dataset3deels:
            raise ValueError("ll_3d_dataset input is not a Dataset3deels object")

        ll_corrected, ll_shifts = ll_3d_dataset.apply_zlp_correction(
            zlp_guess_x=zlp_guess_x,
            fit_window=fit_window,
            fit_to_plane=fit_to_plane,
            fit_to_polynomial=fit_to_polynomial,
            fit_zlp=fit_zlp,
            return_3d_dataset=False,
            return_shifts=True,
        )

        # Synchronize High-Loss energy origin based on median shift
        hl_corrected, hl_shifts = self.apply_zlp_correction(
            zlp_shifts_array=ll_shifts,
            measure_offset=False,
            return_3d_dataset=return_3d_dataset,
            return_shifts=True,
        )

        if return_shifts:
            return hl_corrected, hl_shifts
        else:
            return hl_corrected

    def calculate_thickness_log_ratio(
        self,
        zlp_window=10,
        median_filter_pixels=3,
        fit_zlp=True,
        zlp_guess_x=None,
        plot=True,
        mask=None,
        fit_window_fwhm_multiplier=3.5,
    ):
        """
        Calculates the relative thickness map (t/lambda) using the Log-Ratio method.

        Parameters
        ----------
        zlp_window : float or "auto"
            Half-width (eV) of the window the per-pixel ZLP Gaussian fit is
            evaluated over (also the half-width of the ZLP integration
            window). Default ``10`` (unchanged). Pass ``"auto"`` to size it
            as ``fit_window_fwhm_multiplier * FWHM`` estimated from the
            dataset's mean spectrum -- see ``measure_zlp_offset``.
        mask : ndarray of bool, shape (scan_row, scan_col), optional
            ``True`` marks a pixel to exclude from fitting (e.g. known
            vacuum/dead-detector regions).

        Notes
        -----
        Uses the same per-pixel Gaussian-fit robustness as
        ``measure_zlp_offset``: masked pixels, fit failures
        (``RuntimeError``/``ValueError``, or an all-NaN window), and
        boundary-pinned fits are all left as ``NaN`` rather than filled with
        the imprecise argmax position. Since this method (unlike
        ``measure_zlp_offset``) has no ``fit_to_plane`` toggle but always
        needs a per-pixel ZLP position for the intensity-window integration
        below, any remaining ``NaN`` gaps are filled -- and only those gaps,
        not the successfully measured pixels -- by interpolating a
        ``np.linalg.lstsq`` plane fit over the finite, unmasked pixels
        (same smooth-across-the-scan assumption as ``measure_zlp_offset``).
        If fewer than ``max(6, 5%)`` of pixels are usable, a ``ValueError``
        is raised naming the counts. A summary ``UserWarning`` reports how
        many pixels were masked, failed (by exception type), and
        boundary-pinned.
        """

        def _gaussian_fit(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        scan_row, scan_col, n_energy = self.array.shape
        energy_axis = self.energy_axis

        zlp_window = _resolve_auto_fit_window(
            zlp_window, energy_axis, self.calculate_mean_spectrum(), fit_window_fwhm_multiplier
        )

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (scan_row, scan_col):
                raise ValueError(
                    f"mask shape {mask.shape} does not match scan shape {(scan_row, scan_col)}"
                )
        else:
            mask = np.zeros((scan_row, scan_col), dtype=bool)
        n_masked = int(mask.sum())

        zlp_measured = np.full((scan_row, scan_col), np.nan)
        failure_counts: dict[str, int] = {}
        n_boundary_pinned = 0

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                if mask[i_row, i_col]:
                    continue

                # Apply median filter to discount hot pixels that might spuriously produce the maximum intensity of the spectrum
                if median_filter_pixels > 0:
                    spec_filt = median_filter(self.array[i_row, i_col, :], median_filter_pixels)
                else:
                    spec_filt = self.array[i_row, i_col, :]

                if not fit_zlp:
                    try:
                        zlp_crude_idx = int(np.nanargmax(spec_filt))
                        zlp_measured[i_row, i_col] = energy_axis[zlp_crude_idx]
                    except ValueError as exc:
                        failure_counts[type(exc).__name__] = (
                            failure_counts.get(type(exc).__name__, 0) + 1
                        )
                    continue

                # Use initial guess for ZLP to define window for Gaussian fitting. If zlp_guess_x=None (default) use the maximum value of the spectrum
                try:
                    if zlp_guess_x is not None:
                        zlp_crude_idx = int(np.argmin(np.abs(energy_axis - zlp_guess_x)))
                    else:
                        zlp_crude_idx = int(np.nanargmax(spec_filt))
                except ValueError as exc:
                    failure_counts[type(exc).__name__] = (
                        failure_counts.get(type(exc).__name__, 0) + 1
                    )
                    continue

                mu0 = energy_axis[zlp_crude_idx]

                lo = mu0 - zlp_window
                hi = mu0 + zlp_window

                x_mask = (energy_axis >= lo) & (energy_axis <= hi)

                xw = energy_axis[x_mask]
                yw = spec_filt[x_mask]

                A0 = spec_filt[zlp_crude_idx]
                sigma0 = zlp_window / 2

                p0 = (A0, mu0, sigma0)

                bounds = (
                    (
                        0.0,
                        lo,
                        1e-12,
                    ),
                    (
                        np.inf,
                        hi,
                        np.inf,
                    ),
                )

                try:
                    popt, _ = curve_fit(_gaussian_fit, xw, yw, p0=p0, bounds=bounds)
                except (RuntimeError, ValueError) as exc:
                    failure_counts[type(exc).__name__] = (
                        failure_counts.get(type(exc).__name__, 0) + 1
                    )
                    continue

                mu = popt[1]
                tol = max(1e-3, 0.01 * (hi - lo))
                if abs(mu - lo) < tol or abs(mu - hi) < tol:
                    n_boundary_pinned += 1
                    continue

                zlp_measured[i_row, i_col] = mu

        n_total = scan_row * scan_col
        n_used = int(np.sum(np.isfinite(zlp_measured)))
        n_failed = sum(failure_counts.values())
        min_required = max(6, int(np.ceil(0.05 * n_total)))

        if n_used < min_required:
            raise ValueError(
                f"Only {n_used}/{n_total} pixels produced a usable ZLP fit "
                f"(masked={n_masked}, failed={n_failed} {failure_counts}, "
                f"boundary_pinned={n_boundary_pinned}); need at least "
                f"{min_required}. Check zlp_window (try zlp_window='auto') "
                f"and/or mask."
            )

        if n_masked or n_failed or n_boundary_pinned:
            warnings.warn(
                f"calculate_thickness_log_ratio: {n_used}/{n_total} pixels "
                f"used (zlp_window={zlp_window:.4g} eV); {n_masked} masked, "
                f"{n_failed} failed {failure_counts}, {n_boundary_pinned} "
                f"boundary-pinned (rejected).",
                UserWarning,
            )

        if n_used < n_total:
            # Fill only the gaps (masked/failed/boundary-pinned pixels) by
            # interpolating a plane fit over the pixels that were
            # successfully measured -- successfully measured pixels are
            # left untouched.
            row_data, col_data = np.meshgrid(
                np.arange(scan_row), np.arange(scan_col), indexing="ij"
            )
            valid = np.isfinite(zlp_measured)
            rows_valid = row_data[valid].astype(float)
            cols_valid = col_data[valid].astype(float)
            y_valid = zlp_measured[valid]
            design = np.column_stack([rows_valid, cols_valid, np.ones_like(rows_valid)])
            coeffs, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
            filled = coeffs[0] * row_data + coeffs[1] * col_data + coeffs[2]
            zlp_measured = np.where(valid, zlp_measured, filled)

        I_zlp = np.zeros((scan_row, scan_col))

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                I_zlp[i_row, i_col] = np.sum(
                    self.array[
                        i_row,
                        i_col,
                        np.where(energy_axis >= (zlp_measured[i_row, i_col] - zlp_window / 2))[0][
                            0
                        ] : np.where(energy_axis <= (zlp_measured[i_row, i_col] + zlp_window / 2))[
                            0
                        ][-1],
                    ]
                )

        # print(f"Calculating thickness for {self.name}...")

        # Integrate intensity of ZLP and entire spectrum separately, and calculate t/lambda
        I_total = np.sum(self.array, axis=2)

        t_over_lambda = np.log1p((I_total) / (I_zlp))

        # Remove NaN matrix elements
        t_over_lambda = np.nan_to_num(t_over_lambda, nan=0.0, posinf=0.0, neginf=0.0)
        t_over_lambda = np.clip(t_over_lambda, 0, 4.0)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            im = ax1.imshow(t_over_lambda, cmap="viridis", origin="upper")
            ax1.set_title(r"Relative Thickness Map ($t/\lambda$)")
            plt.colorbar(im, ax=ax1, label=r"$t/\lambda$")

            ax2.hist(t_over_lambda.flatten(), bins=50, color="steelblue", alpha=0.7, ec="k")
            ax2.axvline(
                np.mean(t_over_lambda),
                color="red",
                ls="--",
                label=f"Mean: {np.mean(t_over_lambda):.2f}",
            )
            ax2.set_title("Thickness Distribution")
            ax2.set_xlabel(r"$t/\lambda$")
            ax2.legend()

            plt.tight_layout()
            plt.show()

        return t_over_lambda
