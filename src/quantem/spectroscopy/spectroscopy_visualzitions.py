import contextlib
import io
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_coeffs, savgol_filter
from scipy.stats import norm, pearsonr

from quantem.core.visualization import show_2d
from quantem.spectroscopy.utils import (
    _validate_pre_edge_window,
    detect_peaks_in_range,
    summarize_energy_windows,
    summarize_map_diagnostics,
)


def plot_attached_spectrum(
    self,
    spectrum_index=0,
    display_energy_range=None,
    display_intensity_range=None,
):
    """
    Parameters
    ----------
    display_energy_range : (float, float), optional
        (lo, hi) to zoom the x-axis into. Display-only -- the full spectrum
        is still plotted; this only changes what's visible. Defaults to the
        full energy range (no zoom).
    display_intensity_range : (float, float), optional
        (lo, hi) to zoom the y-axis into. Display-only, same caveat as
        ``display_energy_range``.
    """
    fig, (ax_spec) = plt.subplots(1, 1, figsize=(12, 4))

    ds = self.attached_spectra[spectrum_index]
    energy = ds.origin[0] + ds.sampling[0] * np.arange(ds.shape[0])
    ax_spec.plot(energy, ds.array, linewidth=1.5)

    if self.dataset_type == "xeds":
        ax_spec.set_xlabel("Energy (keV)")
    elif self.dataset_type == "eels":
        ax_spec.set_xlabel("Energy (eV)")
    ax_spec.set_ylabel("Intensity")
    ax_spec.set_title(f"Spectrum in index {spectrum_index}")
    ax_spec.grid(True, alpha=0.1)

    if display_energy_range is not None:
        e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
        ax_spec.set_xlim(e_lo, e_hi)

    if display_intensity_range is not None:
        i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
        if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
            raise ValueError(
                f"display_intensity_range must be (lo, hi) with lo < hi, got {display_intensity_range!r}"
            )
        ax_spec.set_ylim(i_lo, i_hi)

    fig.tight_layout()
    plt.show()


def _plot_pca_results(
    self,
    components,
    loadings,
    explained_variance_ratio,
    n_show: int = 4,
    title: str = "",
):
    """
    Plot PCA results including scree plot, components, and loadings.

    Parameters
    ----------
    components : NDArray
        Principal component spectra
    loadings : NDArray
        Spatial loadings for each component
    explained_variance_ratio : NDArray
        Explained variance ratios
    n_show : int
        Number of components to show
    title : str, optional
        Extra line(s) written under the figure titles (e.g. which data / background subtraction the PCA is of).
    """
    fig, (ax_scree, ax_components) = plt.subplots(1, 2, figsize=(12, 4))
    cumsum_var = np.cumsum(explained_variance_ratio)
    component_numbers = np.arange(1, len(explained_variance_ratio) + 1)

    ax_scree.bar(
        component_numbers,
        explained_variance_ratio * 100,
        alpha=0.6,
        label="Individual",
    )
    ax_scree.plot(component_numbers, cumsum_var * 100, "ro-", label="Cumulative")
    ax_scree.set_xlabel("Component Number")
    ax_scree.set_ylabel("Explained Variance (%)")
    ax_scree.set_title("Scree Plot")
    ax_scree.legend()
    ax_scree.grid(True, alpha=0.3)

    energy_sampling = float(self.sampling[2])
    energy_origin = float(self.origin[2])
    energy_axis = energy_origin + energy_sampling * np.arange(components.shape[1])

    for i in range(n_show):
        ax_components.plot(
            energy_axis,
            components[i],
            label=f"PC{i + 1} ({explained_variance_ratio[i] * 100:.1f}%)",
        )
    ax_components.set_xlabel("Energy")
    ax_components.set_ylabel("Component")
    ax_components.set_title("Principal Component Spectra")
    ax_components.legend()
    ax_components.grid(True, alpha=0.3)

    fig.suptitle("PCA Analysis" + (f"\n{title}" if title else ""), fontsize=10)
    fig.tight_layout()
    plt.show()

    show_2d(
        [loadings[i] for i in range(n_show)],
        title=[
            f"Loading {i + 1} ({explained_variance_ratio[i] * 100:.1f}%)" for i in range(n_show)
        ],
        cmap="RdBu_r",
        cbar=True,
        scalebar={
            "sampling": float(self.sampling[1]),
            "units": str(self.units[1]),
        },
    )
    if title:
        plt.gcf().suptitle(title, fontsize=8)
    plt.show()


def show_mean_spectrum(
    self,
    roi=None,
    roi_cal=None,
    energy_range=None,
    mask=None,
    intensity_range=None,
    normalize=False,
    **kwargs,
):
    """
    Plot the mean spectrum from a spatial ROI in a 3D spectroscopy cube (Y, X, E).

    Parameters
    ----------
    roi : list or tuple, optional
        Region of interest as [y, x, dy, dx] where:
        - y, x: top-left pixel coordinates
        - dy, dx: height and width of ROI
        Use None for default values:
        - [y, None, dy, None] = row y with height dy, full width
        - [None, x, None, dx] = column x with width dx, full height
        - [y, x, None, None] = from (y,x) to bottom-right corner
        If roi=None, uses full image. Can also be [y, x] for single pixel.
    energy_range : list or tuple, optional
        Energy range to display as [min_energy, max_energy] in keV.
    mask : array, optional
        Boolean mask for pixel selection.
    intensity_range : 2-tuple, None
        If not None, sets intensity range on spectrum plot
    normalize : bool, optional
        If ``True``, scale the mean spectrum to the range [0, 1]. If
        ``False``, plot the mean spectrum in original intensity units.
    Returns
    -------
    (fig, ax) : tuple
        The Matplotlib Figure and Axes of the spectrum plot.
    """

    # CALCULATE MEAN SPECTRUM FOR GIVEN ROI AND ENERGY RANGE --------------------------

    y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)

    energy_range_for_calc = None if energy_range is None else list(energy_range)
    spec = self.calculate_mean_spectrum(
        roi=roi,
        roi_cal=roi_cal,
        energy_range=energy_range_for_calc,
        mask=mask,
        normalize=normalize,
    )

    E = np.asarray(self.energy_axis, dtype=float)

    if mask is not None:
        E = E[np.asarray(mask, dtype=bool)]

    if energy_range is not None:
        indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
        E = E[indices]

    # PLOTTING ---------------------------------------------------------------------------

    # Create subplot layout: image on left, spectrum on right
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))

    # LEFT PLOT: Show sum image with ROI highlighted
    # Create sum image across all energy channels (or masked channels)
    if mask is not None:
        sum_img = np.asarray(self.array, dtype=float)[:, :, np.asarray(mask, dtype=bool)].sum(
            axis=2
        )
        title_suffix = " (masked energies)"
    else:
        sum_img = np.asarray(self.array, dtype=float).sum(axis=2)
        title_suffix = ""

    map_title = f"Integrated Intensity Map{title_suffix}"
    show_2d(
        sum_img,
        figax=(fig, ax_img),
        title=map_title,
        cmap="viridis",
        cbar=True,
        show_ticks=True,
        scalebar={
            "sampling": float(self.sampling[1]),
            "units": str(self.units[1]),
        },
        **kwargs,
    )
    # Highlight the ROI with a rectangle
    rect = Rectangle(
        (x - 0.5, y - 0.5), dx, dy, linewidth=2, edgecolor="red", facecolor="none", alpha=0.8
    )
    ax_img.add_patch(rect)

    # RIGHT PLOT: Show spectrum
    ax_spec.plot(E, spec, linewidth=1.5, color="k")
    if self.dataset_type == "xeds":
        ax_spec.set_xlabel("Energy (keV)")
    else:
        ax_spec.set_xlabel("Energy (eV)")
    ax_spec.set_ylabel("Normalized intensity" if normalize else "Intensity")
    ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
    ax_spec.grid(True, alpha=0.1)
    if intensity_range is not None:
        ax_spec.set_ylim([intensity_range[0], intensity_range[1]])

    fig.tight_layout()
    return fig, (ax_img, ax_spec)


_MAP_NORMALIZATIONS = ("percentile", "minmax", "zscore", "robust_z", "relative")


def _normalize_energy_map(energy_map, mode, percentiles=(1.0, 99.0)):
    """Rescale one 2D energy-window map so maps from different windows are comparable.

    Every mode is computed from this map alone, so a bright window (e.g.
    near the ZLP tail) and a dim one (e.g. a weak feature at 2 eV) end up on
    the same footing and their *spatial contrast* can be compared directly.
    NaN/inf pixels are ignored when computing the statistics.

    ``"percentile"`` : clip to the (``percentiles``) range, then map to [0, 1].
        Robust to hot pixels/spikes; the default.
    ``"minmax"``     : (m - min) / (max - min) -> [0, 1].
    ``"zscore"``     : (m - mean) / std, i.e. units of standard deviations.
    ``"robust_z"``   : (m - median) / (1.4826 * MAD). Same units as ``"zscore"``
        (a Gaussian-noise pixel is ~N(0, 1)) but the median/MAD are not
        inflated by a few hot pixels or spikes, so genuine outliers stand out
        instead of inflating their own yardstick. Falls back to the std when
        the MAD is 0 (a map that is mostly identical values).
    ``"relative"``   : m / mean(m), i.e. fold-change vs. the map's own mean
        (1.0 = average pixel). Only meaningful when the mean is not ~0.
    """
    m = np.asarray(energy_map, dtype=float)
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        raise ValueError("energy map has no finite pixels to normalize")

    if mode in ("percentile", "minmax"):
        if mode == "percentile":
            p_lo, p_hi = float(percentiles[0]), float(percentiles[1])
            if not (0.0 <= p_lo < p_hi <= 100.0):
                raise ValueError(
                    f"percentiles must satisfy 0 <= lo < hi <= 100, got {percentiles!r}"
                )
            lo, hi = np.percentile(finite, [p_lo, p_hi])
        else:
            lo, hi = finite.min(), finite.max()
        if hi <= lo:  # flat map: no contrast to show
            return np.zeros_like(m)
        return np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    if mode == "zscore":
        std = finite.std()
        return (m - finite.mean()) / std if std > 0 else np.zeros_like(m)
    if mode == "robust_z":
        med = np.median(finite)
        scale = 1.4826 * np.median(np.abs(finite - med))
        if scale == 0:  # sparse map: >half the pixels identical -> use std instead
            scale = finite.std()
        return (m - med) / scale if scale > 0 else np.zeros_like(m)
    if mode == "relative":
        mean = finite.mean()
        if mean == 0 or not np.isfinite(mean):
            raise ValueError("map_normalization='relative' needs a non-zero map mean")
        return m / mean
    raise ValueError(
        f"map_normalization must be None or one of {_MAP_NORMALIZATIONS}, got {mode!r}"
    )


def show_energy_window_map(
    self,
    energy_window=None,
    roi=None,
    roi_cal=None,
    mask=None,
    cmap="viridis",
    show=True,
    display_energy_range=None,
    normalize=True,
    vmin=None,
    vmax=None,
    map_normalization="percentile",
    normalization_percentiles=(1.0, 99.0),
):
    """Show a spatial map integrated over a selected energy window.

    This is a complementary view to ``show_mean_spectrum``:
    - ``show_mean_spectrum`` answers *what energies are present*.
    - ``show_energy_window_map`` answers *where a chosen energy range is present*.

    Parameters
    ----------
    energy_window : list[float] | tuple[float, float] | None
        Energy interval [emin, emax] to integrate. If None, use the
        full calibrated energy range of the dataset.
    roi : list | tuple | None, optional
        ROI as ``[y, x]`` or ``[y, x, dy, dx]`` (with ``None`` defaults),
        used only for overlay rectangle.
    mask : array-like | None, optional
        Optional boolean mask over energy channels. If provided, it is
        combined with ``energy_window``.
    cmap : str, optional
        Matplotlib colormap for the map.
    show : bool, optional
        If True, call ``plt.show()``.
    display_energy_range : list[float] | tuple[float, float] | None, optional
        (lo, hi) energy range to zoom the spectrum subplot's x-axis into.
        Purely a display crop -- does not affect ``energy_window`` (what's
        integrated into the map) or the map itself. Defaults to the full
        (masked) energy axis, i.e. no zoom.
    normalize : bool, optional
        If True (default), divide the summed window intensity by the
        number of integrated channels, giving a mean intensity/channel
        instead of a raw sum. Without this, a wider ``energy_window``
        trivially sums more channels and produces larger values purely
        from its width, making maps from differently sized windows not
        comparable on the same color scale. Set False to get the raw
        per-pixel sum instead.
    vmin, vmax : float | None, optional
        Explicit color-scale limits for the map, forwarded to ``show_2d``.
        Pass the same ``vmin``/``vmax`` across multiple calls (e.g. one
        shared max computed up front) to put several energy-window maps
        on an identical color scale for direct visual comparison. If
        None (default), each call auto-scales to its own map's min/max.
        These apply to the map *after* ``map_normalization``, so with the
        default they are in normalized units (e.g. 0-1).
    map_normalization : {"percentile", "minmax", "zscore", "robust_z", "relative"} | None, optional
        Rescales each window's map on its own, so windows with very
        different intensity ranges (bright near the ZLP tail, dim at a
        weak feature) can be compared by their spatial contrast. Unlike
        ``normalize`` (which only divides out the window *width*), this
        removes the window's overall intensity *scale*:

        - ``"percentile"`` (default): clip to ``normalization_percentiles``
          then map to [0, 1]; robust to hot pixels and spikes.
        - ``"minmax"``: map the map's min..max to [0, 1].
        - ``"zscore"``: (map - mean) / std.
        - ``"robust_z"``: (map - median) / (1.4826 * MAD). Like ``"zscore"``
          but not inflated by hot pixels/spikes, so real outliers keep their
          full significance instead of inflating their own std. Best choice
          for sparse, spike-dominated maps.
        - ``"relative"``: map / mean, i.e. fold-change vs. the average pixel.
        - ``None``: keep the raw mean-intensity/channel (or summed) values.

        A shared ``vmin``/``vmax`` computed from several normalized maps
        (the pattern used in the notebooks) stays consistent, since every
        call normalizes the same way.
    normalization_percentiles : (float, float), optional
        Lower/upper percentile used by ``map_normalization="percentile"``.
        Default (1, 99).

    Returns
    -------
    tuple
        ``(fig, (ax_map, ax_spec), energy_map)`` where ``energy_map`` is the 2D
        array that is displayed, i.e. **after** ``map_normalization``. Pass
        ``map_normalization=None`` to get the raw integrated intensities.
    """
    y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)
    has_roi_overlay = any(val is not None for val in (roi, roi_cal))
    if map_normalization is False:
        map_normalization = None
    if map_normalization is not None and map_normalization not in _MAP_NORMALIZATIONS:
        raise ValueError(
            f"map_normalization must be None or one of {_MAP_NORMALIZATIONS}, "
            f"got {map_normalization!r}"
        )

    dE = float(self.sampling[2])
    E0 = float(self.origin[2]) if hasattr(self, "origin") else 0.0
    E = E0 + dE * np.arange(self.shape[2])

    if energy_window is None:
        emin = float(np.min(E))
        emax = float(np.max(E))
    else:
        if len(energy_window) != 2:
            raise ValueError("energy_window must be [min_energy, max_energy]")

        emin = float(energy_window[0])
        emax = float(energy_window[1])
        if not np.isfinite(emin) or not np.isfinite(emax) or emin >= emax:
            raise ValueError(
                "Invalid energy_window. Expected [min_energy, max_energy] with min < max"
            )

    window_mask = (E >= emin) & (E <= emax)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.shape[2],):
            raise ValueError(
                f"Mask shape {mask.shape} does not match energy axis shape ({self.shape[2]},)"
            )
        window_mask = window_mask & mask

    if not np.any(window_mask):
        raise ValueError("No energy channels selected. Adjust energy_window or mask")

    arr = np.asarray(self.array, dtype=float)
    energy_map = arr[:, :, window_mask].sum(axis=-1)
    if normalize:
        energy_map = energy_map / float(np.count_nonzero(window_mask))
    if map_normalization is not None:
        energy_map = _normalize_energy_map(
            energy_map, map_normalization, percentiles=normalization_percentiles
        )

    spec = self.calculate_mean_spectrum(
        roi=roi,
        roi_cal=roi_cal,
        mask=mask,
        attach_mean_spectrum=False,
    )
    if mask is not None:
        E_spec = E[mask]
    else:
        E_spec = E

    unit_label = "keV" if str(self.dataset_type).lower() == "xeds" else "eV"
    map_kind = "mean intensity/channel" if normalize else "summed intensity"
    if map_normalization is not None:
        p_lo, p_hi = normalization_percentiles
        map_kind = {
            "percentile": f"normalized 0-1, clipped {p_lo:g}-{p_hi:g} pct",
            "minmax": "normalized 0-1 (min-max)",
            "zscore": "z-score",
            "robust_z": "robust z-score (median/MAD)",
            "relative": "relative to map mean",
        }[map_normalization]
    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))
    show_2d_kwargs = {}
    if vmin is not None:
        show_2d_kwargs["vmin"] = vmin
    if vmax is not None:
        show_2d_kwargs["vmax"] = vmax
    show_2d(
        energy_map,
        figax=(fig, ax_map),
        title=f"Energy-Window Map [{emin:.3f}, {emax:.3f}] {unit_label} ({map_kind})",
        cmap=cmap,
        cbar=True,
        show_ticks=True,
        scalebar={
            "sampling": float(self.sampling[1]),
            "units": str(self.units[1]),
        },
        **show_2d_kwargs,
    )

    if has_roi_overlay:
        rect = Rectangle(
            (x - 0.5, y - 0.5),
            dx,
            dy,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            alpha=0.8,
        )
        ax_map.add_patch(rect)

    ax_spec.plot(E_spec, spec, linewidth=1.5, color="k")
    ax_spec.axvspan(emin, emax, color="orange", alpha=0.2, label="Selected window")
    ax_spec.set_xlabel(f"Energy ({unit_label})")
    ax_spec.set_ylabel("Intensity")
    ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
    ax_spec.grid(True, alpha=0.1)
    ax_spec.legend(loc="best")

    if display_energy_range is not None:
        d_lo, d_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(d_lo) and np.isfinite(d_hi) and d_lo < d_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
        ax_spec.set_xlim(d_lo, d_hi)
        # rescale y to what is visible: a steep edge of the (background-subtracted) spectrum just outside the
        # zoomed x-range would otherwise set the y-scale and flatten everything inside it
        visible = (E_spec >= d_lo) & (E_spec <= d_hi) & np.isfinite(spec)
        if np.any(visible):
            y_lo, y_hi = float(np.min(spec[visible])), float(np.max(spec[visible]))
            pad = 0.06 * (y_hi - y_lo) if y_hi > y_lo else max(abs(y_hi), 1.0) * 0.06
            ax_spec.set_ylim(y_lo - pad, y_hi + pad)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, (ax_map, ax_spec), energy_map


def _plot_background_subtraction(
    self,
    energy_axis,
    input_spectrum,
    background_spectrum,
    subtracted_spectrum,
    fit_mode,
    show_subtracted,
    display_energy_range=None,
    display_intensity_range=None,
):
    """
    Parameters
    ----------
    display_energy_range : (float, float), optional
        (lo, hi) to zoom the x-axis into. Display-only -- the full
        input/background/subtracted spectra are still plotted; this only
        changes what's visible. Defaults to the full energy range (no zoom).
    display_intensity_range : (float, float), optional
        (lo, hi) to zoom the y-axis into. Display-only, same caveat as
        ``display_energy_range``.
    """
    fig, (ax_specbacksub) = plt.subplots(1, 1, figsize=(12, 4))

    ax_specbacksub.plot(energy_axis, input_spectrum, linewidth=1.2, label="Input")
    ax_specbacksub.plot(energy_axis, background_spectrum, linewidth=1.2, label="Background")
    if show_subtracted:
        ax_specbacksub.plot(
            energy_axis,
            subtracted_spectrum,
            linewidth=1.5,
            label="Background-subtracted",
        )
    if self.dataset_type == "xeds":
        ax_specbacksub.set_xlabel("Energy (keV)")
    else:
        ax_specbacksub.set_xlabel("Energy (eV)")
    ax_specbacksub.set_ylabel("Intensity")
    ax_specbacksub.set_title(f"Background-subtracted spectrum from ROI ({fit_mode})")
    ax_specbacksub.grid(True, alpha=0.1)
    ax_specbacksub.legend()

    if display_energy_range is not None:
        e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
        ax_specbacksub.set_xlim(e_lo, e_hi)

    if display_intensity_range is not None:
        i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
        if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
            raise ValueError(
                f"display_intensity_range must be (lo, hi) with lo < hi, got {display_intensity_range!r}"
            )
        ax_specbacksub.set_ylim(i_lo, i_hi)

    fig.tight_layout()
    plt.show()


def show_spectrum_images(
    self, x_ray_lines=None, return_fig=False, return_maps=False, method="integration", **kwargs
):
    """Display cached spectrum images.

    Parameters
    ----------
    x_ray_lines : str | sequence[str] | None, optional
        Selectors to filter which images are shown.  If ``None``, one
        panel per element is displayed.
    return_fig : bool, optional
        If ``True``, return ``(fig, ax)``.
    method : {"integration", "fit"}, optional
        Which cache to read from: integration-based maps or PyTorch
        fit-based maps.
    **kwargs
        Forwarded to :func:`show_2d` (e.g. ``cmap``).

    Returns
    -------
    tuple[Figure, Axes] | None
        Only returned when *return_fig* is ``True``.

    Raises
    ------
    ValueError
        If no cached spectrum images exist for the chosen *method*.
    """
    spectrum_images = self._get_spectrum_images(method)
    if not spectrum_images:
        raise ValueError("No spectrum images found. Run generate_spectrum_images(...) first.")

    line_map = {str(k): np.asarray(getattr(v, "array", v)) for k, v in spectrum_images.items()}
    labels = list(line_map)
    labels_by_element = type(self)._group_labels_by_element(labels)

    def sum_maps(lbls):
        return np.sum([line_map[lbl] for lbl in lbls], axis=0)

    specs = type(self)._normalize_specs(x_ray_lines, param_name="x_ray_lines", allow_none=True)
    if not specs:
        titles = sorted(labels_by_element)
        images = [sum_maps(labels_by_element[t]) for t in titles]
    else:
        selected = [
            type(self)._select_labels(str(raw), labels=labels, labels_by_element=labels_by_element)
            for raw in specs
        ]
        if any(not s for s in selected):
            bad = next(raw for raw, s in zip(specs, selected) if not s)
            raise ValueError(f"No spectrum images matched selector '{bad}'")
        images = [line_map[s[0]] if len(s) == 1 else sum_maps(s) for s in selected]
        titles = [s[0] if len(s) == 1 else str(raw).strip() for raw, s in zip(specs, selected)]

    fig, ax = show_2d(
        images,
        title=titles,
        cmap=kwargs.pop("cmap", "magma"),
        scalebar={"sampling": self.sampling[1], "units": self.units[1]},
        returnfig=True,
        **kwargs,
    )

    if return_maps and hasattr(self, "_map_to_dataset2d"):
        images = [
            self._map_to_dataset2d(image, name=str(title)) for image, title in zip(images, titles)
        ]

    if return_fig and return_maps:
        return (fig, ax), (images, titles)
    elif return_fig:
        return fig, ax
    elif return_maps:
        return images, titles


def plot_absolute_zlp_shift(dataset, search_window=(-10, 10)):
    """
    Calculates the ZLP shift per pixel and plots the absolute deviation from 0.0 eV.
    """
    data = dataset.array

    # Generate energy axis
    energies = np.asarray(dataset.energy_axis, dtype=float)

    # Mask energy window for peak finding
    mask = (energies > search_window[0]) & (energies < search_window[1])
    search_energies = energies[mask]

    # Calculate peak map and absolute deviation
    peak_indices = np.argmax(data[:, :, mask], axis=2)
    zlp_map_ev = search_energies[peak_indices]
    absolute_shift = np.abs(zlp_map_ev)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(absolute_shift, cmap="magma", origin="lower")

    plt.colorbar(im, ax=ax, label="Absolute Shift (eV)")
    ax.set_title(f"Absolute ZLP Deviation: {dataset.name}")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()

    return absolute_shift


def visualize_thickness_windows(dataset, zlp_window=(-3.0, 3.0), total_window=(-3.0, 75.0)):
    """
    Visualizes integration windows for I0 (ZLP) and It (Total).
    Returns a configuration dictionary for the calculation step.
    """
    # 1. Extract Energy and Mean Spectrum
    data = dataset.array
    mean_spec = np.mean(data, axis=(0, 1))

    # Use built-in energy axis if available, else generate from metadata
    if hasattr(dataset, "energy_axis"):
        energy = np.asarray(dataset.energy_axis, dtype=float)
    else:
        energy = dataset.origin[2] + np.arange(dataset.shape[2]) * dataset.sampling[2]

    # 2. Find indices for the windows
    zlp_idx = (
        np.argmin(np.abs(energy - zlp_window[0])),
        np.argmin(np.abs(energy - zlp_window[1])),
    )
    tot_idx = (
        np.argmin(np.abs(energy - total_window[0])),
        np.argmin(np.abs(energy - total_window[1])),
    )

    # 3. Create the Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy, mean_spec, "k-", lw=1.5, label="Mean Spectrum", zorder=5)

    # Highlight Windows
    z_mask = (energy >= zlp_window[0]) & (energy <= zlp_window[1])
    t_mask = (energy >= total_window[0]) & (energy <= total_window[1])

    ax.fill_between(
        energy[z_mask], 0, mean_spec[z_mask], color="red", alpha=0.3, label="$I_0$ (ZLP)"
    )
    ax.fill_between(
        energy[t_mask], 0, mean_spec[t_mask], color="blue", alpha=0.1, label="$I_t$ (Total)"
    )

    ax.axvline(0, color="green", lw=1.5, ls=":", label="0 eV")
    ax.set_title(f"QuantEM: Integration Windows ({dataset.name})", fontweight="bold")
    ax.set_xlabel("Energy Loss (eV)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(energy[0], total_window[1] + 20)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "zlp_idx": zlp_idx,
        "total_idx": tot_idx,
        "zlp_val": zlp_window,
        "total_val": total_window,
    }


def interpret_thickness_quality(t_over_lambda, a=0.3, b=1, c=2, dataset=None):
    """
    Performs a scientific quality assessment on the calculated t/lambda map.

    The Physical Meaning of the ThresholdsThe t/lambda value represents the average number of inelastic scattering events
    an electron undergoes.
    Vacuum (< a):
        (default a = 0.3)
        In pure vacuum, t/lambda should be 0. In practice, values up to ~0.3 often indicate the presence of thin carbon support films,
        surface contamination, or detector noise. Measurements in this regime are highly sensitive to ZLP (Zero Loss Peak) estimation errors.

    Thin (a <t/lambda < b):
        (default b = 1)
        The "Sweet Spot" for EELS. At t/lambda ~1, the probability of a single inelastic scattering event is maximized.
        In this regime, core-loss edges are sharp and clearly visible without the immediate need for complex mathematical
        deconvolution (e.g., Fourier-Log) to remove multiple scattering effects.

    Medium (b < t/lambda < c):
        (default c = 2)
        Multiple scattering begins to dominate the spectrum. The plural scattering of plasmons creates "ghost" peaks
        that overlap with higher-energy chemical edges. While data is still usable, quantitative analysis typically
        requires plural scattering correction for high accuracy.

    Thick (t/lambda > c):
        The "Multiple Scattering Regime.
        " Most electrons have undergone three or more scattering events, resulting in a "spectral soup"
        where fine-structure details and high-resolution chemical information are significantly broadened or lost.
    """

    name = dataset.name if dataset else "Dataset"

    # Classification Masks
    vacuum = t_over_lambda < a
    thin = (t_over_lambda >= a) & (t_over_lambda < b)
    medium = (t_over_lambda >= b) & (t_over_lambda < c)
    thick = t_over_lambda >= c

    print(f"\n{'=' * 20} QUANTEM INTERPRETATION: {name} {'=' * 20}")
    for label, mask in [
        ("Vacuum (<0.3)", vacuum),
        ("Thin (0.3-1.0)", thin),
        ("Medium (1.0-2.0)", medium),
        ("Thick (>2.0)", thick),
    ]:
        pct = 100 * np.sum(mask) / t_over_lambda.size
        print(f"  {label:20}: {pct:5.1f}%")

    # Plotting Classification
    classified = np.zeros_like(t_over_lambda)
    classified[thin] = 1
    classified[medium] = 2
    classified[thick] = 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(classified, cmap="RdYlGn_r", origin="lower")
    ax1.set_title("Region Classification")
    cbar = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Vacuum", "Thin", "Medium", "Thick"])

    t_masked = np.copy(t_over_lambda)
    t_masked[vacuum] = np.nan
    im2 = ax2.imshow(t_masked, cmap="viridis", origin="lower")
    ax2.set_title("Sample-Only Thickness")
    plt.colorbar(im2, ax=ax2, label=r"$t/\lambda$")

    plt.tight_layout()
    plt.show()


def plot_absolute_thickness(t_lambda_map, mfp_nm, dataset=None):
    """
    Converts relative thickness to nanometers and visualizes the absolute map.
    """
    thickness_nm = t_lambda_map * mfp_nm
    name = dataset.name if dataset else "Sample"

    # Mask vacuum for better visualization contrast
    display_map = np.copy(thickness_nm)
    display_map[t_lambda_map < 0.1] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Physical Analysis: {name}", fontsize=14)

    im = ax1.imshow(display_map, cmap="magma", origin="lower")
    ax1.set_title("Absolute Thickness (nm)")
    plt.colorbar(im, ax=ax1, label="nm")

    valid_data = thickness_nm[t_lambda_map >= 0.1].flatten()
    ax2.hist(valid_data, bins=50, color="firebrick", alpha=0.7, ec="k")
    ax2.axvline(
        np.nanmean(display_map),
        color="blue",
        ls="--",
        label=f"Mean: {np.nanmean(display_map):.1f} nm",
    )
    ax2.set_title("Physical Distribution")
    ax2.set_xlabel("Thickness (nm)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(
        f"\nQuantEM Absolute Report:\n  Mean: {np.nanmean(display_map):.2f} nm\n  MFP:  {mfp_nm:.2f} nm"
    )
    return thickness_nm


def plot_dual_eels_picker(
    ll,
    hl,
    coords=None,
    title="QuantEM: Dual-EELS Analysis",
    display_energy_range=None,
    display_intensity_range=None,
):
    """
    Dual-EELS Picker with starting coordinates.

    coords, when provided, is interpreted as (scan_row, scan_col).

    Parameters
    ----------
    display_energy_range : (float, float), optional
        (lo, hi) to zoom both spectrum panels' x-axis into. Display-only --
        the full LL/HL spectra are still plotted; this only changes what's
        visible, and stays fixed as you click around the maps. Defaults to
        the full energy range (no zoom).
    display_intensity_range : (float, float), optional
        (lo, hi) to fix both spectrum panels' y-axis to. Display-only, same
        caveat as ``display_energy_range``. If not given, each panel
        auto-rescales its y-axis to the clicked spectrum's max, as before.
    """
    # 1. Setup Data
    if display_energy_range is not None:
        e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
    if display_intensity_range is not None:
        i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
        if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
            raise ValueError(
                f"display_intensity_range must be (lo, hi) with lo < hi, "
                f"got {display_intensity_range!r}"
            )

    sum_ll = np.sum(ll.array, axis=2)
    sum_hl = np.sum(hl.array, axis=2)
    energy_ll = np.asarray(ll.energy_axis, dtype=float)
    energy_hl = np.asarray(hl.energy_axis, dtype=float)

    # 2. Handle Initial Coordinates
    if coords is not None:
        i_row, i_col = coords
    else:
        i_row, i_col = ll.shape[0] // 2, ll.shape[1] // 2

    # 3. Create Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{title}\n(Click on maps to update spectra)", fontsize=16)
    ax_map_ll, ax_spec_ll = axes[0, 0], axes[0, 1]
    ax_map_hl, ax_spec_hl = axes[1, 0], axes[1, 1]

    # Plot Maps & Markers
    ax_map_ll.imshow(sum_ll, cmap="viridis", origin="lower")
    (marker_ll,) = ax_map_ll.plot(i_col, i_row, "r+", ms=15, mew=2)

    ax_map_hl.imshow(sum_hl, cmap="magma", origin="lower")
    (marker_hl,) = ax_map_hl.plot(i_col, i_row, "r+", ms=15, mew=2)

    # Plot Initial Spectra
    (line_ll,) = ax_spec_ll.plot(energy_ll, ll.array[i_row, i_col, :], color="tab:blue")
    (line_hl,) = ax_spec_hl.plot(energy_hl, hl.array[i_row, i_col, :], color="tab:red")

    if display_energy_range is not None:
        ax_spec_ll.set_xlim(e_lo, e_hi)
        ax_spec_hl.set_xlim(e_lo, e_hi)
    if display_intensity_range is not None:
        ax_spec_ll.set_ylim(i_lo, i_hi)
        ax_spec_hl.set_ylim(i_lo, i_hi)

    def update_plots(i_row, i_col):
        marker_ll.set_data([i_col], [i_row])
        marker_hl.set_data([i_col], [i_row])

        new_ll = ll.array[i_row, i_col, :]
        new_hl = hl.array[i_row, i_col, :]
        line_ll.set_ydata(new_ll)
        line_hl.set_ydata(new_hl)

        # Rescale (unless a fixed display_intensity_range was requested)
        if display_intensity_range is None:
            ax_spec_ll.set_ylim(0, np.max(new_ll) * 1.1)
            ax_spec_hl.set_ylim(0, np.max(new_hl) * 1.1)

        ax_spec_ll.set_title(f"LL Spectrum at ({i_row}, {i_col})")
        ax_spec_hl.set_title(f"HL Spectrum at ({i_row}, {i_col})")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes in [ax_map_ll, ax_map_hl]:
            i_col, i_row = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= i_row < ll.shape[0] and 0 <= i_col < ll.shape[1]:
                update_plots(i_row, i_col)

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_spec_ll.set_title(f"LL Spectrum at ({i_row}, {i_col})")
    ax_spec_hl.set_title(f"HL Spectrum at ({i_row}, {i_col})")

    plt.tight_layout()
    plt.close(fig)  # Prevents double-plotting in VS Code
    return fig


def plot_quantem_diagnostic(dataset, zlp_window=5.0, title_suffix=""):
    """
    QuantEM Diagnostic Dashboard: Visualizes mean spectra, spatial variation,
    and Zero Loss Peak (ZLP) centering accuracy.

    1. Global Average Spectrum (Top Left): Shows the mean intensity across the entire scan.
    It is used to check the signal-to-noise ratio and see if the Zero Loss Peak (ZLP) is roughly centered at 0 eV.
    2. Spatial Variation (Top Right): Plots spectra from a 5x5 grid of pixels across your sample.
    This helps you see if the energy shift or intensity changes drastically from one side of the scan to the other
    (e.g., due to sample thickness changes or beam drift).
    3. Integrated Intensity Map (Bottom Left): A spatial image of the total counts.
    This is your "search image" to help you correlate the spectral data with the physical structure of your sample.
    4. ZLP Alignment Detail (Bottom Right): A high-zoom view of the energy region around 0 eV of the Mean Spectrum.
    It includes a dashed green line at the "Target 0" to show exactly how much residual calibration error remains
    after your alignment.

    Parameters:
    -----------
    dataset : QuantEM Object
        The EELS dataset containing .array, .origin, and .sampling attributes.
    zlp_window : float, optional
        The energy range (± eV) to display in the ZLP zoom plot. Default is 5.0.
    title_suffix : str, optional
        Additional text to append to the figure title (e.g., "(RAW)" or "(Aligned)").

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for further manipulation or saving.
    """
    data = dataset.array
    energy = np.asarray(dataset.energy_axis, dtype=float)

    mean_spec = np.mean(data, axis=(0, 1))
    zlp_pos = energy[np.argmax(mean_spec)]
    sum_img = np.sum(data, axis=2)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    fig.suptitle(f"QuantEM Diagnostic: {dataset.name} {title_suffix}", fontsize=16)

    # 1. Mean Spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(energy, mean_spec, color="black", label="Mean")
    ax1.axvline(0, color="green", ls=":", label="Target")
    ax1.set_title("Global Average Spectrum")
    ax1.legend()

    # 2. Spatial Variability
    ax2 = fig.add_subplot(gs[0, 1])
    # Take a 5x5 grid for better representation than 3x3
    yy, xx = np.meshgrid(
        np.linspace(0, data.shape[0] - 1, 5, dtype=int),
        np.linspace(0, data.shape[1] - 1, 5, dtype=int),
    )
    for y, x in zip(yy.flatten(), xx.flatten()):
        ax2.plot(energy, data[y, x, :], alpha=0.3, lw=0.5)
    ax2.set_title("Spatial Variation (Grid Samples)")

    # 3. Map
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(sum_img, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax3)
    ax3.set_title("Integrated Intensity")

    # 4. ZLP Zoom
    ax4 = fig.add_subplot(gs[1, 1])
    mask = (energy > zlp_pos - zlp_window) & (energy < zlp_pos + zlp_window)
    ax4.plot(energy[mask], mean_spec[mask], lw=2)
    ax4.axvline(0, color="green", ls=":")
    ax4.set_title("ZLP Alignment Detail")
    plt.close(fig)

    return fig


def plot_zlp_drift_diagnostics(dataset, title="ZLP Drift Analysis"):
    """
    QuantEM Diagnostic: Maps the ZLP position and calculates the drift distribution.
    Uses scipy.stats for Gaussian fitting.
    """
    data = dataset.array
    energy = np.asarray(dataset.energy_axis, dtype=float)

    # 1. Mask and find peak per pixel
    search_mask = (energy > -2.0) & (energy < 2.0)
    search_energies = energy[search_mask]
    peak_indices = np.argmax(data[:, :, search_mask], axis=2)
    zlp_map = search_energies[peak_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"QuantEM: {dataset.name} - {title}", fontsize=16)

    # Plot A: Map
    im = ax1.imshow(zlp_map, cmap="RdYlBu_r", origin="lower")
    plt.colorbar(im, ax=ax1, label="Energy Shift (eV)")

    # Plot B: Histogram + Scipy Fit
    flat_pos = zlp_map.flatten()
    mu, std = norm.fit(flat_pos)  # Professional scipy fitting

    ax2.hist(flat_pos, bins=30, density=True, alpha=0.6, color="skyblue")
    x_range = np.linspace(np.min(flat_pos), np.max(flat_pos), 100)
    ax2.plot(
        x_range,
        norm.pdf(x_range, mu, std),
        color="darkred",
        lw=2,
        label=f"Fit: μ={mu:.3f} eV, σ={std:.3f} eV",
    )
    ax2.legend()

    plt.tight_layout()

    plt.close(fig)

    return fig


def plot_near_zlp_transitions_fit(
    fit_result,
    dataset=None,
    title=None,
    display_energy_range=None,
    display_intensity_range=None,
    display_residual_range=None,
):
    """
    Plot diagnostics for ``Dataset3deels.fit_near_zlp_transitions()``.

    Call directly (not as a bound dataset method -- ``fit_result``'s
    position as the first argument, not ``dataset``, would conflict with
    the module's usual self-binding convention)::

        fit_result = eels_hl_despiked.fit_near_zlp_transitions(peaks, ...)
        fig = plot_near_zlp_transitions_fit(fit_result, dataset=eels_hl_despiked)

    Top panel overlays the raw spectrum, the total fit, the ZLP-tail
    component alone, and each peak component alone. Bottom panel is the
    residual (raw - total fit) -- the important diagnostic: if a region is
    genuinely explained by the ZLP tail plus the fitted peaks, the residual
    there should be small and unstructured; a real unexplained feature will
    show up there instead of being silently absorbed into a peak.

    Parameters
    ----------
    fit_result : dict
        The dict returned by ``fit_near_zlp_transitions()``.
    dataset : Dataset3deels, optional
        Only used for the title (``dataset.name``) if ``title`` isn't given.
    title : str, optional
        Plot title. Defaults to ``f"Near-ZLP transition fit: {dataset.name}"``
        if ``dataset`` is given, else a generic title.
    display_energy_range : (float, float), optional
        (lo, hi) eV to zoom the top panel's x-axis into (the residual panel
        shares this x-axis, so it zooms too). Purely a display crop -- the
        full spectrum/fit/residual are still plotted and returned; this
        only changes what's visible. Useful for peeking at the low-amplitude
        peak components against the ZLP tail's much larger dynamic range.
        Defaults to the full fitted energy range (no zoom).
    display_intensity_range : (float, float), optional
        (lo, hi) to zoom the top panel's y-axis into (does not affect the
        residual panel, which has its own, unrelated scale). Same
        display-only caveat as ``display_energy_range``.
    display_residual_range : (float, float), optional
        (lo, hi) to zoom the bottom (residual) panel's y-axis into. Separate
        from ``display_intensity_range`` since the residual is typically a
        much smaller scale than the raw intensity. Same display-only
        caveat as ``display_energy_range``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    energy = fit_result["energy"]
    spectrum = fit_result["spectrum"]
    total_fit = fit_result["total_fit"]
    residual = fit_result["residual"]
    components = fit_result["components"]

    if title is None:
        title = (
            f"Near-ZLP transition fit: {dataset.name}"
            if dataset is not None
            else "Near-ZLP transition fit"
        )

    fig, (ax_fit, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_fit.plot(energy, spectrum, color="black", lw=1.3, label="Raw spectrum")
    ax_fit.plot(energy, total_fit, color="crimson", lw=1.8, label="Total fit")
    for name, comp in components.items():
        style = "--" if name == "zlp_tail" else "-."
        ax_fit.plot(
            energy,
            comp["curve"],
            style,
            lw=1.2,
            label=f"{name} (c={comp['center']:.3f} eV, fwhm={comp['fwhm']:.3f} eV)",
        )
    ax_fit.set_ylabel("Intensity")
    ax_fit.set_title(title)
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True, alpha=0.15)

    ax_resid.axhline(0, color="gray", lw=0.8)
    ax_resid.plot(energy, residual, color="steelblue", lw=1.0)
    ax_resid.set_xlabel("Energy (eV)")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(True, alpha=0.15)

    if display_energy_range is not None:
        e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
        ax_fit.set_xlim(e_lo, e_hi)  # ax_resid shares this x-axis (sharex=True)

    if display_intensity_range is not None:
        i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
        if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
            raise ValueError(
                f"display_intensity_range must be (lo, hi) with lo < hi, got {display_intensity_range!r}"
            )
        ax_fit.set_ylim(i_lo, i_hi)

    if display_residual_range is not None:
        r_lo, r_hi = float(display_residual_range[0]), float(display_residual_range[1])
        if not (np.isfinite(r_lo) and np.isfinite(r_hi) and r_lo < r_hi):
            raise ValueError(
                f"display_residual_range must be (lo, hi) with lo < hi, got {display_residual_range!r}"
            )
        ax_resid.set_ylim(r_lo, r_hi)

    fig.tight_layout()
    plt.close(fig)

    return fig


def _odd_clipped(value, lo, hi):
    """Round to the nearest odd int, then clip into [lo, hi] (lo/hi assumed odd)."""
    v = int(round(value))
    if v % 2 == 0:
        v += 1
    return max(lo, min(hi, v))


def find_maximum_and_shoulder(
    x,
    y,
    x_min: float,
    x_max: float,
    min_x_separation: float,
    *,
    prominence_fraction: float = 0.075,
    edge_margin_fraction: float = 0.05,
    stability_fraction: float = 0.05,
    smoothing_change_fraction: float = 0.20,
    polyorder: int = 3,
    show: bool = True,
    hover_annotations: bool = True,
) -> Dict[str, Any]:
    """
    Identify the main maximum and (optionally) one shoulder in noisy 1D data
    (x, y), restricted to [x_min, x_max]. General-purpose -- x/y need not be
    an EELS spectrum; call directly, it doesn't take a dataset.

    Procedure:
    1. Restrict to [x_min, x_max].
    2. Smooth with a Savitzky-Golay filter (preserves broad peak shapes
       better than a Gaussian). If x is unevenly spaced, interpolate onto a
       uniform grid first.
    3. Search increasing SavGol window lengths (from a small fraction of
       the point count up to ~75%) and pick the *smallest* one whose
       smoothed curve has exactly one locally-maximal peak with prominence
       >= `prominence_fraction` (default 7.5%, i.e. within the requested
       5-10%) of the smoothed curve's total range in [x_min, x_max] --
       "small noise-induced local maxima" get smoothed away before that
       point; larger windows aren't used once one dominant peak remains.
    4. Re-analyze at window lengths +/- `smoothing_change_fraction`
       (default 20%) around the selected one. A feature (main maximum or
       shoulder) only counts as valid if its x-position is stable (within
       `stability_fraction` of the [x_min, x_max] span) across all three.
    5. Main maximum = the highest accepted local maximum of the selected
       smoothed curve.
    6-8. Shoulder = a 2nd-derivative sign change that is >= `min_x_separation`
       from the main maximum, outside `edge_margin_fraction` of either
       boundary, sits on a real (non-negligible) slope (not a flat region),
       and is position-stable under the +/-20% smoothing check. If nothing
       satisfies all of that, no shoulder is reported (never forced).

    Parameters
    ----------
    x, y : ndarray
        1D data (need not be sorted or evenly spaced -- both are handled).
    x_min, x_max : float
        Restrict analysis to this range of x.
    min_x_separation : float
        Minimum x-distance a shoulder candidate must keep from the main
        maximum to be considered a distinct feature.
    prominence_fraction : float, optional
        Minimum peak prominence, as a fraction of the smoothed curve's own
        range in [x_min, x_max], for a local maximum to be accepted.
        Default 0.075.
    edge_margin_fraction : float, optional
        Reject any maximum/shoulder within this fraction of the
        [x_min, x_max] span from either boundary. Default 0.05.
    stability_fraction : float, optional
        A feature's x-position must stay within this fraction of the
        [x_min, x_max] span across the +/-`smoothing_change_fraction`
        smoothing check to count as stable. Default 0.05.
    smoothing_change_fraction : float, optional
        Relative change in SavGol window length used for the stability
        check. Default 0.20 (+/-20%).
    polyorder : int, optional
        Savitzky-Golay polynomial order. Default 3.
    show : bool, optional
        Display the 5-panel diagnostic figure. Default True.
    hover_annotations : bool, optional
        Adds a mouse-hover tooltip to plot 2 (the main max / shoulder
        markers) showing that point's (x, y) as you move the cursor near
        it, instead of only reading it off the legend text. This needs an
        interactive matplotlib backend to actually fire mouse-move events
        -- with the default static/Agg backend the plot renders exactly as
        before and the tooltip just never appears. In a notebook, run
        `%matplotlib widget` (ipympl) once before this cell to enable it.
        Pass False to skip this and always get the plain static plot.
        Default True.

    Returns
    -------
    dict with keys:
        smoothing_method, smoothing_params,
        main_maximum ({"x", "y"} or None),
        main_maximum_prominence,
        shoulder ({"x", "y"} or None),
        shoulder_stability_statement,
        main_maximum_stable (bool),
        notes (list of str -- caveats, e.g. if the ideal single-peak
        smoothing level was never reached).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} and {y.shape}")
    if not (x_min < x_max):
        raise ValueError(f"x_min must be < x_max; got x_min={x_min}, x_max={x_max}")

    order = np.argsort(x)
    x, y = x[order], y[order]
    mask = (x >= x_min) & (x <= x_max) & ~np.isnan(y)
    x_r, y_r = x[mask], y[mask]

    notes: List[str] = []

    if len(x_r) < polyorder + 3:
        print(
            f"  -> not enough data points in [{x_min}, {x_max}] "
            f"({len(x_r)} found) to smooth/analyze."
        )
        return {
            "smoothing_method": "savgol",
            "smoothing_params": None,
            "main_maximum": None,
            "main_maximum_prominence": None,
            "shoulder": None,
            "shoulder_stability_statement": "No resolvable shoulder (insufficient data).",
            "main_maximum_stable": False,
            "notes": ["insufficient data points in range"],
        }

    # ---- 1/2. restrict + interpolate onto a uniform grid if needed ----
    diffs = np.diff(x_r)
    uniform = bool(np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-12)) if len(diffs) else True
    if not uniform:
        x_grid = np.linspace(x_r[0], x_r[-1], len(x_r))
        y_grid = np.interp(x_grid, x_r, y_r)
        notes.append("x was unevenly spaced -- interpolated onto a uniform grid before smoothing")
    else:
        x_grid, y_grid = x_r, y_r

    n = len(x_grid)
    span = x_max - x_min
    edge_margin = edge_margin_fraction * span
    stability_tol = stability_fraction * span

    min_window = _odd_clipped(
        max(polyorder + 2, 0.01 * n),
        polyorder + 2 if (polyorder + 2) % 2 else polyorder + 3,
        n if n % 2 else n - 1,
    )
    max_window = _odd_clipped(0.75 * n, min_window, n if n % 2 else n - 1)
    if max_window < min_window:
        max_window = min_window

    def _prominent_maxima(smoothed):
        y_range = float(smoothed.max() - smoothed.min())
        min_prom = prominence_fraction * y_range if y_range > 0 else 0.0
        idx, props = find_peaks(smoothed, prominence=min_prom)
        return idx, props.get("prominences", np.array([]))

    # ---- 3. search increasing window lengths for the lowest one giving
    # exactly one dominant, prominent local maximum ----
    fractions = [0.01, 0.015, 0.02, 0.03, 0.045, 0.07, 0.1, 0.15, 0.22, 0.33, 0.5, 0.75]
    candidate_windows = sorted(
        {min_window, max_window} | {_odd_clipped(f * n, min_window, max_window) for f in fractions}
    )

    results = []  # (window_length, smoothed, idx, prominences)
    for w in candidate_windows:
        smoothed_w = savgol_filter(y_grid, window_length=w, polyorder=polyorder)
        idx_w, prom_w = _prominent_maxima(smoothed_w)
        results.append((w, smoothed_w, idx_w, prom_w))

    selected = next((r for r in results if len(r[2]) == 1), None)
    if selected is None:
        multi = [r for r in results if len(r[2]) >= 1]
        if multi:
            selected = multi[0]
            notes.append(
                f"never reached exactly one dominant maximum (best: {len(selected[2])} at "
                f"window_length={selected[0]}); using its highest peak as the main maximum"
            )
        else:
            selected = results[-1]
            notes.append(
                "no local maximum cleared the prominence threshold at any smoothing level; "
                "falling back to the global maximum of the most-smoothed curve"
            )

    w_sel, smoothed_sel, idx_sel, prom_sel = selected
    if len(idx_sel) == 0:
        main_idx = int(np.argmax(smoothed_sel))
        main_prom = None
    else:
        best = int(np.argmax(smoothed_sel[idx_sel]))
        main_idx = int(idx_sel[best])
        main_prom = float(prom_sel[best])

    main_max_x = float(x_grid[main_idx])
    main_max_y = float(smoothed_sel[main_idx])

    # ---- 4. +/-20% smoothing-strength neighbors, for stability checks and
    # for the "nearby smoothing strengths" plot ----
    w_minus = _odd_clipped(w_sel * (1 - smoothing_change_fraction), min_window, max_window)
    w_plus = _odd_clipped(w_sel * (1 + smoothing_change_fraction), min_window, max_window)
    smoothed_minus = savgol_filter(y_grid, window_length=w_minus, polyorder=polyorder)
    smoothed_plus = savgol_filter(y_grid, window_length=w_plus, polyorder=polyorder)

    def _main_max_x(smoothed):
        idx, _ = _prominent_maxima(smoothed)
        if len(idx) == 0:
            return float(x_grid[int(np.argmax(smoothed))])
        best = idx[int(np.argmax(smoothed[idx]))]
        return float(x_grid[best])

    main_x_minus = _main_max_x(smoothed_minus)
    main_x_plus = _main_max_x(smoothed_plus)
    main_maximum_stable = (
        abs(main_x_minus - main_max_x) <= stability_tol
        and abs(main_x_plus - main_max_x) <= stability_tol
    )
    if not main_maximum_stable:
        notes.append(
            f"main maximum position shifts beyond {stability_tol:.3g} "
            f"({stability_fraction:.0%} of range) under +/-{smoothing_change_fraction:.0%} "
            f"smoothing changes -- treat its exact position with caution"
        )

    # ---- 6/7. shoulder candidates ----
    #
    # A 2nd derivative amplifies noise far more than the signal itself, so
    # the window_length that already gives "one dominant maximum" in Y
    # (a low-frequency criterion) is typically nowhere near clean enough
    # for derivative work -- so the shoulder search re-runs its own
    # increasing-smoothing search starting from w_sel, independent of it.
    #
    # SavGol's own polynomial-fit derivatives (deriv=1/2) are used instead
    # of np.gradient on the smoothed curve -- a finite difference of an
    # already-smoothed curve still amplifies noise far more than SavGol's
    # own local least-squares derivative at the same window length.
    #
    # Candidates are gated and ranked by absolute SLOPE-DROP magnitude
    # (|dy/dx|'s local maximum minus its value at the candidate), not by
    # the raw Y change nearby -- that alternative is dominated by however
    # large the *main* peak happens to be anywhere near its own flank,
    # which buries a real but weaker, more separated shoulder every time.
    # The noise floor for that comparison is estimated by running the
    # exact same SavGol derivative filter over the (raw - smoothed)
    # residuals -- linearity means that shows directly how much spurious
    # slope pure noise contributes at this window length.
    dx = float(np.median(np.diff(x_grid))) if n > 1 else 1.0
    n_local = max(4, int(round(0.5 * min_x_separation / dx))) if dx > 0 else 4
    min_shoulder_snr = 4.0

    def _shoulder_derivs(w):
        smoothed_w = savgol_filter(y_grid, window_length=w, polyorder=polyorder)
        dy_w = savgol_filter(y_grid, window_length=w, polyorder=polyorder, deriv=1, delta=dx)
        d2y_w = savgol_filter(y_grid, window_length=w, polyorder=polyorder, deriv=2, delta=dx)
        return smoothed_w, dy_w, d2y_w

    # Every 2nd-derivative zero crossing is a stationary point of the 1st
    # derivative, and there are two kinds: a LOCAL MAXIMUM of |dy/dx| (the
    # steepest point of an otherwise plain monotonic flank -- e.g. every
    # single Gaussian has exactly two of these, at mu +/- sigma, with no
    # second feature involved at all), or a LOCAL MINIMUM of |dy/dx| (the
    # slope genuinely slows down / flattens there before continuing --
    # exactly the "flattening, slope change, or weak secondary feature"
    # signature the spec asks for). Only the second kind is a real shoulder.
    def _shoulder_candidates(w, smoothed_w, dy_w, d2y_w):
        residuals_w = y_grid - smoothed_w
        dy_noise = savgol_filter(
            residuals_w, window_length=w, polyorder=polyorder, deriv=1, delta=dx
        )
        dy_noise_sigma = 1.4826 * float(np.median(np.abs(dy_noise - np.median(dy_noise))))
        slope_drop_threshold = min_shoulder_snr * dy_noise_sigma if dy_noise_sigma > 0 else 0.0

        raw = []
        for i in range(1, len(d2y_w)):
            s0, s1 = d2y_w[i - 1], d2y_w[i]
            if s0 * s1 >= 0:
                continue
            frac = abs(s0) / (abs(s0) + abs(s1)) if (abs(s0) + abs(s1)) > 0 else 0.5
            xc = float(x_grid[i - 1] + frac * (x_grid[i] - x_grid[i - 1]))
            if abs(xc - main_max_x) < min_x_separation:
                continue
            if (xc - x_min) < edge_margin or (x_max - xc) < edge_margin:
                continue

            lo_i, hi_i = max(0, i - n_local), min(len(dy_w) - 1, i + n_local)
            local_abs_dy = np.abs(dy_w[lo_i : hi_i + 1])
            slope_here = abs(dy_w[i])
            if local_abs_dy.max() <= 0 or slope_here > 0.85 * local_abs_dy.max():
                continue  # a slope-magnitude LOCAL MAX -> the peak's own flank inflection

            slope_drop = local_abs_dy.max() - slope_here
            if slope_drop < slope_drop_threshold:
                continue  # doesn't clear the noise floor for this window length
            raw.append((xc, slope_drop))

        # merge crossings within min_x_separation of each other (one real
        # shoulder can produce more than one raw crossing), keeping the
        # strongest of each cluster
        raw.sort(key=lambda c: c[0])
        merged: List[Tuple[float, float]] = []
        for xc, drop in raw:
            if merged and (xc - merged[-1][0]) < min_x_separation:
                if drop > merged[-1][1]:
                    merged[-1] = (xc, drop)
            else:
                merged.append((xc, drop))
        merged.sort(key=lambda c: -c[1])
        return merged

    # Search increasing window lengths, starting at w_sel, for one where the
    # (merged) shoulder-candidate list has settled to exactly 1 or 2. This
    # needs a much finer step than the Y-level search above -- the window
    # range where a real shoulder is cleanly resolved (neither buried in
    # derivative noise nor smoothed away entirely) can be narrow, and the
    # coarse geometric `candidate_windows` list can jump straight over it
    # (e.g. straight from "several candidates" to "zero candidates").  So
    # this steps by ~8% at a time, and if it *does* jump from >2 straight to
    # 0, it bisects between those two window lengths to land inside the gap.
    def _try(w):
        smoothed_w, dy_w, d2y_w = _shoulder_derivs(w)
        return w, smoothed_w, dy_w, d2y_w, _shoulder_candidates(w, smoothed_w, dy_w, d2y_w)

    w_shoulder, smoothed_w, dy_w, d2y_w, shoulder_merged = _try(w_sel)
    prev = None
    if not (1 <= len(shoulder_merged) <= 2):
        w = w_sel
        found = False
        for _ in range(40):
            prev = (w, smoothed_w, dy_w, d2y_w, shoulder_merged)
            w_next = _odd_clipped(w * 1.08, min_window, max_window)
            if w_next <= w:
                break
            w, smoothed_w, dy_w, d2y_w, shoulder_merged = _try(w_next)
            if 1 <= len(shoulder_merged) <= 2:
                w_shoulder = w
                found = True
                break
            if len(shoulder_merged) == 0 and len(prev[4]) > 2:
                # jumped from several candidates straight to none -- bisect
                lo_w = prev[0]
                hi_w = w
                for _ in range(12):
                    mid = _odd_clipped((lo_w + hi_w) / 2, min_window, max_window)
                    if mid <= lo_w or mid >= hi_w:
                        break
                    mid_state = _try(mid)
                    if 1 <= len(mid_state[4]) <= 2:
                        w_shoulder, smoothed_w, dy_w, d2y_w, shoulder_merged = mid_state
                        found = True
                        break
                    if len(mid_state[4]) > 2:
                        lo_w = mid
                    else:
                        hi_w = mid
                break
            if w >= max_window:
                break
        if not found:
            w_shoulder = w
            notes.append(
                f"shoulder candidates never settled to 1-2 at any smoothing level tried "
                f"(ended at window_length={w_shoulder} with {len(shoulder_merged)} candidates); "
                f"reporting its best candidate only"
            )

    dy_sel, d2y_sel = dy_w, d2y_w  # for the derivative plots below

    shoulder = None
    shoulder_stability_statement = "No resolvable shoulder."
    if shoulder_merged:
        shoulder_x_candidate = shoulder_merged[0][0]

        w_shoulder_minus = _odd_clipped(
            w_shoulder * (1 - smoothing_change_fraction), min_window, max_window
        )
        w_shoulder_plus = _odd_clipped(
            w_shoulder * (1 + smoothing_change_fraction), min_window, max_window
        )

        def _best_shoulder_x(w):
            sm, dyw, d2yw = _shoulder_derivs(w)
            merged = _shoulder_candidates(w, sm, dyw, d2yw)
            return merged[0][0] if merged else None

        def _best_shoulder_x_near(target_w):
            # The window range where a given shoulder resolves cleanly can
            # be narrow; requiring a hit at exactly the +/-20% window can
            # miss it by a handful of points even though the feature is
            # genuinely present just next door. Try the target first, then
            # a small neighborhood around it, before giving up.
            x = _best_shoulder_x(target_w)
            if x is not None:
                return x, target_w
            for frac in (0.95, 1.05, 0.90, 1.10, 0.85, 1.15):
                w_try = _odd_clipped(target_w * frac, min_window, max_window)
                if w_try == target_w:
                    continue
                x = _best_shoulder_x(w_try)
                if x is not None:
                    return x, w_try
            return None, target_w

        sx_minus, w_minus_used = _best_shoulder_x_near(w_shoulder_minus)
        sx_plus, w_plus_used = _best_shoulder_x_near(w_shoulder_plus)
        stable = (
            sx_minus is not None
            and sx_plus is not None
            and abs(sx_minus - shoulder_x_candidate) <= stability_tol
            and abs(sx_plus - shoulder_x_candidate) <= stability_tol
        )
        if stable:
            shoulder_y = float(np.interp(shoulder_x_candidate, x_grid, smoothed_sel))
            shoulder = {"x": shoulder_x_candidate, "y": shoulder_y}
            shoulder_stability_statement = (
                f"Shoulder at x={shoulder_x_candidate:.4g} is stable: present within "
                f"{stability_tol:.3g} of this position at both window_length={w_minus_used} "
                f"({sx_minus:.4g}) and window_length={w_plus_used} ({sx_plus:.4g})."
            )
        else:
            shoulder_stability_statement = (
                "A candidate shoulder was found at the selected smoothing level but its "
                "position was not stable within +/-20% smoothing changes, so it is not "
                "reported. No resolvable shoulder."
            )

    print(f"Smoothing: Savitzky-Golay, window_length={w_sel}, polyorder={polyorder}")
    print(
        f"Main maximum: x={main_max_x:.4g}, y={main_max_y:.4g}"
        + (
            f", prominence={main_prom:.4g}"
            if main_prom is not None
            else " (prominence threshold not met)"
        )
    )
    print(
        f"Shoulder: x={shoulder['x']:.4g}, y={shoulder['y']:.4g}" if shoulder else "Shoulder: none"
    )
    print(shoulder_stability_statement)
    for note in notes:
        print(f"  note: {note}")

    if show:
        fig, axes = plt.subplots(5, 1, figsize=(9, 22), sharex=True)

        # 1. raw data + smoothed curve
        axes[0].plot(x_r, y_r, "k.", ms=3, alpha=0.4, label="raw data")
        axes[0].plot(x_grid, smoothed_sel, "b-", lw=1.8, label=f"smoothed (window_length={w_sel})")
        axes[0].set_ylabel("y")
        axes[0].set_title("1. Raw data and smoothed curve")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 2. detected main maximum + shoulder
        axes[1].plot(x_grid, smoothed_sel, "b-", lw=1.5)
        axes[1].plot(
            [main_max_x],
            [main_max_y],
            "r^",
            ms=10,
            label=f"main max ({main_max_x:.3g}, {main_max_y:.3g})",
        )
        if shoulder:
            axes[1].plot(
                [shoulder["x"]],
                [shoulder["y"]],
                "gD",
                ms=9,
                label=f"shoulder ({shoulder['x']:.3g}, {shoulder['y']:.3g})",
            )

        if hover_annotations:
            # Plain matplotlib event handling -- no extra dependency (e.g.
            # mplcursors isn't installed here). Only fires with an
            # interactive backend (e.g. `%matplotlib widget` via ipympl);
            # with the static/Agg backend this sets up harmlessly and the
            # tooltip just never appears since no mouse-move events occur.
            hover_points = [
                (main_max_x, main_max_y, f"main max\n({main_max_x:.4g}, {main_max_y:.4g})")
            ]
            if shoulder:
                hover_points.append(
                    (
                        shoulder["x"],
                        shoulder["y"],
                        f"shoulder\n({shoulder['x']:.4g}, {shoulder['y']:.4g})",
                    )
                )
            tooltip = axes[1].annotate(
                "",
                xy=(0, 0),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.95),
                arrowprops=dict(arrowstyle="->"),
                fontsize=9,
                visible=False,
                zorder=10,
            )
            hover_pixel_radius = 20

            def _on_hover(event, _ax=axes[1], _pts=hover_points, _tip=tooltip):
                if event.inaxes is not _ax or event.x is None:
                    if _tip.get_visible():
                        _tip.set_visible(False)
                        fig.canvas.draw_idle()
                    return
                best, best_dist = None, None
                for px, py, label in _pts:
                    sx, sy = _ax.transData.transform((px, py))
                    d = ((sx - event.x) ** 2 + (sy - event.y) ** 2) ** 0.5
                    if best_dist is None or d < best_dist:
                        best_dist, best = d, (px, py, label)
                if best is not None and best_dist <= hover_pixel_radius:
                    px, py, label = best
                    _tip.xy = (px, py)
                    _tip.set_text(label)
                    if not _tip.get_visible():
                        _tip.set_visible(True)
                    fig.canvas.draw_idle()
                elif _tip.get_visible():
                    _tip.set_visible(False)
                    fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", _on_hover)

        axes[1].set_ylabel("y")
        axes[1].set_title("2. Detected main maximum and shoulder")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # 3. first derivative (from the shoulder-analysis smoothing level,
        # which is typically higher than w_sel -- see note above)
        axes[2].plot(x_grid, dy_sel, "m-", lw=1.2)
        axes[2].axhline(0, color="gray", ls="--", lw=1)
        axes[2].set_ylabel("dy/dx")
        axes[2].set_title(f"3. First derivative (window_length={w_shoulder})")
        axes[2].grid(True, alpha=0.3)

        # 4. second derivative
        axes[3].plot(x_grid, d2y_sel, "c-", lw=1.2)
        axes[3].axhline(0, color="gray", ls="--", lw=1)
        if shoulder:
            axes[3].axvline(shoulder["x"], color="g", ls=":", lw=1.5)
        axes[3].set_ylabel("d2y/dx2")
        axes[3].set_title(f"4. Second derivative (window_length={w_shoulder})")
        axes[3].grid(True, alpha=0.3)

        # 5. nearby smoothing strengths
        axes[4].plot(
            x_grid,
            smoothed_minus,
            "--",
            color="tab:orange",
            lw=1.2,
            label=f"window_length={w_minus} (-{smoothing_change_fraction:.0%})",
        )
        axes[4].plot(
            x_grid,
            smoothed_sel,
            "-",
            color="tab:blue",
            lw=1.8,
            label=f"window_length={w_sel} (selected)",
        )
        axes[4].plot(
            x_grid,
            smoothed_plus,
            "--",
            color="tab:green",
            lw=1.2,
            label=f"window_length={w_plus} (+{smoothing_change_fraction:.0%})",
        )
        axes[4].axvline(main_x_minus, color="tab:orange", ls=":", lw=1, alpha=0.7)
        axes[4].axvline(main_max_x, color="tab:blue", ls=":", lw=1, alpha=0.7)
        axes[4].axvline(main_x_plus, color="tab:green", ls=":", lw=1, alpha=0.7)
        axes[4].set_xlabel("x")
        axes[4].set_ylabel("y")
        axes[4].set_title("5. Results at nearby smoothing strengths")
        axes[4].legend(loc="best", fontsize=8)
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "smoothing_method": "savgol",
        "smoothing_params": {
            "window_length": w_sel,
            "polyorder": polyorder,
            "window_length_minus": w_minus,
            "window_length_plus": w_plus,
            "shoulder_window_length": w_shoulder,
        },
        "main_maximum": {"x": main_max_x, "y": main_max_y},
        "main_maximum_prominence": main_prom,
        "shoulder": shoulder,
        "shoulder_stability_statement": shoulder_stability_statement,
        "notes": notes,
    }


def _wrap_title(text, width=80):
    """Wrap a long plot-title line at `width` characters."""
    return textwrap.fill(str(text), width=width)


def detect_broad_bump_via_slope_change(
    dataset,
    energy_range: Tuple[float, float],
    smoothing_window_eV: float = 0.4,
    polyorder: int = 3,
    min_prominence: Optional[float] = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Find a very broad, low-amplitude bump riding on a monotonically
    decaying spectrum (e.g. sitting on the ZLP tail's own decay) that never
    produces an actual local maximum in the raw data -- so it's invisible
    to `detect_peaks_in_range()`, including its "shoulder" fallback, which
    both ultimately require a real extremum of intensity or of the 1st
    derivative. This instead looks at where the *2nd derivative* (the
    curvature) itself dips: a genuine bend where the decay's slope stops
    steepening and briefly flattens/reverses because of the underlying
    bump, before resuming its plain decay. A plain monotonic decay with no
    hidden bump has a 2nd derivative with no such dip.

    This is a qualitative, not quantitative, bump *locator* -- use its
    `candidates[i]["center_eV"]` to seed where to center a real physical
    model (e.g. a Gaussian on top of a fitted background), not as the
    bump's actual amplitude or width.

    Never modifies `dataset`.

    Parameters
    ----------
    dataset : Dataset3deels-like
        Any dataset with `.energy_axis`, `.calculate_mean_spectrum()`, and
        `.sampling` (its eV/channel spacing is `sampling[2]`).
    energy_range : (float, float)
        (lo, hi) eV window to restrict analysis to.
    smoothing_window_eV : float, default 0.4
        Savitzky-Golay window width in eV, converted to an odd number of
        samples using this dataset's own eV/channel spacing
        (`dataset.sampling[2]`).
    polyorder : int, default 3
        Savitzky-Golay polynomial order, shared by all three passes
        (spectrum, 1st derivative, 2nd derivative) so they stay directly
        comparable.
    min_prominence : float, optional
        Minimum prominence (in the smoothed 2nd derivative's own units) a
        curvature dip must clear to be reported. If None (default), every
        local minimum of the 2nd derivative in the (edge-trimmed) range is
        returned unfiltered (scipy.signal.find_peaks's own default
        handling) -- pass an explicit threshold to keep only dips whose
        prominence clears it.
    plot : bool, default True
        Show the 3-panel diagnostic figure (raw+smoothed spectrum; slope;
        curvature with candidate dips marked).

    Returns
    -------
    dict with keys:
        "energy", "spectrum_smoothed", "first_derivative",
        "second_derivative" : the windowed (not edge-trimmed) arrays.
        "candidates" : list of
            {"center_eV", "dip_prominence", "slope_change_amplitude"},
            sorted by center_eV ascending (energy position, not
            prominence) -- every candidate found is a real local minimum
            of the curvature, not ranked as "more/less real" by strength.
            `slope_change_amplitude` is the difference between the
            smoothed 1st derivative's local extrema immediately
            bracketing the dip -- an honestly qualitative indicator of how
            much the slope visibly bent, not a rigorous amplitude fit.
    """
    lo, hi = float(energy_range[0]), float(energy_range[1])
    if not (lo < hi):
        raise ValueError(f"energy_range must be (lo, hi) with lo < hi, got {energy_range!r}")

    energy_axis = np.asarray(dataset.energy_axis, dtype=float)
    mean_spec = np.asarray(dataset.calculate_mean_spectrum(), dtype=float)
    mask = (energy_axis >= lo) & (energy_axis <= hi)
    E = energy_axis[mask]
    I_win = mean_spec[mask]
    if len(E) < 5:
        raise ValueError(
            f"energy_range={energy_range!r} contains only {len(E)} sample(s); need at least "
            "a handful of channels to smooth and differentiate."
        )

    spacing = float(dataset.sampling[2])
    if not (np.isfinite(spacing) and spacing > 0):
        raise ValueError(f"dataset eV/channel spacing (sampling[2]={spacing!r}) must be positive.")

    # ---- 2. smoothing_window_eV -> odd sample count at this dataset's own
    # spacing, validated against both polyorder and energy_range itself
    # (rather than silently clipped -- a silently-shrunk window would
    # quietly change what "0.4 eV" means without the caller noticing).
    window_samples = int(round(smoothing_window_eV / spacing))
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples <= polyorder + 1:
        raise ValueError(
            f"smoothing_window_eV={smoothing_window_eV} eV converts to {window_samples} "
            f"samples at this dataset's spacing ({spacing:.4g} eV/channel), which is not "
            f"> polyorder+1={polyorder + 1}; widen smoothing_window_eV or lower polyorder."
        )
    if window_samples > len(E):
        raise ValueError(
            f"smoothing_window_eV={smoothing_window_eV} eV converts to {window_samples} "
            f"samples at this dataset's spacing ({spacing:.4g} eV/channel), which does not "
            f"fit inside energy_range={energy_range!r} ({len(E)} samples available)."
        )

    # ---- 3. three SavGol passes over the same windowed spectrum: the
    # smoothed spectrum itself (deriv=0), the smoothed slope (deriv=1),
    # and the smoothed curvature (deriv=2) -- same window/polyorder for
    # all three so they're directly comparable, `delta` set to the actual
    # eV spacing so the derivatives are in real dI/dE, d2I/dE2 units.
    smoothed = savgol_filter(
        I_win, window_length=window_samples, polyorder=polyorder, delta=spacing
    )
    first_derivative = savgol_filter(
        I_win, window_length=window_samples, polyorder=polyorder, deriv=1, delta=spacing
    )
    second_derivative = savgol_filter(
        I_win, window_length=window_samples, polyorder=polyorder, deriv=2, delta=spacing
    )

    # ---- 4. trim a half-window margin off each edge before searching for
    # candidates -- a SavGol derivative near a boundary is fit from a
    # lopsided/incomplete window and is not trustworthy there.
    half_window = window_samples // 2
    n = len(E)
    trim_lo, trim_hi = half_window, n - half_window
    if trim_hi - trim_lo < 3:
        raise ValueError(
            f"energy_range={energy_range!r} is too narrow relative to "
            f"smoothing_window_eV={smoothing_window_eV} eV -- after trimming a half-window "
            f"margin off each edge only {max(0, trim_hi - trim_lo)} sample(s) remain to "
            "search for candidates."
        )

    # ---- 5. candidates = local MINIMA of the 2nd derivative (find_peaks on
    # its negation) -- a genuine dip/bend in curvature, not a zero
    # crossing (which marks an inflection with no actual curvature
    # extremum, and fires on every plain monotonic decay too).
    #
    # prominence=0.0 here is not a real filter (every local extremum has
    # prominence >= 0) -- it's only how find_peaks is told to actually
    # compute and report prominences for every peak it finds under its own
    # default handling, rather than silently collapsing to one "best"
    # candidate. An explicit min_prominence is what does the real
    # filtering.
    prominence_threshold = min_prominence if min_prominence is not None else 0.0
    neg_curvature = -second_derivative[trim_lo:trim_hi]
    peak_idx, props = find_peaks(neg_curvature, prominence=prominence_threshold)
    dip_prominences = props["prominences"]
    full_idx = peak_idx + trim_lo

    # 1st derivative's own local extrema (both signs) -- used to bracket
    # each candidate and report how much the slope visibly bent.
    extrema_idx = np.sort(
        np.concatenate([find_peaks(first_derivative)[0], find_peaks(-first_derivative)[0]])
    )

    raw_candidates = []  # (index, dip_prominence, slope_change_amplitude)
    for i, prom in zip(full_idx, dip_prominences):
        left = extrema_idx[extrema_idx < i]
        right = extrema_idx[extrema_idx > i]
        left_i = int(left[-1]) if len(left) else 0
        right_i = int(right[0]) if len(right) else n - 1
        slope_change_amplitude = abs(float(first_derivative[right_i] - first_derivative[left_i]))
        raw_candidates.append((int(i), float(prom), slope_change_amplitude))
    raw_candidates.sort(key=lambda t: t[0])  # by energy position, not prominence

    candidates = [
        {"center_eV": float(E[i]), "dip_prominence": prom, "slope_change_amplitude": amp}
        for i, prom, amp in raw_candidates
    ]

    print(
        f"detect_broad_bump_via_slope_change: window_samples={window_samples} "
        f"({smoothing_window_eV} eV at {spacing:.4g} eV/channel spacing), polyorder={polyorder}, "
        f"energy_range={energy_range!r}"
    )
    if candidates:
        for c in candidates:
            print(
                f"  -> candidate at {c['center_eV']:.4g} eV: dip_prominence="
                f"{c['dip_prominence']:.4g}, slope_change_amplitude={c['slope_change_amplitude']:.4g}"
            )
    else:
        print("  -> no candidates found.")

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        axes[0].plot(E, I_win, "k.", ms=3, alpha=0.4, label="raw mean spectrum")
        axes[0].plot(
            E, smoothed, "b-", lw=1.6, label=f"SavGol smoothed (window_length={window_samples})"
        )
        axes[0].set_ylabel("Intensity")
        axes[0].set_title("1. Raw spectrum and smoothed spectrum")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(E, first_derivative, "m-", lw=1.2, label="smoothed 1st derivative (slope)")
        axes[1].axhline(0, color="gray", ls="--", lw=1)
        for (i, prom, _amp), c in zip(raw_candidates, candidates):
            axes[1].axvline(c["center_eV"], color="tab:red", ls=":", lw=1.5)
            axes[1].annotate(
                f"{c['center_eV']:.3g} eV\nprominence={prom:.3g}",
                xy=(c["center_eV"], float(first_derivative[i])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )
        axes[1].set_ylabel("dI/dE")
        axes[1].set_title("2. Slope (1st derivative) -- candidate positions marked")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            E, second_derivative, "c-", lw=1.2, label="smoothed 2nd derivative (curvature)"
        )
        axes[2].axhline(0, color="gray", ls="--", lw=1)
        for (i, prom, _amp), c in zip(raw_candidates, candidates):
            y_bottom = float(second_derivative[i])
            y_top = y_bottom + prom
            axes[2].plot([c["center_eV"]], [y_bottom], "rv", ms=9, zorder=5)
            axes[2].annotate(
                "",
                xy=(c["center_eV"], y_top),
                xytext=(c["center_eV"], y_bottom),
                arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.3),
            )
            axes[2].annotate(
                f"{c['center_eV']:.3g} eV\nprominence={prom:.3g}",
                xy=(c["center_eV"], y_top),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )
        axes[2].set_xlabel("Energy loss (eV)")
        axes[2].set_ylabel("d2I/dE2")
        axes[2].set_title("3. Curvature (2nd derivative) -- candidate dip + prominence bracket")
        axes[2].legend(loc="best", fontsize=8)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            "Broad bump detection via 2nd-derivative slope change "
            "-- NOT a peak/shoulder fit (see detect_peaks_in_range for that)",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    return {
        "energy": E,
        "spectrum_smoothed": smoothed,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative,
        "candidates": candidates,
    }


def build_hidden_bump_spatial_maps(
    eels_hl_despiked,
    eels_ll,
    thickness_map,
    adf,
    candidate_energies_eV: Sequence[float],
    window_samples: int = 45,
    polyorder: int = 3,
    zlp_search_half_width_eV: float = 0.2,
    median_filter_pixels: int = 3,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Per-pixel spatial maps of fixed broad-bump candidate energies (e.g. the
    ones `detect_broad_bump_via_slope_change()` found on the mean
    spectrum), built directly from the 2nd-derivative/slope-change
    signature at the pixel level -- no ZLP model is fitted.

    Why the ZLP peak height comes from `eels_ll`, not `eels_hl_despiked`
    itself: `eels_hl_despiked`'s own energy axis starts a bit above 0 eV
    (dual-EELS acquisition -- LL carries the ZLP/thickness signal, HL is
    read out starting just past it) -- there is no "window around E=0"
    inside `eels_hl_despiked` to measure a ZLP peak height from. Only
    `eels_ll`'s energy axis actually spans E=0, so each pixel's ZLP peak
    height is measured from that SAME pixel's `eels_ll` spectrum, then
    used to normalize that pixel's `eels_hl_despiked` spectrum, before any
    derivative is taken -- controlling for per-pixel thickness/intensity
    differences up front.

    The ZLP peak height itself reuses the crude-ZLP convention already
    used (twice) in `Dataset3deels` -- `measure_zlp_offset()` and
    `calculate_thickness_log_ratio()` both median-filter each pixel's
    spectrum first ("to discount hot pixels that might spuriously produce
    the maximum intensity") before taking its max as the crude per-pixel
    ZLP estimate. Reused verbatim here (same `median_filter_pixels=3`
    default), just reporting the max VALUE (the peak height) within a
    narrow +/-`zlp_search_half_width_eV` window around E=0, rather than
    the position a Gaussian fit would refine it to, or the integrated
    intensity `calculate_thickness_log_ratio` sums.

    Per pixel:
      1. `height` = max of the median-filtered `eels_ll` spectrum within
         `|E| <= zlp_search_half_width_eV`.
      2. `normalized` = that pixel's `eels_hl_despiked` spectrum / `height`.
      3. 2nd derivative of `normalized` via the same Savitzky-Golay
         `window_samples`/`polyorder` `detect_broad_bump_via_slope_change`
         uses (`delta` = this dataset's own eV/channel spacing), over the
         full spectrum.
      4. At each fixed candidate energy, a dip "prominence" = the max of
         the 2nd derivative in a small local window (+/- `window_samples
         // 2` samples -- the same edge-trim half-window
         `detect_broad_bump_via_slope_change` uses) around that energy,
         minus the 2nd derivative's own value there. Left signed, not
         clipped at 0 -- a negative value means the curvature is locally
         convex (no dip at all) at that pixel, which is real information
         about how much weaker/absent the feature is there, not noise to
         discard.

    A pixel is NaN-masked (never silently filled) if: its `eels_ll` ZLP
    peak height is non-finite, or so close to zero (<= 1e-6 * the median
    finite height across the scan) that normalizing by it is meaningless;
    its raw `eels_hl_despiked` spectrum contains a non-finite value; the
    normalized spectrum contains a non-finite value; or the
    Savitzky-Golay filter itself raises.

    Parameters
    ----------
    eels_hl_despiked, eels_ll : Dataset3deels
        ZLP-corrected high-loss (despiked) and low-loss datasets from the
        SAME scan -- must share `(scan_row, scan_col)`.
    thickness_map : ndarray, shape (scan_row, scan_col)
        For the visual comparison panel and the thickness correlation
        cross-check.
    adf : ndarray, shape (scan_row, scan_col), or None
        For the visual comparison panel and the ADF correlation
        cross-check. Pass None to skip both (e.g. `raw.adf is None`).
    candidate_energies_eV : sequence of float
        The fixed energies (eV) to sample the per-pixel 2nd derivative at,
        typically ``candidates[i]["center_eV"]`` from
        ``detect_broad_bump_via_slope_change()``.
    window_samples, polyorder : int, default 45, 3
        Savitzky-Golay parameters -- must match
        `detect_broad_bump_via_slope_change`'s own choice for these to
        mean the same thing spatially that they meant on the mean
        spectrum.
    zlp_search_half_width_eV : float, default 0.2
        Half-width (eV) of the window around E=0 in `eels_ll` the ZLP peak
        height is searched in.
    median_filter_pixels : int, default 3
        Same hot-pixel-discounting median filter size
        `measure_zlp_offset` / `calculate_thickness_log_ratio` use.
    show : bool, default True
        Render the multi-panel figure (thickness, ADF if available, and
        the candidate maps).

    Returns
    -------
    dict with keys:
        "map_<E>eV" -- one ndarray (scan_row, scan_col) per candidate
            energy, with "." replaced by "p" in the key
            (e.g. 1.649 -> "map_1p649eV").
        "failed_pixel_counts" -- dict of the same map-name keys, each
            mapping to the SAME shared failure count (a failure is a
            per-pixel property of the whole normalized-spectrum/2nd-
            derivative computation, not per candidate energy).
        "correlation_with_thickness", "correlation_with_adf" -- dict of
            map-name -> Pearson r (NaN-pair-dropped). "correlation_with_adf"
            is empty if `adf` is None.
    """
    energy_axis = np.asarray(eels_hl_despiked.energy_axis, dtype=float)
    spacing = float(eels_hl_despiked.sampling[2])
    if not (np.isfinite(spacing) and spacing > 0):
        raise ValueError(
            f"eels_hl_despiked eV/channel spacing (sampling[2]={spacing!r}) must be positive."
        )
    if window_samples % 2 == 0 or window_samples <= polyorder + 1:
        raise ValueError(
            f"window_samples={window_samples} must be odd and > polyorder+1={polyorder + 1}."
        )
    half_window = window_samples // 2
    n_energy = len(energy_axis)

    candidate_idx = []
    for e in candidate_energies_eV:
        idx = int(np.argmin(np.abs(energy_axis - e)))
        if idx < half_window or idx >= n_energy - half_window:
            raise ValueError(
                f"candidate energy {e} eV (index {idx}) is within the half-window "
                f"({half_window} samples) edge margin of eels_hl_despiked's own energy "
                "range -- the 2nd derivative there is unreliable near the boundary."
            )
        candidate_idx.append(idx)

    hl_array = np.asarray(eels_hl_despiked.array, dtype=float)
    ll_array = np.asarray(eels_ll.array, dtype=float)
    if hl_array.shape[:2] != ll_array.shape[:2]:
        raise ValueError(
            f"eels_hl_despiked scan shape {hl_array.shape[:2]} does not match eels_ll "
            f"scan shape {ll_array.shape[:2]} -- must be the same scan."
        )
    scan_row, scan_col, _ = hl_array.shape

    ll_energy_axis = np.asarray(eels_ll.energy_axis, dtype=float)
    zlp_window_mask = np.abs(ll_energy_axis) <= zlp_search_half_width_eV
    if not np.any(zlp_window_mask):
        raise ValueError(
            f"No eels_ll energy channels fall within +/-{zlp_search_half_width_eV} eV of "
            "E=0 -- widen zlp_search_half_width_eV."
        )

    # ---- 1. per-pixel ZLP peak height from eels_ll (max of a median-
    # filtered spectrum within a narrow window around E=0) -- see the
    # docstring for why eels_ll and not eels_hl_despiked itself.
    zlp_height = np.full((scan_row, scan_col), np.nan)
    for i in range(scan_row):
        for j in range(scan_col):
            spec = ll_array[i, j, :]
            if median_filter_pixels > 0:
                spec = median_filter(spec, median_filter_pixels)
            window_vals = spec[zlp_window_mask]
            if np.any(np.isfinite(window_vals)):
                zlp_height[i, j] = float(np.nanmax(window_vals))

    finite_heights = zlp_height[np.isfinite(zlp_height)]
    zlp_epsilon = 1e-6 * float(np.nanmedian(finite_heights)) if finite_heights.size else 0.0

    maps = [np.full((scan_row, scan_col), np.nan) for _ in candidate_idx]
    n_failed = 0
    failure_reasons: Dict[str, int] = {}

    def _fail(reason):
        nonlocal n_failed
        n_failed += 1
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    # ---- 2/3. per-pixel: normalize by that pixel's own ZLP height, SavGol
    # 2nd derivative, sample it at each fixed candidate energy.
    for i in range(scan_row):
        for j in range(scan_col):
            height = zlp_height[i, j]
            if not np.isfinite(height) or height <= zlp_epsilon:
                _fail("zlp_height_near_zero_or_nan")
                continue

            spectrum = hl_array[i, j, :]
            if not np.all(np.isfinite(spectrum)):
                _fail("non_finite_raw_spectrum")
                continue

            normalized = spectrum / height
            if not np.all(np.isfinite(normalized)):
                _fail("non_finite_normalized_spectrum")
                continue

            try:
                second_derivative = savgol_filter(
                    normalized,
                    window_length=window_samples,
                    polyorder=polyorder,
                    deriv=2,
                    delta=spacing,
                )
            except ValueError:
                _fail("savgol_failed")
                continue

            for k, idx in enumerate(candidate_idx):
                lo_i, hi_i = idx - half_window, idx + half_window + 1
                baseline = float(np.max(second_derivative[lo_i:hi_i]))
                maps[k][i, j] = baseline - float(second_derivative[idx])

    def _key(e):
        return f"map_{f'{e:.3f}'.replace('.', 'p')}eV"

    map_names = [_key(e) for e in candidate_energies_eV]
    result_maps = dict(zip(map_names, maps))
    failed_pixel_counts = {name: n_failed for name in map_names}

    n_total = scan_row * scan_col
    print(
        f"build_hidden_bump_spatial_maps: {n_total - n_failed}/{n_total} pixels usable "
        f"(window_samples={window_samples}, polyorder={polyorder}, "
        f"zlp_search_half_width_eV={zlp_search_half_width_eV})"
    )
    if n_failed:
        print(f"  -> {n_failed} pixel(s) NaN-masked: {failure_reasons}")
    for name in map_names:
        print(f"  -> {name}: {failed_pixel_counts[name]} failed pixel(s)")

    # ---- correlation cross-check against thickness and ADF -- even after
    # normalizing by ZLP peak height, multiple-scattering broadening in
    # thicker regions could still leave a residual thickness correlation
    # in the curvature SHAPE itself, not just its amplitude.
    thickness_map = np.asarray(thickness_map, dtype=float)
    adf_map = np.asarray(adf, dtype=float) if adf is not None else None

    correlation_with_thickness: Dict[str, float] = {}
    correlation_with_adf: Dict[str, float] = {}
    for name, m in zip(map_names, maps):
        valid = np.isfinite(m) & np.isfinite(thickness_map)
        correlation_with_thickness[name] = (
            float(pearsonr(m[valid], thickness_map[valid])[0])
            if valid.sum() >= 2
            else float("nan")
        )
        if adf_map is not None:
            valid = np.isfinite(m) & np.isfinite(adf_map)
            correlation_with_adf[name] = (
                float(pearsonr(m[valid], adf_map[valid])[0]) if valid.sum() >= 2 else float("nan")
            )

    print("Pearson correlation vs. thickness_map:")
    for name in map_names:
        print(f"  -> {name}: r={correlation_with_thickness[name]:.3f}")
    if adf_map is not None:
        print("Pearson correlation vs. ADF:")
        for name in map_names:
            print(f"  -> {name}: r={correlation_with_adf[name]:.3f}")
    else:
        print("ADF not available (adf=None) -- skipped correlation_with_adf.")

    if show:
        panels: List[Any] = []
        titles: List[str] = []
        cmaps: List[str] = []

        panels.append(thickness_map)
        titles.append("Thickness (t/λ)")
        cmaps.append("viridis")

        if adf_map is not None:
            panels.append(adf_map)
            titles.append("ADF")
            cmaps.append("gray")

        for name, e, m in zip(map_names, candidate_energies_eV, maps):
            panels.append(m)
            titles.append(f"{e:.3f} eV dip prominence")
            cmaps.append("magma")

        # NaN pixels render fully transparent under matplotlib's default
        # "bad" color, and show_2d's default norm (2nd-98th percentile,
        # per panel) already filters NaN/inf before computing its limits
        # -- no extra handling needed here.
        fig, axs = show_2d(panels, title=titles, cmap=cmaps, cbar=True)
        fig.suptitle(
            "Per-pixel spatial maps of validated bump candidates (2nd-derivative/slope-change)",
            fontsize=10,
        )
        plt.show()

    return {
        **result_maps,
        "failed_pixel_counts": failed_pixel_counts,
        "correlation_with_thickness": correlation_with_thickness,
        "correlation_with_adf": correlation_with_adf,
    }


def detect_peaks_whole_range(
    dataset,
    *,
    exclude_windows=(),
    energy_range=None,
    smoothing_window_eV: float = 2.0,
    polyorder: int = 3,
    prominence_sigma: float = 5.0,
    min_prominence_fraction: float = 0.15,
    min_prominence: Optional[float] = None,
    min_distance_eV: float = 1.0,
    edge_guard_eV: float = 0.5,
    check_reproducibility: bool = True,
    reproducibility_tol_eV: float = 0.15,
    title: str = "",
    display_energy_range=None,
    show: bool = True,
):
    """
    General peak detection on the mean spectrum over a WHOLE range (e.g. the entire high-loss axis), skipping
    `exclude_windows` (typically the background-fit area, where the background-subtracted spectrum is zero by
    construction). Unlike `detect_broad_bump_via_slope_change()` -- which looks for a faint bend riding on a
    smooth decaying tail and needs a very smooth curve -- this finds ordinary local maxima of a
    Savitzky-Golay-smoothed mean spectrum, so it works on a whole noisy core-loss spectrum.

    `check_reproducibility=True` (default) also runs the same detection independently on two interleaved
    pixel subsets (a checkerboard split -- not a left/right split, so a real spatial gradient across the scan
    doesn't bias one half) and flags each peak `reproducible: True/False`: found in both halves within
    `reproducibility_tol_eV`. This is how a genuine, sharp, very-high-SNR "peak" was previously traced to a
    fixed detector pattern rather than a real spectral feature -- its per-pixel amplitude didn't track the
    local signal, but a plain significance/prominence test alone could not tell it apart from a real edge
    feature. A real edge or resonance is present in (most of) the pixels and survives the split; a fixed
    detector artifact, present identically in every pixel, also survives it (this check does NOT catch that
    case -- cross-check a suspiciously narrow, very high `snr` peak's per-pixel amplitude against local
    signal/counts by hand, as was done here). What it DOES catch is a peak driven by a handful of outlier
    pixels (e.g. one spike-affected region), which typically appears strongly in only one of the two halves.

    Noise: sigma_y = 1.4826 * MAD of (mean spectrum - its smoothed curve); the noise left in the smoothed curve
    is sigma_y times the norm of the smoothing filter's coefficients. That statistical noise is tiny for a mean over
    hundreds of pixels, while a real spectrum carries a systematic ripple (detector pattern, correlated noise) many
    times larger, so a peak must ALSO have a prominence of at least `min_prominence_fraction` of the smoothed curve's
    dynamic range in the searched region. The threshold is max(`prominence_sigma` x noise,
    `min_prominence_fraction` x range), or the absolute `min_prominence` if given. A peak must further be `min_distance_eV` from a stronger peak, and
    lie at least `edge_guard_eV` away from an excluded window / the ends of the range (the smoother is unreliable
    there). Each contiguous allowed stretch is searched separately so a peak cannot straddle an excluded window.

    A wider `smoothing_window_eV` gives a smoother curve (fewer, broader peaks): correlated noise in core-loss data
    needs 2-4 eV; broad features may need more.

    Returns ``(peaks, figure_or_None)``; `peaks` is a list of dicts sorted by energy with ``energy_eV``,
    ``height`` (of the smoothed curve), ``prominence``, ``fwhm_eV`` (width at half prominence), ``snr``
    (prominence / smoothed-curve noise) and ``reproducible`` (True/False if `check_reproducibility`, else
    None). The figure shows the raw mean, the smoothed curve, the peaks (labelled, hollow marker = not
    reproducible) and the excluded windows (red).
    """
    e = np.asarray(dataset.energy_axis, dtype=float)
    dE = float(np.median(np.diff(e)))
    n = int(round(smoothing_window_eV / dE))
    n += n % 2 == 0
    n = max(n, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if n > len(e):
        raise ValueError(
            f"smoothing_window_eV={smoothing_window_eV} eV needs {n} channels, the axis has only {len(e)}"
        )
    lo_r, hi_r = (
        (float(e[0]), float(e[-1]))
        if energy_range is None
        else (float(energy_range[0]), float(energy_range[1]))
    )
    allowed = (e >= lo_r) & (e <= hi_r)
    for lo, hi in exclude_windows:
        allowed &= ~((e >= lo - edge_guard_eV) & (e <= hi + edge_guard_eV))
    guard = int(round(edge_guard_eV / dE))
    allowed[: n // 2 + guard] = False
    allowed[len(e) - n // 2 - guard :] = False
    dist = max(int(round(min_distance_eV / dE)), 1)

    def _find(mean_spec):
        """Same detection logic, parameterized on the mean spectrum -- reused for the full mean and each
        checkerboard half. Returns (peaks_without_reproducibility_field, sigma_s, dyn_range, smooth)."""
        smooth = savgol_filter(mean_spec, n, polyorder)
        resid = (mean_spec - smooth)[allowed]
        sigma_y = (
            1.4826 * float(np.median(np.abs(resid - np.median(resid))))
            if resid.size
            else float("nan")
        )
        gain = float(np.linalg.norm(savgol_coeffs(n, polyorder)))
        sigma_s = sigma_y * gain
        searched = smooth[allowed]
        dyn_range = float(searched.max() - searched.min()) if searched.size else 0.0
        prom = (
            float(min_prominence)
            if min_prominence is not None
            else max(prominence_sigma * sigma_s, min_prominence_fraction * dyn_range)
        )
        found = []
        idx = np.flatnonzero(allowed)
        if idx.size:
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            for seg in np.split(idx, splits):
                if len(seg) < 5:
                    continue
                y = smooth[seg]
                pk, props = find_peaks(y, prominence=prom, distance=dist)
                if len(pk):
                    widths = peak_widths(y, pk, rel_height=0.5)[0] * dE
                    for j, k in enumerate(pk):
                        found.append(
                            dict(
                                energy_eV=float(e[seg][k]),
                                height=float(y[k]),
                                prominence=float(props["prominences"][j]),
                                fwhm_eV=float(widths[j]),
                                snr=float(props["prominences"][j] / sigma_s)
                                if sigma_s > 0
                                else float("inf"),
                            )
                        )
        return found, sigma_s, dyn_range, smooth

    m = np.asarray(dataset.calculate_mean_spectrum(), dtype=float)
    peaks, sigma_s, dyn_range, smooth = _find(m)
    peaks.sort(key=lambda d: d["energy_eV"])

    if check_reproducibility and peaks:
        arr = np.asarray(dataset.array, dtype=float)
        ny, nx = arr.shape[:2]
        checker = np.add.outer(np.arange(ny), np.arange(nx)) % 2 == 0
        half_peaks = []
        for mask in (checker, ~checker):
            if mask.sum() < 2:  # e.g. a 1x1 dataset: nothing to split
                half_peaks.append([])
                continue
            hp, *_ = _find(arr[mask].mean(axis=0))
            half_peaks.append(hp)
        for d in peaks:
            d["reproducible"] = all(
                any(abs(d["energy_eV"] - h["energy_eV"]) <= reproducibility_tol_eV for h in hp)
                for hp in half_peaks
            )
    elif peaks:
        for d in peaks:
            d["reproducible"] = None  # not checked

    print(
        f"peak detection over {lo_r:g}-{hi_r:g} eV excluding {[tuple(w) for w in exclude_windows]}: smoothing {smoothing_window_eV} eV "
        f"({n} channels, order {polyorder}), noise of smoothed curve {sigma_s:.3g}, range {dyn_range:.3g}, min prominence "
        f"{max(prominence_sigma * sigma_s, min_prominence_fraction * dyn_range) if min_prominence is None else min_prominence:.3g}"
    )
    if peaks:
        print(
            f"{'energy (eV)':>12s} {'height':>9s} {'prominence':>11s} {'FWHM (eV)':>10s} {'prom/noise':>11s}  reproducible (both pixel halves)"
        )
        for d in peaks:
            rep = "-" if d["reproducible"] is None else ("yes" if d["reproducible"] else "NO")
            print(
                f"{d['energy_eV']:12.2f} {d['height']:9.3g} {d['prominence']:11.3g} {d['fwhm_eV']:10.2f} {d['snr']:11.1f}  {rep}"
            )
        if check_reproducibility and not all(d["reproducible"] for d in peaks):
            print(
                "  -> peaks flagged NOT reproducible are driven by a subset of pixels (e.g. one hot/spike-affected "
                "region), not a signal present across the scan -- treat with more suspicion than the others."
            )
    else:
        print("  -> no peaks found.")

    fig = None
    if show:
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(e, m, color="0.65", lw=0.8, label="mean spectrum")
        ax.plot(
            e,
            np.where(allowed, smooth, np.nan),
            "b-",
            lw=1.6,
            label=f"smoothed ({smoothing_window_eV:g} eV, order {polyorder})",
        )
        ax.plot(
            e,
            np.where(allowed, np.nan, smooth),
            color="0.5",
            lw=1.0,
            ls=":",
            label="smoothed (not searched)",
        )
        for lo, hi in exclude_windows:
            ax.axvspan(lo, hi, color="red", alpha=0.12)
        rep = [d for d in peaks if d["reproducible"] is not False]
        not_rep = [d for d in peaks if d["reproducible"] is False]
        ax.plot(
            [d["energy_eV"] for d in rep],
            [d["height"] for d in rep],
            "rv",
            ms=8,
            label="detected peaks",
        )
        if not_rep:
            ax.plot(
                [d["energy_eV"] for d in not_rep],
                [d["height"] for d in not_rep],
                "v",
                ms=9,
                mfc="none",
                mec="darkred",
                mew=1.6,
                label="NOT reproducible (both pixel halves)",
            )
        for d in peaks:
            ax.annotate(
                f"{d['energy_eV']:.1f}",
                (d["energy_eV"], d["height"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="darkred",
            )
        ax.set_xlim(*(display_energy_range or (lo_r, hi_r)))
        vis = (e >= ax.get_xlim()[0]) & (e <= ax.get_xlim()[1])
        if vis.any():
            y0, y1 = float(np.nanmin(m[vis])), float(np.nanmax(m[vis]))
            ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.12 * (y1 - y0))
        ax.set_xlabel("Energy loss (eV)")
        ax.set_ylabel("Mean intensity")
        ax.set_title(
            _wrap_title(
                f"{title}: general peak detection (red = excluded from the search)".strip(" :"),
                110,
            ),
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
    return peaks, fig


def plot_energy_windows_summary(
    dataset, windows, *, title="", display_energy_range=None, show=True
):
    """
    Two panels: the mean spectrum with the energy windows shaded and numbered (dotted line = position of the
    window's maximum), and the per-window signal (mean over pixels +/- standard error) as bars with the
    t-value (signal / standard error) written on them. Prints the table from `summarize_energy_windows()`.
    Returns ``(rows, figure_or_None)``.
    """
    rows = summarize_energy_windows(dataset, windows)
    e = np.asarray(dataset.energy_axis, dtype=float)
    mean_spec = np.asarray(dataset.calculate_mean_spectrum(), dtype=float)
    print(
        f"{'window (eV)':>16s} {'mean':>9s} {'sem':>8s} {'t':>7s} {'pixels>0':>9s} {'spatial CV':>11s}  peak of the mean spectrum"
    )
    for r in rows:
        if r["n_channels"]:
            print(
                f"{r['window'][0]:7.2f}-{r['window'][1]:<8.2f} {r['mean']:9.4g} {r['sem']:8.3g} {r['t']:7.1f} "
                f"{r['frac_positive']:9.2f} {r['spatial_cv']:11.1f}  {r['peak_value']:.3g} at {r['peak_eV']:.2f} eV"
            )
    fig = None
    if show:
        lo_all = min(w[0] for w in windows)
        hi_all = max(w[1] for w in windows)
        pad = 0.35 * (hi_all - lo_all)
        xlim = display_energy_range or (
            max(float(e[0]), lo_all - pad),
            min(float(e[-1]), hi_all + pad),
        )
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1])
        )
        ax1.plot(e, mean_spec, "k-", lw=1.3)
        colors = plt.cm.tab10(np.arange(len(windows)))
        for i, (r, (lo, hi)) in enumerate(zip(rows, windows)):
            ax1.axvspan(lo, hi, color=colors[i], alpha=0.18)
            ax1.text(
                0.5 * (lo + hi),
                0.97,
                str(i + 1),
                transform=ax1.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=11,
                weight="bold",
                color=colors[i],
            )
            if r["n_channels"]:
                ax1.axvline(r["peak_eV"], color=colors[i], ls=":", lw=1.2)
        ax1.set_xlim(*xlim)
        vis = (e >= xlim[0]) & (e <= xlim[1])
        if vis.any():
            y0, y1 = float(mean_spec[vis].min()), float(mean_spec[vis].max())
            ax1.set_ylim(y0 - 0.06 * (y1 - y0), y1 + 0.12 * (y1 - y0))
        ax1.axhline(0, color="gray", ls="--", lw=0.8)
        ax1.set_xlabel("Energy loss (eV)")
        ax1.set_ylabel("Mean intensity")
        ax1.set_title("mean spectrum and the energy windows")
        ax1.grid(True, alpha=0.3)
        vals = [r.get("mean", np.nan) for r in rows]
        errs = [r.get("sem", np.nan) for r in rows]
        ax2.bar(
            range(1, len(rows) + 1),
            vals,
            yerr=errs,
            color=colors[: len(rows)],
            alpha=0.8,
            capsize=4,
        )
        for i, r in enumerate(rows):
            if r["n_channels"]:
                ax2.text(
                    i + 1,
                    r["mean"] + (r["sem"] if r["mean"] >= 0 else -r["sem"]),
                    f"t={r['t']:.1f}",
                    ha="center",
                    va="bottom" if r["mean"] >= 0 else "top",
                    fontsize=8,
                )
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xticks(range(1, len(rows) + 1))
        ax2.set_xticklabels([f"{w[0]:g}-{w[1]:g}" for w in windows], fontsize=8, rotation=20)
        ax2.set_ylabel("window mean +/- s.e.m. over pixels")
        ax2.set_title("signal in each window")
        ax2.grid(True, axis="y", alpha=0.3)
        if title:
            fig.suptitle(_wrap_title(title, 130), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
        plt.show()
    return rows, fig


def show_energy_windows_with_peaks(
    eels_hl_bgsub,
    energy_windows: Sequence[Tuple[float, float]],
    *,
    target_edge: Optional[float] = None,
    display_energy_range: Optional[Tuple[float, float]] = None,
    detect_peaks: bool = True,
    peak_detection_method: str = "auto",
    peak_kwargs: Optional[dict] = None,
    show_peak_markers: bool = False,
    show_peak_labels: bool = False,
    show_second_derivative: bool = False,
    show_peak_table: bool = True,
    hover_annotations: bool = True,
    enable_click_labels: bool = True,
    click_label_snap_to_spectrum: bool = True,
    click_label_precision: int = 2,
    title_suffix: str = "",
) -> Dict[Tuple[float, float], List[float]]:
    """
    NON-interactive preview (no prompts): plot the mean background-subtracted
    HL spectrum with `energy_windows` shaded and a smoothed-curve overlay per
    window, run peak detection (via detect_peaks_in_range()) and print a
    summary regardless of what's drawn, and return the detected-peaks dict
    -- a visual/printed aid for hand-editing the `energy_windows` tuple used
    by the static show_energy_window_map() loop. Does not change or replace
    that loop.

    Default display is deliberately minimal: spectrum + smoothed overlay +
    shaded windows + a peak-coordinate table underneath -- no vertical peak
    lines, no in-plot text labels, no derivative subplot. Detection always
    runs and is always printed/returned the same way regardless of these
    display toggles; only what gets *drawn* changes.

    Parameters
    ----------
    eels_hl_bgsub : Dataset3deels
        Background-subtracted HL dataset.
    energy_windows : sequence of (lo_eV, hi_eV)
        The windows to shade (and, if detect_peaks, search for peaks in) --
        e.g. pass the notebook's `energy_windows` tuple directly.
    target_edge : float, optional
        Reference vertical line only.
    display_energy_range : (float, float), optional
        (lo, hi) eV to zoom the plotted x-axis into. Purely a display crop
        -- the full mean spectrum is still plotted and windows outside this
        range are still shaded if they happen to be visible; peak detection
        always runs over the full `energy_windows` regardless of this
        range. Defaults to the dataset's full energy axis (no zoom).
    detect_peaks : bool, default True
        If False, just shades the windows -- no detection, no peak summary,
        no peak table regardless of `show_peak_table`.
    peak_detection_method : "max" | "local_maxima" | "shoulder" | "auto"
        Passed to detect_peaks_in_range() for every window.
    peak_kwargs : dict, optional
        Method-specific kwargs (prominence/height/distance for local_maxima;
        smooth_window/zero_crossing_threshold for shoulder), passed to
        detect_peaks_in_range() for every window.
    show_peak_markers : bool, default False
        Draw a dashed vertical line at each detected peak.
    show_peak_labels : bool, default False
        Draw a boxed, rotated "X.XX eV" label at each detected peak
        (staggered to increasing heights when peaks are close together in
        energy, so labels don't overlap, with the y-axis expanded to keep
        them fully inside the axes). Implies `show_peak_markers=True`
        regardless of that argument's own value -- a label with no line
        pointing at it isn't useful.
    show_second_derivative : bool, default False
        If True and peak_detection_method is "shoulder" or "auto", adds a
        subplot showing the (smoothed) 2nd derivative used by the shoulder
        method, for debugging/validation.
    show_peak_table : bool, default True
        Adds a compact table beneath the plot(s) listing every window and
        its detected peak energies in plain text. Each row's border is
        color-matched to that window's shading color. No-ops if
        `detect_peaks=False` or there are no windows.
    hover_annotations : bool, default True
        Adds a mouse-hover tooltip to the main spectrum axes showing the
        nearest (energy, intensity) point on the mean spectrum as you move
        the cursor anywhere over it -- not just near a marker -- plus which
        energy window (if any) and nearest detected peak (if close enough).
        Plain matplotlib event handling only (no extra dependency); needs an
        interactive backend to actually fire mouse-move events -- with the
        default static/Agg backend the plot renders exactly the same and
        the tooltip just never appears. In a notebook, run
        `%matplotlib widget` (ipympl) once before this cell to enable it.
    enable_click_labels : bool, default True
        On the main spectrum axes only (same interactive-backend caveat as
        `hover_annotations` -- a no-op under the static Agg backend):
        left-click anywhere on the axes to permanently pin a coordinate
        label there; right-click an existing pinned label to delete just
        that one; press 'c' (with the plot focused) to clear all of them.
        New labels are staggered away from existing ones that are close in
        energy, same idea as the peak labels, to avoid overlap. Purely a
        display feature -- never touches peak detection or the returned
        dict.
    click_label_snap_to_spectrum : bool, default True
        If True, a click's label position snaps to the nearest actual
        (energy, intensity) sample on the mean spectrum (found via
        np.searchsorted), so the pinned coordinate is always a real data
        point. If False, the label uses the raw clicked position instead.
    click_label_precision : int, default 2
        Decimal places shown in pinned-label text, e.g. `(539.37, 12.51)`.

    The smoothed curve overlaid per window uses the exact same
    (window-width-adaptive, unless overridden) sigma detect_peaks_in_range()
    uses internally -- what you see is what was detected on.

    Returns
    -------
    dict {(lo, hi): [peak_eV, ...]} -- empty list per window if
    detect_peaks=False or none were found.
    """
    peak_kwargs = dict(peak_kwargs or {})
    energy_axis = np.asarray(eels_hl_bgsub.energy_axis, dtype=float)
    mean_spec = eels_hl_bgsub.calculate_mean_spectrum()

    # Shared label-placement scale, reused by both the (optional) peak
    # labels and the click-pinned labels below, so both use the same
    # "how close counts as overlapping" / "how far apart to stagger" sense
    # of scale.
    x_span = float(energy_axis[-1] - energy_axis[0]) or 1.0
    close_threshold = 0.035 * x_span
    data_range = float(mean_spec.max() - mean_spec.min()) or 1.0
    label_step = 0.10 * data_range

    # A label with no marker pointing at it isn't useful.
    show_peak_markers = show_peak_markers or show_peak_labels

    want_second_deriv_plot = show_second_derivative and peak_detection_method in (
        "shoulder",
        "auto",
    )
    want_table = show_peak_table and detect_peaks and len(energy_windows) > 0

    n_plot_axes = 2 if want_second_deriv_plot else 1
    n_total = n_plot_axes + (1 if want_table else 0)
    height_ratios = [3] * n_plot_axes + (
        [max(1.0, 0.42 * len(energy_windows) + 0.6)] if want_table else []
    )
    fig, all_axes = plt.subplots(
        n_total,
        1,
        figsize=(9, 5 * n_plot_axes + (height_ratios[-1] if want_table else 0)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    all_axes = np.atleast_1d(all_axes)
    plot_axes = list(all_axes[:n_plot_axes])
    table_ax = all_axes[n_plot_axes] if want_table else None
    for a in plot_axes[1:]:
        a.sharex(plot_axes[0])

    ax = plot_axes[0]
    ax.plot(energy_axis, mean_spec, "k-", lw=1.2, label="Mean HL spectrum (bg-subtracted)")

    colors = plt.cm.tab10.colors
    detected: Dict[Tuple[float, float], List[float]] = {}
    smoothed_by_window: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    all_peak_points: List[Tuple[float, float, str]] = []  # (x, y, color) across every window

    print(
        "Detected peaks in energy windows:"
        if detect_peaks
        else "Energy windows (peak detection disabled):"
    )
    for i, (lo, hi) in enumerate(energy_windows):
        color = colors[i % len(colors)]
        ax.axvspan(lo, hi, alpha=0.25, color=color, label=f"[{lo:.2f}, {hi:.2f}] eV")

        # Same adaptive-sigma formula detect_peaks_in_range() uses
        # internally (window-width-scaled unless smooth_window is
        # explicitly overridden in peak_kwargs), so the overlaid curve is
        # exactly what detection saw for this window, not an approximation.
        win_mask = (energy_axis >= lo) & (energy_axis <= hi)
        n_window_points = int(np.sum(win_mask))
        smooth_window_fraction = peak_kwargs.get("smooth_window_fraction", 0.04)
        default_sigma = (
            max(2, round(smooth_window_fraction * n_window_points)) if n_window_points else 2
        )
        win_sigma = peak_kwargs.get("smooth_window", default_sigma)
        win_smoothed = gaussian_filter1d(mean_spec, sigma=win_sigma) if win_sigma else mean_spec
        smoothed_by_window[(lo, hi)] = (win_smoothed, win_mask)

        ax.plot(
            energy_axis[win_mask],
            win_smoothed[win_mask],
            color=color,
            lw=2.0,
            alpha=0.9,
            label="smoothed (per window)" if i == 0 else None,
        )

        if not detect_peaks:
            detected[(lo, hi)] = []
            print(f"  Window ({lo}, {hi}) eV")
            continue

        # Detection (and its console printout) always runs, unaffected by
        # what's drawn -- only the code below this point decides what to
        # plot from the results.
        peaks = detect_peaks_in_range(
            mean_spec, energy_axis, (lo, hi), method=peak_detection_method, **peak_kwargs
        )
        detected[(lo, hi)] = peaks
        print(
            f"  Window ({lo}, {hi}) eV: {[round(p, 2) for p in peaks]} eV"
            if peaks
            else f"  Window ({lo}, {hi}) eV: [no peaks detected]"
        )

        if show_peak_markers:
            for p in peaks:
                ax.axvline(p, color=color, linestyle="--", linewidth=1, alpha=0.85)
                if show_peak_labels:
                    y = float(np.interp(p, energy_axis, mean_spec))
                    all_peak_points.append((p, y, color))

    # Place every peak's label in one pass across ALL windows together (not
    # per-window) so labels that are close in energy but belong to
    # different windows still get staggered relative to each other, not
    # just within their own window.
    if show_peak_labels and all_peak_points:
        all_peak_points.sort(key=lambda t: t[0])
        max_stagger_levels = 5

        levels = []
        last_x = None
        level = 0
        for x_p, _, _ in all_peak_points:
            if last_x is not None and (x_p - last_x) < close_threshold:
                level = (level + 1) % max_stagger_levels
            else:
                level = 0
            levels.append(level)
            last_x = x_p

        for (x_p, y_p, color), level in zip(all_peak_points, levels):
            label_y = y_p + (level + 1) * label_step
            ax.annotate(
                f"{x_p:.2f} eV",
                xy=(x_p, y_p),
                xytext=(x_p, label_y),
                textcoords="data",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.85,
                    linewidth=1.3,
                ),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
                zorder=6,
            )

        # Reserve enough headroom (based on the tallest stack of labels
        # actually used) so labels sit fully inside the axes -- not cut off
        # at the top -- instead of relying on autoscale to guess.
        max_level_used = max(levels)
        ax.set_ylim(
            mean_spec.min() - 0.05 * data_range,
            mean_spec.max() + (max_level_used + 2) * label_step,
        )

    if target_edge is not None:
        ax.axvline(
            target_edge,
            color="tab:orange",
            ls=":",
            lw=2,
            label=f"target_edge = {float(target_edge):.2f} eV",
        )

    ax.set_xlabel("Energy loss (eV)")
    ax.set_ylabel("Intensity")
    ax.set_title(
        "Energy windows for chemical maps"
        + (" + detected peaks" if detect_peaks else "")
        + (f"\n{_wrap_title(title_suffix, 85)}" if title_suffix else ""),
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    if display_energy_range is not None:
        d_lo, d_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(d_lo) and np.isfinite(d_hi) and d_lo < d_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
        # ax2 (if present) shares its x-axis with ax, so this zooms both.
        ax.set_xlim(d_lo, d_hi)

    if hover_annotations:
        # Plain matplotlib event handling -- no extra dependency (e.g.
        # mplcursors isn't installed here). Only fires with an interactive
        # backend (e.g. `%matplotlib widget` via ipympl); with the static
        # Agg backend this sets up harmlessly and the tooltip just never
        # appears since no mouse-move events occur.
        hover_annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.95),
            arrowprops=dict(arrowstyle="->"),
            fontsize=9,
            visible=False,
            zorder=10,
        )
        hover_windows = list(energy_windows)
        hover_peak_tolerance = 0.02 * (float(energy_axis[-1] - energy_axis[0]) or 1.0)

        def _on_hover(event):
            if event.inaxes is not ax or event.xdata is None:
                if hover_annot.get_visible():
                    hover_annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            idx = int(np.searchsorted(energy_axis, event.xdata))
            idx = min(max(idx, 0), len(energy_axis) - 1)
            if idx > 0 and abs(energy_axis[idx - 1] - event.xdata) < abs(
                energy_axis[idx] - event.xdata
            ):
                idx -= 1
            e_val = float(energy_axis[idx])
            i_val = float(mean_spec[idx])

            lines = [f"Energy: {e_val:.2f} eV", f"Intensity: {i_val:.2f}"]

            for lo, hi in hover_windows:
                if lo <= e_val <= hi:
                    lines.append(f"Window: [{lo:.2f}, {hi:.2f}] eV")
                    break

            all_peaks_flat = [p for peaks in detected.values() for p in peaks]
            if all_peaks_flat:
                nearest_peak = min(all_peaks_flat, key=lambda p: abs(p - e_val))
                if abs(nearest_peak - e_val) <= hover_peak_tolerance:
                    lines.append(f"Nearest detected peak: {nearest_peak:.2f} eV")

            hover_annot.xy = (e_val, i_val)
            hover_annot.set_text("\n".join(lines))
            if not hover_annot.get_visible():
                hover_annot.set_visible(True)
            fig.canvas.draw_idle()

        hover_callback_id = fig.canvas.mpl_connect("motion_notify_event", _on_hover)
        # Keep strong references so the callback and annotation aren't
        # garbage-collected once this function returns.
        fig._eels_hover_callback = hover_callback_id
        fig._eels_hover_annotation = hover_annot

    if enable_click_labels:
        # Manual click-to-pin coordinate labels on the main spectrum axes
        # only. Plain matplotlib event handling, same reasoning as the
        # hover tooltip above: harmless under a static/Agg backend (the
        # callbacks just never fire), needs an interactive backend (e.g.
        # `%matplotlib widget`) to actually respond to clicks/key presses.
        # Purely a display feature -- never touches detection or `detected`.
        pinned_labels: List[dict] = []
        click_hit_radius_px = 25

        def _pin_label_at(x_click, y_click):
            if click_label_snap_to_spectrum:
                idx = int(np.searchsorted(energy_axis, x_click))
                idx = min(max(idx, 0), len(energy_axis) - 1)
                if idx > 0 and abs(energy_axis[idx - 1] - x_click) < abs(
                    energy_axis[idx] - x_click
                ):
                    idx -= 1
                x_val = float(energy_axis[idx])
                y_val = float(mean_spec[idx])
            else:
                x_val = float(x_click)
                y_val = (
                    float(y_click)
                    if y_click is not None
                    else float(np.interp(x_click, energy_axis, mean_spec))
                )

            # Avoid overlap: pick the lowest stagger level not already used
            # by an existing pinned label close to this one in energy.
            used_levels = {
                e["level"] for e in pinned_labels if abs(e["x"] - x_val) < close_threshold
            }
            level = 0
            while level in used_levels:
                level += 1
            label_y = y_val + (level + 1) * label_step

            artist = ax.annotate(
                f"({x_val:.{click_label_precision}f}, {y_val:.{click_label_precision}f})",
                xy=(x_val, y_val),
                xytext=(x_val, label_y),
                textcoords="data",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9,
                    linewidth=1.1,
                ),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.7),
                zorder=8,
            )
            pinned_labels.append(
                {"x": x_val, "y": y_val, "level": level, "label_y": label_y, "artist": artist}
            )

            cur_bottom, cur_top = ax.get_ylim()
            needed_top = label_y + 0.05 * data_range
            if needed_top > cur_top:
                ax.set_ylim(cur_bottom, needed_top)

        def _delete_nearest_pinned_label(event):
            if not pinned_labels:
                return
            best_i, best_dist = None, None
            for i, entry in enumerate(pinned_labels):
                px, py = ax.transData.transform((entry["x"], entry["label_y"]))
                d = ((px - event.x) ** 2 + (py - event.y) ** 2) ** 0.5
                if best_dist is None or d < best_dist:
                    best_dist, best_i = d, i
            if best_i is not None and best_dist <= click_hit_radius_px:
                pinned_labels[best_i]["artist"].remove()
                del pinned_labels[best_i]

        def _on_click(event):
            if event.inaxes is not ax or event.xdata is None:
                return
            if event.button == 1:  # left click -> pin a new label
                _pin_label_at(event.xdata, event.ydata)
                fig.canvas.draw_idle()
            elif event.button == 3:  # right click -> delete nearest pinned label
                _delete_nearest_pinned_label(event)
                fig.canvas.draw_idle()

        def _on_key(event):
            if event.key == "c" and pinned_labels:
                for entry in pinned_labels:
                    entry["artist"].remove()
                pinned_labels.clear()
                fig.canvas.draw_idle()

        click_callback_id = fig.canvas.mpl_connect("button_press_event", _on_click)
        key_callback_id = fig.canvas.mpl_connect("key_press_event", _on_key)
        # Keep strong references so callbacks/state aren't garbage-collected
        # once this function returns.
        fig._eels_click_label_callbacks = [click_callback_id, key_callback_id]
        fig._eels_pinned_labels = pinned_labels

    if want_second_deriv_plot:
        ax2 = plot_axes[1]
        # Per-window, using each window's own smoothed curve (same one
        # plotted above and used by detect_peaks_in_range()) rather than one
        # fixed-sigma curve for the whole spectrum -- otherwise this
        # wouldn't match what detection actually saw for narrow vs. wide
        # windows.
        for i, (lo, hi) in enumerate(energy_windows):
            win_smoothed, win_mask = smoothed_by_window[(lo, hi)]
            second = np.gradient(np.gradient(win_smoothed, energy_axis), energy_axis)
            ax2.plot(energy_axis[win_mask], second[win_mask], color=colors[i % len(colors)], lw=1)
            ax2.axvspan(lo, hi, alpha=0.15, color=colors[i % len(colors)])
        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.set_xlabel("Energy loss (eV)")
        ax2.set_ylabel("2nd derivative")
        ax2.set_title("2nd derivative (shoulder-method debugging)")
        ax2.grid(True, alpha=0.3)

    if want_table:
        table_ax.axis("off")
        col_labels = ["Energy window (eV)", "Detected peak(s) (eV)"]
        cell_text = []
        for i, (lo, hi) in enumerate(energy_windows):
            peaks = detected.get((lo, hi), [])
            peak_str = ", ".join(f"{p:.2f}" for p in peaks) if peaks else "no peaks detected"
            cell_text.append([f"[{lo:.2f}, {hi:.2f}]", peak_str])

        tbl = table_ax.table(
            cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
        for i in range(len(energy_windows)):
            color = colors[i % len(colors)]
            for col in (0, 1):
                cell = tbl[(i + 1, col)]  # +1: row 0 is the header
                cell.set_edgecolor(color)
                cell.set_linewidth(2)
        table_ax.set_title("Peak coordinate summary", fontsize=10)

    plt.tight_layout()
    plt.show()

    return detected


def plot_background_fit_ranges(
    before,
    after,
    background,
    *,
    fit_windows,
    ranges=((None,),),
    n_pixels: int = 3,
    exclude_windows=(),
    title: str = "",
    show: bool = True,
):
    """
    Look at a background fit and its subtraction over SEVERAL energy ranges, to spot artifacts (over- or
    under-subtraction, ringing at a fit-window edge, a step where the model stops being valid, binning
    blocks, ...).

    Figure 1 -- one column per energy range in `ranges` (each a (lo, hi) eV tuple, or None = full axis):
        top    mean spectrum before subtraction (black) and the mean fitted background (red), y-scaled to
               what is visible in that range;
        bottom mean spectrum after subtraction, with the zero line, again y-scaled to the visible range.
        The fit windows are shaded green in every panel. For a one-sided subtraction (`background=None`) the
        bottom panel also shows the mean *without* the per-pixel clip at 0 (dashed): where the two differ, the
        clip is inflating the mean of the saved result (the mean of max(noise, 0) is > 0).
    Figure 2 -- the same three curves (before / background / after) for `n_pixels` individual pixels picked by
        quantile of the intensity inside the fit windows (dim / typical / bright), over the first range, so a
        problem that averages out in the mean but hits single pixels is visible.

    Parameters
    ----------
    before, after : Dataset3deels-like
        Spectra before and after the subtraction (same shape and energy axis).
    background : ndarray (row, col, energy) or None
        The fitted background cube (`subtract_background_two_sided(..., return_details=True)["background"]`).
        If None (one-sided subtraction: a single curve subtracted from every pixel, result clipped at 0) the
        curve is recovered from `before - after` as the per-channel maximum over pixels.
    fit_windows : sequence of (lo, hi)
        Windows the fit was made on (shaded green).
    exclude_windows : sequence of (lo, hi)
        Ranges kept out of the fit (shaded red).
    """
    energy = np.asarray(before.energy_axis, dtype=float)
    e_after = np.asarray(after.energy_axis, dtype=float)
    data = np.asarray(before.array, dtype=float)
    res = np.asarray(after.array, dtype=float)
    idx = np.abs(energy[None, :] - e_after[:, None]).argmin(
        axis=1
    )  # nearest channel (axes differ by rounding)
    clipped_curve = None
    if background is None:
        # No per-pixel background given (one-sided subtraction: ONE fitted curve for the whole cube, and the result is
        # clipped at 0 per pixel). before - after equals the curve only where a pixel was not clipped, and a clipped
        # pixel gives before - 0 = before < curve -- so the per-channel MAXIMUM over pixels recovers the curve.
        curve = np.full(energy.shape, np.nan)
        curve[idx] = (data[:, :, idx] - res).reshape(-1, len(idx)).max(axis=0)
        bg_full = np.broadcast_to(curve, data.shape).copy()
        clipped_curve = curve
    else:
        bg_full = np.asarray(background, dtype=float)
    res_full = np.full_like(data, np.nan)
    res_full[:, :, idx] = res

    mean_before = np.nanmean(data.reshape(-1, data.shape[2]), axis=0)
    mean_bg = np.nanmean(bg_full.reshape(-1, data.shape[2]), axis=0)
    mean_after = np.nanmean(res_full.reshape(-1, data.shape[2]), axis=0)

    def _rng(r):
        return (
            (float(energy[0]), float(energy[-1]))
            if r is None or r[0] is None
            else (float(r[0]), float(r[1]))
        )

    def _scale(ax, x, *ys, lo_zero=False):
        vis = (x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1])
        vals = np.concatenate([np.asarray(y)[vis][np.isfinite(np.asarray(y)[vis])] for y in ys])
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            if lo_zero:
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            pad = 0.06 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)

    figs = []
    if show:
        rr = [_rng(r if not isinstance(r, tuple) or len(r) == 2 else None) for r in ranges]
        fig, axes = plt.subplots(2, len(rr), figsize=(4.6 * len(rr) + 1.5, 7.5), squeeze=False)
        for j, (lo, hi) in enumerate(rr):
            for i in (0, 1):
                ax = axes[i, j]
                for wl, wh in fit_windows:
                    ax.axvspan(wl, wh, color="green", alpha=0.15)
                for wl, wh in exclude_windows:
                    ax.axvspan(wl, wh, color="red", alpha=0.10)
                ax.set_xlim(lo, hi)
                ax.grid(True, alpha=0.3)
            axes[0, j].plot(energy, mean_before, "k-", lw=1.2, label="before (mean)")
            axes[0, j].plot(energy, mean_bg, "r-", lw=1.2, label="fitted background (mean)")
            _scale(axes[0, j], energy, mean_before, mean_bg)
            axes[0, j].set_title(f"{lo:g} - {hi:g} eV")
            axes[1, j].plot(
                energy, mean_after, "b-", lw=1.2, label="after (mean of the saved result)"
            )
            ys = [mean_after]
            if (
                clipped_curve is not None
            ):  # show what the mean would be without the per-pixel clip at 0
                mean_unclipped = mean_before - clipped_curve
                axes[1, j].plot(
                    energy,
                    mean_unclipped,
                    "g--",
                    lw=1.0,
                    label="mean before - fitted curve (no clip)",
                )
                ys.append(mean_unclipped)
            axes[1, j].axhline(0, color="gray", ls="--", lw=1)
            _scale(axes[1, j], energy, *ys, lo_zero=True)
            axes[1, j].set_xlabel("Energy loss (eV)")
        axes[0, 0].set_ylabel("Intensity")
        axes[1, 0].set_ylabel("after subtraction")
        axes[0, 0].legend(fontsize=7, loc="best")
        axes[1, 0].legend(fontsize=7, loc="best")
        fig.suptitle(
            _wrap_title(
                f"{title}: background fit and subtraction, several ranges (green = fit windows, red = excluded from the fit)".strip(
                    " :"
                ),
                150,
            ),
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        plt.show()
        figs.append(fig)

        if n_pixels > 0:
            in_fit = np.zeros(energy.shape, dtype=bool)
            for wl, wh in fit_windows:
                in_fit |= (energy >= wl) & (energy <= wh)
            level = np.nanmean(data[:, :, in_fit], axis=-1).ravel()
            order = np.argsort(level)
            picks = [order[int(q * (len(order) - 1))] for q in np.linspace(0.1, 0.9, n_pixels)]
            lo, hi = rr[0]
            fig2, axs = plt.subplots(n_pixels, 2, figsize=(11, 3.0 * n_pixels), squeeze=False)
            for k, flat in enumerate(picks):
                r_, c_ = np.unravel_index(flat, data.shape[:2])
                for ax in axs[k]:
                    for wl, wh in fit_windows:
                        ax.axvspan(wl, wh, color="green", alpha=0.15)
                    for wl, wh in exclude_windows:
                        ax.axvspan(wl, wh, color="red", alpha=0.10)
                    ax.set_xlim(lo, hi)
                    ax.grid(True, alpha=0.3)
                axs[k, 0].plot(energy, data[r_, c_], "k-", lw=1, label="before")
                axs[k, 0].plot(energy, bg_full[r_, c_], "r-", lw=1, label="background")
                _scale(axs[k, 0], energy, data[r_, c_], bg_full[r_, c_])
                axs[k, 1].plot(energy, res_full[r_, c_], "b-", lw=1, label="after")
                axs[k, 1].axhline(0, color="gray", ls="--", lw=1)
                _scale(axs[k, 1], energy, res_full[r_, c_], lo_zero=True)
                axs[k, 0].set_ylabel(f"pixel ({r_}, {c_})")
                if k == 0:
                    axs[k, 0].legend(fontsize=7)
                    axs[k, 1].legend(fontsize=7)
            axs[-1, 0].set_xlabel("Energy loss (eV)")
            axs[-1, 1].set_xlabel("Energy loss (eV)")
            fig2.suptitle(
                _wrap_title(
                    f"{title}: single pixels (dim / typical / bright in the fit windows), {lo:g}-{hi:g} eV".strip(
                        " :"
                    ),
                    110,
                ),
                fontsize=10,
            )
            fig2.tight_layout(rect=(0, 0, 1, 0.93))
            plt.show()
            figs.append(fig2)
    return figs


def compare_background_methods(
    dataset,
    *,
    target_edge: float,
    pre_edge_range: Tuple[float, float],
    methods: Sequence[Tuple[str, str, int]] = (
        ("powerlaw", "powerlaw", 0),
        ("linear", "linear", 1),
        ("polynomial deg 2", "polynomial", 2),
        ("polynomial deg 3", "polynomial", 3),
    ),
    windows: Sequence[Tuple[float, float]] = (),
    markers: Sequence[float] = (),
    two_sided_windows: Optional[Sequence[Tuple[float, float]]] = None,
    display_energy_range: Optional[Tuple[float, float]] = None,
    display_intensity_range: Optional[Tuple[float, float]] = None,
    display_subtracted_range: Optional[Tuple[float, float]] = None,
    title: str = "",
    show: bool = True,
) -> Dict[str, Any]:
    """
    Side-by-side look at the pre-edge background fits available in
    `subtract_background_limited_preedge()` -- power law, linear and polynomial
    (degree 2, 3) -- on the same fit window, for one dataset.

    Top panel: the mean spectrum with each method's fitted background over the
    display range (the fit window is shaded). Bottom panel: mean spectrum minus
    each fitted background (unclipped, so an over-subtraction shows as negative
    values instead of being hidden by the clip to 0 that the real per-pixel
    subtraction applies), with the map `windows` shaded and any `markers` (eV)
    drawn as dotted lines. The fits are done on the mean spectrum with the same
    functional forms (`A*E^-r`, `polyfit`), so they show what every pixel's fit
    looks like on average.

    The returned table also reports, from the *real* per-pixel subtraction
    (`subtract_background_limited_preedge`), the largest fraction of pixels
    that end up exactly 0 in any of `windows` -- a flat-map check -- and the RMS
    of the fit residual inside the fit window.

    `two_sided_windows` (e.g. ``((0.5, 0.8), (2.4, 3.2))``) adds two more curves
    for comparison: ``A*E^-r + c`` and the curved ``A*E^-r*exp(s ln(E)^2) + c``
    fitted to the mean spectrum in windows on BOTH sides of the features, so the
    background is interpolated across them instead of extrapolated from one side
    (see `subtract_background_two_sided()` for the real per-pixel subtraction).

    Returns ``{"table": [...], "background": {label: array}, "figure": fig}``.
    """
    energy = np.asarray(dataset.energy_axis, dtype=float)
    mean_spec = np.asarray(dataset.calculate_mean_spectrum(), dtype=float)
    lo, hi = float(pre_edge_range[0]), float(pre_edge_range[1])
    fit = (energy >= lo) & (energy <= hi)
    if fit.sum() < 4:
        raise ValueError(
            f"pre_edge_range {pre_edge_range} contains only {int(fit.sum())} channels"
        )

    backgrounds, table = {}, []
    for label, method, degree in methods:
        row = dict(method=label, note="")
        try:
            if method == "powerlaw":
                ok = fit & (energy > 0) & (mean_spec > 0)
                slope, intercept = np.polyfit(np.log(energy[ok]), np.log(mean_spec[ok]), 1)
                bgc = np.where(energy > 0, np.exp(intercept) * np.abs(energy) ** slope, np.nan)
                row["note"] = f"A={np.exp(intercept):.3g}, r={-slope:.2f}"
            else:
                coef = np.polyfit(energy[fit], mean_spec[fit], int(degree))
                bgc = np.polyval(coef, energy)
            backgrounds[label] = bgc
            row["fit_rms"] = float(np.sqrt(np.nanmean((mean_spec[fit] - bgc[fit]) ** 2)))
        except Exception as exc:  # a failed fit must not stop the comparison
            row["note"] = f"fit failed: {type(exc).__name__}"
            row["fit_rms"] = float("nan")
        # real per-pixel subtraction -> flat-map check
        zero = float("nan")
        if windows:
            try:
                _validate_pre_edge_window(lo, hi, energy, target_edge)
                with contextlib.redirect_stdout(io.StringIO()):
                    sub = dataset.subtract_background_limited_preedge(
                        target_edge=target_edge,
                        pre_edge_range=(lo, hi),
                        method=method,
                        polynomial_degree=int(degree) if method == "polynomial" else 2,
                        show=False,
                    )
                e2, a2 = np.asarray(sub.energy_axis), np.asarray(sub.array)
                if a2.size == 0 or not np.isfinite(a2).any():
                    raise ValueError("background subtraction produced an empty / all-NaN result")
                zero = 0.0
                for wl, wh in windows:
                    k = (e2 >= wl) & (e2 <= wh)
                    if k.any():
                        zero = max(zero, float(np.mean(a2[:, :, k].sum(axis=-1) == 0)))
            except Exception as exc:
                row["note"] += f" subtraction failed: {type(exc).__name__}"
        row["worst_zero_pixel_fraction"] = zero
        table.append(row)

    if two_sided_windows:
        sel = np.zeros_like(energy, dtype=bool)
        for wl, wh in two_sided_windows:
            sel |= (energy >= wl) & (energy <= wh)
        sel &= energy > 0
        forms = (
            (
                "two-sided power law + const",
                lambda x, a, r, c: a * x ** (-r) + c,
                (float(np.max(mean_spec[sel])), 2.0, 0.0),
                1,
            ),
            (
                "two-sided curved power law + const",
                lambda x, a, r, s_, c: a * x ** (-r) * np.exp(s_ * np.log(x) ** 2) + c,
                (float(np.max(mean_spec[sel])), 2.0, 0.0, 0.0),
                2,
            ),
        )
        for label, f, p0, _ in forms:
            try:
                popt, _ = curve_fit(f, energy[sel], mean_spec[sel], p0=p0, maxfev=50000)
                backgrounds[label] = np.where(
                    energy > 0, f(np.abs(energy) + (energy <= 0), *popt), np.nan
                )
                table.append(
                    dict(
                        method=label,
                        note=f"windows {tuple(tuple(w) for w in two_sided_windows)}, params={np.round(popt, 3).tolist()} (mean spectrum only)",
                        fit_rms=float(
                            np.sqrt(np.nanmean((mean_spec[sel] - backgrounds[label][sel]) ** 2))
                        ),
                        worst_zero_pixel_fraction=float("nan"),
                    )
                )
            except Exception as exc:
                table.append(
                    dict(
                        method=label,
                        note=f"fit failed: {type(exc).__name__}",
                        fit_rms=float("nan"),
                        worst_zero_pixel_fraction=float("nan"),
                    )
                )

    fig = None
    if show:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
        xlim = display_energy_range or (float(energy[0]), float(energy[-1]))
        ax1.plot(energy, mean_spec, "k-", lw=1.4, label="mean spectrum")
        ax1.axvspan(lo, hi, color="green", alpha=0.15, label=f"fit window {lo:g}-{hi:g} eV")
        for label, bgc in backgrounds.items():
            ax1.plot(energy, bgc, lw=1.3, label=f"{label} fit")
        ax1.set_xlim(*xlim)
        if display_intensity_range is not None:
            ax1.set_ylim(*display_intensity_range)
        ax1.set_ylabel("Intensity")
        ax1.set_title(f"{title} background fits on the mean spectrum".strip())
        ax1.legend(fontsize=8, loc="best")
        ax1.grid(True, alpha=0.3)

        for label, bgc in backgrounds.items():
            ax2.plot(energy, mean_spec - bgc, lw=1.3, label=label)
        ax2.axhline(0, color="gray", lw=1, ls="--")
        for wl, wh in windows:
            ax2.axvspan(wl, wh, color="orange", alpha=0.12)
        for m in markers:
            ax2.axvline(m, color="k", ls=":", lw=0.9)
        ax2.axvspan(lo, hi, color="green", alpha=0.10)
        for wl, wh in two_sided_windows or ():
            ax1.axvspan(wl, wh, color="purple", alpha=0.08)
            ax2.axvspan(wl, wh, color="purple", alpha=0.08)
        if display_subtracted_range is not None:
            ax2.set_ylim(*display_subtracted_range)
        ax2.set_xlabel("Energy loss (eV)")
        ax2.set_ylabel("mean - fitted background")
        ax2.set_title("Subtracted mean spectrum per method (unclipped; orange = map windows)")
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    print(f"{'method':18s} {'fit RMS':>9s} {'worst zero-pixel frac in windows':>34s}  note")
    for r in table:
        print(
            f"{r['method']:18s} {r['fit_rms']:9.4g} {r['worst_zero_pixel_fraction']:34.2f}  {r['note']}"
        )
    return {"table": table, "background": backgrounds, "figure": fig}


def plot_map_diagnostics(
    maps,
    *,
    adf=None,
    thickness_map=None,
    n_perm: int = 200,
    seed: int = 0,
    title: str = "",
    show: bool = True,
):
    """
    Print and plot :func:`summarize_map_diagnostics` for a ``{(lo, hi): 2D map}`` dict:
    a table, plus (if ``show``) a bar chart of each window's Moran's I z-score (dashed
    line at the one-sided p = 0.05 threshold, z = 1.645) and, where given, its
    correlation with the ADF / thickness map.

    Returns ``(rows, figure_or_None)`` with the rows of
    :func:`summarize_map_diagnostics`. Call directly, not as a bound method.
    """
    windows = list(maps.keys())
    rows = summarize_map_diagnostics(
        maps, adf=adf, thickness_map=thickness_map, n_perm=n_perm, seed=seed
    )

    print(
        f"{'window (eV)':>14s} {'moran I':>9s} {'z':>7s} {'p':>8s}"
        + ("  corr(ADF) r [p]" if adf is not None else "")
        + ("  corr(thickness) r [p]" if thickness_map is not None else "")
    )
    for r in rows:
        line = f"{r['window'][0]:6.2f}-{r['window'][1]:<6.2f} {r['moran_i']:9.3f} {r['z']:7.2f} {r['p']:8.3g}"
        if "corr_adf" in r:
            line += f"   {r['corr_adf']['r']:+.2f} [{r['corr_adf']['p']:.2g}]"
        if "corr_thickness" in r:
            line += f"       {r['corr_thickness']['r']:+.2f} [{r['corr_thickness']['p']:.2g}]"
        print(line)
    sig = [r for r in rows if np.isfinite(r["z"]) and r["z"] >= 1.645]
    if sig:
        print(
            "  -> spatially coherent (z >= 1.645, one-sided p <= 0.05): "
            + ", ".join(f"[{r['window'][0]:g}, {r['window'][1]:g}] eV" for r in sig)
        )
    else:
        print(
            "  -> no window shows spatial structure clearly above what shuffling the same pixels would give."
        )

    fig = None
    if show:
        n_extra = int(adf is not None) + int(thickness_map is not None)
        fig, axes = plt.subplots(1, 1 + n_extra, figsize=(4.5 * (1 + n_extra), 4), squeeze=False)
        axes = axes[0]
        labels = [f"{w[0]:g}-{w[1]:g}" for w in windows]
        colors = [
            "tab:green" if (np.isfinite(r["z"]) and r["z"] >= 1.645) else "0.6" for r in rows
        ]
        axes[0].bar(range(len(rows)), [r["z"] for r in rows], color=colors)
        axes[0].axhline(1.645, color="red", ls="--", lw=1, label="p=0.05 (one-sided)")
        axes[0].set_xticks(range(len(rows)))
        axes[0].set_xticklabels(labels, fontsize=8, rotation=20)
        axes[0].set_ylabel("Moran's I z-score")
        axes[0].set_title("Spatial coherence per window")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="y", alpha=0.3)
        j = 1
        for key, ylabel in (
            ("corr_adf", "r with ADF"),
            ("corr_thickness", "r with thickness map"),
        ):
            if any(key in r for r in rows):
                vals = [r[key]["r"] for r in rows]
                sigc = ["tab:red" if r[key]["p"] < 0.05 else "0.6" for r in rows]
                axes[j].bar(range(len(rows)), vals, color=sigc)
                axes[j].axhline(0, color="gray", lw=0.8)
                axes[j].set_xticks(range(len(rows)))
                axes[j].set_xticklabels(labels, fontsize=8, rotation=20)
                axes[j].set_ylabel(ylabel)
                axes[j].set_title(f"{ylabel} (red = p<0.05)")
                axes[j].grid(True, axis="y", alpha=0.3)
                j += 1
        if title:
            fig.suptitle(_wrap_title(title, 130), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92 if title else 1))
        plt.show()
    return rows, fig


def plot_despike_preview(
    energy_axis,
    mean_spec,
    mean_spec_corrected,
    ranges,
    display_energy_range=None,
    display_intensity_range=None,
    display_residual_range=None,
):
    """Three panels: the mean spectrum with the spike ranges shaded, original vs.
    despiked, and the correction residual (zero outside the ranges). Used by
    :meth:`Dataset3deels.despike`; returns the figure."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    colors = ("red", "orange")

    # 1. original spectrum with chosen spike ranges shaded
    axes[0].plot(energy_axis, mean_spec, "k-", lw=1.2, label="Mean HL spectrum (original)")
    for i, (lo, hi) in enumerate(ranges):
        axes[0].axvspan(
            lo,
            hi,
            color=colors[i % len(colors)],
            alpha=0.3,
            label=f"spike {i + 1}: [{lo:.2f}, {hi:.2f}] eV",
        )
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("1. Original spectrum with chosen spike ranges")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. original vs. despiked mean spectrum overlaid
    axes[1].plot(energy_axis, mean_spec, "k-", lw=1.2, alpha=0.55, label="original")
    axes[1].plot(energy_axis, mean_spec_corrected, "b-", lw=1.4, label="despiked")
    for lo, hi in ranges:
        axes[1].axvspan(lo, hi, color="red", alpha=0.12)
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("2. Original vs. despiked mean spectrum")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. correction residual -- should be exactly zero outside the chosen
    # ranges, which is the direct visual proof that nothing else changed
    axes[2].plot(energy_axis, mean_spec_corrected - mean_spec, "m-", lw=1.0)
    axes[2].axhline(0, color="gray", ls="--", lw=1)
    for lo, hi in ranges:
        axes[2].axvspan(lo, hi, color="red", alpha=0.12)
    axes[2].set_xlabel("Energy loss (eV)")
    axes[2].set_ylabel("corrected - original")
    axes[2].set_title("3. Correction residual (should be zero outside the shaded ranges)")
    axes[2].grid(True, alpha=0.3)

    if display_energy_range is not None:
        e_lo, e_hi = float(display_energy_range[0]), float(display_energy_range[1])
        if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi):
            raise ValueError(
                f"display_energy_range must be (lo, hi) with lo < hi, got {display_energy_range!r}"
            )
        axes[0].set_xlim(e_lo, e_hi)  # axes[1], axes[2] share this x-axis (sharex=True)

    if display_intensity_range is not None:
        i_lo, i_hi = float(display_intensity_range[0]), float(display_intensity_range[1])
        if not (np.isfinite(i_lo) and np.isfinite(i_hi) and i_lo < i_hi):
            raise ValueError(
                f"display_intensity_range must be (lo, hi) with lo < hi, "
                f"got {display_intensity_range!r}"
            )
        axes[0].set_ylim(i_lo, i_hi)
        axes[1].set_ylim(i_lo, i_hi)

    if display_residual_range is not None:
        r_lo, r_hi = float(display_residual_range[0]), float(display_residual_range[1])
        if not (np.isfinite(r_lo) and np.isfinite(r_hi) and r_lo < r_hi):
            raise ValueError(
                f"display_residual_range must be (lo, hi) with lo < hi, "
                f"got {display_residual_range!r}"
            )
        axes[2].set_ylim(r_lo, r_hi)

    plt.tight_layout()
    plt.show()
    return fig
