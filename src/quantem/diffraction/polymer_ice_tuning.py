"""Interactive tuning and diagnostics for the polymer ice flagger.

Plotting and inspection helpers for choosing IceFlaggerParams against a real
dataset. They live here rather than in a notebook so that any notebook, anywhere,
can import them without carrying a copy:

    from quantem.diffraction.polymer_ice_tuning import (
        selection_box,          # tune the sharpness ceilings, returns updated params
        orientation_histograms, # did the flagger take ice and leave your peaks?
        ice_split_widget,       # inspect kept / removed peaks interactively
        probe_peaks,            # per-peak FWHM at one scan position, with the cuts
    )

Every function takes the BraggPeaksPolymer and an IceFlaggerParams explicitly and
returns its results; none mutate ``bp``. They are free functions rather than more
methods on BraggPeaksPolymer, which is already large -- the algorithms live in
``polymer_ice``, and this module is only the interactive layer over them.

``ice_split_widget`` needs the optional ``quantem.widget`` package; its import is
deferred into the function so this module stays importable without it.
"""

from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from quantem.diffraction.polymer_ice import measure_peak_widths, sharpness_mask

__all__ = [
    "IcePeakView",
    "ice_split_widget",
    "orientation_histograms",
    "peak_widths",
    "probe_peaks",
    "selection_box",
]

_WIDTH_CACHE: dict[tuple, dict] = {}


def peak_widths(bp, params, *, use_cache=True):
    """Radial/annular FWHM for every peak in the ice q band, across the scan.

    Cached on the q band and intensity field, which are the only inputs that
    change the measurement, so sweeping the ceilings costs nothing.
    """
    key = (id(bp), params.q_target_invA, params.dq_invA, params.intensity_field,
           params.sharpness_window_theta_deg, params.sharpness_smooth_sigma_bins)
    if use_cache and key in _WIDTH_CACHE:
        return _WIDTH_CACHE[key]
    print("measuring peak widths over the q band (cached until the q band changes)...")
    widths = bp.measure_ice_peak_widths(params=params)
    _WIDTH_CACHE[key] = widths
    return widths


def selection_box(bp, params, *, max_width_r_invA=None, max_width_theta_deg=None,
                  fig_path=None, use_cache=True):
    """Preview the sharpness gate as a box in (width, intensity), and return the params.

    Assign the result back to your params object; nothing is mutated in place::

        params_ice_flagger = selection_box(bp, params_ice_flagger,
                                           max_width_theta_deg=8.0)

    The ceilings are ANDed, and an unset axis is not gated. Ice is annularly
    sharp -- compact dots and radial streaks alike -- while polymer at the same q
    is an annularly broad arc, so ``max_width_theta_deg`` is normally the only one
    you need; a radial ceiling mostly rejects the streaks.
    """
    tuned = dataclasses.replace(
        params,
        max_width_r_invA=max_width_r_invA,
        max_width_theta_deg=max_width_theta_deg,
    )
    widths = peak_widths(bp, tuned, use_cache=use_cache)
    width_r = widths["width_r_invA"]
    width_theta = widths["width_theta_deg"]
    intensity = widths["intensity"]

    # A percentile cutoff is resolved per scan inside detect_ice, so preview it as
    # "no floor" rather than guessing the value it will take.
    floor = tuned.intensity_cutoff if tuned.intensity_cutoff is not None else -np.inf
    floor_label = "none" if not np.isfinite(floor) else f"{floor:.4g}"
    selected = sharpness_mask(width_r, width_theta, tuned) & (intensity >= floor)
    total = len(width_r)
    print(f"box selects {selected.sum()} of {total} q-band peaks "
          f"({100 * selected.sum() / max(total, 1):.1f}%)")

    # Where each population sits. If a width's two rows look alike, that axis does
    # not separate the populations and its ceiling buys nothing.
    bright, dim = intensity >= floor, intensity < floor
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for label, values, digits in (("radial FWHM (1/Å)", width_r, 4),
                                  ("annular FWHM (deg)", width_theta, 2)):
        print(f"{label} quantiles {quantiles}")
        for group_label, group in ((f"intensity >= {floor_label}", bright),
                                   (f"intensity <  {floor_label}", dim)):
            finite = group & np.isfinite(values)
            print(f"  {group_label:<22} (n={finite.sum():>7}):",
                  np.round(np.quantile(values[finite], quantiles), digits)
                  if finite.any() else "none")

    fig, ((ax_r, ax_t), (ax_rt, ax_map)) = plt.subplots(2, 2, figsize=(12.5, 9))

    def width_panel(ax, values, ceiling, xlabel):
        finite = np.isfinite(values)
        hist = ax.hist2d(values[finite], intensity[finite], bins=(80, 80),
                         norm=LogNorm(), cmap="magma")
        fig.colorbar(hist[3], ax=ax, label="count (log)")
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        bottom = y_lo if not np.isfinite(floor) else floor
        if np.isfinite(floor):
            ax.axhline(floor, color="cyan", ls="--", lw=2)
        edge = x_hi if ceiling is None else ceiling
        if ceiling is not None:
            ax.axvline(ceiling, color="cyan", ls="--", lw=2)
        ax.add_patch(plt.Rectangle((x_lo, bottom), edge - x_lo, y_hi - bottom,
                                   facecolor="cyan", alpha=0.15, edgecolor="none"))
        ax.set(xlabel=xlabel, ylabel=tuned.intensity_field,
               title=f"{xlabel} vs intensity"
                     + ("" if ceiling is not None else "  (no ceiling set)"))

    width_panel(ax_r, width_r, tuned.max_width_r_invA, "radial FWHM (1/Å)")
    width_panel(ax_t, width_theta, tuned.max_width_theta_deg, "annular FWHM (deg)")

    # The plane the gate cuts in. Ceilings are ANDed, so the keep-region is the
    # corner under both; with no radial ceiling it is the band below the annular one.
    plane = np.isfinite(width_r) & np.isfinite(width_theta) & bright
    hist = ax_rt.hist2d(width_r[plane], width_theta[plane], bins=(80, 80),
                        norm=LogNorm(), cmap="magma")
    fig.colorbar(hist[3], ax=ax_rt, label="count (log)")
    x_lo, x_hi = ax_rt.get_xlim()
    y_lo, y_hi = ax_rt.get_ylim()
    r_edge = x_hi if tuned.max_width_r_invA is None else tuned.max_width_r_invA
    t_edge = y_hi if tuned.max_width_theta_deg is None else tuned.max_width_theta_deg
    if tuned.max_width_r_invA is not None:
        ax_rt.axvline(tuned.max_width_r_invA, color="cyan", ls="--", lw=2)
    if tuned.max_width_theta_deg is not None:
        ax_rt.axhline(tuned.max_width_theta_deg, color="cyan", ls="--", lw=2)
    ax_rt.add_patch(plt.Rectangle((x_lo, y_lo), r_edge - x_lo, t_edge - y_lo,
                                  facecolor="cyan", alpha=0.15, edgecolor="none"))
    ax_rt.set(xlabel="radial FWHM (1/Å)", ylabel="annular FWHM (deg)",
              title=f"radial vs annular FWHM  (intensity >= {floor_label})")

    # Ice should be compact blobs on the scan, not sprinkled everywhere.
    selected_map = np.zeros(bp.polar_peaks.shape, dtype=int)
    np.add.at(selected_map, (widths["iy"][selected], widths["ix"][selected]), 1)
    image = ax_map.imshow(selected_map, cmap="inferno", interpolation="nearest")
    fig.colorbar(image, ax=ax_map, label="selected peaks per position")
    ax_map.set_title("selected peaks across the scan")
    ax_map.axis("off")

    fig.suptitle(f"selection box — {selected.sum()} of {total} q-band peaks", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if fig_path is not None:
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.show()

    print(f"\nreturned params: max_width_r_invA={tuned.max_width_r_invA}, "
          f"max_width_theta_deg={tuned.max_width_theta_deg}, "
          f"intensity_cutoff={tuned.intensity_cutoff}")
    print(f"detect_ice can flag at most these {selected.sum()} peaks; the six-fold "
          "alignment then drops those not sitting in an aligned pattern.")
    return tuned


def orientation_histograms(bp, ice_result, params, *, theta_step_deg=2.0,
                           orientation_offset_degrees=0.0, fig_path=None):
    """Ice-band orientation histograms for all / kept / removed peaks.

    Run BEFORE the filter cell: the removed set needs the unfiltered peaks.
    Crystalline ice is a sharp compact blob in the scan image and a narrow spike
    in the angular histogram; polymer signal is diffuse and broad. Returns the
    three histograms as a dict, keyed 'all peaks' / 'kept (ice removed)' /
    'removed (ice)'.
    """
    q_window = (params.q_target_invA - params.dq_invA,
                params.q_target_invA + params.dq_invA)

    def histogram(polar_peaks, peak_intensities):
        # make_orientation_histogram reads bp.polar_peaks / bp.peak_intensities,
        # so swap the subset in, measure, and restore. upsample_factor=1 keeps
        # this cheap and at scan resolution.
        saved = (bp.polar_peaks, bp.peak_intensities)
        bp.polar_peaks, bp.peak_intensities = polar_peaks, peak_intensities
        try:
            return bp.make_orientation_histogram(
                radial_ranges=np.array([q_window]),
                upsample_factor=1, theta_step_deg=theta_step_deg,
                sigma_x=0.0, sigma_y=0.0, sigma_theta=3.0,
                orientation_offset_degrees=orientation_offset_degrees,
                normalize_intensity_image=False, normalize_intensity_stack=False,
                progress_bar=False)[0]
        finally:
            bp.polar_peaks, bp.peak_intensities = saved

    hists = {
        "all peaks": histogram(bp.polar_peaks, bp.peak_intensities),
        "kept (ice removed)": histogram(ice_result.filter(bp.polar_peaks),
                                        ice_result.filter(bp.peak_intensities)),
        "removed (ice)": histogram(ice_result.filter(bp.polar_peaks, invert=True),
                                   ice_result.filter(bp.peak_intensities, invert=True)),
    }

    theta = np.arange(0, 180, theta_step_deg)
    vmax = max(float(h.max()) for h in hists.values()) or 1.0
    ymax = max(float(h.sum(axis=(0, 1)).max()) for h in hists.values()) or 1.0
    fig, axes = plt.subplots(len(hists), 2, figsize=(9, 3.2 * len(hists)),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    for (label, hist), (ax_map, ax_hist) in zip(hists.items(), np.atleast_2d(axes)):
        image = ax_map.imshow(hist.max(axis=2), cmap="inferno", vmin=0, vmax=vmax,
                              interpolation="nearest")
        ax_map.set_title(f"{label} — max over theta", fontsize=9)
        ax_map.axis("off")
        fig.colorbar(image, ax=ax_map, fraction=0.046)
        ax_hist.plot(theta, hist.sum(axis=(0, 1)), lw=1.2)
        ax_hist.set(xlim=(0, 180), ylim=(0, 1.05 * ymax), xlabel="theta (deg)",
                    ylabel="summed intensity")
        ax_hist.set_title(f"{label} — angular histogram", fontsize=9)
    fig.suptitle(f"Ice band q = {q_window[0]:.3f}–{q_window[1]:.3f} 1/Å "
                 f"(d = {1 / q_window[1]:.2f}–{1 / q_window[0]:.2f} Å)", fontsize=10)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.show()
    return hists


class IcePeakView:
    """Stand-in for ``bp`` exposing one side of the ice split.

    ``show_polymer_4DSTEM`` is duck-typed and reads the peak vectors live on every
    cursor move, so the subset must stay visible for the widget's lifetime and
    cannot be restored after construction. Delegating instead of assigning onto
    ``bp`` keeps ``bp`` itself pristine.
    """

    def __init__(self, base, cartesian, intensities, polar):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "peak_coordinates_cartesian", cartesian)
        object.__setattr__(self, "peak_intensities", intensities)
        object.__setattr__(self, "polar_peaks", polar)

    def __getattr__(self, name):
        # Reached only for attributes not set above, i.e. everything but the peaks.
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        # The widget's "Save settings" writes to its source object; keep those
        # writes on the view so they never land on bp.
        object.__setattr__(self, name, value)


def ice_split_widget(bp, ice_result, params, *, view="removed", ice_hists=None,
                     map_view="match", **widget_kwargs):
    """Open the interactive viewer on one side of the ice split. ``bp`` is untouched.

    ``view`` is 'removed' | 'kept' | 'all'. ``map_view`` follows ``view`` by
    default; pin it to one of the same names to hold a fixed backdrop while
    toggling the overlay. Pass ``ice_hists`` from :func:`orientation_histograms`
    to use the ice-band orientation image as the context map.
    """
    from quantem.widget import show_polymer_4DSTEM

    peaks = (bp.peak_coordinates_cartesian, bp.peak_intensities, bp.polar_peaks)
    if view == "all":
        subset = peaks
    elif view in ("kept", "removed"):
        subset = tuple(ice_result.filter(v, invert=view == "removed") for v in peaks)
    else:
        raise ValueError(f"view must be 'removed', 'kept' or 'all'; got {view!r}")

    map_key = {"all": "all peaks", "kept": "kept (ice removed)", "removed": "removed (ice)"}
    resolved = view if map_view == "match" else map_view
    if resolved not in map_key:
        raise ValueError(f"map_view must be 'match', 'removed', 'kept' or 'all'; "
                         f"got {map_view!r}")
    if ice_hists is not None:
        context_map = ice_hists[map_key[resolved]].max(axis=2)
        map_label = f"{resolved} orientation"
    else:
        context_map = ice_result.flagged_peaks_count_map.astype(float)
        map_label = "flagged count — pass ice_hists for the orientation map"

    def count(vector):
        total = 0
        for iy in range(vector.shape[0]):
            for ix in range(vector.shape[1]):
                rows = vector[iy, ix].array
                if rows is not None:
                    total += len(rows)
        return total

    print(f"showing '{view}' peaks: {count(subset[0])} of {count(peaks[0])} total "
          "(bp itself is untouched)")
    # Pass the view, not bp: bp.show_widget() would bind bp as self.
    return show_polymer_4DSTEM(
        IcePeakView(bp, *subset),
        intensity_map=context_map,
        title=f"ice split — {view} peaks (map: {map_label})",
        show_inset=True,
        sharpness_params=params,
        **widget_kwargs,
    )


def probe_peaks(bp, params, ry, rx, *, n_show=4, q_band_only=True):
    """Measure every peak at one scan position and plot the cuts behind each FWHM.

    Use it to check the automated width against a peak you can see:
    ``probe_peaks(bp, params_ice_flagger, ice_widget.pos_ry, ice_widget.pos_rx)``.
    Returns the per-peak arrays as a dict.
    """
    polar_data = bp.polar_data
    image = np.asarray(polar_data["intensity"])[ry, rx]
    r_axis = np.asarray(polar_data["r_invA"])[:, 0]
    theta_axis = np.asarray(polar_data["theta"])[0, :]

    peaks = np.asarray(bp.polar_peaks[ry, rx].array)
    intensities = np.asarray(bp.peak_intensities[ry, rx].array)
    q = peaks[:, bp.polar_peaks.fields.index("r_invA")]
    theta = peaks[:, bp.polar_peaks.fields.index("theta")]
    values = intensities[:, bp.peak_intensities.fields.index(params.intensity_field)]

    if q_band_only:
        keep = np.abs(q - params.q_target_invA) <= params.dq_invA
        q, theta, values = q[keep], theta[keep], values[keep]
    if not len(q):
        raise ValueError(f"no peaks at ({ry},{rx})"
                         + (" in the q band" if q_band_only else ""))

    width_r, width_theta = measure_peak_widths(q, theta, image, r_axis, theta_axis,
                                               params=params)
    gate_r, gate_t = params.max_width_r_invA, params.max_width_theta_deg
    print(f"position ({ry}, {rx}) -- {len(q)} peaks"
          f"{' in the q band' if q_band_only else ''}\n")
    print(f"{'#':>3} {'q 1/Å':>8} {'d Å':>7} {'theta°':>8} {'intens':>8} "
          f"{'radFWHM':>9} {'annFWHM':>9}  gate")
    for k in np.argsort(-values):
        pass_r = gate_r is None or width_r[k] <= gate_r
        pass_t = gate_t is None or width_theta[k] <= gate_t
        verdict = ("sharp" if (pass_r and pass_t) else
                   "broad-r" if not pass_r and pass_t else
                   "broad-t" if pass_r else "broad-rt")
        print(f"{k:>3} {q[k]:>8.4f} {1 / max(q[k], 1e-9):>7.2f} "
              f"{np.rad2deg(theta[k]):>8.1f} {values[k]:>8.4f} "
              f"{width_r[k]:>9.4f} {width_theta[k]:>9.1f}  {verdict}")

    order = np.argsort(-values)[:n_show]
    theta_step = float(np.rad2deg(theta_axis[1] - theta_axis[0]))
    r_step = float(r_axis[1] - r_axis[0])
    fig, axes = plt.subplots(len(order), 2, figsize=(11, 2.6 * len(order)), squeeze=False)
    for row, k in enumerate(order):
        r_bin = int(np.clip(round((q[k] - r_axis[0]) / r_step), 0, len(r_axis) - 1))
        theta_bin = int(round(np.mod(np.rad2deg(theta[k]), 180.0) / theta_step))
        theta_bin %= len(theta_axis)

        ax = axes[row][0]
        ax.plot(r_axis, image[:, theta_bin], lw=1)
        ax.axvline(q[k], color="tab:red", ls=":")
        ax.axvspan(q[k] - width_r[k] / 2, q[k] + width_r[k] / 2, color="tab:red", alpha=0.15)
        ax.set(xlim=(q[k] - 6 * max(width_r[k], r_step), q[k] + 6 * max(width_r[k], r_step)),
               xlabel="q (1/Å)", ylabel="intensity",
               title=f"peak {k}: radial cut, FWHM={width_r[k]:.4f} 1/Å")

        ax = axes[row][1]
        center = np.mod(np.rad2deg(theta[k]), 180.0)
        ax.plot(np.rad2deg(theta_axis), image[r_bin, :], lw=1)
        ax.axvline(center, color="tab:red", ls=":")
        ax.axvspan(center - width_theta[k] / 2, center + width_theta[k] / 2,
                   color="tab:red", alpha=0.15)
        ax.set(xlim=(0, 180), xlabel="theta (deg)", ylabel="intensity",
               title=f"peak {k}: annular cut, FWHM={width_theta[k]:.1f}°")
    fig.tight_layout()
    plt.show()
    return {"q_invA": q, "theta_rad": theta, "intensity": values,
            "width_r_invA": width_r, "width_theta_deg": width_theta}
