import contextlib
import csv
import io
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import pearsonr


def _read_csv_without_preamble(path: Union[Path, str]) -> list[str]:
    """Return CSV lines starting after leading citation/comment/blank lines."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.lower().startswith("citation:")
        ):
            return lines[index:]

    raise ValueError(f"{path} does not contain a CSV header")


def _parse_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


def load_xray_lines_database(path: Union[Path, str]) -> dict[str, dict[str, dict[str, float]]]:
    """Load X-ray lines CSV into the legacy element->line metadata mapping."""
    elements: dict[str, dict[str, dict[str, float]]] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}

    reader = csv.DictReader(_read_csv_without_preamble(path))
    for row in reader:
        element = str(row.get("element", "")).strip()
        line_name = str(row.get("line", "")).strip()
        if not element or not line_name:
            continue

        energy_kev = _parse_float(row, ("energy_keV", "energy (keV)", "energy"))
        if energy_kev is None:
            energy_ev = _parse_float(row, ("energy_eV", "energy (eV)"))
            if energy_ev is None:
                continue
            energy_kev = energy_ev / 1000.0

        weight = _parse_float(row, ("weight", "relative_intensity"))
        if weight is None:
            weight = 0.0

        element_lines = elements.setdefault(element, {})
        key = (element, line_name)
        if line_name in element_lines:
            duplicate_counts[key] = duplicate_counts.get(key, 1) + 1
            line_name = f"{line_name}__{duplicate_counts[key]}"

        element_lines[line_name] = {
            "energy (keV)": energy_kev,
            "weight": weight,
        }

    return elements


def load_eels_edges_database(path: Union[Path, str]) -> dict[str, dict[str, dict[str, object]]]:
    """Load EELS edge CSV into the legacy element->edge metadata mapping."""
    elements: dict[str, dict[str, dict[str, object]]] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}

    reader = csv.DictReader(_read_csv_without_preamble(path))
    fieldnames = set(reader.fieldnames or [])
    required_columns = ("symbol", "edge_label", "edge_energy_eV")
    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required EELS edge columns: {', '.join(missing_columns)}"
        )

    for row in reader:
        element_symbol = str(row.get("symbol", "")).strip()
        if not element_symbol:
            continue

        energy_ev = _parse_float(row, ("edge_energy_eV", "onset_energy (eV)", "energy_eV"))
        if energy_ev is None:
            continue

        edge_label = str(row.get("edge_label", "")).strip()
        element_edges = elements.setdefault(element_symbol, {})
        edge_name = f"{energy_ev:g} eV"
        key = (element_symbol, edge_name)
        if edge_name in element_edges:
            duplicate_counts[key] = duplicate_counts.get(key, 1) + 1
            edge_name = f"{edge_name}__{duplicate_counts[key]}"

        edge_info: dict[str, object] = {
            "onset_energy (eV)": energy_ev,
        }
        if edge_label:
            edge_info["edge_label"] = edge_label

        atomic_number = _parse_float(row, ("atomic_number",))
        if atomic_number is not None:
            edge_info["atomic_number"] = (
                int(atomic_number) if atomic_number.is_integer() else atomic_number
            )

        element_name = str(row.get("element", "")).strip()
        if element_name:
            edge_info["element"] = element_name

        element_edges[edge_name] = edge_info

    return elements


def spatial_coherence(map2d: np.ndarray, *, n_perm: int = 200, seed: int = 0) -> dict:
    """
    Is there more spatial structure in a 2D map than the same pixel values arranged
    at random would give -- i.e. do neighbouring pixels actually agree with each
    other -- or does it just look like noise however colourful it is?

    Uses Moran's I (4-neighbour rook adjacency: up/down/left/right, no diagonals, no
    wraparound) -- +1 = neighbours are as similar as possible (patches), 0 = no
    relationship, -1 = neighbours alternate (checkerboard). Significance is a
    permutation test: the same pixel values are shuffled to random positions
    ``n_perm`` times, Moran's I recomputed each time, and the real value is compared
    against that null distribution (``z``, and a one-sided ``p`` -- real spatial
    structure raises I, it essentially never lowers it, so only the upper tail is
    tested). NaN pixels are excluded from every term.

    Parameters
    ----------
    map2d : ndarray
        2D map to test. NaN pixels are excluded.
    n_perm : int, optional
        Number of permutations for the null distribution. Default 200.
    seed : int, optional
        Seed for the permutation RNG. Default 0.

    Returns
    -------
    dict
        ``{"moran_i", "z", "p", "n_valid"}``. ``p`` below ~0.05 with ``n_valid`` in
        at least the hundreds is decent evidence of real coherence; with only a
        handful of valid pixels this test has little power and ``p`` should not be
        trusted either way.
    """
    x = np.asarray(map2d, dtype=float)
    valid = np.isfinite(x)
    n = int(valid.sum())
    if n < 4:
        return dict(moran_i=float("nan"), z=float("nan"), p=float("nan"), n_valid=n)
    mean = float(x[valid].mean())
    xm = np.where(valid, x - mean, 0.0)

    def _moran(xm_arr, valid_arr):
        num, w = 0.0, 0
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a = xm_arr
            b = np.roll(xm_arr, shift=(dy, dx), axis=(0, 1))
            mask = valid_arr & np.roll(valid_arr, shift=(dy, dx), axis=(0, 1))
            if dy == 1:
                mask[0, :] = False
            elif dy == -1:
                mask[-1, :] = False
            if dx == 1:
                mask[:, 0] = False
            elif dx == -1:
                mask[:, -1] = False
            num += float(np.sum(a[mask] * b[mask]))
            w += int(mask.sum())
        denom = float(np.sum(xm_arr[valid_arr] ** 2))
        return (n / w) * (num / denom) if w > 0 and denom > 0 else float("nan")

    i_obs = _moran(xm, valid)
    rng = np.random.default_rng(seed)
    flat = xm[valid]
    null = np.empty(n_perm)
    xm_p = xm.copy()
    idx = np.flatnonzero(valid.ravel())
    for k in range(n_perm):
        shuffled = flat.copy()
        rng.shuffle(shuffled)
        xm_p_flat = xm_p.ravel()
        xm_p_flat[idx] = shuffled
        null[k] = _moran(xm_p_flat.reshape(xm_p.shape), valid)
    null = null[np.isfinite(null)]
    if null.size < 2 or not np.isfinite(i_obs):
        return dict(moran_i=i_obs, z=float("nan"), p=float("nan"), n_valid=n)
    z = float((i_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    p = float((np.sum(null >= i_obs) + 1) / (null.size + 1))  # one-sided, +1/+1 avoids p=0
    return dict(moran_i=float(i_obs), z=z, p=p, n_valid=n)


def correlate_with_reference(map2d: np.ndarray, reference2d: np.ndarray) -> dict:
    """
    Pearson correlation between a map and a reference image of the same shape (e.g.
    an ADF or a thickness map) -- a quick check of whether a pattern is simply
    following material contrast (more/thicker material -> more signal everywhere)
    rather than showing something specific to that map. NaN pixels in either image
    are excluded pairwise.

    Parameters
    ----------
    map2d, reference2d : ndarray
        2D maps of the same shape.

    Returns
    -------
    dict
        ``{"r", "p", "n"}``. ``r`` near 0 does not by itself prove the map is real
        signal (it could still be pure noise) -- pair with :func:`spatial_coherence`.

    Raises
    ------
    ValueError
        If ``map2d`` and ``reference2d`` have different shapes.
    """
    a = np.asarray(map2d, dtype=float).ravel()
    b = np.asarray(reference2d, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: map2d {np.shape(map2d)} vs reference2d {np.shape(reference2d)}"
        )
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return dict(r=float("nan"), p=float("nan"), n=n)
    r, p = pearsonr(a[mask], b[mask])
    return dict(r=float(r), p=float(p), n=n)


def detect_peaks_in_range(
    spectrum_data,
    energy_axis,
    energy_range: Tuple[float, float],
    method: str = "auto",
    **kwargs,
) -> List[float]:
    """
    Find peaks in ``spectrum_data`` (1D intensity) restricted to ``energy_range`` =
    (lo_eV, hi_eV) of ``energy_axis`` (1D, same length as ``spectrum_data``).

    Strict selection policy -- it is better to return no peak than to label noise as
    one:

    1. The *whole* spectrum is Gaussian-smoothed once (``smooth_window`` sigma; by
       default 4% of this window's own point count, floored at 2 -- see
       ``smooth_window_fraction``; pass 0/None to disable), then sliced to the
       window. Smoothing the full trace first (not the windowed slice) avoids
       fabricating a bump at the window's own edges, and scaling the default per
       window means a narrow window isn't smoothed as aggressively as a wide one.
    2. A robust noise estimate is derived from the (raw - smoothed) residuals over
       the whole spectrum: ``noise_sigma = 1.4826 * MAD``.
    3. A "local maximum" must be a real ``find_peaks()`` maximum on the smoothed
       data with prominence >= ``noise_prominence_factor * noise_sigma`` (default
       factor 5) unless you pass an explicit ``prominence`` (or ``height``/
       ``distance``) yourself, AND it must sit at least ``edge_margin_eV`` away from
       both window edges (default 5% of the window width).
    4. A "shoulder" candidate (2nd-derivative sign change) is accepted only if ALL
       of: the window has no accepted local maximum; the 1st derivative has a
       genuine local extremum there (its magnitude is within 20% of the largest
       |1st derivative| in its neighborhood, not just a wobble); the intensity
       change across that neighborhood clears ``min_shoulder_snr * noise_sigma``
       (default factor 3, or an explicit ``zero_crossing_threshold``); the slope
       there isn't ~0 (not a flat region); it's outside ``edge_margin_eV`` of both
       window edges; and nearby crossings within ``min_shoulder_separation_eV``
       (default 5% of the window width) are merged into one.

    Parameters
    ----------
    spectrum_data : ndarray
        1D intensity, same length as ``energy_axis``.
    energy_axis : ndarray
        1D energy axis (eV).
    energy_range : (float, float)
        (lo, hi) eV window to restrict peak detection to.
    method : str, optional
        - ``"max"`` : the single global maximum of the smoothed data -- always
          returns exactly one point.
        - ``"local_maxima"`` : only accepted local maxima (rule 3 above).
        - ``"shoulder"`` : only accepted genuine shoulders (rule 4 above) -- if a
          window already has an accepted local maximum, this returns ``[]`` rather
          than double-counting it as a shoulder too.
        - ``"auto"`` (default) : local maxima first; if (and only if) none are
          accepted, evaluate shoulder candidates as a fallback (and say so). When a
          local maximum IS accepted, shoulder candidates are never even computed.
    **kwargs
        All optional, all keyword-only:

        - ``smooth_window`` : Gaussian sigma (in points) applied to the whole
          spectrum before detection. Overrides the adaptive default below with one
          fixed value for every window.
        - ``smooth_window_fraction`` : fraction of this window's own point count
          used as ``smooth_window`` when you don't supply one (floored at 2
          points). Default 0.04 (4%) -- a wider window gets proportionally more
          smoothing, a narrower one less.
        - ``prominence``, ``height``, ``distance`` : passed straight to
          ``scipy.signal.find_peaks()`` for local maxima; override the
          noise-derived default prominence when given.
        - ``noise_prominence_factor`` : multiplier on the robust noise estimate
          used as the default ``prominence`` when you don't supply one. Default 5.
        - ``min_shoulder_snr`` : multiplier on the robust noise estimate a
          shoulder's local intensity change must clear. Default 3.
        - ``min_shoulder_separation_eV`` : merge distance (eV) for nearby shoulder
          crossings. Default 5% of the window width.
        - ``edge_margin_eV`` : reject any maximum/shoulder within this many eV of
          either window edge. Default 5% of the window width.
        - ``zero_crossing_threshold`` : explicit override for the shoulder
          intensity threshold (in place of ``min_shoulder_snr * noise_sigma``).

    Returns
    -------
    List of peak energies (eV), sorted ascending.

    Notes
    -----
    Edge cases (empty range, no/NaN data in range, unknown method) print a one-line
    warning and return ``[]`` rather than raising, since this is a preview aid, not
    a hard requirement.
    """
    lo, hi = energy_range
    if not (lo < hi):
        print(f"  -> peak detection skipped for window ({lo}, {hi}) eV: empty range (lo >= hi).")
        return []

    spectrum_data = np.asarray(spectrum_data, dtype=float)
    energy_axis = np.asarray(energy_axis, dtype=float)
    mask = (energy_axis >= lo) & (energy_axis <= hi)
    if not np.any(mask):
        print(f"  -> peak detection skipped for window ({lo}, {hi}) eV: no data in range.")
        return []

    E = energy_axis[mask]
    if np.all(np.isnan(spectrum_data[mask])):
        print(f"  -> peak detection skipped for window ({lo}, {hi}) eV: all values are NaN.")
        return []

    # Smooth the FULL spectrum once, then slice. A single fixed sigma across every
    # window is wrong: a narrow window and a wide window need very different
    # amounts of smoothing relative to their own width, so the default scales with
    # *this window's* point count: smooth_window_fraction (default 4%) of the
    # window's points, floored at 2. Pass an explicit smooth_window to override
    # this and use one fixed sigma for every window instead.
    n_window_points = len(E)
    smooth_window_fraction = kwargs.get("smooth_window_fraction", 0.04)
    default_sigma = max(2, round(smooth_window_fraction * n_window_points))
    sigma = kwargs.get("smooth_window", default_sigma)
    full_smoothed = gaussian_filter1d(spectrum_data, sigma=sigma) if sigma else spectrum_data
    I_win = full_smoothed[mask]
    n = len(I_win)

    # Robust noise estimate: MAD of (raw - smoothed), whole trace.
    residuals = spectrum_data - full_smoothed
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    noise_sigma = 1.4826 * mad

    edge_margin_eV = kwargs.get("edge_margin_eV", 0.05 * (hi - lo))

    def _within_edge_margin(e):
        return (e - lo) < edge_margin_eV or (hi - e) < edge_margin_eV

    # Local maxima: real find_peaks() maxima, noise-scaled prominence.
    def _local_maxima_candidates():
        default_prominence = kwargs.get("noise_prominence_factor", 5.0) * noise_sigma
        prominence = kwargs.get(
            "prominence", default_prominence if default_prominence > 0 else None
        )
        idx, _ = find_peaks(
            I_win,
            prominence=prominence,
            height=kwargs.get("height"),
            distance=kwargs.get("distance"),
        )
        return sorted(float(E[i]) for i in idx if not _within_edge_margin(float(E[i])))

    # Shoulders: strict multi-condition acceptance.
    def _shoulder_candidates():
        first = np.gradient(I_win, E)
        second = np.gradient(first, E)

        min_shoulder_snr = kwargs.get("min_shoulder_snr", 3.0)
        min_sep = kwargs.get("min_shoulder_separation_eV", 0.05 * (hi - lo))
        intensity_threshold = kwargs.get(
            "zero_crossing_threshold",
            min_shoulder_snr * noise_sigma if noise_sigma > 0 else 0.0,
        )
        half_win = max(2, int(round(sigma)) if sigma else 2)
        half_win = min(half_win, max(1, n // 4))

        accepted: List[float] = []
        for i in range(1, n):
            s0, s1 = second[i - 1], second[i]
            if s0 * s1 >= 0:
                continue  # not a sign change -> not a candidate at all

            frac = abs(s0) / (abs(s0) + abs(s1)) if (abs(s0) + abs(s1)) > 0 else 0.5
            e_cross = float(E[i - 1] + frac * (E[i] - E[i - 1]))
            if _within_edge_margin(e_cross):
                continue

            i0, i1 = max(0, i - half_win), min(n - 1, i + half_win)

            # (a) the 1st derivative must have a genuine local extremum here (near
            # the largest |slope| in its neighborhood) -- a real shoulder is the
            # steepest point of a transition, not a random wobble that happens to
            # cross zero in the 2nd derivative.
            local_first_abs = np.abs(first[i0 : i1 + 1])
            slope_here = abs(first[i])
            if local_first_abs.max() <= 0 or slope_here < 0.8 * local_first_abs.max():
                continue

            # (b) the intensity change across the neighborhood must clear the
            # noise-derived threshold.
            if abs(I_win[i1] - I_win[i0]) < intensity_threshold:
                continue

            # (c) reject a ~flat region (negligible slope).
            if slope_here < 1e-12:
                continue

            accepted.append(e_cross)

        # merge nearby crossings so one physical shoulder -> one energy
        accepted.sort()
        merged: List[float] = []
        for c in accepted:
            if not merged or (c - merged[-1]) > min_sep:
                merged.append(c)
        return merged

    if method == "max":
        return [float(E[int(np.nanargmax(I_win))])]

    if method == "local_maxima":
        return _local_maxima_candidates()

    if method == "shoulder":
        # A window with a genuine local maximum isn't a "shoulder" window.
        if _local_maxima_candidates():
            return []
        return _shoulder_candidates()

    if method == "auto":
        local_maxima = _local_maxima_candidates()
        if local_maxima:
            return local_maxima  # shoulder candidates are not even computed
        peaks = _shoulder_candidates()
        if peaks:
            print(f"  -> window ({lo}, {hi}) eV: using shoulder method, no local maxima found.")
        return peaks

    raise ValueError(
        f"Unknown method {method!r}; choose 'max', 'local_maxima', 'shoulder', or 'auto'."
    )


def crop_energy_range(dataset, lo_eV: float, hi_eV: float):
    """
    Crop a 3D spectroscopy dataset to the energy channels inside [lo_eV, hi_eV].

    A plain, non-interactive crop: no prompts, no plot. Retained channel values are
    unchanged; ``dataset`` itself is not modified.

    Parameters
    ----------
    lo_eV, hi_eV : float
        Energy bounds (eV), lo_eV < hi_eV.

    Returns
    -------
    Dataset3dspectroscopy
        Cropped copy of ``dataset``.

    Raises
    ------
    ValueError
        If ``lo_eV >= hi_eV``, or the requested range does not overlap the
        dataset's own energy axis at all. (Contrast with
        :meth:`fit_windows_inside_axis`, which silently moves a window that
        doesn't fit -- this is for a deliberate, explicit range, so it errors
        instead.)
    """
    e = np.asarray(dataset.energy_axis, dtype=float)
    lo_eV, hi_eV = float(lo_eV), float(hi_eV)
    if not (lo_eV < hi_eV):
        raise ValueError(f"lo_eV must be < hi_eV, got ({lo_eV}, {hi_eV})")
    if hi_eV < e[0] or lo_eV > e[-1]:
        raise ValueError(
            f"[{lo_eV}, {hi_eV}] eV does not overlap the dataset's axis "
            f"[{e[0]:.3f}, {e[-1]:.3f}] eV"
        )
    first_idx = min(max(int(np.searchsorted(e, lo_eV, side="left")), 0), len(e) - 1)
    last_idx = min(max(int(np.searchsorted(e, hi_eV, side="right")) - 1, 0), len(e) - 1)
    out = dataset.crop(crop_widths=((first_idx, last_idx + 1),), axes=(2,))
    out.name = (
        f"{dataset.name} (cropped to [{float(e[first_idx]):.2f}, {float(e[last_idx]):.2f}] eV)"
    )
    return out


def fit_windows_inside_axis(
    dataset, windows: List[Tuple[float, float]], min_width_eV: float = 0.25
) -> List[Tuple[float, float]]:
    """
    Make a list of (lo, hi) background-fit windows usable on ``dataset``'s energy axis.

    Windows are clipped to the axis; a window entirely below the axis start (e.g. a
    0.5-0.8 eV tail window on an axis that begins at 1.2 eV) is replaced by a
    same-width window at the start of the axis; a window above the axis end is
    dropped; windows narrower than ``min_width_eV`` after clipping are dropped.

    Parameters
    ----------
    windows : list of (float, float)
        Candidate (lo, hi) windows in eV.
    min_width_eV : float, optional
        Minimum retained window width after clipping. Default 0.25.

    Returns
    -------
    list of (float, float)
        Adapted windows, sorted ascending.
    """
    e = np.asarray(dataset.energy_axis, dtype=float)
    a, b = float(e[0]) + 0.02, float(e[-1])
    out = []
    for lo, hi in windows:
        width = float(hi) - float(lo)
        if hi <= a:
            lo2, hi2 = a + 0.03, a + 0.03 + width
        elif lo >= b:
            continue
        else:
            lo2, hi2 = max(float(lo), a), min(float(hi), b)
        if hi2 - lo2 >= min_width_eV:
            out.append((round(lo2, 3), round(hi2, 3)))
    return sorted(out)


def suggest_spike_ranges(
    dataset,
    *,
    threshold_sigma: float = 12.0,
    min_sigma: float = 4.0,
    medfilt_channels: int = 11,
    sigma_window_channels: Optional[int] = None,
    merge_gap_channels: int = 4,
    pad_channels: int = 2,
    max_width_eV: float = 1.0,
    exclude_zlp_eV: float = 1.0,
    edge_guard_channels: int = 15,
    max_ranges: int = 8,
) -> List[Tuple[float, float]]:
    """
    Suggest narrow detector-spike intervals from a dataset's mean spectrum.

    The mean spectrum minus its own running median (``medfilt_channels`` wide)
    leaves only structure narrower than that window. Its noise level sigma is the
    robust global 1.4826 x MAD of that residual (or, with ``sigma_window_channels``,
    a running median of |residual|, floored at 30 % of the global value -- this
    tracks noise that varies along the spectrum but also makes real structure in
    smooth, low-noise stretches look like spikes, so it is off by default).
    Channels with |residual| > ``min_sigma`` * sigma are grouped (gaps up to
    ``merge_gap_channels`` merge -- a spike usually comes with an undershoot), and
    a group is reported only if

    * its strongest channel exceeds ``threshold_sigma`` * sigma (high enough to
      leave real but weaker narrow features alone, e.g. a ~9 sigma, 0.1-0.2 eV
      wide peak),
    * it is narrower than ``max_width_eV`` (a broad peak is not a spike),
    * it does not overlap the zero-loss peak (|E| <= ``exclude_zlp_eV``), and
    * it is not within ``edge_guard_channels`` of either end of the axis, where
      the running median is unreliable.

    Groups are padded by ``pad_channels`` and overlapping ranges merged. If more
    than ``max_ranges`` survive, the spectrum is almost certainly noise or real
    structure rather than a few spikes, so a warning is issued and nothing is
    returned.

    This is a *suggestion*: the mean spectrum only shows artifacts that sit at the
    same channel in most pixels, and interpolating across a spike also removes any
    real signal under it -- check the ranges on the spectrum before using them.

    Returns
    -------
    list of (float, float)
        (lo_eV, hi_eV) spike intervals, ascending.
    """
    energy = np.asarray(dataset.energy_axis, dtype=float)
    mean_spec = np.asarray(dataset.calculate_mean_spectrum(), dtype=float)
    resid = mean_spec - median_filter(mean_spec, size=int(medfilt_channels) | 1, mode="nearest")
    dev = np.abs(resid - np.median(resid))
    sigma_global = 1.4826 * float(np.median(dev))
    if not np.isfinite(sigma_global) or sigma_global <= 0:
        return []
    if sigma_window_channels is None:
        sigma = np.full_like(resid, sigma_global)
    else:
        sigma = np.maximum(
            1.4826 * median_filter(dev, size=int(sigma_window_channels) | 1, mode="nearest"),
            0.3 * sigma_global,
        )

    flagged = np.where(np.abs(resid) > min_sigma * sigma)[0]
    groups: List[List[int]] = []
    for i in flagged:
        if groups and i - groups[-1][-1] <= merge_gap_channels:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])

    n = len(energy)
    ranges: List[Tuple[float, float]] = []
    for g in groups:
        if np.max(np.abs(resid[g]) / sigma[g]) < threshold_sigma:
            continue
        lo_i, hi_i = g[0], g[-1]
        if lo_i < edge_guard_channels or hi_i > n - 1 - edge_guard_channels:
            continue
        if energy[hi_i] - energy[lo_i] > max_width_eV:
            continue
        if energy[hi_i] >= -exclude_zlp_eV and energy[lo_i] <= exclude_zlp_eV:
            continue
        lo_i, hi_i = max(lo_i - pad_channels, 1), min(hi_i + pad_channels, n - 2)
        lo, hi = float(energy[lo_i]), float(energy[hi_i])
        if ranges and lo <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
        else:
            ranges.append((lo, hi))
    if len(ranges) > max_ranges:
        warnings.warn(
            f"suggest_spike_ranges: {len(ranges)} candidate ranges (> max_ranges={max_ranges}) -- "
            "this looks like noisy or structured data rather than a few detector spikes; "
            "returning no suggestions."
        )
        return []
    return [(round(lo, 3), round(hi, 3)) for lo, hi in ranges]


# Elements whose edges auto_hl_config() scores by default, with the shell label used
# for reporting. Onsets come from the EELS edge database (eels_edges.csv).
HL_CANDIDATE_EDGES = {
    "Si": "L2,3",
    "P": "L2,3",
    "S": "L2,3",
    "Cl": "L2,3",
    "K": "L2,3",
    "Ca": "L2,3",
    "C": "K",
    "N": "K",
    "O": "K",
    "F": "K",
}


def _candidate_edge_onsets(
    element_info: dict, candidate_edges: dict, min_onset_eV: float
) -> dict[str, float]:
    """{"<symbol> <shell>": onset_eV} -- lowest "major" database edge >= min_onset_eV per element."""
    onsets = {}
    for symbol, shell in candidate_edges.items():
        majors = [
            float(info["onset_energy (eV)"])
            for info in element_info.get(symbol, {}).values()
            if info.get("edge_label") == "major"
            and float(info["onset_energy (eV)"]) >= min_onset_eV
        ]
        if not majors:
            continue
        onset = min(majors)
        name = f"{symbol} {shell}" if shell else f"{symbol} {onset:g} eV"
        onsets[name] = onset
    return onsets


def auto_hl_config(
    dataset,
    *,
    low_loss_windows: Sequence[Tuple[float, float]] = ((0.8, 1.4), (1.4, 1.8), (1.8, 2.2)),
    candidate_edges: Optional[dict] = None,
    report_only: Sequence[str] = ("K L2,3",),
) -> dict[str, Any]:
    """
    Pick background / energy-window settings for a high-loss cube whose energy
    range is not known in advance (carbon K, oxygen K, S/P/Cl L2,3, or even a
    low-loss-range window).

    * If the usable axis starts below 50 eV the "HL" cube is really a low-loss
      range spectrum: pre-edge just above the axis start, windows =
      ``low_loss_windows`` that fit inside the axis (else generic ones).
    * Otherwise each candidate edge whose onset sits at least 12 eV inside the
      axis is scored by its jump ratio in the mean spectrum: median post-edge
      signal (onset + 3 to + 15 eV) over a power law extrapolated from the
      pre-edge (onset - 32 to - 8 eV). The highest eligible score wins. Pre-edge
      = (onset - 32, onset - 8) clipped to the axis, windows = (onset, onset +
      3.5) and (onset + 5, onset + 16).

    Parameters
    ----------
    low_loss_windows : sequence of (float, float), optional
        Energy windows to use when the cube turns out to be a low-loss range.
    candidate_edges : dict, optional
        ``{element_symbol: shell_label}`` to score. Each onset is the element's
        lowest "major" edge >= 50 eV in the dataset's edge database
        (:meth:`load_element_info`). A ``None`` shell label names the edge
        ``"<symbol> <onset> eV"``. Default :data:`HL_CANDIDATE_EDGES`.
    report_only : sequence of str, optional
        Edge names that are scored and reported in ``scores`` but never picked.
        Default ``("K L2,3",)``: its pre-edge window reaches into carbon K's rise
        whenever both are in the axis, which flipped the winner on a noise-level
        margin on real data (1.086 vs 1.083) with no evidence of potassium.

    Returns
    -------
    dict
        ``kind`` ("core-loss" or "low-loss-range"), ``edge`` (name or None),
        ``target_edge`` (eV), ``pre_edge_range``, ``windows``, ``scores``
        ({edge: jump ratio} for every scored candidate), ``valid_start_eV``
        (start of the usable axis after skipping dead leading channels) and
        ``confident`` (False when the picked edge's jump ratio is < 1.04 -- the
        pick is then just the least-bad candidate).
    """
    e = np.asarray(dataset.energy_axis, dtype=float)
    m = np.asarray(dataset.calculate_mean_spectrum(), dtype=float)
    hi_ax = float(e[-1])
    # Dead leading channels (mean ~0 for the first few eV) ruin a pre-edge fit, so
    # the usable axis starts at the first channel from which the next 10 all carry
    # at least 30 % of the typical level.
    typical = float(np.median(m[: min(len(m), 300)]))
    if typical > 0:
        ok = np.array([bool(np.all(m[i : i + 10] > 0.3 * typical)) for i in range(len(m) - 9)])
    else:
        ok = np.array([True])
    lead = int(np.argmax(ok)) if ok.any() else 0
    lo_ax = float(e[lead])

    if lo_ax < 50.0:
        s0 = max(lo_ax, 0.0)
        pre = (round(max(0.15, s0 + 0.05), 3), round(max(0.60, s0 + 0.50), 3))
        wins = [w for w in low_loss_windows if w[0] >= pre[1] and w[1] <= hi_ax]
        if not wins:
            wins = [
                (round(pre[1] + 0.4, 3), round(pre[1] + 2.4, 3)),
                (round(pre[1] + 2.4, 3), round(pre[1] + 6.4, 3)),
            ]
        return dict(
            kind="low-loss-range",
            edge=None,
            target_edge=round(pre[1] + 0.5, 3),
            pre_edge_range=pre,
            windows=tuple(wins),
            scores={},
            confident=True,
            valid_start_eV=lo_ax,
        )

    def _jump(onset):
        pre = (e >= max(lo_ax + 0.05, onset - 32.0)) & (e <= onset - 8.0) & (e > 0) & (m > 0)
        post = (e >= onset + 3.0) & (e <= onset + 15.0) & (m > 0)
        if pre.sum() < 20 or post.sum() < 10:
            return np.nan
        slope, intercept = np.polyfit(np.log(e[pre]), np.log(m[pre]), 1)
        pred = np.exp(intercept + slope * np.log(e[post]))
        return float(np.median(m[post] / pred))

    onsets = _candidate_edge_onsets(
        dataset.load_element_info() or {},
        HL_CANDIDATE_EDGES if candidate_edges is None else candidate_edges,
        min_onset_eV=50.0,
    )
    scores = {name: _jump(on) for name, on in onsets.items() if lo_ax + 12 <= on <= hi_ax - 12}
    scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    eligible = {k: v for k, v in scores.items() if k not in report_only}
    if eligible:
        edge = max(eligible, key=eligible.get)
        onset = onsets[edge]
    else:  # no candidate edge inside the window: treat the middle of the axis as the "edge"
        edge, onset = None, 0.5 * (lo_ax + hi_ax)
    pre = (round(max(lo_ax + 0.05, onset - 32.0), 3), round(onset - 8.0, 3))
    wins = (
        (round(onset, 3), round(onset + 3.5, 3)),
        (round(onset + 5.0, 3), round(onset + 16.0, 3)),
    )
    return dict(
        kind="core-loss",
        edge=edge,
        target_edge=float(onset),
        pre_edge_range=pre,
        windows=wins,
        scores={k: round(v, 3) for k, v in scores.items()},
        confident=bool(eligible) and max(eligible.values()) >= 1.04,
        valid_start_eV=lo_ax,
    )


def auto_ll_pre_edge_range(
    dataset,
    *,
    target_edge: float,
    windows: Sequence[Tuple[float, float]],
    method: str = "powerlaw",
    fallback_methods: Sequence[str] = ("polynomial",),
    candidates: Optional[Sequence[Tuple[float, float]]] = None,
    max_zero_frac: float = 0.05,
):
    """
    Choose a low-loss background pre-edge window (and, if needed, method) that
    does not over-subtract.

    A power law fitted to the steep ZLP tail can overshoot the real spectrum; the
    subtraction then clips to exactly 0 and the maps come out flat and empty. This
    tries each candidate window (clipped to the energy axis and kept below
    ``target_edge`` and ``windows``; default candidates start close to the ZLP
    tail and move away from it) with ``method``, then with each of
    ``fallback_methods``, running :meth:`subtract_background_limited_preedge`, and
    scores every attempt by the largest fraction of pixels that are exactly 0 in
    any of ``windows`` after subtraction. The first attempt at or below
    ``max_zero_frac`` wins; otherwise the one with the lowest score.

    Parameters
    ----------
    target_edge : float
        Edge onset (eV) passed to the background subtraction.
    windows : sequence of (float, float)
        Energy windows (eV) the maps will be integrated over.
    method : str, optional
        First background method to try. Default "powerlaw".
    fallback_methods : sequence of str, optional
        Methods tried, in order, if no window passes with ``method``.
    candidates : sequence of (float, float), optional
        Pre-edge windows to try. Default: a few windows just above the ZLP tail.
    max_zero_frac : float, optional
        Largest acceptable fraction of exactly-zero pixels in any window.

    Returns
    -------
    ((float, float), str, list)
        The chosen pre-edge window, the method used, and a table of
        ``(method, window, worst_zero_fraction)`` for every attempt (a failed or
        all-NaN fit scores 1.0).
    """
    energy = np.asarray(dataset.energy_axis, dtype=float)
    lo_ax, hi_ax = float(energy[0]), float(energy[-1])
    win_lo = max(
        min(w[0] for w in windows), lo_ax + 0.5
    )  # windows below the axis start are clipped anyway
    hi_limit = min(float(target_edge) - 0.05, win_lo - 0.02, hi_ax)
    if candidates is None:
        candidates = [(0.15, 0.60), (0.20, 0.70), (0.30, 0.70), (0.40, 0.75), (0.45, 0.78)]
        if lo_ax > 0.1:  # the axis itself starts above the ZLP-tail region
            candidates = [
                (lo_ax + 0.05, lo_ax + 0.45),
                (lo_ax + 0.1, lo_ax + 0.6),
                (lo_ax + 0.2, lo_ax + 0.8),
            ]
    valid = [(max(lo, lo_ax + 0.02), min(hi, hi_limit)) for lo, hi in candidates]
    valid = [(round(lo, 3), round(hi, 3)) for lo, hi in valid if hi - lo >= 0.15]
    if not valid:
        valid = [(round(lo_ax + 0.05, 3), round(lo_ax + 0.5, 3))]

    table = []
    for meth in (method, *fallback_methods):
        for win in valid:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    arr = np.asarray(
                        dataset.subtract_background_limited_preedge(
                            target_edge=target_edge,
                            pre_edge_range=win,
                            method=meth,
                            show=False,
                            return_dataset=False,
                        )
                    )
            except Exception:
                table.append((meth, win, 1.0))
                continue
            if not np.isfinite(arr).any():
                table.append((meth, win, 1.0))
                continue
            worst = 0.0
            for lo, hi in windows:
                k = (energy >= lo) & (energy <= hi)
                if k.any():
                    worst = max(worst, float(np.mean(arr[:, :, k].sum(axis=-1) == 0)))
            table.append((meth, win, round(worst, 3)))
            if worst <= max_zero_frac:
                return win, meth, table
    best = min(table, key=lambda t: t[2])
    return best[1], best[0], table


def robustness_check_near_zlp(
    dataset,
    peaks: List[dict],
    energy_range: Tuple[float, float] = (-1.9, 0.85),
    n_refits: int = 4,
    seed: int = 42,
    max_spread_eV: float = 0.02,
) -> dict:
    """
    Fit :meth:`fit_near_zlp_transitions` once, then refit ``n_refits`` more times
    from perturbed initial guesses (same bounds) and report the spread of each
    component's fitted center.

    Any center that pins at a bound, or whose spread across refits exceeds
    ``max_spread_eV``, is flagged -- only a center that lands in the same place
    regardless of the starting guess is a confident detection.

    Parameters
    ----------
    peaks : list of dict
        Peak specs as accepted by :meth:`fit_near_zlp_transitions` (each with
        ``name``, ``center_guess``, ``center_bounds``, ``width_guess``,
        ``width_bounds``).
    energy_range : (float, float), optional
        Energy window passed to :meth:`fit_near_zlp_transitions`.
    n_refits : int, optional
        Number of additional perturbed-initial-guess refits.
    seed : int, optional
        RNG seed for the perturbations.
    max_spread_eV : float, optional
        Center spread above which a component is flagged. Default 0.02.

    Returns
    -------
    dict
        The base (unperturbed) fit result, plus a ``"robustness"`` entry:
        ``{name: {"centers": [...], "spread": float, "flags": [...]}}``.
    """
    rng = np.random.default_rng(seed)
    base = dataset.fit_near_zlp_transitions(peaks, energy_range=energy_range)
    centers_by_name = {name: [comp["center"]] for name, comp in base["components"].items()}

    for _ in range(n_refits):
        trial_peaks = []
        for peak in peaks:
            lo, hi = peak["center_bounds"]
            wlo, whi = peak["width_bounds"]
            p = dict(peak)
            p["center_guess"] = float(lo + rng.uniform(0.2, 0.8) * (hi - lo))
            p["width_guess"] = float(wlo + rng.uniform(0.2, 0.8) * (whi - wlo))
            trial_peaks.append(p)
        try:
            trial = dataset.fit_near_zlp_transitions(
                trial_peaks,
                energy_range=energy_range,
                zlp_center_guess=float(rng.uniform(-0.08, 0.08)),
                zlp_width_guess=float(rng.uniform(0.02, 0.9)),
            )
        except RuntimeError as exc:
            print(f"  refit failed to converge: {exc}")
            continue
        for name, comp in trial["components"].items():
            centers_by_name[name].append(comp["center"])

    print("=== Fitted components (main fit) ===")
    for name, comp in base["components"].items():
        print(
            f"{name:12s} shape={comp['shape']:12s} "
            f"center={comp['center']:.4f}+/-{comp['center_stderr']:.4f} eV  "
            f"fwhm={comp['fwhm']:.4f}+/-{comp['fwhm_stderr']:.4f} eV  "
            f"amp={comp['amplitude']:.4g}  area={comp['area']:.4g}"
        )
    print(f"residual std: {np.std(base['residual']):.4f}")

    print(f"\n--- Robustness: center spread across {n_refits} perturbed refits ---")
    bounds_by_name = {p["name"]: p["center_bounds"] for p in peaks}
    robustness = {}
    for name, centers in centers_by_name.items():
        centers = np.array(centers)
        spread = float(centers.max() - centers.min())
        flags = []
        bounds = bounds_by_name.get(name)
        if bounds is not None:
            tol = 1e-3
            if abs(centers.min() - bounds[0]) < tol or abs(centers.max() - bounds[1]) < tol:
                flags.append("PINNED AT BOUND")
        if spread > max_spread_eV:
            flags.append(f"SPREAD {spread:.4f} eV > {max_spread_eV} eV")
        flag_str = f"  <-- {'; '.join(flags)} -- NOT a confident detection" if flags else "  OK"
        print(f"{name:12s} centers={np.round(centers, 4).tolist()}{flag_str}")
        robustness[name] = {"centers": centers.tolist(), "spread": spread, "flags": flags}

    base["robustness"] = robustness
    return base


def _block_mean(arr, factor):
    """Average `factor` x `factor` real-space blocks of a (ny, nx, n_energy) cube. Edge blocks that are not
    full are averaged over the pixels they contain. Returns (binned, iy, ix) where iy / ix map every original
    row / column to its block index."""
    arr = np.asarray(arr, dtype=float)
    ny, nx = arr.shape[:2]
    starts_y, starts_x = np.arange(0, ny, factor), np.arange(0, nx, factor)
    summed = np.add.reduceat(np.add.reduceat(arr, starts_y, axis=0), starts_x, axis=1)
    counts = np.outer(np.diff(np.append(starts_y, ny)), np.diff(np.append(starts_x, nx)))
    return summed / counts[..., None], np.arange(ny) // factor, np.arange(nx) // factor


def subtract_background_two_sided(
    dataset,
    fit_windows: Sequence[Tuple[float, float]],
    *,
    form: str = "powerlaw_const",
    polynomial_degree: int = 2,
    bin_factor: int = 1,
    free_offset: bool = True,
    clip_negative: bool = False,
    exclude_windows: Sequence[Tuple[float, float]] = (),
    return_details: bool = False,
):
    """
    Background subtraction with fit windows on BOTH sides of the features of interest (interpolation across
    them instead of extrapolation from one side), for any 3D spectroscopy dataset (LL, HL, ...).

    The functional shape is fixed on the *mean* spectrum, and only the few linear coefficients are fitted per
    pixel -- far more stable at low counts than an independent nonlinear fit in every pixel:

    * ``form="powerlaw_const"`` : A*E^-r + c. r (and c) are fitted on the mean spectrum in `fit_windows`; each
      pixel then gets its own A and (if `free_offset`) its own c by linear least squares. With
      ``free_offset=False`` the per-pixel model is A*(E^-r + c_mean/A_mean): the mean's shape, scaled.
    * ``form="powerlaw_curved"``: A*E^-r*exp(s*ln(E)^2) + c -- a power law whose exponent changes slowly with
      energy (a parabola in log-log). A ZLP tail is not a pure power law over an eV or more (its wings bend), so
      this often leaves a flatter baseline than ``powerlaw_const`` when the windows are far apart; r, s, c come
      from the mean, A (and c if `free_offset`) per pixel. More flexible => check that it does not absorb real
      broad features (it needs windows on both sides of them).
    * ``form="powerlaw"``       : A*E^-r with r from the mean, A per pixel.
    * ``form="polynomial"``     : per-pixel polynomial of `polynomial_degree` (fitted directly per pixel).

    ``bin_factor > 1`` averages `bin_factor` x `bin_factor` pixels **before the fitting**: the coefficients are
    fitted on the binned spectra (less noise) and each original pixel takes the coefficients of its block; the
    background is then subtracted from the original, full-resolution spectra. So the output keeps the input's
    spatial shape (use ``dataset.bin(bin_factor, axes=(0, 1), reducer="mean")`` if you want a binned cube
    instead -- note that it drops incomplete edge blocks, which this function keeps).

    ``exclude_windows`` -- (lo, hi) eV ranges that are NEVER used for the fit, even where they overlap
    `fit_windows` (e.g. a real peak, a plasmon, a reaction feature): they are removed from the fit mask. They
    are still subtracted from like everything else, only the fit ignores them.

    Power-law forms need E > 0 in the fit windows. Channels at E <= 0 (pre-ZLP) get no subtraction (the model
    is undefined there); near E ~ 0+ the power law is huge, so crop the ZLP region afterwards (the notebooks
    do: LL is trimmed to >= 0.3 eV). Negative results are kept by default (`clip_negative=False`) -- clipping
    to 0 is what turns an over-subtraction into flat, empty maps.

    Returns the background-subtracted dataset (a new object, input untouched); with ``return_details=True``
    also a dict with the fitted shape (``r``, ``c``), the windows, the binned block count and the per-pixel
    background cube ``background`` (same shape as the input).
    """
    energy = np.asarray(dataset.energy_axis, dtype=float)
    data = np.asarray(dataset.array, dtype=float)
    wins = [(float(lo), float(hi)) for lo, hi in fit_windows]
    if not wins:
        raise ValueError("fit_windows must contain at least one (lo, hi) window")
    for lo, hi in wins:
        if not lo < hi:
            raise ValueError(f"fit window must have lo < hi, got ({lo}, {hi})")
        if lo < energy[0] or hi > energy[-1]:
            raise ValueError(
                f"fit window ({lo}, {hi}) eV is outside the energy axis [{energy[0]:.3f}, {energy[-1]:.3f}] eV "
                "(fit_windows_inside_axis() can adapt it)"
            )
    fit = np.zeros(energy.shape, dtype=bool)
    for lo, hi in wins:
        fit |= (energy >= lo) & (energy <= hi)
    excl = [(float(lo), float(hi)) for lo, hi in exclude_windows]
    for lo, hi in excl:
        fit &= ~((energy >= lo) & (energy <= hi))
    n_params = {
        "powerlaw": 1,
        "powerlaw_const": 2 if free_offset else 1,
        "powerlaw_curved": 2 if free_offset else 1,
        "polynomial": int(polynomial_degree) + 1,
    }
    if form not in n_params:
        raise ValueError(f"form must be one of {sorted(n_params)}, got {form!r}")
    if fit.sum() < n_params[form] + (5 if form == "powerlaw_curved" else 3):
        raise ValueError(
            f"only {int(fit.sum())} channels left in the fit windows after exclude_windows -- too few for form={form!r}"
        )
    if form != "polynomial" and np.any(energy[fit] <= 0):
        raise ValueError("power-law forms need fit windows entirely above 0 eV")

    bin_factor = int(bin_factor)
    if bin_factor > 1:
        fit_cube, iy, ix = _block_mean(data, bin_factor)
    else:
        fit_cube, iy, ix = data, np.arange(data.shape[0]), np.arange(data.shape[1])
    ny_b, nx_b = fit_cube.shape[:2]
    Y = fit_cube[:, :, fit].reshape(ny_b * nx_b, -1)  # (n_binned_pixels, n_fit_channels)

    mean_spec = data.reshape(-1, data.shape[2]).mean(axis=0)
    details = dict(
        windows=wins,
        exclude_windows=excl,
        fit_mask=fit,
        form=form,
        bin_factor=bin_factor,
        r=None,
        c=None,
        s=None,
    )
    if form == "polynomial":
        basis_fit = np.vander(energy[fit], int(polynomial_degree) + 1, increasing=True)
        basis_all = np.vander(energy, int(polynomial_degree) + 1, increasing=True)
    else:
        s_curv = 0.0
        y_fit = mean_spec[fit]
        a0 = float(np.max(y_fit))
        chain = {
            "powerlaw_curved": ("powerlaw_curved", "powerlaw_const", "powerlaw"),
            "powerlaw_const": ("powerlaw_const", "powerlaw"),
            "powerlaw": ("powerlaw",),
        }[form]
        for (
            form_try
        ) in chain:  # a nonlinear fit that does not converge falls back to the next simpler shape
            try:
                if form_try == "powerlaw":
                    slope, _ = np.polyfit(
                        np.log(energy[fit]), np.log(np.clip(y_fit, 1e-12, None)), 1
                    )
                    r, c_mean, a_mean = -float(slope), 0.0, 1.0
                elif form_try == "powerlaw_const":
                    f = lambda x, a, r_, c_: a * x ** (-r_) + c_  # noqa: E731
                    popt, _ = curve_fit(f, energy[fit], y_fit, p0=(a0, 2.0, 0.0), maxfev=20000)
                    a_mean, r, c_mean = (float(v) for v in popt)
                else:
                    f = lambda x, a, r_, s_, c_: a * x ** (-r_) * np.exp(s_ * np.log(x) ** 2) + c_  # noqa: E731
                    popt, _ = curve_fit(
                        f, energy[fit], y_fit, p0=(a0, 2.0, 0.0, 0.0), maxfev=50000
                    )
                    a_mean, r, s_curv, c_mean = (float(v) for v in popt)
                break
            except (RuntimeError, ValueError):
                continue
        else:
            raise RuntimeError(
                "none of the tail shapes could be fitted to the mean spectrum in the fit windows"
            )
        if form_try != form:
            warnings.warn(
                f"subtract_background_two_sided: form {form!r} did not converge on the mean spectrum; used {form_try!r}"
            )
        form = form_try
        details.update(r=r, c=c_mean, s=s_curv, form=form)

        def shape(x):
            # x<=0 (pre-ZLP) channels are masked out by the outer np.where anyway, but numpy still evaluates
            # both branches eagerly -- clip to a tiny positive value first so that branch never sees 0/negative
            # input (log(0), 0**-r) and warns.
            xs = np.where(x > 0, x, 1e-6)
            return np.where(x > 0, xs ** (-r) * np.exp(s_curv * np.log(xs) ** 2), 0.0)

        g_fit, g_all = shape(energy[fit]), shape(energy)
        with_offset = form in ("powerlaw_const", "powerlaw_curved")
        if with_offset and free_offset:
            basis_fit = np.stack([g_fit, np.ones_like(g_fit)], axis=1)
            basis_all = np.stack([g_all, np.where(energy > 0, 1.0, 0.0)], axis=1)
        else:  # single shape scaled per pixel (offset, if any, is part of the mean's shape)
            off = c_mean / a_mean if with_offset else 0.0
            basis_fit = (g_fit + off)[:, None]
            basis_all = np.where(energy > 0, g_all + off, 0.0)[:, None]

    coef = Y @ np.linalg.pinv(basis_fit).T  # (n_binned_pixels, n_params)
    bg_binned = (coef @ basis_all.T).reshape(ny_b, nx_b, -1)
    background = bg_binned[iy][:, ix]  # nearest-neighbour back to the original pixel grid
    result = data - background
    if clip_negative:
        result = np.clip(result, 0.0, None)

    out = dataset.copy()
    out.array = result
    out.name = f"{dataset.name} (two-sided {form} background subtracted)"
    if return_details:
        details["background"] = background
        details["n_blocks"] = ny_b * nx_b
        return out, details
    return out


def summarize_energy_windows(dataset, windows):
    """
    Per energy window, what does the (background-subtracted) data actually contain? For every window the
    per-pixel window mean (mean over the window's channels, as in the energy-window maps) is taken and summarised:

    * ``mean`` / ``sem`` -- mean over pixels and its standard error (spatial std / sqrt(n_pixels));
    * ``t`` -- mean / sem, how many standard errors the window's signal sits from 0 (>~3 = clearly there);
    * ``frac_positive`` -- fraction of pixels with a positive window mean (~0.5 = nothing but noise);
    * ``spatial_cv`` -- spatial std / |mean|: >> 1 means the window is dominated by pixel-to-pixel noise, so its
      map will look like noise even if the mean is significant;
    * ``peak_eV`` / ``peak_value`` -- position and height of the maximum of the mean spectrum inside the window.

    Returns a list of dicts (one per window, in order).
    """
    e = np.asarray(dataset.energy_axis, dtype=float)
    arr = np.asarray(dataset.array, dtype=float)
    mean_spec = arr.reshape(-1, arr.shape[2]).mean(axis=0)
    rows = []
    for lo, hi in windows:
        k = (e >= lo) & (e <= hi)
        if not k.any():
            rows.append(dict(window=(lo, hi), n_channels=0))
            continue
        m = arr[:, :, k].mean(axis=-1).ravel()
        sd = float(m.std(ddof=1)) if m.size > 1 else float("nan")
        sem = sd / np.sqrt(m.size)
        mean = float(m.mean())
        j = int(np.argmax(mean_spec[k]))
        rows.append(
            dict(
                window=(float(lo), float(hi)),
                n_channels=int(k.sum()),
                mean=mean,
                sem=sem,
                t=mean / sem if sem > 0 else float("nan"),
                frac_positive=float(np.mean(m > 0)),
                spatial_cv=sd / abs(mean) if mean != 0 else float("inf"),
                peak_eV=float(e[k][j]),
                peak_value=float(mean_spec[k][j]),
            )
        )
    return rows


def summarize_map_diagnostics(
    maps: dict,
    *,
    adf: Optional[np.ndarray] = None,
    thickness_map: Optional[np.ndarray] = None,
    n_perm: int = 200,
    seed: int = 0,
) -> List[dict]:
    """
    For every ``{(lo, hi): 2D map}`` in ``maps``: is the map showing real spatial
    structure, or does it just look interesting?

    Combines :func:`spatial_coherence` (more structure than the same pixel values
    shuffled at random -- Moran's I, permutation z/p) with
    :func:`correlate_with_reference` against the ADF and/or thickness map, if given
    (is the pattern just following material contrast or thickness).

    This asks a different question than the window t-value of
    :func:`summarize_energy_windows`: t asks whether the window's average signal is
    above zero; this asks whether the signal varies coherently in space. A real,
    uniform feature can have high t and low Moran's I; a weak, zero-mean feature
    can still be spatially structured.

    Returns
    -------
    list of dict
        One per window, in ``maps`` order: ``window``, ``moran_i``, ``z``, ``p``,
        ``n_valid``, and ``corr_adf`` / ``corr_thickness`` ({r, p, n}) when the
        corresponding reference was given. :func:`plot_map_diagnostics` prints and
        plots the same rows.
    """
    rows = []
    for w in maps:
        emap = maps[w]
        sc = spatial_coherence(emap, n_perm=n_perm, seed=seed)
        row = dict(window=w, **sc)
        if adf is not None:
            row["corr_adf"] = correlate_with_reference(emap, adf)
        if thickness_map is not None:
            row["corr_thickness"] = correlate_with_reference(emap, thickness_map)
        rows.append(row)
    return rows


def _validate_pre_edge_window(lo, hi, energy_axis, target_edge=None) -> Tuple[float, float]:
    """(lo, hi) must be finite, lo < hi, fully inside ``energy_axis``, and hi < ``target_edge``."""
    energy_axis = np.asarray(energy_axis, dtype=float)
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"Background window bounds must be finite; got lo={lo} eV, hi={hi} eV")
    if not lo < hi:
        raise ValueError(f"Lower bound must be < upper bound; got lo={lo:.3f} eV, hi={hi:.3f} eV")
    axis_lo, axis_hi = float(energy_axis[0]), float(energy_axis[-1])
    if lo < axis_lo or hi > axis_hi:
        raise ValueError(
            f"Window [{lo:.3f}, {hi:.3f}] eV is not fully within the energy axis "
            f"[{axis_lo:.3f}, {axis_hi:.3f}] eV"
        )
    if target_edge is not None:
        target_edge = float(target_edge)
        if not np.isfinite(target_edge):
            raise ValueError(f"target_edge must be finite; got {target_edge}")
        if not hi < target_edge:
            raise ValueError(
                f"Upper bound {hi:.3f} eV must be below target_edge={target_edge:.3f} eV "
                "(the pre-edge fitting window must not reach the edge onset)."
            )
    return lo, hi


def _despike_apply(array, energy_axis, ranges):
    """
    Replace the channels inside each (lo, hi) range with a linear
    interpolation between the clean channel immediately before and
    immediately after the range, independently for every spectrum in
    `array` (whatever its leading dims are -- works for a 1D mean spectrum
    or the full (row, col, energy) cube, since energy is always the last
    axis and everything else broadcasts).
    """
    out = np.array(array, copy=True)
    for lo, hi in ranges:
        idx = np.where((energy_axis >= lo) & (energy_axis <= hi))[0]
        i0, i1 = int(idx[0]), int(idx[-1])
        left, right = i0 - 1, i1 + 1
        x_left, x_right = energy_axis[left], energy_axis[right]
        y_left = out[..., left]
        y_right = out[..., right]
        seg_x = energy_axis[i0 : i1 + 1]
        frac = (seg_x - x_left) / (x_right - x_left)
        out[..., i0 : i1 + 1] = y_left[..., None] + frac * (y_right - y_left)[..., None]
    return out


def bin_spatial(dataset, factor: int = 2):
    """
    Average ``factor`` x ``factor`` real-space pixels into one, to raise the
    signal-to-noise of each spectrum at the cost of spatial resolution.

    Returns a new dataset (the input is untouched) with the spatial sampling
    multiplied by ``factor``. Incomplete edge blocks are averaged over the pixels
    they contain, so no row or column is lost -- unlike ``Dataset.bin()``, which
    drops the remainder when the scan size is not a multiple of ``factor``.
    ``factor=1`` returns a copy.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    out = dataset.copy()
    if factor == 1:
        return out
    binned, _, _ = _block_mean(dataset.array, factor)
    out.array = binned
    sampling = np.array(dataset.sampling, dtype=float)
    sampling[:2] *= factor
    out.sampling = sampling
    out.name = f"{dataset.name} (binned {factor}x{factor})"
    return out
