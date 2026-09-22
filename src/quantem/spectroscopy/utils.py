import csv
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
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
