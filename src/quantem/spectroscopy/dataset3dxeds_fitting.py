import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences


def peak_autoid(
    self,
    roi=None,
    roi_cal=None,
    energy_range=None,
    elements=None,
    ignore_elements=None,
    ignore_range=None,
    tolerance=0.15,
    threshold=None,
    noise_percentile=75,
    min_line_weight=0.0,
    mask=None,
    show_text=True,
    peaks=15,
    mode=None,
    line=None,
    return_details=False,
    intensity_range=None,
):
    """Identify likely elements by matching XEDS spectrum peaks to known lines.

    This routine keeps the matching logic intentionally direct:
    calculate a mean spectrum, find local maxima above an optional intensity
    threshold, match each peak to database lines within an energy tolerance,
    and rank elements by the quality of those matches.

    Parameters
    ----------
    roi, roi_cal : sequence or None, optional
        Spatial region used to calculate the mean spectrum. See
        ``show_mean_spectrum`` for ROI formats.
    energy_range : sequence[float] or None, optional
        Energy range ``[emin, emax]`` in keV to analyze.
    elements : str or sequence[str] or None, optional
        Element or element-line selectors to search, such as ``"Fe"``,
        ``"Fe K"``, or ``["Cu", "Zn"]``. If omitted, all database elements
        are considered.
    ignore_elements : str or sequence[str] or None, optional
        Elements to exclude from matching.
    ignore_range : sequence[float] or None, optional
        Energy interval ``[emin, emax]`` in keV where detected peaks are
        ignored.
    tolerance : float, optional
        Maximum allowed energy difference in keV between a detected peak
        and a database line. This controls line matching, not peak finding.
    threshold : float, "mean", or None, optional
        Minimum mean-spectrum intensity required for a peak to be
        considered. Use ``"mean"`` to require peaks above the average
        spectrum intensity. If ``None``, no intensity threshold is applied.
    noise_percentile : float or None, optional
        Percentile intensity used as the SNR denominator. The default
        ``75`` uses the 75th percentile of the mean-spectrum intensity. If
        ``None``, the mean finite intensity is used.
    min_line_weight : float, optional
        Minimum database line weight required for a line to be considered.
    mask : ndarray or None, optional
        Boolean mask forwarded to ``calculate_mean_spectrum``.
    show_text : bool, optional
        If ``True``, label matched plotted peaks.
    peaks : int or None, optional
        Maximum number of peaks to plot and print in the table. Matching is
        still performed on all peaks that pass ``threshold``.
    mode : {"autofill", "elements_only", "elements_preferred"} or None, optional
        Search strategy when ``elements`` or saved ``model_elements`` are
        available. ``"elements_only"`` restricts matching to those elements,
        ``"elements_preferred"`` searches all elements but ranks matching
        requested/saved elements before other candidates, and ``"autofill"``
        searches all elements. If ``None``, defaults to ``"elements_only"``
        when element context is available and ``"autofill"`` otherwise.
    line : float or sequence[float] or None, optional
        Reference energy line(s) in keV to draw as dashed black vertical
        markers.
    return_details : bool, optional
        If ``True``, return a dictionary with figure, axes, peaks, matches,
        alternatives, and element scores.
    intensity_range : sequence[float] or None, optional
        Spectrum y-axis limits as ``[ymin, ymax]``. If ``None``, the limits
        are chosen automatically with extra space below the spectrum for
        unmatched-peak markers.

    Returns
    -------
    tuple or dict
        By default returns ``(fig, (ax_img, ax_spec))``. If
        ``return_details`` is ``True``, returns a details dictionary.
    """
    type(self)._ensure_element_info()
    all_info = type(self).element_info or {}
    ignored_elements = set(
        map(str, type(self)._normalize_specs(ignore_elements, allow_none=True) or [])
    )
    min_line_weight = max(float(min_line_weight), 0.0)

    requested_edge_filters = type(self)._parse_element_selectors(
        elements, allow_none=True, param_name="elements"
    )

    def model_edge_filters():
        model_elements = getattr(self, "model_elements", {}) or {}
        filters = {}
        for element_name, selected_lines in model_elements.items():
            element_name = str(element_name)
            if not isinstance(selected_lines, dict) or not selected_lines:
                filters[element_name] = None
                continue

            selected = set(map(str, selected_lines.keys()))
            all_lines = set(map(str, (all_info.get(element_name) or {}).keys()))
            filters[element_name] = None if all_lines and selected >= all_lines else selected
        return filters or None

    saved_edge_filters = model_edge_filters()

    def merge_edge_filters(primary, secondary):
        if primary is None:
            return secondary
        if secondary is None:
            return primary

        merged = {str(k): (None if v is None else set(map(str, v))) for k, v in primary.items()}
        for element_name, selectors in secondary.items():
            element_name = str(element_name)
            if element_name not in merged or selectors is None or merged[element_name] is None:
                merged[element_name] = None if selectors is None else set(map(str, selectors))
            else:
                merged[element_name].update(map(str, selectors))
        return merged or None

    edge_filters = merge_edge_filters(saved_edge_filters, requested_edge_filters)
    requested_elements = set(edge_filters) if edge_filters else None

    mode_name = (
        str(mode) if mode is not None else ("elements_only" if requested_elements else "autofill")
    )
    mode_name = str(mode_name).strip().lower()
    valid_modes = {"autofill", "elements_only", "elements_preferred"}
    if mode_name not in valid_modes:
        raise ValueError("mode must be one of: autofill, elements_only, elements_preferred")
    if mode_name in {"elements_only", "elements_preferred"} and not requested_elements:
        raise ValueError(
            f"mode={mode_name!r} requires elements to be specified or saved in model_elements"
        )

    search_elements = requested_elements if mode_name == "elements_only" else None
    preferred_elements = requested_elements if mode_name == "elements_preferred" else set()

    fig, (ax_img, ax_spec) = self.show_mean_spectrum(
        roi=roi,
        roi_cal=roi_cal,
        energy_range=energy_range,
        mask=mask,
        intensity_range=intensity_range,
        data_type="xeds",
        show=False,
    )
    spec = np.asarray(
        self.calculate_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            mask=mask,
        ),
        dtype=float,
    )
    energy_axis = np.asarray(self.energy_axis, dtype=float)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != energy_axis.shape:
            raise ValueError(
                f"Mask shape {mask_arr.shape} does not match energy axis shape "
                f"{energy_axis.shape}."
            )
        energy_axis = energy_axis[mask_arr]

    if energy_range is not None:
        keep = (float(energy_range[0]) <= energy_axis) & (energy_axis <= float(energy_range[1]))
        energy_axis = energy_axis[keep]

    if spec.shape != energy_axis.shape:
        raise ValueError(
            "Energy axis length does not match mean spectrum length after filtering. "
            f"Got len(E)={len(energy_axis)} and len(spec)={len(spec)}."
        )

    def in_ignore_range(value):
        return (
            ignore_range is not None
            and len(ignore_range) == 2
            and float(ignore_range[0]) <= float(value) <= float(ignore_range[1])
        )

    def noise_level(values, percentile=75):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 1.0
        if percentile is None:
            noise = float(np.mean(values))
        else:
            percentile = float(percentile)
            if not 0 <= percentile <= 100:
                raise ValueError("noise_percentile must be between 0 and 100, or None")
            noise = float(np.percentile(values, percentile))
        return noise if np.isfinite(noise) and noise > 0 else 1.0

    def resolve_threshold(value):
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() != "mean":
                raise ValueError("threshold must be a number, 'mean', or None")
            finite = spec[np.isfinite(spec)]
            return float(np.mean(finite)) if finite.size else None
        threshold_value = float(value)
        if not np.isfinite(threshold_value):
            raise ValueError("threshold must be finite")
        return threshold_value

    noise = noise_level(spec, noise_percentile)
    threshold_value = resolve_threshold(threshold)
    peak_indices, _ = find_peaks(spec, height=threshold_value)
    prominences = (
        peak_prominences(spec, peak_indices)[0]
        if len(peak_indices)
        else np.asarray([], dtype=float)
    )
    peak_rows = []
    for idx, prominence in zip(peak_indices, prominences):
        energy = float(energy_axis[int(idx)])
        if in_ignore_range(energy):
            continue
        height = float(spec[int(idx)])
        snr = height / noise
        peak_rows.append((int(idx), height, energy, float(snr), float(prominence)))

    peak_rows.sort(key=lambda row: row[4], reverse=True)
    all_peaks = [(idx, height, energy, snr) for idx, height, energy, snr, _ in peak_rows]
    display_peaks = all_peaks if peaks is None else all_peaks[: max(int(peaks), 0)]

    def candidate_matches(peak_energy, snr, allowed_elements=None):
        candidates = []
        for element_name, lines in all_info.items():
            element_name = str(element_name)
            if element_name in ignored_elements:
                continue
            if allowed_elements is not None and element_name not in allowed_elements:
                continue
            for line_name, line_info in (lines or {}).items():
                if not type(self)._line_allowed_for_element(
                    element_name, str(line_name), edge_filters
                ):
                    continue
                try:
                    line_energy = float(
                        line_info["energy (keV)"]
                        if "energy (keV)" in line_info
                        else line_info["energy"]
                    )
                    line_weight = float(line_info.get("weight", 0.5))
                except (TypeError, ValueError, KeyError):
                    continue
                distance = abs(float(peak_energy) - line_energy)
                if line_weight < min_line_weight or distance > float(tolerance):
                    continue
                score = type(self)._peak_confidence(snr, line_weight, distance, float(tolerance))
                candidates.append(
                    {
                        "element": element_name,
                        "line": str(line_name),
                        "energy": line_energy,
                        "weight": line_weight,
                        "distance": distance,
                        "score": float(score),
                    }
                )
        if mode_name == "elements_preferred" and preferred_elements:
            candidates.sort(
                key=lambda item: (
                    not (
                        str(item["element"]) in preferred_elements and float(item["score"]) > 0.0
                    ),
                    -float(item["score"]),
                )
            )
        else:
            candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    peak_matches = []
    alternatives_by_peak = {}
    for peak_idx, height, peak_energy, snr in all_peaks:
        matches = candidate_matches(peak_energy, snr, search_elements)
        alternatives_by_peak[int(peak_idx)] = matches[:3]
        if not matches:
            continue
        best = matches[0]
        peak_matches.append(
            (
                peak_idx,
                height,
                peak_energy,
                snr,
                best["element"],
                f"{best['element']} {best['line']}",
                best["distance"],
                best["line"],
                best["weight"],
                best["score"],
            )
        )

    element_confidence: dict[str, float] = {}
    for _, _, _, _, element, _, _, _, _, score in peak_matches:
        element_confidence[element] = element_confidence.get(element, 0.0) + float(score)

    detected_elements = set(element_confidence)
    match_by_idx = {int(match[0]): match for match in peak_matches}

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
    ]
    element_color_map = {
        element: palette[i % len(palette)]
        for i, element in enumerate(sorted(detected_elements or (search_elements or [])))
    }

    y_min = float(np.nanmin(spec)) if len(spec) else 0.0
    y_max = float(np.nanmax(spec)) if len(spec) else 1.0
    y_span = max(y_max - y_min, abs(y_max), 1.0)
    label_y = 0.96

    table_rows = []
    for peak_idx, height, peak_energy, snr in display_peaks:
        match = match_by_idx.get(int(peak_idx))
        if match is None:
            ax_spec.plot(
                [peak_energy],
                [y_min - 0.04 * y_span],
                marker="|",
                markersize=5,
                color="gray",
                linestyle="None",
            )
            table_rows.append((peak_energy, height, snr, "Unmatched", "-", "-"))
            continue

        element = str(match[4])
        line_name = str(match[7])
        color = element_color_map.get(element, "black")
        ax_spec.axvline(peak_energy, color=color, linestyle="-", alpha=0.55, linewidth=1.5)
        if show_text:
            ax_spec.text(
                peak_energy,
                label_y,
                f"{element} {line_name}",
                transform=ax_spec.get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=90,
                fontsize=9,
                color=color,
                clip_on=True,
            )

        labels = [
            f"{m['element']} {m['line']} ({m['energy']:.3f})"
            for m in alternatives_by_peak[int(peak_idx)]
        ]
        labels = labels + ["-"] * (3 - len(labels))
        table_rows.append((peak_energy, height, snr, labels[0], labels[1], labels[2]))

    if line is not None:
        x_min, x_max = ax_spec.get_xlim()
        ref_energies = [line] if isinstance(line, (int, float)) else list(line)
        for ref_energy in ref_energies:
            try:
                ref_energy = float(ref_energy)
            except (TypeError, ValueError):
                continue
            if x_min <= ref_energy <= x_max:
                ax_spec.axvline(ref_energy, color="black", linestyle="--", linewidth=1.2, zorder=3)
        ax_spec.set_xlim(x_min, x_max)

    if intensity_range is None:
        current_bottom, current_top = ax_spec.get_ylim()
        ax_spec.set_ylim(bottom=min(current_bottom, y_min - 0.10 * y_span), top=current_top)
    fig.tight_layout()
    plt.show()

    print(
        f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} "
        f"{'Best Match':<24} {'Alt 2':<24} {'Alt 3':<24}"
    )
    print("-" * 112)
    for peak_energy, height, snr, best_match, alt_2, alt_3 in sorted(table_rows):
        print(
            f"{peak_energy:<12.3f} {height:<12.2f} {snr:<8.1f} "
            f"{best_match:<24} {alt_2:<24} {alt_3:<24}"
        )
    print("-" * 112)
    print(f"Matched {len(peak_matches)} peaks; displayed {len(display_peaks)} prominent peaks.\n")

    if return_details:
        return {
            "figure": fig,
            "axes": (ax_img, ax_spec),
            "detected_elements": sorted(detected_elements),
            "element_confidence": element_confidence,
            "display_peaks": display_peaks,
            "peak_matches": peak_matches,
            "peak_alternatives": alternatives_by_peak,
            "mode": mode_name,
            "threshold": threshold_value,
            "noise": noise,
            "noise_percentile": noise_percentile,
        }
    return fig, (ax_img, ax_spec)


def _fit_mean_model_pytorch(
    self,
    *args,
    **kwargs,
):
    """Compatibility wrapper for the simplified XEDS fitter."""
    from quantem.spectroscopy.xeds_fitting import _fit_mean_model_pytorch as _impl

    return _impl(self, *args, **kwargs)


def fit_spectrum_mean_pytorch(
    self,
    *args,
    **kwargs,
):
    """Compatibility wrapper for the simplified XEDS fitter."""
    from quantem.spectroscopy.xeds_fitting import fit_spectrum_mean_pytorch as _impl

    return _impl(self, *args, **kwargs)


def fit_spectrum_pytorch(
    self,
    *args,
    **kwargs,
):
    """Compatibility wrapper for the simplified XEDS fitter."""
    from quantem.spectroscopy.xeds_fitting import fit_spectrum_pytorch as _impl

    return _impl(self, *args, **kwargs)
