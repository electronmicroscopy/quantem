"""Every line MAPED prints, and the table of shifts.

Kept apart from the computation so a script that sets ``verbose=False`` runs
the same code, and so the wording lives in one place.
"""

import time


def header(maped) -> str:
    """``MAPED: 7 tilts, 512 x 512 scan, 192 x 192 detector, on cuda:0``."""
    scan_rows, scan_cols, detector_rows, detector_cols = maped.shape
    return (
        f"MAPED: {len(maped.tilt_names)} tilts, {scan_rows} x {scan_cols} scan, "
        f"{detector_rows} x {detector_cols} detector, on {maped.device}"
    )


def aligned(maped, iterations: int, started: float) -> str:
    """The largest shifts found, and the time since ``started``.

    The shifts are read before the clock, since reading them waits for the
    alignment to finish.
    """
    detector = float(maped.diffraction_shifts.abs().max())
    scan = float(maped.real_space_shifts.abs().max())
    return (
        f"Aligned in {time.perf_counter() - started:.1f} s: largest detector shift {detector:.2f} px, "
        f"largest scan shift {scan:.2f} px after {iterations} iterations"
    )


def crop(maped, scan_box: tuple[int, int, int, int]) -> str:
    """The scan box every tilt covers, and how to keep the full scan instead."""
    row0, row1, col0, col1 = scan_box
    return (
        f"Cropped to rows {row0}:{row1}, columns {col0}:{col1} "
        f"({row1 - row0} x {col1 - col0} of {maped.shape[0]} x {maped.shape[1]}), "
        f"where all {len(maped.tilt_names)} tilts overlap. crop=False keeps the full scan."
    )


def merged(merged, seconds: float) -> str:
    """Time, stored size and rounding error of the complete merge, in counts."""
    precision = merged.metadata.get("precision")
    if precision is None:
        return f"Merged in {seconds:.1f} s: exact float32"
    return (
        f"Merged in {seconds:.1f} s: {merged.resident_bytes / 2**30:.2f} GiB as scaled 16-bit, "
        f"rounding RMSE {precision['rmse']:.4f} and at most {precision['max_abs_error']:.4f} counts"
    )


def shift_table(maped) -> dict[str, list[str] | list[float] | list[int]]:
    """Return beam centers and measured shifts in pixels, grouped by column.

    Returns
    -------
    dict
        Tilt labels and available row/column shifts as lists in acquisition
        order. All values are plain Python objects.

    Examples
    --------
    >>> maped.align_diffraction(show=False)
    >>> maped.align_images(show=False)
    >>> maped.report()
    """
    columns = {"Tilt": maped.tilt_names.copy()}
    for attribute, label in (
        ("beam_centers", "Beam center"),
        ("diffraction_shifts", "Detector"),
        ("real_space_shifts", "Scan"),
    ):
        shifts_t = getattr(maped, attribute, None)
        if shifts_t is not None:
            columns[f"{label} row (px)"] = shifts_t[:, 0].detach().cpu().tolist()
            columns[f"{label} col (px)"] = shifts_t[:, 1].detach().cpu().tolist()
    return columns
