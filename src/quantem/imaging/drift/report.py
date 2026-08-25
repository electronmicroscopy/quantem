"""Regional alignment reports for :class:`DriftCorrection`.

This code owns checkpoint selection, fixed-mask NCC calculation, and
DataFrame assembly. The public entry point remains ``DriftCorrection.report``;
keeping the implementation here separates scientific reporting from solver
state and optimization code.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

import quantem.imaging.drift.core.strip as strip
import quantem.imaging.drift.core.warping as warping
from quantem.imaging.drift.core import knots as drift_knots


def record_error(correction, mode: int, warped: torch.Tensor | None = None):
    """Record per-scan disagreement for a completed correction stage."""
    if warped is not None:
        mean = warped.mean(dim=0)
        differences = torch.mean(
            torch.abs(warped - mean[None]), dim=(1, 2)
        ).cpu().numpy()
    else:
        warping.ensure_warped_images(correction)
        mean = np.mean(correction.imgs_warped.array, axis=0)
        differences = np.mean(
            np.abs(correction.imgs_warped.array - mean[None]), axis=(1, 2)
        )
    current = np.hstack((mode, np.mean(differences), differences))
    if not hasattr(correction, "error_track"):
        correction.error_track = current[None, :]
    else:
        correction.error_track = np.vstack((correction.error_track, current))


def report(
    self,
    *,
    stages: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare alignment quality across completed correction stages.

    Every stage is measured through one final common-coverage mask, so changes
    in NCC reflect registration rather than a changing set of padded edge
    pixels. The regional scores expose improvements or regressions that a
    single whole-image NCC can hide.

    Parameters
    ----------
    stages : sequence of str or None, default None
        Checkpoints selected from ``"initial"``, ``"affine"``, ``"strip"``,
        and ``"current"``. ``None`` includes every completed distinct stage.

    Returns
    -------
    pandas.DataFrame
        Common, top, middle, bottom, and coverage measurements by stage.

    Examples
    --------
    >>> drift.correct_affine(show_combined=False)
    >>> drift.report()
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "report() requires a prepared alignment. Run correct_affine() first."
        )

    snapshots = [("initial", "before")]
    if hasattr(self, "_knots_after_affine"):
        snapshots.append(("affine", "affine"))
    if hasattr(self, "_knots_after_strip"):
        snapshots.append(("strip", "strip"))

    last_snapshot = (
        getattr(self, "_knots_after_strip", None)
        or getattr(self, "_knots_after_affine", None)
        or self._initial_knots
    )
    current_is_distinct = any(
        not torch.equal(current, previous)
        for current, previous in zip(self.knots, last_snapshot, strict=True)
    )
    if current_is_distinct:
        has_nonrigid = hasattr(self, "error_track") and np.any(
            np.asarray(self.error_track)[:, 0] == 2
        )
        snapshots.append(("current", "nonrigid" if has_nonrigid else "current"))

    if stages is not None:
        requested = tuple(str(stage) for stage in stages)
        valid = {"initial", "affine", "strip", "current"}
        unknown = sorted(set(requested) - valid)
        if unknown:
            raise ValueError(
                f"Unknown report stages {unknown}. Choose from {sorted(valid)}."
            )
        labels = dict(snapshots)
        missing = [stage for stage in requested if stage not in labels]
        if missing:
            raise ValueError(
                f"Report stages are not available yet: {missing}. "
                f"Available stages: {sorted(labels)}."
            )
        snapshots = [(stage, labels[stage]) for stage in requested]
    fixed_set = frozenset({0}) if self._reference_mode else frozenset()
    # One final common mask is held fixed across every row. Score changes then
    # measure alignment rather than a changing population of edge pixels.
    comparison_mask = np.asarray(self.coverage_mask(), dtype=bool)
    rows: list[dict[str, float | str]] = []
    for stage, label in snapshots:
        knots = drift_knots.stage_knots(
            self, None if stage == "current" else stage
        )
        stack = (
            warping.reference_scan_stack(self, knots)
            if self._reference_mode
            else warping.co_registered_scan_stack(
                self,
                fixed_set=fixed_set,
                solve_translation=False,
                knots=knots,
            )
        )
        if self._reference_mode:
            comparisons = [(stack[0], stack[index]) for index in range(1, len(stack))]
        elif len(stack) == 2:
            comparisons = [(stack[0], stack[1])]
        else:
            comparisons = [
                (
                    np.mean(
                        [stack[j] for j in range(len(stack)) if j != index],
                        axis=0,
                    ).astype(np.float32),
                    stack[index],
                )
                for index in range(len(stack))
            ]
        scores = [
            strip.region_ncc(
                reference,
                moving,
                comparison_mask,
                device=self._device,
            )
            for reference, moving in comparisons
        ]
        row: dict[str, float | str] = {"stage": label}
        for column in ("common", "top", "middle", "bottom", "mask_frac"):
            row[column] = float(np.mean([score[column] for score in scores]))
        rows.append(row)

    report = pd.DataFrame.from_records(rows).set_index("stage")
    report.index.name = (
        "Fixed-reference stage" if self._reference_mode else "Mutual stage"
    )
    return report.rename(
        columns={
            "common": "Common NCC",
            "top": "Top third",
            "middle": "Middle third",
            "bottom": "Bottom third",
            "mask_frac": "Coverage",
        }
    )
