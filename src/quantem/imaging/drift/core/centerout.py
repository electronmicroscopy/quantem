import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm.auto import tqdm

from quantem.imaging.drift import plot as drift_plot
from quantem.imaging.drift.core.knots import transform_row_numpy
from quantem.imaging.drift.core.warping import warp_and_translate


def solve_rows_center_out(
    correction,
    index: int,
    image_ref: NDArray,
    knots_init: NDArray,
    max_optimize_iterations: int = 4,
    step_cap_px: float = 8.0,
    increment_alpha: float = 0.4,
    step_alpha: float = 0.1,
    step_clip_px: float = 0.1,
    relaxation: float = 0.25,
) -> NDArray:
    """Solve one scan line at a time, outward from the center of the slow axis."""
    height, width = int(correction.shape[1]), int(correction.shape[2])
    scan_fast = np.asarray(correction.scan_fast[index], dtype=float)
    perp = np.array([scan_fast[1], -scan_fast[0]])
    image = np.asarray(correction.imgs[index].array, dtype=float)
    input_shape = tuple(int(size) for size in correction.imgs[index].shape[:2])
    options = {"maxiter": max_optimize_iterations} if max_optimize_iterations else {}

    num_rows = knots_init.shape[1]
    center = num_rows // 2
    position_init = perp @ knots_init[:, :, 0]
    mean_step = float(np.mean(np.diff(position_init)))

    knots = knots_init.copy()
    state = {
        side: {"increment": None, "count": 0, "last_b": None, "step": None}
        for side in (-1, 1)
    }

    order = [center]
    for offset in range(1, num_rows):
        order += [
            row for row in (center + offset, center - offset) if 0 <= row < num_rows
        ]

    for row in order:
        sign = 1 if row >= center else -1
        side = state[sign]

        def cost(ab):
            knots_row = knots_init[:, row, :] + (perp * ab[0] + scan_fast * ab[1])[:, None]
            row_coords, col_coords = transform_row_numpy(
                knots_row, scan_fast, input_shape
            )
            row_coords = np.clip(row_coords, 0, height - 1.001)
            col_coords = np.clip(col_coords, 0, width - 1.001)
            row_floor = np.floor(row_coords).astype(int)
            col_floor = np.floor(col_coords).astype(int)
            frac_row = row_coords - row_floor
            frac_col = col_coords - col_floor
            warped = (
                image_ref[row_floor, col_floor] * (1 - frac_row) * (1 - frac_col)
                + image_ref[row_floor + 1, col_floor] * frac_row * (1 - frac_col)
                + image_ref[row_floor, col_floor + 1] * (1 - frac_row) * frac_col
                + image_ref[row_floor + 1, col_floor + 1] * frac_row * frac_col
            )
            return float(np.sum((warped.ravel() - image[row, :]) ** 2))

        # Bias-corrected EMA
        b_last = 0.0 if side["last_b"] is None else side["last_b"]
        side["increment"] = (
            increment_alpha * b_last + (1 - increment_alpha) * (side["increment"] or 0.0)
        )
        b_pred = side["increment"] / (1 - (1 - increment_alpha) ** (side["count"] + 1))
        if row != center:
            side["count"] += 1

        if row == center:
            prediction = np.array([0.0, b_pred])
        else:
            neighbor = row - sign
            position_neighbor = float(perp @ knots[:, neighbor, 0])
            step = (
                mean_step
                if side["step"] is None
                else float(
                    np.clip(
                        side["step"], mean_step - step_clip_px, mean_step + step_clip_px
                    )
                )
            )
            prediction = np.array(
                [position_neighbor + sign * step - position_init[row], b_pred]
            )

        bounds = [(value - step_cap_px, value + step_cap_px) for value in prediction]
        solution = minimize(
            cost, prediction, method="L-BFGS-B", bounds=bounds, options=options
        ).x
        ab = prediction + relaxation * (solution - prediction)
        knots[:, row, :] = (
            knots_init[:, row, :] + (perp * ab[0] + scan_fast * ab[1])[:, None]
        )

        side["last_b"] = float(ab[1])
        if row == center:
            state[-sign]["last_b"] = float(ab[1])
        else:
            realized = sign * (float(perp @ knots[:, row, 0]) - position_neighbor)
            side["step"] = (
                realized
                if side["step"] is None
                else (1 - step_alpha) * side["step"] + step_alpha * realized
            )

    return knots


def _smooth_knot_residual(
    knots: NDArray,
    sigma: float,
    poly_order: int,
) -> NDArray:
    """Gaussian-smooth each knot track after removing a polynomial trend."""
    smoothed = knots.copy()
    rows = np.arange(knots.shape[1])
    for dim in range(knots.shape[0]):
        for knot_index in range(knots.shape[2]):
            track = knots[dim, :, knot_index]
            trend = np.polyval(np.polyfit(rows, track, deg=poly_order), rows)
            smoothed[dim, :, knot_index] = (
                gaussian_filter(track - trend, sigma=sigma) + trend
            )
    return smoothed


def correct_nonrigid_center_out(
    self,
    num_sweeps: int = 8,
    max_optimize_iterations: int = 4,
    step_cap_px: float = 8.0,
    increment_alpha: float = 0.4,
    step_alpha: float = 0.1,
    step_clip_px: float = 0.1,
    relaxation: float = 0.25,
    knot_smoothing_sigma: float = 16.0,
    knot_smoothing_poly_order: int = 1,
    update_fraction: float | None = None,
    max_image_shift: float | None = 32.0,
    lowpass: float = 0.0,
    cost_taper: int = 0,
    subpixel: str = "dft",
    verbose: bool = True,
    show_combined: bool = True,
    show_scans: bool = False,
    show_knots: bool = True,
    **kwargs,
):
    """Non-rigid drift correction from the center out, minimizing cases of unit cell skips"""

    if not hasattr(self, "knots"):
        raise RuntimeError(
            "correct_nonrigid_center_out() requires preprocess()"
        )
    num_images = int(self.shape[0])
    device, dtype = self._device, self._dtype

    warped_t = warp_and_translate(
        self,
        max_image_shift,
        upsample_factor=8,
        solve_translation=False,
        lowpass=lowpass,
        ramp=cost_taper,
        subpixel=subpixel,
    )
    error_buffer = []

    for _ in tqdm(
        range(num_sweeps), desc="Solving nonrigid drift (center-out)", disable=not verbose
    ):
        warped_np = warped_t.detach().cpu().numpy().astype(float)
        for index in range(num_images):
            image_ref = np.delete(warped_np, index, axis=0).mean(axis=0)
            knots_init = self.knots[index].detach().cpu().numpy().astype(float)
            knots_updated = solve_rows_center_out(
                self,
                index,
                image_ref,
                knots_init,
                max_optimize_iterations=max_optimize_iterations,
                step_cap_px=step_cap_px,
                increment_alpha=increment_alpha,
                step_alpha=step_alpha,
                step_clip_px=step_clip_px,
                relaxation=relaxation,
            )
            if knot_smoothing_sigma and knot_smoothing_sigma > 0:
                knots_updated = _smooth_knot_residual(
                    knots_updated, knot_smoothing_sigma, knot_smoothing_poly_order
                )
            if update_fraction is not None:
                knots_updated = knots_init + (knots_updated - knots_init) * update_fraction
            self.knots[index][...] = torch.tensor(
                knots_updated, dtype=dtype, device=device
            )

        warped_t = warp_and_translate(
            self,
            max_image_shift,
            upsample_factor=8,
            solve_translation=True,
            lowpass=lowpass,
            ramp=cost_taper,
            subpixel=subpixel,
        )
        images_mean = warped_t.mean(dim=0)
        error_buffer.append(
            torch.mean(torch.abs(warped_t - images_mean[None]), dim=(1, 2))
        )

    self._images_warped_stale = True
    self._max_image_shift_cached = max_image_shift
    if error_buffer:
        errors_np = torch.stack(error_buffer).cpu().numpy()
        mode_col = np.full((len(errors_np), 1), 2.0)
        mean_col = errors_np.mean(axis=1, keepdims=True)
        new_rows = np.hstack((mode_col, mean_col, errors_np))
        if not hasattr(self, "error_track"):
            self.error_track = new_rows
        else:
            self.error_track = np.vstack((self.error_track, new_rows))

    drift_plot.show_after_step(
        self,
        "non-rigid",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    return self
