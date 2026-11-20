"""Polymer-workflow extensions to :mod:`quantem.diffraction.polar_transform`.

Per-scan origin finding, descan fitting, the centered mean diffraction pattern,
and polar conversion of detected peaks, built on Karen Ehrhardt's angular-uniformity
origin finders and polar transform.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction.peak_detection import find_central_beam_from_peaks
from quantem.diffraction.polar_transform import (
    find_origin_angular_descent,
    find_origin_angular_grid,
    polar_transform,
)

OriginMethod = Literal["descent", "grid"]

__all__ = [
    "OriginMethod",
    "as_dataset4dstem",
    "centered_dp_mean",
    "find_central_beams_4d",
    "find_origin",
    "fit_origin_roi",
    "polar_transform_pattern",
    "polar_transform_peaks",
]


def as_dataset4dstem(data) -> Dataset4dstem:
    """Wrap a 2D pattern or 4D array for :func:`polar_transform`, which takes a Dataset4dstem.

    A 2D pattern becomes a 1x1 scan, so pass ``scan_pos=(0, 0)`` to get ``(n_phi, n_r)`` back.
    """
    if isinstance(data, Dataset4dstem):
        return data
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[None, None]
    return Dataset4dstem.from_array(array)


def polar_transform_pattern(pattern, origin, **kwargs) -> NDArray:
    """Polar-transform a single 2D diffraction pattern into a numpy ``(n_phi, n_r)`` array.

    ``kwargs`` are forwarded to :func:`polar_transform`.
    """
    polar = polar_transform(
        as_dataset4dstem(pattern), origin_array=origin, scan_pos=(0, 0), **kwargs
    )
    return polar.detach().cpu().numpy()


def find_origin(
    data,
    *,
    method: OriginMethod = "descent",
    ellipse_params: tuple[float, float, float] | None = None,
    radial_min: float = 4.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    num_annular_bins: int = 180,
    n_phi: int = 120,
    two_fold_rotation_symmetry: bool = False,
    kpow: float = 0.0,
    device: str = "cpu",
    batch_size: int = 16,
    local_margin: int = 40,
) -> NDArray:
    """Estimate diffraction-pattern origins as ``(scan_y, scan_x, 2)`` row/col pixels."""
    data = as_dataset4dstem(data)
    if method == "descent":
        return find_origin_angular_descent(
            data,
            ellipse_params=ellipse_params,
            radial_min=radial_min,
            radial_max=radial_max,
            n_phi=n_phi,
            radial_step=radial_step,
            kpow=kpow,
            device=device,
        )
    if method == "grid":
        return find_origin_angular_grid(
            data,
            ellipse_params=ellipse_params,
            num_annular_bins=num_annular_bins,
            radial_min=radial_min,
            radial_max=radial_max,
            radial_step=radial_step,
            two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            device=device,
            batch_size=batch_size,
            local_margin=local_margin,
        )
    raise ValueError(f"method must be 'descent' or 'grid', got {method!r}.")


def find_central_beams_from_peaks_4d(
    dataset,
    peaks,
    scan_mask: ArrayLike = None,
    intensity_threshold=0.3,
    distance_weight=0.5,
    sampling_radius=2,
    debug=False,
    use_tqdm=True,
):
    """Previous central-beam heuristic based on detected peak locations."""
    scan_y, scan_x, det_y, det_x = dataset.shape
    centers = np.zeros((2, scan_y, scan_x))

    # Handle scan_mask
    if scan_mask is None:
        scan_mask = np.ones((scan_y, scan_x), dtype=bool)
    else:
        scan_mask = np.asarray(scan_mask, dtype=bool)

    iterator = tqdm(range(scan_y), disable=not use_tqdm, desc="Finding centers")

    for i in iterator:
        for j in range(scan_x):
            if not scan_mask[i, j]:
                continue
            if peaks[i, j] is None:
                print(f"None at i={i}, j={j}")
            centers[:, i, j] = find_central_beam_from_peaks(
                peak_coords=peaks[i, j],
                peak_intensities=None,
                image_shape=(det_y, det_x),
                intensity_threshold=intensity_threshold,
                distance_weight=distance_weight,
                debug=debug,
                image=dataset[i, j].array.squeeze(),
                sampling_radius=sampling_radius,
            )
    return centers


def find_central_beams_4d(
    dataset,
    scan_mask: ArrayLike = None,
    intensity_threshold=0.3,
    distance_weight=0.5,
    sampling_radius=2,
    debug=False,
    use_tqdm=True,
    center_method: str = "descent",
    radial_min: float = 4.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    num_annular_bins: int = 180,
    n_phi: int = 120,
    kpow: float = 0.0,
    ellipse_params: tuple[float, float, float] | None = None,
    center_device: str | None = None,
    center_batch_size: int = 16,
    local_margin: int = 40,
    fallback_to_peaks: bool = True,
    peaks=None,
    default_device: str = "cpu",
):
    """
    Fast central beam finding for entire 4D dataset.

    Parameters:
    -----------
    scan_mask : ArrayLike, optional
        Boolean mask (Ry, Rx) indicating which positions to process
    use_tqdm : bool
        Show progress bar

    Returns:
    --------
    centers : ndarray, shape (2, scan_y, scan_x)
        Center coordinates (y, x) for each scan position
    """
    center_method = center_method.lower()
    if center_method == "peaks":
        return find_central_beams_from_peaks_4d(
            dataset,
            peaks,
            scan_mask=scan_mask,
            intensity_threshold=intensity_threshold,
            distance_weight=distance_weight,
            sampling_radius=sampling_radius,
            debug=debug,
            use_tqdm=use_tqdm,
        )
    if center_method not in ("descent", "grid"):
        raise ValueError(
            f"center_method must be 'descent', 'grid', or 'peaks', got {center_method!r}."
        )

    scan_y, scan_x, _det_y, _det_x = dataset.shape
    if scan_mask is None:
        scan_mask_arr = np.ones((scan_y, scan_x), dtype=bool)
    else:
        scan_mask_arr = np.asarray(scan_mask, dtype=bool)
        if scan_mask_arr.shape != (scan_y, scan_x):
            raise ValueError(
                f"scan_mask shape {scan_mask_arr.shape} must match {(scan_y, scan_x)}"
            )

    device = center_device if center_device is not None else default_device
    try:
        origins = find_origin(
            dataset,
            method=center_method,
            ellipse_params=ellipse_params,
            radial_min=radial_min,
            radial_max=radial_max,
            radial_step=radial_step,
            num_annular_bins=num_annular_bins,
            n_phi=n_phi,
            kpow=kpow,
            device=device,
            batch_size=center_batch_size,
            local_margin=local_margin,
        )
    except Exception as exc:
        if not fallback_to_peaks:
            raise
        warnings.warn(
            "Angular-uniformity center finding failed; falling back to "
            f"peak-based central-beam heuristic. Original error: {exc}",
            stacklevel=2,
        )
        return find_central_beams_from_peaks_4d(
            dataset,
            peaks,
            scan_mask=scan_mask_arr,
            intensity_threshold=intensity_threshold,
            distance_weight=distance_weight,
            sampling_radius=sampling_radius,
            debug=debug,
            use_tqdm=use_tqdm,
        )

    if origins.shape != (scan_y, scan_x, 2):
        raise ValueError(
            f"Origin finder returned shape {origins.shape}; expected {(scan_y, scan_x, 2)}."
        )
    centers = np.moveaxis(np.asarray(origins, dtype=float), -1, 0)
    centers[:, ~scan_mask_arr] = 0.0
    return centers


def fit_origin_roi(measured, mask, fit_method="plane", clip_sigma=5.0, n_iter=2):
    """Fit the smooth CoM-origin background over the ROI (scan mask) only.

    ``measured`` is the per-pattern CoM origin ``(Ry, Rx, 2)``; ``mask`` is the
    ``(Ry, Rx)`` scan ROI. Fitting over the whole scan lets out-of-ROI patterns
    (vacuum/substrate, whose CoM is meaningless) drag the plane, leaving a uniform
    residual inside the ROI. Each component is fit independently: ``"plane"`` does an
    ordinary least-squares ``z = a + b*row + c*col`` with ``n_iter`` sigma-clip passes
    to reject hot/dead-pixel CoM outliers; ``"constant"`` uses the ROI mean. Returns
    the fitted field ``(Ry, Rx, 2)`` evaluated at every scan position.
    """
    measured = np.asarray(measured, dtype=float)
    Ry, Rx, ncomp = measured.shape
    m0 = np.asarray(mask, dtype=bool)
    yy, xx = np.mgrid[0:Ry, 0:Rx].astype(float)
    fitted = np.empty_like(measured)
    for c in range(ncomp):
        z = measured[..., c]
        if fit_method == "constant":
            ref = z[m0] if m0.any() else z
            fitted[..., c] = float(np.nanmean(ref))
            continue
        use = m0 & np.isfinite(z)
        plane = np.full((Ry, Rx), float(np.nanmean(z[use])) if use.any() else 0.0)
        for _ in range(max(1, n_iter)):
            if int(use.sum()) < 3:
                break
            A = np.stack([np.ones(int(use.sum())), xx[use], yy[use]], axis=1)
            coef, *_ = np.linalg.lstsq(A, z[use], rcond=None)
            plane = coef[0] + coef[1] * xx + coef[2] * yy
            resid = z - plane
            s = float(np.std(resid[use]))
            if s == 0:
                break
            new_use = m0 & np.isfinite(z) & (np.abs(resid) < clip_sigma * s)
            if int(new_use.sum()) == int(use.sum()) or int(new_use.sum()) < 8:
                break
            use = new_use
        fitted[..., c] = plane
    return fitted


def centered_dp_mean(dataset, image_centers, com_model=None):
    """Mean diffraction pattern with every pattern shifted so its central beam
    lands at the detector center.

    Averaging the raw patterns smears the diffraction ring by the descan drift and
    leaves the central beam off-center, which biases an ellipse fit toward the
    central beam. Aligning each pattern first yields a sharp, concentric ring.

    When a ``CenterOfMassOriginModel`` is available (it holds the 4D tensor on-device),
    each ROI pattern is SUB-PIXEL shifted (bilinear) by ``image_centers`` -- the
    angular-uniformity BEAM center -- so the beam lands exactly on the detector center,
    and the mean is accumulated in batches. We center by ``image_centers`` rather than
    the CoM/descan origin because the CoM is the centroid of the whole pattern: any
    ring or background asymmetry pulls it a few px off the actual beam, which would
    leave the beam off-center in the mean. Otherwise (no CoM model) we fall back to
    translating the plain mean DP by the average center offset.

    Sub-pixel matters: plain integer rolls leave a per-pattern residual of up to 0.5 px
    that does NOT average out when the center spread is narrow (all patterns round to
    the same integer), so the mean beam ends up biased off-center by a fraction of a
    pixel. Bilinear splatting removes that bias.

    We deliberately avoid ``CenterOfMassOriginModel.shift_origin_to`` here: it
    materialises a full second copy of the 4D stack *plus* a per-pattern sampling
    grid, which OOMs on large scans. The bilinear accumulator needs only a
    ``(Qy, Qx)`` buffer plus one batch of patterns at a time.
    """
    Qy, Qx = dataset.shape[-2:]
    center = ((Qy - 1) / 2.0, (Qx - 1) / 2.0)
    ic = np.asarray(image_centers, dtype=float)  # (2, Ry, Rx); 0 outside the scan mask
    valid = (ic[0] != 0) | (ic[1] != 0)
    if com_model is not None and valid.any():
        import torch

        with torch.no_grad():
            flat = com_model.tensor.reshape(-1, Qy, Qx)
            dev = flat.device
            # Center by the authoritative per-pattern beam centers (image_centers, from
            # the angular-uniformity finder), NOT the CoM plane: the CoM is the centroid
            # of the whole pattern, so ring/background asymmetry pulls it a few px off
            # the beam, and centering by it would leave the beam off-center. Only ROI
            # (in-mask, non-zero) patterns are averaged.
            oy = torch.as_tensor(ic[0].ravel(), dtype=torch.float, device=dev)
            ox = torch.as_tensor(ic[1].ravel(), dtype=torch.float, device=dev)
            keep = torch.as_tensor(valid.ravel(), device=dev).nonzero(as_tuple=False).squeeze(1)
            sy = center[0] - oy  # continuous shift -> detector center, (y, x)
            sx = center[1] - ox
            acc = torch.zeros((Qy, Qx), dtype=torch.float32, device=dev)
            batch = 256
            for bstart in range(0, int(keep.numel()), batch):
                bidx = keep[bstart : bstart + batch]
                chunk = flat[bidx].float()  # (b, Qy, Qx), ROI patterns only
                fyb = torch.floor(sy[bidx])
                fxb = torch.floor(sx[bidx])
                gy = sy[bidx] - fyb
                gx = sx[bidx] - fxb
                floor_pairs = torch.stack([fyb.long(), fxb.long()], dim=1)  # (b, 2)
                # bilinear weights for the 4 integer-shift corners around the fraction
                corner_w = {
                    (0, 0): (1 - gy) * (1 - gx),
                    (0, 1): (1 - gy) * gx,
                    (1, 0): gy * (1 - gx),
                    (1, 1): gy * gx,
                }
                # descan drift is smooth -> few distinct floor shifts per batch:
                # weight-sum each group, then roll each of the 4 corners.
                uniq, inv = torch.unique(floor_pairs, dim=0, return_inverse=True)
                for k in range(int(uniq.shape[0])):
                    m = inv == k
                    cg = chunk[m]  # (g, Qy, Qx)
                    fy = int(uniq[k, 0])
                    fx = int(uniq[k, 1])
                    for (dy, dx), wt in corner_w.items():
                        s = (cg * wt[m][:, None, None]).sum(0)
                        acc += torch.roll(s, shifts=(fy + dy, fx + dx), dims=(0, 1))
            dp = (acc / max(int(keep.numel()), 1)).detach().cpu().numpy()
        return np.asarray(dp, dtype=float)
    # Fallback: translate the raw mean DP so the average beam center is centered.
    # image_centers is 0 outside the scan mask (find_central_beams_4d), so average
    # over valid (non-zero) positions only to avoid a bias toward the origin.
    from scipy.ndimage import shift as ndi_shift

    dp = np.asarray(dataset.get_dp_mean().array, dtype=float)
    valid = (image_centers[0] != 0) | (image_centers[1] != 0)
    if not valid.any():
        valid = np.ones_like(image_centers[0], dtype=bool)
    dy = center[0] - float(image_centers[0][valid].mean())
    dx = center[1] - float(image_centers[1][valid].mean())
    return ndi_shift(dp, (dy, dx), order=1, mode="constant", cval=0.0)


def polar_transform_peaks(
    cartesian_vector,
    centers: NDArray,
    *,
    scan_mask: NDArray | None = None,
    x_field: str | list[str] = ["x_pixels", "x"],
    y_field: str | list[str] = ["y_pixels", "y"],
    sampling_conversion_factor: float | None = None,
    two_fold_rotation_symmetry: bool = True,
    ellipse_params: tuple[float, float, float] | None = None,
    r_unit: str = "pixels",
    theta_unit: str = "radians",
    name_suffix: str = "_polar",
    use_tqdm: bool = True,
):
    """Transform Cartesian peak coordinates with Karen's polar convention.

    Peaks remain one-to-one with the input rows. Under two-fold symmetry, partner
    peaks are folded to the same theta coordinate with ``theta % pi`` but are not
    aggregated.
    """
    from quantem.core.datastructures import Vector

    if isinstance(cartesian_vector, np.ndarray) and cartesian_vector.dtype == object:
        cartesian_vector = cartesian_vector.item()
    if not isinstance(cartesian_vector, Vector):
        raise TypeError(f"Expected Vector, got {type(cartesian_vector)}")

    def find_field(field_options, available_fields):
        fields = [field_options] if isinstance(field_options, str) else field_options
        return next((f for f in fields if f in available_fields), None)

    x_field_found = find_field(x_field, cartesian_vector.fields)
    y_field_found = find_field(y_field, cartesian_vector.fields)
    if x_field_found is None or y_field_found is None:
        raise ValueError(
            "Could not find x/y coordinate fields in Vector. "
            f"Available fields: {cartesian_vector.fields}"
        )

    n_scan_y, n_scan_x = cartesian_vector.shape
    centers = _standardize_centers(centers, n_scan_y, n_scan_x)
    if scan_mask is None:
        scan_mask = np.ones((n_scan_y, n_scan_x), dtype=bool)
    else:
        scan_mask = np.asarray(scan_mask, dtype=bool)
        if scan_mask.shape != (n_scan_y, n_scan_x):
            raise ValueError(
                f"scan_mask shape {scan_mask.shape} must match {(n_scan_y, n_scan_x)}"
            )
    if sampling_conversion_factor is None:
        sampling_conversion_factor = 1.0

    x_idx = cartesian_vector.fields.index(x_field_found)
    y_idx = cartesian_vector.fields.index(y_field_found)
    extra_indices = [
        idx for idx in range(len(cartesian_vector.fields)) if idx not in (x_idx, y_idx)
    ]
    # ``theta`` is folded when two_fold_rotation_symmetry is set, which maps each
    # Friedel pair onto one angle. ``theta_unfolded`` keeps the full 0-2pi angle so
    # that information is not lost; peaks are small, so the extra column is cheap.
    output_fields = ["r_pixels", "theta", "r_invA", "theta_unfolded"] + [
        cartesian_vector.fields[idx] for idx in extra_indices
    ]
    output_units = [r_unit, theta_unit, "1/Å", theta_unit] + [
        cartesian_vector.units[idx] for idx in extra_indices
    ]
    polar_vector = Vector.from_shape(
        shape=(n_scan_y, n_scan_x),
        fields=output_fields,
        units=output_units,
        name=cartesian_vector.name + name_suffix,
        dtype=torch.float64,
    )

    theta_period = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    iterator = tqdm(range(n_scan_y), disable=not use_tqdm, desc="Polar transform peaks")
    for i in iterator:
        for j in range(n_scan_x):
            if not scan_mask[i, j]:
                polar_vector[i, j] = np.zeros((0, len(output_fields)))
                continue

            cartesian_data = cartesian_vector[i, j].numpy()
            if len(cartesian_data) == 0:
                polar_vector[i, j] = np.zeros((0, len(output_fields)))
                continue

            center_y, center_x = centers[i, j]
            dx = cartesian_data[:, x_idx] - center_x
            dy = cartesian_data[:, y_idx] - center_y
            r_pixels, theta = _cartesian_offsets_to_polar(dx, dy, ellipse_params)
            theta_unfolded = np.mod(theta, 2.0 * np.pi)
            theta = np.mod(theta, theta_period)
            r_invA = r_pixels * sampling_conversion_factor

            polar_data = np.column_stack([r_pixels, theta, r_invA, theta_unfolded])
            if extra_indices:
                polar_data = np.column_stack([polar_data, cartesian_data[:, extra_indices]])
            polar_vector[i, j] = polar_data

    return polar_vector


def _standardize_centers(centers, scan_y: int, scan_x: int) -> NDArray:
    centers = np.asarray(centers, dtype=float)
    if centers.shape == (2,):
        out = np.empty((scan_y, scan_x, 2), dtype=float)
        out[...] = centers
        return out
    if centers.shape == (scan_y, scan_x, 2):
        return centers
    if centers.shape == (2, scan_y, scan_x):
        return np.moveaxis(centers, 0, -1)
    raise ValueError(
        f"centers must have shape (2,), {(scan_y, scan_x, 2)}, "
        f"or {(2, scan_y, scan_x)}, got {centers.shape}."
    )


def _cartesian_offsets_to_polar(
    dx: NDArray,
    dy: NDArray,
    ellipse_params: tuple[float, float, float] | None,
) -> tuple[NDArray, NDArray]:
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    if ellipse_params is None:
        return np.hypot(dx, dy), np.arctan2(dy, dx)
    if len(ellipse_params) != 3:
        raise ValueError("ellipse_params must be (a, b, theta_deg).")

    a, b, theta_deg = ellipse_params
    theta = np.deg2rad(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = dx * cos_t + dy * sin_t
    v_prime = -dx * sin_t + dy * cos_t
    scaled_u = (b / a) * u
    r_pixels = np.hypot(scaled_u, v_prime)
    phi = np.arctan2(v_prime, scaled_u) + theta
    return r_pixels, phi
