"""
Tomography geometry conventions.

Frame: x along the detector columns, y along the rows, z along the beam travel. This is the
lab frame of `quantem.core.utils.rotations` (row, col, z toward the source) rotated by 180
degrees about the row + col diagonal.

Pose: each projection has angles (z1, x, z3) in degrees, x being the tilt. Lab ray coordinates
map into the specimen as ``p_specimen = Rz(-z1) @ Rx(x) @ Rz(-z3) @ p_lab``, rotating about the
array centre (n - 1) / 2. `quat_from_tomography_angles` gives the same pose as a core
orientation. The INR rays (`TomographyINRDataset.transform_batch_rays`), `rot_ZXZ` and
`radon_torch` / `iradon_torch` all follow it.

Volumes: `rot_ZXZ` takes [C, Z, Y, X]. Reconstructions (`obj_view`, SIRT / FBP objects) are
[C, X, Z, Y].
"""

import torch
import torch.nn.functional as F

from quantem.core.utils.rotations import quat_from_tomography_angles, quat_to_matrix

# Tomography axes (x = col, y = row, z = beam travel) in the core lab frame (row, col, z).
_TOMO_TO_LAB = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
)

# --- Projection Operator Utils ---


def rot_ZXZ(mags, z1, x, z3, device=None, mode="bilinear"):
    """Rotate a [C, Z, Y, X] volume into the lab frame of a tomography (z1, x, z3) pose.

    Uses the same pose as the INR rays, so ``rot_ZXZ(vol, z1, x, z3).sum(-3)`` is the
    projection they see. The rotation is rigid in voxel units about the array centre, also
    for non-cubic volumes. Nothing in the reconstructions calls it; it makes tilt series of
    phantoms (the tests and the ``tomography_00_generate_phantom`` notebook).

    Args:
        mags: Volume of shape [C, Z, Y, X].
        z1, x, z3: Pose angles in degrees, scalars or tensors of shape (T,).
        device: Unused, kept for compatibility. The volume's device is used.
        mode: Interpolation mode of `grid_sample`.

    Returns:
        [C, Z, Y, X] for a single pose, [T, C, Z, Y, X] for T poses.
    """
    z1, x, z3 = (torch.as_tensor(a, device=mags.device) for a in (z1, x, z3))
    R = quat_to_matrix(quat_from_tomography_angles(z1, x, z3)).reshape(-1, 3, 3)
    P = _TOMO_TO_LAB.to(mags.device)
    # p_specimen = M @ p_lab, in tomography axes
    M = (P.T @ R.transpose(-1, -2) @ P).to(mags.dtype)

    C, D, H, W = mags.shape
    half = torch.tensor(
        [(W - 1) / 2, (H - 1) / 2, (D - 1) / 2], device=mags.device, dtype=mags.dtype
    )
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(n, device=mags.device, dtype=mags.dtype) for n in (D, H, W)),
        indexing="ij",
    )
    p_lab = torch.stack((xx, yy, zz), dim=-1) - half  # voxel units, (x, y, z)
    grid = torch.einsum("tij,dhwj->tdhwi", M, p_lab) / half
    out = F.grid_sample(mags.expand(len(M), -1, -1, -1, -1), grid, mode=mode, align_corners=True)
    return out[0] if len(M) == 1 else out


def tv_loss_1d(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    1D Total Variation Loss.

    Encourages piecewise smoothness by penalizing differences between
    adjacent elements.

    Args:
        x:         Input tensor of shape (N, C, L) or (N, L) or (L,)
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar loss (or per-sample tensor if reduction='none')
    """
    # Difference between adjacent elements along the last dimension
    diff = x[..., 1:] - x[..., :-1]  # shape: (..., L-1)
    tv = diff.abs()  # L1 variant  ← most common

    if reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()
    elif reduction == "none":
        return tv
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")
