"""Fast polyhedral template matching for 3D atomic models.

Every site is compared against one or more :class:`PolyhedralTemplate` objects
to find the rotation that best aligns the template's neighbor vectors with the
site's measured neighbor vectors.  The search is fully vectorized with PyTorch
(CPU or GPU), so all sites are processed in chunks rather than one at a time.

Algorithm (per site, for one template)
--------------------------------------
1. **Trial rotations.**  A reference pair of adjacent template vectors
   ``(t_a, t_b)`` defines an orthonormal frame.  Every ordered pair of measured
   neighbor vectors ``(p_i, p_j)`` defines another frame; the rotation mapping
   one frame onto the other is a trial orientation.  Pairs whose angle differs
   from the template pair angle by more than ``angle_tolerance`` are skipped.
2. **Scoring.**  Each rotated template vector is matched to its nearest
   measured neighbor; the score is the sum of ``clip(1 - d / score_radius, 0, 1)``
   over template vectors, so it ranges from 0 to the number of template
   vectors.  The best-scoring trial is kept.
3. **Refinement.**  Template vectors are assigned one-to-one to their nearest
   neighbors within ``score_radius`` and the rotation is re-fit with a batched
   Kabsch (SVD) solve.  The final score uses only these one-to-one matches, so
   a template vector cannot be "explained" by a neighbor already used.  A second least-squares solve yields the affine
   deformation gradient ``F`` (template -> measured), from which local strain
   is derived.

All neighbor vectors must be given in **nearest-neighbor bond-length units**
(i.e. divided by the mean NN distance), matching the template convention.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm.auto import tqdm

from quantem.atoms.templates import PolyhedralTemplate

__all__ = ["match_template", "TemplateMatch"]


class TemplateMatch(dict):
    """Result of :func:`match_template` for one template (a dict with fixed keys).

    Keys
    ----
    ``score`` : ``(N,)`` soft score in ``[0, 1]`` (normalized by template size).
    ``score_strained`` : ``(N,)`` score after the affine (strain-allowing) fit.
    ``num_matched`` : ``(N,)`` number of template vectors matched one-to-one.
    ``rmsd`` : ``(N,)`` RMS distance of matched vectors (NN units).
    ``rotation`` : ``(N, 3, 3)`` rotation ``R`` with ``p ~ R t`` (lab <- template).
    ``deformation`` : ``(N, 3, 3)`` deformation gradient ``F`` with ``p ~ F t``.
    ``matched`` : ``(N, M)`` neighbor-slot index matched to each template vector, or -1.
    """


def _frames(u: Tensor, v: Tensor) -> Tensor:
    """Right-handed orthonormal frames (columns) from batched vector pairs."""
    e1 = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v2 = v - e1 * (v * e1).sum(-1, keepdim=True)
    e2 = v2 / v2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def _soft_score(dmin: Tensor, score_radius: float) -> Tensor:
    return (1.0 - dmin / score_radius).clamp(0.0, 1.0).sum(-1)


def _kabsch(t: Tensor, p: Tensor, w: Tensor, fallback: Tensor) -> Tensor:
    """Batched Kabsch rotation with ``p ~ R t``; weights ``w`` select matches."""
    h = (t * w[..., None]).transpose(1, 2) @ p  # (N,3,3) = sum w t p^T
    h_cpu = h.detach().to("cpu").to(torch.float64)
    u, _, vt = torch.linalg.svd(h_cpu)
    v = vt.transpose(1, 2)
    d = torch.det(v @ u.transpose(1, 2))
    sign = torch.ones_like(d)
    sign[d < 0] = -1.0
    v = v.clone()
    v[:, :, 2] *= sign[:, None]
    r = (v @ u.transpose(1, 2)).to(fallback.device, fallback.dtype)
    enough = (w.sum(-1) >= 3).to(fallback.device)
    return torch.where(enough[:, None, None], r, fallback)


def _deformation(t: Tensor, p: Tensor, w: Tensor, fallback: Tensor, min_matched: int) -> Tensor:
    """Batched least-squares deformation gradient ``F`` with ``p ~ F t``."""
    tw = t * w[..., None]
    a = tw.transpose(1, 2) @ t  # (N,3,3) sum w t t^T
    b = tw.transpose(1, 2) @ p  # (N,3,3) sum w t p^T
    eye = torch.eye(3, device=t.device, dtype=t.dtype)
    a = a + 1e-6 * eye
    # F^T = A^{-1} B  ->  F = B^T A^{-1}
    ft = torch.linalg.solve(a.to("cpu").to(torch.float64), b.to("cpu").to(torch.float64))
    f = ft.transpose(1, 2).to(t.device, t.dtype)
    enough = w.sum(-1) >= min_matched
    return torch.where(enough[:, None, None], f, fallback)


def _assign_one_to_one(
    dist: Tensor, valid: Tensor, score_radius: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Match each template vector to its nearest valid neighbor, one-to-one.

    Parameters
    ----------
    dist : Tensor
        ``(N, M, K)`` distances between rotated template vectors and neighbors.
    valid : Tensor
        ``(N, K)`` neighbor validity mask.

    Returns
    -------
    dmin, j, keep : Tensor
        ``(N, M)`` nearest distance, neighbor slot index and match mask.
    """
    dist = dist.masked_fill(~valid[:, None, :], float("inf"))
    dmin, j = dist.min(-1)
    keep = dmin < score_radius
    n, _, k = dist.shape
    best = torch.full((n, k), float("inf"), device=dist.device, dtype=dist.dtype)
    best = best.scatter_reduce(1, j, dmin.masked_fill(~keep, float("inf")), reduce="amin")
    keep = keep & (dmin <= best.gather(1, j) + 1e-6)
    return dmin, j, keep


def match_template(
    dxyz: NDArray | Tensor,
    valid: NDArray | Tensor,
    template: PolyhedralTemplate,
    score_radius: float = 0.5,
    angle_tolerance: float = 30.0,
    num_pair_neighbors: int | None = None,
    num_refine: int = 2,
    chunk_size: int = 512,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> TemplateMatch:
    """Match one polyhedral template to every site.

    Parameters
    ----------
    dxyz : array or Tensor
        ``(N, K, 3)`` neighbor vectors in NN bond-length units, sorted by
        distance.
    valid : array or Tensor
        ``(N, K)`` boolean mask of usable neighbors (e.g. inside a radial cutoff).
    template : PolyhedralTemplate
        Template to match.
    score_radius : float
        Matching radius (NN units).  A template vector contributes
        ``1 - d / score_radius`` to the score when its nearest neighbor is at
        distance ``d``.
    angle_tolerance : float
        Trial neighbor pairs are only tested if their angle is within this many
        degrees of the reference template pair angle.
    num_pair_neighbors : int, optional
        Trial rotations are built from ordered pairs among the first this many
        neighbors.  Default ``min(K, M + 2)`` with ``M`` the template size.
    num_refine : int
        Number of assign-then-Kabsch refinement iterations.
    chunk_size : int
        Sites processed per batch (memory ~ ``chunk_size * P * M * K`` floats).
    device : str or torch.device, optional
        Torch device.  Default: ``quantem.core.config`` device, or CPU.
    progress : bool
        Show a progress bar.

    Returns
    -------
    TemplateMatch
        See :class:`TemplateMatch` for keys.  All values are NumPy arrays.
    """
    if device is None:
        from quantem.core import config

        device = config.get_device()
    device = torch.device(device)
    dtype = torch.float32

    p_all = torch.as_tensor(np.asarray(dxyz), dtype=dtype)
    valid_all = torch.as_tensor(np.asarray(valid), dtype=torch.bool)
    n, k, _ = p_all.shape
    t = torch.as_tensor(template.vectors, dtype=dtype, device=device)
    m = t.shape[0]
    k_pair = min(k, m + 2) if num_pair_neighbors is None else min(k, int(num_pair_neighbors))

    # reference template pair: vector 0 and its closest first-shell partner
    t_r = t.norm(dim=-1)
    first = t_r <= template.shells[0] * (1 + 1e-3)
    cos_t = (t @ t[0]) / (t_r * t_r[0])
    cos_t[0] = -2.0
    cos_t[~first] = -2.0
    j_ref = int(torch.argmax(cos_t))
    ref_angle = float(torch.arccos(cos_t[j_ref].clamp(-1, 1)))
    ref_frame = _frames(t[0][None], t[j_ref][None])[0]  # (3,3)
    tol = math.radians(angle_tolerance)
    ref_len_a, ref_len_b = float(t_r[0]), float(t_r[j_ref])

    ii, jj = torch.meshgrid(torch.arange(k_pair), torch.arange(k_pair), indexing="ij")
    pair_mask = ii != jj
    ii, jj = ii[pair_mask].to(device), jj[pair_mask].to(device)
    num_pairs = ii.numel()

    score = torch.zeros(n, dtype=dtype)
    score_strained = torch.zeros(n, dtype=dtype)
    num_matched = torch.zeros(n, dtype=torch.int64)
    rmsd = torch.full((n,), float("nan"), dtype=dtype)
    rotation = torch.eye(3, dtype=dtype).repeat(n, 1, 1)
    deformation = torch.eye(3, dtype=dtype).repeat(n, 1, 1)
    matched = torch.full((n, m), -1, dtype=torch.int64)
    min_matched_affine = max(4, m // 3)

    ranges = range(0, n, chunk_size)
    for start in tqdm(ranges, desc=f"match {template.name}", disable=not progress):
        sl = slice(start, min(start + chunk_size, n))
        p = p_all[sl].to(device)  # (Nc,K,3)
        pv = valid_all[sl].to(device)  # (Nc,K)
        nc = p.shape[0]
        p_safe = torch.where(pv[..., None], p, torch.full_like(p, 1e3))

        # ---- trial rotations from neighbor pairs -------------------------
        pi, pj = p[:, ii], p[:, jj]  # (Nc,P,3)
        li, lj = pi.norm(dim=-1), pj.norm(dim=-1)
        cos_ij = (pi * pj).sum(-1) / (li * lj).clamp_min(1e-12)
        ang = torch.arccos(cos_ij.clamp(-1, 1))
        ok = (ang - ref_angle).abs() < tol
        ok &= pv[:, ii] & pv[:, jj]
        ok &= ((li - ref_len_a).abs() < 0.35 * ref_len_a) & (
            (lj - ref_len_b).abs() < 0.35 * ref_len_b
        )
        frames = _frames(pi, pj)  # (Nc,P,3,3)
        rot = frames @ ref_frame.T  # (Nc,P,3,3): p ~ R t
        t_rot = torch.einsum("npab,mb->npma", rot, t)  # (Nc,P,M,3)
        dist = torch.cdist(
            t_rot.reshape(nc * num_pairs, m, 3), p_safe.repeat_interleave(num_pairs, 0)
        )
        dmin = dist.view(nc, num_pairs, m, k).min(-1).values
        trial_score = _soft_score(dmin, score_radius).masked_fill(~ok, -1.0)
        best = trial_score.argmax(dim=1)
        r_best = rot[torch.arange(nc, device=device), best]  # (Nc,3,3)
        del dist, dmin, t_rot, rot, frames

        # ---- refinement: one-to-one assignment + Kabsch ------------------
        t_b = t[None].expand(nc, m, 3)
        for _ in range(max(1, num_refine)):
            t_rot = t_b @ r_best.transpose(1, 2)
            dmin, j, keep = _assign_one_to_one(torch.cdist(t_rot, p_safe), pv, score_radius)
            w = keep.to(dtype)
            p_m = p_safe.gather(1, j[..., None].expand(-1, -1, 3))
            r_best = _kabsch(t_b, p_m, w, r_best)

        t_rot = t_b @ r_best.transpose(1, 2)
        dmin, j, keep = _assign_one_to_one(torch.cdist(t_rot, p_safe), pv, score_radius)
        w = keep.to(dtype)
        p_m = p_safe.gather(1, j[..., None].expand(-1, -1, 3))
        f_best = _deformation(t_b, p_m, w, r_best, min_matched_affine)
        t_def = t_b @ f_best.transpose(1, 2)
        dmin_def = (
            torch.cdist(t_def, p_safe).masked_fill(~pv[:, None, :], float("inf")).min(-1).values
        )

        n_match = keep.sum(-1)
        dmin_1to1 = dmin.masked_fill(~keep, float("inf"))
        _, _, keep_def = _assign_one_to_one(torch.cdist(t_def, p_safe), pv, score_radius)
        dmin_def = dmin_def.masked_fill(~keep_def, float("inf"))
        score[sl] = (_soft_score(dmin_1to1, score_radius) / m).cpu()
        score_strained[sl] = (_soft_score(dmin_def, score_radius) / m).cpu()
        num_matched[sl] = n_match.cpu()
        sq = (dmin**2 * w).sum(-1) / n_match.clamp_min(1).to(dtype)
        rmsd[sl] = torch.where(n_match > 0, sq.sqrt(), torch.full_like(sq, float("nan"))).cpu()
        rotation[sl] = r_best.cpu()
        deformation[sl] = f_best.cpu()
        matched[sl] = torch.where(keep, j, torch.full_like(j, -1)).cpu()

    return TemplateMatch(
        score=score.numpy(),
        score_strained=score_strained.numpy(),
        num_matched=num_matched.numpy(),
        rmsd=rmsd.numpy(),
        rotation=rotation.numpy(),
        deformation=deformation.numpy(),
        matched=matched.numpy(),
    )
