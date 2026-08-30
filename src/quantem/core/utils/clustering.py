"""Density-based clustering (DBSCAN) in pure torch, and Vector helpers.

No external clustering dependency: neighbors are found with blockwise
distance computations pruned by a sliding sorted window along the widest
dimension, and cluster connectivity is resolved by min-label propagation
with path compression. Runs on CPU or any torch device.
"""

from __future__ import annotations

import numpy as np
import torch


def dbscan(
    points,
    eps: float,
    min_samples: int,
    device: str | torch.device = "cpu",
    block: int = 2048,
    sort_by_size: bool = True,
    max_rounds: int = 200,
) -> np.ndarray:
    """DBSCAN cluster labels for a point set.

    Same semantics as the standard algorithm: points with at least
    `min_samples` neighbors within `eps` (self included) are core points;
    core points within `eps` of each other share a cluster; non-core points
    within `eps` of a core point join that core's cluster (ties resolved to
    the NEAREST core, deterministically); everything else is noise (-1).

    Parameters
    ----------
    points : array-like (N, D)
        Point coordinates. Scale the columns beforehand to weight
        dimensions (the metric is plain Euclidean).
    eps : float
        Neighborhood radius.
    min_samples : int
        Neighbors (including the point itself) required for a core point.
    device : str | torch.device, default="cpu"
        Torch device for the distance computations.
    block : int, default=2048
        Rows per distance block; lower to reduce memory.
    sort_by_size : bool, default=True
        Renumber clusters largest-first (label 0 is the biggest cluster).

    Returns
    -------
    np.ndarray
        (N,) integer labels, -1 for noise.
    """
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
    N, D = pts.shape
    if N == 0:
        return np.zeros(0, dtype=int)

    # sort along the widest dimension so neighbor candidates live in a
    # contiguous window of the sorted order
    spans = pts.max(dim=0).values - pts.min(dim=0).values
    d0 = int(spans.argmax())
    order = torch.argsort(pts[:, d0])
    ps = pts[order]
    key = ps[:, d0].contiguous()

    def block_candidates(i0: int, i1: int) -> tuple[int, int]:
        lo = float(key[i0]) - eps
        hi = float(key[i1 - 1]) + eps
        j0 = int(torch.searchsorted(key, torch.tensor(lo, device=device)))
        j1 = int(torch.searchsorted(key, torch.tensor(hi, device=device), right=True))
        return j0, j1

    # pass 1: neighbor counts -> core points
    counts = torch.zeros(N, dtype=torch.long, device=device)
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        j0, j1 = block_candidates(i0, i1)
        d = torch.cdist(ps[i0:i1], ps[j0:j1])
        counts[i0:i1] = (d <= eps).sum(dim=1)
    core = counts >= min_samples

    # pass 2: min-label propagation among core points
    labels = torch.arange(N, dtype=torch.long, device=device)
    labels[~core] = -1
    for _ in range(max_rounds):
        changed = False
        for i0 in range(0, N, block):
            i1 = min(i0 + block, N)
            if not bool(core[i0:i1].any()):
                continue
            j0, j1 = block_candidates(i0, i1)
            d = torch.cdist(ps[i0:i1], ps[j0:j1])
            adj = (d <= eps) & core[i0:i1, None] & core[None, j0:j1]
            lab_nb = torch.where(
                adj, labels[j0:j1][None, :], torch.full_like(d, N, dtype=torch.long)
            )
            new = lab_nb.min(dim=1).values
            cur = labels[i0:i1]
            upd = core[i0:i1] & (new < cur)
            if bool(upd.any()):
                labels[i0:i1] = torch.where(upd, new, cur)
                changed = True
        # path compression: labels point at representative indices
        for _ in range(32):
            comp = torch.where(core, labels[labels.clamp_min(0)], labels)
            if bool(torch.equal(comp, labels)):
                break
            labels = comp
        if not changed:
            break

    # pass 3: border points join the nearest core cluster within eps
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        bmask = ~core[i0:i1]
        if not bool(bmask.any()):
            continue
        j0, j1 = block_candidates(i0, i1)
        d = torch.cdist(ps[i0:i1], ps[j0:j1])
        d = torch.where(
            core[None, j0:j1], d, torch.full_like(d, torch.inf)
        )
        d_min, j_min = d.min(dim=1)
        near = bmask & (d_min <= eps)
        if bool(near.any()):
            lab = labels[i0:i1]
            lab[near] = labels[j0 + j_min[near]]
            labels[i0:i1] = lab

    # back to input order, compact label ids
    out = torch.full((N,), -1, dtype=torch.long, device=device)
    out[order] = labels
    out_np = out.cpu().numpy()
    uniq, inv = np.unique(out_np[out_np >= 0], return_inverse=True)
    if uniq.size:
        if sort_by_size:
            sizes = np.bincount(inv)
            rank = np.empty_like(sizes)
            rank[np.argsort(sizes)[::-1]] = np.arange(sizes.size)
            out_np[out_np >= 0] = rank[inv]
        else:
            out_np[out_np >= 0] = inv
    return out_np


def cluster_vector(
    vector,
    fields,
    eps: float,
    min_samples: int,
    field_scales=None,
    scan_scales=None,
    device: str | torch.device = "cpu",
    label_field: str = "cluster",
):
    """DBSCAN over the rows of a Vector, in any combination of field and
    scan coordinates.

    Builds the clustering space from the named fields (optionally scaled per
    field) plus, when `scan_scales` is given, the scan-grid indices of each
    row (row, col of the cell it belongs to, scaled). Digital dark field
    clustering is the special case fields=(qx, qy) with scan_scales set.

    Parameters
    ----------
    vector : Vector
        Ragged vector over a scan grid.
    fields : sequence of str
        Field names contributing dimensions.
    eps, min_samples :
        DBSCAN parameters (Euclidean metric in the scaled space).
    field_scales : sequence of float | None
        Multiplier per field; default 1.
    scan_scales : (float, float) | None
        If given, append (row * s0, col * s1) of each row's scan cell.
    label_field : str, default="cluster"
        Name of the label field on the returned Vector.

    Returns
    -------
    labeled : Vector
        Copy of `vector` with the integer labels appended as a new field
        (-1 = noise).
    labels : np.ndarray
        The flat label array, aligned with vector.flatten().
    """
    flat = vector.select_fields(*fields).flatten().astype(float)
    if field_scales is not None:
        flat = flat * np.asarray(field_scales, dtype=float)[None, :]
    dims = [flat]
    if scan_scales is not None:
        counts = np.asarray(vector.row_counts(), dtype=int)
        shape = vector.shape[:2]
        cell_r, cell_c = np.divmod(np.arange(counts.size), shape[1])
        rr = np.repeat(cell_r, counts) * float(scan_scales[0])
        cc = np.repeat(cell_c, counts) * float(scan_scales[1])
        dims.append(np.stack([rr, cc], axis=1))
    space = np.concatenate(dims, axis=1)

    labels = dbscan(space, eps=eps, min_samples=min_samples, device=device)

    labeled = vector.copy()
    labeled.add_fields([label_field], units=["index"])
    full = labeled.flatten()
    full[:, -1] = labels
    labeled.set_flattened(full)
    return labeled, labels


def filter_rows(vector, mask):
    """Copy of a ragged Vector keeping only the flattened rows where mask.

    The scan-grid shape is unchanged; rows are dropped from their cells.
    """
    mask = np.asarray(mask, dtype=bool)
    counts = np.asarray(vector.row_counts(), dtype=int)
    flat = vector.flatten()
    starts = np.concatenate([[0], np.cumsum(counts)])
    shape = vector.shape[:2]
    nested = []
    for r in range(shape[0]):
        row = []
        for c in range(shape[1]):
            k = r * shape[1] + c
            sel = mask[starts[k]:starts[k + 1]]
            row.append(flat[starts[k]:starts[k + 1]][sel])
        nested.append(row)
    from quantem.core.datastructures.vector import Vector

    out = Vector.from_data(
        nested, fields=list(vector.fields), units=list(vector.units),
        name=vector.name,
    )
    out.metadata.update(vector.metadata)
    return out
