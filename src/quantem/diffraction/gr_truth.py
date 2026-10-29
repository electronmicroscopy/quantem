"""
Adapted from PDFs_3batch by Arthur McCray (renamed 9/23/2026)

Three-body g3(r, r', theta) histograms of atomic structures (PDF3B), for 3D
cells and for 2D (monolayer / coplanar) structures.

Differences from PDFs_3batch.py
-------------------------------
1. Flat-axis detection: a non-periodic axis along which every atom shares the
   same coordinate is "flat". Centers are no longer trimmed along such an axis
   (a monolayer loses no neighbors above/below the plane), which previously
   emptied `center_inds` and crashed on `max([len(i) for i in inds_all])`.
   For genuinely thick non-periodic axes the trim margin is now measured from
   the atomic extent rather than the cell, so a thin film in a tall vacuum cell
   is handled correctly.
2. Areal density (atoms / A^2) is used when exactly one axis is flat, instead of
   a volumetric density taken from an arbitrary vacuum-padded cell.
3. 2D normalization: the coplanar ideal-gas pair measure is
       4 * pi * sigma^2 * r1 * r2 * dr^2 * dtheta
   i.e. linear in r and with NO sin(theta) factor. Dividing a 2D PADF by
   sin(theta) manufactures spurious intensity at 0 and 180 degrees.
4. `good_vals` now compares the theta bin index against the number of theta
   bins rather than against the literal value 180 (the original was only
   correct for dtheta == 1).

The 3D code path is unchanged; a fully periodic or thick non-periodic cell
behaves exactly as before.
"""

import numpy as np
import time

from scipy.spatial import KDTree
from datetime import timedelta

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np


class PDF3B(object):

    def __init__(self, device="cpu", v=1):
        if isinstance(device, int):
            self._xp = cp
            cp.cuda.runtime.setDevice(device)
        elif device.lower() == "cpu":
            self._xp = np
        self.device = device
        self._v = v
        return

    def pdf3B(
        self,
        atoms,
        dr=0.1,
        dtheta=1,
        hist_r_max=20,
        skip=1,
        batch_size=1,
        v=None,
        pbcs=None,
        subpixel_weights=True,
        flat_tol=1e-6,
    ):
        xp = self._xp
        stime = time.perf_counter()
        v = self._v if v is None else v
        vprint = print if v >= 1 else lambda *a, **k: None
        if skip == None:
            skip = 1
        if pbcs is None:
            pbcs = atoms.get_pbc()
        self.pbcs = pbcs

        bbox_np = np.array(atoms.cell[(0, 1, 2), (0, 1, 2)])
        bbox = xp.array(bbox_np)
        pbcs_np = np.copy(pbcs)
        pbcs = xp.array(pbcs)
        positions_np = np.array(atoms.positions)  # for KD tree

        ### wrap into the cell along periodic dims (KDTree requires 0 <= x < L)
        for i, p in enumerate(pbcs_np):
            if p:
                positions_np[:, i] = np.mod(positions_np[:, i], bbox_np[i])
                positions_np[positions_np[:, i] >= bbox_np[i], i] = 0.0  # fp edge case
        positions = xp.array(positions_np)

        vprint(f"Cell size (A): {bbox_np}")
        vprint(f"Using a max radius of {hist_r_max} A")

        ### FIX 1a: identify flat axes -- non-periodic with zero atomic extent.
        ### A monolayer has no neighbors to lose along such an axis, so every
        ### atom already has a complete neighborhood and must not be trimmed.
        flat_axes = [
            i
            for i in range(3)
            if not pbcs_np[i] and np.ptp(positions_np[:, i]) < flat_tol
        ]
        self.flat_axes = flat_axes
        if len(flat_axes) == 1:
            vprint(
                f"Detected flat (2D) axis {flat_axes[0]}: using areal density "
                "and 2D normalization"
            )
        elif len(flat_axes) > 1:
            vprint(
                f"WARNING: {len(flat_axes)} flat axes {flat_axes} -> structure is "
                "1D or a single point. Falling back to volumetric normalization, "
                "which is not correct for this geometry."
            )

        ### warn if the minimum-image convention is violated in a periodic dim.
        ### _get_vecs_pbcs and query_ball_tree both require L > 2 * hist_r_max
        for i, p in enumerate(pbcs_np):
            if p and bbox_np[i] <= 2 * hist_r_max:
                vprint(
                    f"WARNING: periodic axis {i} has L = {bbox_np[i]:.2f} A "
                    f"<= 2 * hist_r_max = {2 * hist_r_max} A. Minimum-image "
                    "distances and neighbor counts will be wrong -- tile the "
                    "cell first, e.g. atoms * (n, m, 1)."
                )

        ### FIX 1b: get center positions that aren't within r_max of a boundary
        ### without pbcs. Margins are measured from the atomic extent, not the
        ### cell, and flat axes are skipped entirely.
        goods = np.ones(len(positions_np)).astype("bool")
        for i, p in enumerate(pbcs_np):
            if p or i in flat_axes:
                continue
            lo = positions_np[:, i].min()
            hi = positions_np[:, i].max()
            goods *= (positions_np[:, i] > lo + hist_r_max) & (
                positions_np[:, i] < hi - hist_r_max
            )
        center_inds = np.where(goods)[0]
        if len(center_inds) == 0:
            raise ValueError(
                "No valid center atoms: every atom lies within hist_r_max of a "
                "non-periodic boundary. The atomic extent along each finite, "
                f"non-flat axis must exceed 2 * hist_r_max = {2 * hist_r_max} A. "
                "Reduce hist_r_max, use a thicker sample, or enable pbcs."
            )

        ### KD tree for getting neighbors, cpu cuz faster
        start_KD = time.perf_counter()
        bbox_np_pbcs = bbox_np * pbcs_np
        bbox_np_pbcs[pbcs_np.astype(bool)] += 1e-9
        tree_all = KDTree(positions_np, copy_data=True, boxsize=bbox_np_pbcs)
        center_inds_skip = center_inds[::skip]
        tree_centers = KDTree(
            positions_np[center_inds_skip], copy_data=True, boxsize=bbox_np_pbcs
        )
        inds_all = tree_centers.query_ball_tree(tree_all, hist_r_max)
        inds_all = [xp.array(i) for i in inds_all]
        end_KD = time.perf_counter()
        vprint(f"KDTree calc time: {end_KD - start_KD:.1e} s")

        ### Histogram coordinates
        hist_theta = xp.arange(0.0, 180 + dtheta, dtheta)
        hist_r = xp.arange(0.0, hist_r_max + dr, dr)
        num_bins_r = hist_r.shape[0]
        num_bins_theta = hist_theta.shape[0]
        hist_sig = xp.zeros((num_bins_theta, num_bins_r, num_bins_r))

        ### FIX 2: Density -- areal for a 2D structure, volumetric otherwise
        cell = xp.array(atoms.cell)
        if len(flat_axes) == 1:
            i, j = [k for k in range(3) if k != flat_axes[0]]
            area = xp.linalg.norm(xp.cross(cell[i], cell[j]))
            assert area > 0, "in-plane cell area is zero"
            dens = positions.shape[0] / area  # atoms / A^2
            vprint(f"Areal density = {float(dens):.4g} atoms / A^2")
        else:
            volume = xp.abs(xp.sum(cell[:, 0] * xp.cross(cell[:, 1], cell[:, 2])))
            assert volume > 0, "cell volume is zero"
            dens = positions.shape[0] / volume  # atoms / A^3

        ### making big arrays that will be filled
        maxinds = max([len(i) for i in inds_all])
        inds2 = xp.zeros((batch_size, maxinds)).astype("int")

        if hist_r_max > 15:
            split_bincount = True
            off_inds = xp.triu_indices(maxinds, k=1)
        else:
            split_bincount = False
            off_inds = xp.where(~xp.eye(maxinds, dtype=bool))

        ### exact upper bound: one entry per (center, pair) in the batch
        maxval_n = int(off_inds[0].shape[0])
        vals = xp.zeros((3, maxval_n * batch_size)).astype("int")
        if subpixel_weights:
            weights_full = xp.empty(maxval_n * batch_size)
            weights = weights_full
        else:
            weights_full = None
            weights = None

        vprint(f"Num total atoms in sim = {len(positions)}")
        if not np.all(pbcs_np) and len(flat_axes) < 3:
            vprint(f"Num valid center atoms = {len(center_inds)}")
        if skip != 1:
            vprint(
                f"Skip = {skip}, so calculating using {len(center_inds_skip)} atoms as centers"
            )
        vprint("Final shape will be: ", hist_sig.shape)

        for a0 in range(0, len(center_inds_skip), batch_size):
            b0 = min(a0 + batch_size, len(center_inds_skip))
            batch_center_inds = center_inds_skip[a0:b0]
            bsize = len(batch_center_inds)
            bnum_inds = [len(inds) for inds in inds_all[a0:b0]]
            for i in range(bsize):
                inds2[i, : bnum_inds[i]] = inds_all[a0 + i]

            vecs = self._get_vecs_pbcs(
                positions[batch_center_inds],
                positions[inds2[:bsize]],
                bbox=bbox,
                pbcs=pbcs,
            )
            for i in range(bsize):
                vecs[i, bnum_inds[i] :] = 0

            drs = xp.sqrt(xp.sum(vecs**2, axis=-1))

            thetas_full = self._get_thetas_all(vecs, vecs, deg=True)
            thetas = thetas_full[:, off_inds[0], off_inds[1]]
            thetas = thetas[:bsize].ravel()

            dr1s = drs[:, off_inds[0]]
            dr2s = drs[:, off_inds[1]]
            dr1s = dr1s[:bsize].ravel()
            dr2s = dr2s[:bsize].ravel()

            r_ind1 = dr1s / dr
            r_ind2 = dr2s / dr
            theta_ind = thetas / dtheta

            if subpixel_weights:
                ### this isn't perfect, doesn't do corners
                r_floor1 = xp.floor(r_ind1).astype("int")
                dr_ind1 = r_ind1 - r_floor1

                r_floor2 = xp.floor(r_ind2).astype("int")
                dr_ind2 = r_ind2 - r_floor2

                theta_floor = xp.floor(theta_ind).astype("int")
                dtheta_ind = theta_ind - theta_floor

            else:
                r_floor1 = xp.round(r_ind1).astype("int")
                r_floor2 = xp.round(r_ind2).astype("int")
                theta_floor = xp.round(theta_ind).astype("int")

            ### FIX 4: theta_floor is a BIN INDEX, so bound it by the number of
            ### theta bins. The original compared it against the literal 180,
            ### which was only correct for dtheta == 1.
            good_vals = (
                (r_floor1 > 1e-9)
                & (r_floor2 > 1e-9)
                & (theta_floor > 0)
                & (theta_floor < num_bins_theta - 1)
            )
            good_inds = xp.where(good_vals)[0]
            nvals = len(good_inds)
            vals[0, :nvals] = theta_floor[good_inds]
            vals[1, :nvals] = r_floor2[good_inds]
            vals[2, :nvals] = r_floor1[good_inds]
            if subpixel_weights:
                weights_full[:nvals] = (
                    dtheta_ind[good_inds] + dr_ind2[good_inds] + dr_ind1[good_inds]
                ) / 3
                weights = weights_full[:nvals]

            ### Because taking triu, have to do this twice
            ### and doing bincount twice with half indices seems to be ~10% faster than
            ### once with twice as many (for a 20A cutoff), but slower if < 15 A
            hist_sig += self._bincountdd(
                vals[:, :nvals],
                (num_bins_theta, num_bins_r, num_bins_r),
                weights=weights,
            )
            if split_bincount:
                vals[[1, 2]] = vals[[2, 1]]
                hist_sig += self._bincountdd(
                    vals[:, :nvals],
                    (num_bins_theta, num_bins_r, num_bins_r),
                    weights=weights,
                )

        hist_sig /= len(center_inds_skip)

        ### FIX 3: normalization
        gr = hist_sig.copy()
        r1 = hist_r[1:][None, None, ...]
        r2 = hist_r[1:][None, ..., None]
        th = hist_theta[1:-1, None, None]
        if len(flat_axes) == 1:
            ### 2D ideal-gas pair measure:
            ###   sigma^2 * (2 pi r1 dr) * (2 r2 dr dtheta)
            ### -> linear in r, and no sin(theta): the out-of-plane solid-angle
            ### band that produces sin(theta) in 3D does not exist for coplanar
            ### vectors. The factor 2 folds +/- azimuth into [0, 180].
            gr[1:-1, 1:, 1:] /= (
                dens**2
                * dr**2
                * 4  # 2 * 2pi, for all off-diagonal (ordered) pairs; use 2 for triu only
                * xp.pi
                * r1
                * r2
                * np.deg2rad(dtheta)
            )
        else:
            gr[1:-1, 1:, 1:] /= (
                dens**2
                * dr**2
                * 8  # if all off diag indices
                # * 4 # if triu
                * xp.pi**2
                * r1**2
                * r2**2
                * xp.sin(xp.deg2rad(th))
                * np.deg2rad(dtheta)
            )

        ### edge bins are half-width, and bin 0 is excluded by good_vals
        gr[-1] = 0
        gr[0] = 0
        gr[:, 0] = 0
        gr[:, :, 0] = 0

        vprint("-- done --")
        ttime = time.perf_counter() - stime
        vprint(
            f"Total time (h:m:s) {str(timedelta(seconds=round(ttime,3))).rstrip('0')}"
        )
        vprint(f"Center atoms per sec: {len(center_inds_skip) / ttime:.1f}\n")
        return gr

    def _get_vecs_pbcs(self, points, pointslists, bbox, pbcs=[1, 1, 1]):
        xp = self._xp
        if np.any(pbcs):
            assert xp.all(xp.min(pointslists, axis=1) >= 0)
            assert xp.all(xp.max(pointslists, axis=1) <= bbox)
        dif = pointslists - points[:, None]
        for i, ind in enumerate(pbcs):
            if ind:
                dif[:, :, i] = (
                    xp.mod(dif[:, :, i] + bbox[i] * 0.5, bbox[i]) - bbox[i] * 0.5
                )
        return dif

    def _bincountdd(self, vals, nbins, weights=None):
        # based on xp.histogramdd
        """
        vals: array shape (N, D)
        weights: array shape N
        nbins: (D,) tuple (nbinsz, nbinsy, nbinsx)
        based on xp.histogramdd
        """
        xp = self._xp
        vals = xp.floor(vals).astype("int")
        xy = xp.ravel_multi_index(vals, nbins, mode="wrap")
        minlength = int(np.prod(nbins))
        if weights is not None:
            hist = xp.bincount(xy, weights=1 - weights, minlength=minlength).reshape(
                nbins
            )
            xy = xp.ravel_multi_index(vals + 1, nbins, mode="wrap")
            hist += xp.bincount(xy, weights=weights, minlength=minlength).reshape(nbins)
        else:
            hist = xp.bincount(xy, minlength=minlength).reshape(nbins)
        return hist

    def _get_thetas_all(self, v1, v2, deg=True):
        """
        get angles between all vectors, outer product style
        """
        xp = self._xp
        v11 = self._norm_vecs(v1)
        v21 = self._norm_vecs(v2)
        mmul = xp.transpose(xp.matmul(v11, xp.transpose(v21, (0, 2, 1))), (0, 2, 1))
        theta = xp.arccos(xp.clip(mmul, -1.0, 1.0))
        if deg:
            theta = xp.rad2deg(theta)
        return theta

    def _norm_vecs(self, v):
        xp = self._xp
        if xp.ndim(v) == 1:
            return v / xp.sqrt(xp.sum(v**2, axis=-1))[..., None]
        mags = xp.sqrt(xp.sum(v**2, axis=-1))
        bads = xp.where(mags == 0)
        mags[bads] = 1
        normed = v / mags[..., None]
        normed[bads] = 0
        return normed
