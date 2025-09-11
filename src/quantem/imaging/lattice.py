from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d


class Lattice(AutoSerialize):
    """
    Atomic lattice fitting in 2D.
    """

    _token = object()

    def __init__(
        self,
        image: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Lattice.from_data() to instantiate this class.")
        self._image: Dataset2d = image

    # --- Constructors ---
    @classmethod
    def from_data(
        cls,
        image: Union[Dataset2d, NDArray],
        normalize_min: bool = True,
        normalize_max: bool = True,
    ) -> "Lattice":
        if isinstance(image, Dataset2d):
            ds2d = image
        else:
            arr = ensure_valid_array(image, ndim=2)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]
        if normalize_min:
            ds2d.array -= np.min(ds2d.array)
        if normalize_max:
            ds2d.array /= np.max(ds2d.array)
        return cls(image=ds2d, _token=cls._token)

    # --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Union[Dataset2d, NDArray]):
        if isinstance(value, Dataset2d):
            self._image = value
        else:
            arr = ensure_valid_array(value, ndim=2)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                self._image = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                self._image = Dataset2d(arr)  # type: ignore[call-arg]

    # --- Functions ---
    def define_lattice(
        self,
        origin,
        u,
        v,
        block_size: int = -1,
        plot_lattice=True,
        bound_num_vectors=None,
        mask=None,
        refine_lattice=True,
        refine_maxiter: int = 200,
        **kwargs,
    ):
        # Lattice
        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )

        # Refine lattice coordinates
        # Note that we currently assume corners are local maxima
        if refine_lattice:
            from scipy.optimize import minimize

            H, W = self._image.shape  # rows (x), cols (y)
            im = np.asarray(self._image.array, dtype=float)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)  # (x, y)

            corners = np.array(
                [
                    [0.0, 0.0],
                    [float(H), 0.0],
                    [0.0, float(W)],
                    [float(H), float(W)],
                ],
                dtype=float,
            )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))  # (2,2)
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)

            a_min, a_max = int(np.floor(ab[0].min())), int(np.ceil(ab[0].max()))
            b_min, b_max = int(np.floor(ab[1].min())), int(np.ceil(ab[1].max()))

            max_ind = max(abs(a_min), a_max, abs(b_min), b_max)
            steps = (
                [*np.arange(0, max_ind + 1, block_size)[1:], max_ind] if max_ind > 0 else [max_ind]
            )

            PENALTY = 1e10
            H_CLIP = H - 2
            W_CLIP = W - 2

            a_range = np.arange(max(a_min, -max_ind), min(a_max, max_ind) + 1, dtype=np.int32)
            b_range = np.arange(max(b_min, -max_ind), min(b_max, max_ind) + 1, dtype=np.int32)
            aa, bb = np.meshgrid(a_range, b_range, indexing="ij")

            # Pre-compute all masks and bases
            all_masks = {}
            all_bases = {}
            for curr_block_size in steps:
                a_min_blk = max(a_min, -curr_block_size)
                a_max_blk = min(a_max, curr_block_size)
                b_min_blk = max(b_min, -curr_block_size)
                b_max_blk = min(b_max, curr_block_size)

                mask = (
                    (aa >= a_min_blk) & (aa <= a_max_blk) & (bb >= b_min_blk) & (bb <= b_max_blk)
                )

                aa_masked = aa[mask]
                bb_masked = bb[mask]

                all_masks[curr_block_size] = mask
                all_bases[curr_block_size] = np.column_stack(
                    [np.ones(aa_masked.size), aa_masked.ravel(), bb_masked.ravel()]
                )

            # Pre-allocate cache
            max_points = max(basis.shape[0] for basis in all_bases.values())
            x0_cache = np.empty(max_points, dtype=np.int32)
            y0_cache = np.empty(max_points, dtype=np.int32)
            dx_cache = np.empty(max_points, dtype=np.float64)
            dy_cache = np.empty(max_points, dtype=np.float64)

            def bilinear_sum(im_: np.ndarray, xy: np.ndarray) -> float:
                """Sum of bilinearly interpolated intensities at (x,y) points."""

                n_points = xy.shape[0]
                if n_points == 0:
                    return 0.0

                x, y = xy[:, 0], xy[:, 1]

                # Filter points that are within valid bounds for bilinear interpolation
                # Need x in [0, H-2] and y in [0, W-2] so that x+1 and y+1 are valid
                valid_mask = (
                    (x >= 0)
                    & (x <= H_CLIP)
                    & (y >= 0)
                    & (y <= W_CLIP)
                    & np.isfinite(x)
                    & np.isfinite(y)
                )

                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    return -PENALTY

                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                # Use pre-allocated arrays
                x0, y0 = x0_cache[:n_valid], y0_cache[:n_valid]
                dx, dy = dx_cache[:n_valid], dy_cache[:n_valid]

                np.floor(x_valid, out=dx)
                x0[:] = dx.astype(np.int32)
                np.floor(y_valid, out=dy)
                y0[:] = dy.astype(np.int32)

                np.subtract(x_valid, x0, out=dx)
                np.subtract(y_valid, y0, out=dy)

                Ia = im_[x0, y0]
                Ib = im_[x0 + 1, y0]
                Ic = im_[x0, y0 + 1]
                Id = im_[x0 + 1, y0 + 1]

                return np.sum(
                    Ia * (1 - dx) * (1 - dy)
                    + Ib * dx * (1 - dy)
                    + Ic * (1 - dx) * dy
                    + Id * dx * dy
                )

            current_basis = None

            def objective(theta: np.ndarray) -> float:
                # theta is 6-vector -> (3,2) matrix [[r0],[u],[v]]
                lat = theta.reshape(3, 2)
                xy = current_basis @ lat  # (N,2) with columns (x,y)
                # Negative: maximize intensity sum by minimizing its negative
                return -bilinear_sum(im, xy)

            minimize_options = {
                "maxiter": int(refine_maxiter),
                "xtol": 1e-3,
                "ftol": 1e-3,
                "disp": False,
            }

            lat_flat = self._lat.astype(np.float32).reshape(-1)

            for curr_block_size in steps:
                current_basis = all_bases[curr_block_size]

                res = minimize(
                    objective,
                    lat_flat,
                    method="Powell",
                    options=minimize_options,
                )

                # Update for next iteration
                lat_flat = res.x
                self._lat = res.x.reshape(3, 2)

        # plotting
        if plot_lattice:
            fig, ax = show_2d(
                self._image.array,
                returnfig=True,
                **kwargs,
            )

            # Put the image at lowest zorder so overlays sit on top
            if ax.images:
                ax.images[-1].set_zorder(0)

            H, W = self._image.shape  # rows (x), cols (y)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)  # each (x, y) == (row, col)

            # -------------------------------
            # Origin marker (TOP of stack)
            # -------------------------------
            ax.scatter(
                r0[1],
                r0[0],  # (y, x)
                s=60,
                edgecolor=(0, 0, 0),
                facecolor=(0, 0.5, 0),
                marker="s",
                zorder=30,
            )

            # -------------------------------
            # Lattice vectors as arrows
            # -------------------------------
            n_vec = int(bound_num_vectors) if bound_num_vectors is not None else 1

            # draw n_vec arrows for u (red)
            for k in range(1, n_vec + 1):
                tip = r0 + k * u
                ax.arrow(
                    r0[1],
                    r0[0],  # base (y, x)
                    (tip - r0)[1],
                    (tip - r0)[0],  # delta (y, x)
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color="red",
                    zorder=20,
                )

            # draw n_vec arrows for v (cyan)
            for k in range(1, n_vec + 1):
                tip = r0 + k * v
                ax.arrow(
                    r0[1],
                    r0[0],
                    (tip - r0)[1],
                    (tip - r0)[0],
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color=(0.0, 0.7, 1.0),
                    zorder=20,
                )

            # -----------------------------------------
            # Solve for a,b at plot corners (bounds)
            # -----------------------------------------
            if bound_num_vectors is None:
                corners = np.array(
                    [
                        [0.0, 0.0],
                        [float(H), 0.0],
                        [0.0, float(W)],
                        [float(H), float(W)],
                    ]
                )
            else:
                n = float(bound_num_vectors)
                corners = np.array(
                    [
                        r0 - n * u,
                        r0 - n * v,
                        r0 + n * u,
                        r0 + n * v,
                    ],
                    dtype=float,
                )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))  # shape (2,2)
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)

            a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
            b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

            # -----------------------------------------
            # Clipping rectangle (image or custom)
            # -----------------------------------------
            if bound_num_vectors is None:
                x_lo, x_hi = 0.0, float(H)  # rows
                y_lo, y_hi = 0.0, float(W)  # cols
            else:
                # Bounds are the min/max over the provided corners
                x_lo, x_hi = float(np.min(corners[:, 0])), float(np.max(corners[:, 0]))
                y_lo, y_hi = float(np.min(corners[:, 1])), float(np.max(corners[:, 1]))

            def clipped_segment(base: np.ndarray, direction: np.ndarray):
                """Clip base + t*direction to rectangle [x_lo,x_hi] x [y_lo,y_hi]."""
                x0, y0 = base
                dx, dy = direction
                t0, t1 = -np.inf, np.inf
                eps = 1e-12

                # x in [x_lo, x_hi]
                if abs(dx) < eps:
                    if not (x_lo <= x0 <= x_hi):
                        return None
                else:
                    tx0 = (x_lo - x0) / dx
                    tx1 = (x_hi - x0) / dx
                    t_enter, t_exit = (tx0, tx1) if tx0 <= tx1 else (tx1, tx0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                # y in [y_lo, y_hi]
                if abs(dy) < eps:
                    if not (y_lo <= y0 <= y_hi):
                        return None
                else:
                    ty0 = (y_lo - y0) / dy
                    ty1 = (y_hi - y0) / dy
                    t_enter, t_exit = (ty0, ty1) if ty0 <= ty1 else (ty1, ty0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                if t0 > t1:
                    return None

                p1 = base + t0 * direction  # (x, y)
                p2 = base + t1 * direction
                return p1, p2

            # -----------------------------------------
            # Lattice lines (zorder above image)
            # Using x=rows, y=cols: plot(y, x)
            # -----------------------------------------

            # Lines parallel to v (vary a)
            for a in range(a_min, a_max + 1):
                base = r0 + a * u
                seg = clipped_segment(base, v)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color=(0.0, 0.7, 1.0), lw=1, clip_on=True, zorder=10)

            # Lines parallel to u (vary b)
            for b in range(b_min, b_max + 1):
                base = r0 + b * v
                seg = clipped_segment(base, u)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color="red", lw=1, clip_on=True, zorder=10)

            # Axes limits (x=rows vertical; y=cols horizontal)
            ax.set_xlim(y_lo, y_hi)
            ax.set_ylim(x_hi, x_lo)

        return self

    def add_atoms(
        self,
        positions_frac,
        numbers=None,
        intensity_min=None,
        intensity_radius=None,
        plot_atoms=True,
        *,
        edge_min_dist_px=None,
        mask=None,
        contrast_min=None,
        annulus_radii=None,
        **kwargs,
    ):
        self._positions_frac = np.array(positions_frac, dtype=float)
        self._num_sites = self._positions_frac.shape[0]
        self._numbers = (
            np.arange(1, self._num_sites + 1, dtype=int)
            if numbers is None
            else np.array(numbers, dtype=int)
        )

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape  # x=rows, y=cols
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        corners = np.array(
            [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]], dtype=float
        )
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        def _auto_radius_px() -> float:
            S = self._positions_frac
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
        rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
        R_disk = int(np.ceil(r_px))
        R_ring = int(np.ceil(rout))
        edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

        DT = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
            try:
                from scipy.ndimage import distance_transform_edt

                DT = distance_transform_edt(m)
            except Exception:
                DT = None

        def mean_disk(x: float, y: float) -> float:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
            j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
            if vals.size == 0:
                return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
            return float(vals.mean())

        def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
            j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            r2 = dx * dx + dy * dy
            mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
            if vals.size == 0:
                val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                return val, 0.0
            return float(vals.mean()), float(vals.std(ddof=0))

        self.atoms = Vector.from_shape(
            shape=(self._num_sites,),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )

        for a0 in range(self._num_sites):
            da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]
            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T
            xy = basis @ self._lat  # (N,2) in (x,y)

            x, y = xy[:, 0], xy[:, 1]
            in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
            border_ok = (
                (x - edge_thresh >= 0.0)
                & (x + edge_thresh <= H - 1)
                & (y - edge_thresh >= 0.0)
                & (y + edge_thresh <= W - 1)
            )

            if mask is not None:
                if DT is not None:
                    ii = np.clip(np.round(x).astype(int), 0, H - 1)
                    jj = np.clip(np.round(y).astype(int), 0, W - 1)
                    mask_ok = DT[ii, jj] >= edge_thresh
                else:
                    m = np.asarray(mask).astype(bool)
                    mask_ok = m[
                        np.clip(np.round(x).astype(int), 0, H - 1),
                        np.clip(np.round(y).astype(int), 0, W - 1),
                    ]
            else:
                mask_ok = np.ones_like(in_bounds, dtype=bool)

            int_center = np.empty(xy.shape[0], dtype=float)
            for i in range(xy.shape[0]):
                int_center[i] = mean_disk(x[i], y[i])

            keep = in_bounds & border_ok & mask_ok
            if intensity_min is not None:
                keep &= int_center >= float(intensity_min)
            if contrast_min is not None:
                bg_mean = np.empty(xy.shape[0], dtype=float)
                for i in range(xy.shape[0]):
                    bg_mean[i], _ = mean_std_annulus(x[i], y[i])
                keep &= (int_center - bg_mean) >= float(contrast_min)

            if np.any(keep):
                arr = np.vstack(
                    (x[keep], y[keep], basis[keep, 1], basis[keep, 2], int_center[keep])
                ).T
            else:
                arr = np.zeros((0, 5), dtype=float)

            # --- Correct API usage ---
            self.atoms.set_data(arr, a0)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for a0 in range(self._num_sites):
                cell = self.atoms.get_data(a0)
                if isinstance(cell, list) or cell is None or cell.size == 0:
                    continue
                x = self.atoms[a0]["x"]
                y = self.atoms[a0]["y"]
                rgb = site_colors(int(self._numbers[a0]))
                ax.scatter(
                    y,
                    x,
                    s=18,
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=18,
                )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def refine_atoms(
        self,
        fit_radius=None,
        max_nfev: int = 200,
        max_move_px: float | None = None,
        plot_atoms: bool = False,
        **kwargs,
    ):
        import numpy as np

        if not hasattr(self, "atoms"):
            raise ValueError("No atoms to refine. Call add_atoms() first.")

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = np.asarray(getattr(self, "_positions_frac", [[0.0, 0.0]]), dtype=float)
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_fit = float(fit_radius) if fit_radius is not None else _auto_radius_px()
        R = int(np.ceil(r_fit))
        max_move = float(max_move_px) if max_move_px is not None else r_fit

        # Ensure extra fields exist
        needed = [f for f in ("sigma", "int_bg") if f not in self.atoms.fields]
        if needed:
            self.atoms.add_fields(needed)

        # Single lookup of column indices for writing
        idx_x = self.atoms.fields.index("x")
        idx_y = self.atoms.fields.index("y")
        idx_amp = self.atoms.fields.index("int_peak")
        idx_sigma = self.atoms.fields.index("sigma")
        idx_bg = self.atoms.fields.index("int_bg")

        for s in range(self._num_sites):
            row = self.atoms.get_data(s)
            if isinstance(row, list) or row is None or row.size == 0:
                continue

            # Intuitive reads: per-cell field arrays
            x_arr = self.atoms[s]["x"]
            y_arr = self.atoms[s]["y"]

            updated = row.copy()
            for i in range(row.shape[0]):
                x0, y0 = float(x_arr[i]), float(y_arr[i])

                ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
                i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
                j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)
                if i1 <= i0 or j1 <= j0:
                    continue

                patch = im[i0 : i1 + 1, j0 : j1 + 1]

                # broadcast coordinate grids to patch shape
                ii = np.arange(i0, i1 + 1)[:, None]
                jj = np.arange(j0, j1 + 1)[None, :]
                II = np.broadcast_to(ii, patch.shape)
                JJ = np.broadcast_to(jj, patch.shape)

                r2 = (II - x0) ** 2 + (JJ - y0) ** 2
                mask = r2 <= (r_fit * r_fit)
                if not np.any(mask):
                    continue

                vals = patch[mask].astype(float).ravel()
                pmin, pmax = float(vals.min()), float(vals.max())
                bg0 = float(np.median(patch[~mask])) if np.any(~mask) else float(np.median(patch))
                amp0 = max(float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6)
                sig0 = max(r_fit * 0.5, 0.5)

                x_coords = II[mask].astype(float).ravel()
                y_coords = JJ[mask].astype(float).ravel()

                def residual(theta):
                    x_c, y_c, amp, sig, bg = theta
                    sig2 = max(sig, 1e-6) ** 2
                    rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2
                    model = amp * np.exp(-0.5 * rr / sig2) + bg
                    return model - vals

                # movement-limited bounds + image bounds
                x_lb = max(x0 - max_move, 0.0)
                x_ub = min(x0 + max_move, H - 1.0)
                y_lb = max(y0 - max_move, 0.0)
                y_ub = min(y0 + max_move, W - 1.0)

                lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
                ub = [
                    x_ub,
                    y_ub,
                    max(pmax - pmin, amp0 * 4.0),
                    max(2.0 * r_fit, 1.0),
                    pmax + (pmax - pmin),
                ]
                theta0 = [x0, y0, amp0, sig0, bg0]

                res = least_squares(
                    residual,
                    theta0,
                    bounds=(lb, ub),
                    method="trf",
                    loss="soft_l1",
                    max_nfev=int(max_nfev),
                    xtol=1e-6,
                    ftol=1e-6,
                    gtol=1e-6,
                )

                x_c, y_c, amp, sig, bg = res.x
                updated[i, idx_x] = x_c
                updated[i, idx_y] = y_c
                updated[i, idx_amp] = amp
                updated[i, idx_sigma] = sig
                updated[i, idx_bg] = bg

            self.atoms.set_data(updated, s)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for s in range(self._num_sites):
                cell = self.atoms.get_data(s)
                if isinstance(cell, list) or cell is None or cell.size == 0:
                    continue
                xs = self.atoms[s]["x"]
                ys = self.atoms[s]["y"]
                rgb = site_colors(int(self._numbers[s]))
                ax.scatter(
                    ys,
                    xs,
                    s=18,
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=25,
                )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def measure_polarization(
        self,
        measure_ind,
        reference_ind,
        reference_radius=None,
        reference_num=4,
        plot_polarization_vectors: bool = False,
        **plot_kwargs,
    ):
        from scipy.spatial import cKDTree

        # lattice vectors in pixels
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        if reference_radius is None:
            reference_radius = float(min(np.linalg.norm(u), np.linalg.norm(v)))

        # grab cells (skip if empty)
        A_cell = self.atoms.get_data(int(measure_ind))
        B_cell = self.atoms.get_data(int(reference_ind))
        if (
            isinstance(A_cell, list)
            or A_cell is None
            or A_cell.size == 0
            or isinstance(B_cell, list)
            or B_cell is None
            or B_cell.size == 0
        ):
            out = Vector.from_shape(
                shape=(1,),
                fields=("x", "y", "a", "b", "x_ref", "y_ref"),
                units=("px", "px", "ind", "ind", "px", "px"),
                name="polarization",
            )
            out.set_data(np.zeros((0, 6), float), 0)
            return out

        # field access via _CellView
        Ax = self.atoms[int(measure_ind)]["x"]
        Ay = self.atoms[int(measure_ind)]["y"]
        Aa = self.atoms[int(measure_ind)]["a"]
        Ab = self.atoms[int(measure_ind)]["b"]
        Bx = self.atoms[int(reference_ind)]["x"]
        By = self.atoms[int(reference_ind)]["y"]

        # KD-tree on reference coordinates
        tree = cKDTree(np.column_stack([Bx, By]))
        k = int(max(1, reference_num))
        dists, idxs = tree.query(
            np.column_stack([Ax, Ay]),
            k=k,
            distance_upper_bound=float(reference_radius),
            workers=-1,
        )
        if k == 1:  # normalize shapes
            dists = dists[:, None]
            idxs = idxs[:, None]

        x_list, y_list, a_list, b_list, xr_list, yr_list = [], [], [], [], [], []
        for i in range(Ax.shape[0]):
            valid = np.isfinite(dists[i]) & (idxs[i] < Bx.shape[0])
            if np.count_nonzero(valid) < reference_num:
                continue
            order = np.argsort(dists[i][valid])[:reference_num]
            nbr_idx = idxs[i][valid][order].astype(int)
            x_ref = float(np.mean(Bx[nbr_idx]))
            y_ref = float(np.mean(By[nbr_idx]))

            x_list.append(float(Ax[i]))
            y_list.append(float(Ay[i]))
            a_list.append(float(Aa[i]))
            b_list.append(float(Ab[i]))
            xr_list.append(x_ref)
            yr_list.append(y_ref)

        out = Vector.from_shape(
            shape=(1,),
            fields=("x", "y", "a", "b", "x_ref", "y_ref"),
            units=("px", "px", "ind", "ind", "px", "px"),
            name="polarization",
        )
        if len(x_list) == 0:
            out.set_data(np.zeros((0, 6), float), 0)
            return out

        arr = np.column_stack([x_list, y_list, a_list, b_list, xr_list, yr_list]).astype(float)
        out.set_data(arr, 0)

        if plot_polarization_vectors:
            self.plot_polarization_vectors(out, **plot_kwargs)

        return out

    def plot_polarization_vectors(
        self,
        pol_vec: "Vector",
        length_scale: float = 1.0,
        show_image: bool = True,
        figsize=(6, 6),
        subtract_median: bool = False,
        linewidth: float = 1.0,
        tail_width: float = 1.0,
        headwidth: float = 4.0,
        headlength: float = 4.0,
        outline: bool = True,
        outline_width: float = 2.0,
        outline_color: str = "black",
        alpha: float = 1.0,
        show_ref_points: bool = False,
        chroma_boost: float = 2.0,
        use_magnitude_lightness: bool = True,
        ref_marker: str = "o",
        ref_size: float = 20.0,
        ref_edge: str = "k",
        ref_face: str = "none",
        show_colorbar: bool = True,
        disp_color_max: float | None = None,
        phase_offset_deg: float = 180.0,  # red = down
        phase_dir_flip: bool = False,  # flip color direction if desired
        **kwargs,
    ):
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from quantem.core.visualization.visualization_utils import array_to_rgba

        data = pol_vec.get_data(0)
        if isinstance(data, list) or data is None or data.size == 0:
            if show_image:
                fig, ax = show_2d(self._image.array, returnfig=True, figsize=figsize, **kwargs)
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
            H, W = self._image.shape
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_title("polarization" + (" (median subtracted)" if subtract_median else ""))
            plt.tight_layout()
            return fig, ax

        # Fields
        xA = pol_vec[0]["x"]
        yA = pol_vec[0]["y"]
        xR = pol_vec[0]["x_ref"]
        yR = pol_vec[0]["y_ref"]

        # Displacements (rows, cols)
        dr_raw = (xA - xR).astype(float)  # down +
        dc_raw = (yA - yR).astype(float)  # right +

        # --- Unified color mapping (identical across scripts) ---
        dr, dc, amp, disp_cap_px = _compute_polar_color_mapping(
            dr_raw,
            dc_raw,
            subtract_median=subtract_median,
            use_magnitude_lightness=use_magnitude_lightness,
            disp_color_max=disp_color_max,
        )

        # Angle mapping consistent with legend (down=0°, right=+90°, up=180°, left=-90°)
        ang = np.arctan2(dc, dr)
        if phase_dir_flip:
            ang = -ang
        ang += np.deg2rad(phase_offset_deg)

        # Colors
        rgba = array_to_rgba(amp, ang, chroma_boost=chroma_boost)
        colors = rgba.reshape(-1, 4)[:, :3] if rgba.ndim != 2 else rgba[:, :3]

        # Background
        if show_image:
            fig, ax = show_2d(self._image.array, returnfig=True, figsize=figsize, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Draw arrows (colored patch with black stroke beneath via path effects)
        arrowstyle = ArrowStyle.Simple(
            head_length=headlength, head_width=headwidth, tail_width=tail_width
        )
        for i in range(xA.size):
            x0, y0 = float(xA[i]), float(yA[i])
            x1 = x0 + float(dr[i]) * float(length_scale)
            y1 = y0 + float(dc[i]) * float(length_scale)

            arrow = FancyArrowPatch(
                (y0, x0),
                (y1, x1),
                arrowstyle=arrowstyle,
                mutation_scale=1.0,
                linewidth=linewidth,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=alpha,
                zorder=11,
                capstyle="round",
                joinstyle="round",
                shrinkA=0.0,
                shrinkB=0.0,
            )
            if outline:
                arrow.set_path_effects(
                    [
                        pe.Stroke(linewidth=linewidth + outline_width, foreground=outline_color),
                        pe.Normal(),
                    ]
                )
            ax.add_patch(arrow)

        if show_ref_points:
            ax.scatter(
                yR,
                xR,
                s=ref_size,
                marker=ref_marker,
                facecolors=ref_face,
                edgecolors=ref_edge,
                linewidths=1.0,
                zorder=12,
            )

        H, W = self._image.shape
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title("polarization" + (" (median subtracted)" if subtract_median else ""))
        plt.tight_layout()

        # Circular legend (same mapping and label)
        if show_colorbar:
            divider = make_axes_locatable(ax)
            ax_c = divider.append_axes("right", size="28%", pad="6%")

            N = 256
            yy = np.linspace(-1, 1, N)
            xx = np.linspace(-1, 1, N)
            YY, XX = np.meshgrid(yy, xx, indexing="ij")
            rr = np.sqrt(XX**2 + YY**2)
            disk = rr <= 1.0

            ang_grid = np.arctan2(XX, -YY)
            if phase_dir_flip:
                ang_grid = -ang_grid
            ang_grid += np.deg2rad(phase_offset_deg)

            amp_grid = np.clip(rr, 0, 1)
            rgba_grid = array_to_rgba(amp_grid, ang_grid, chroma_boost=chroma_boost)
            rgba_grid[~disk] = 0.0

            ax_c.imshow(
                rgba_grid, origin="lower", extent=(-1, 1, -1, 1), interpolation="nearest", zorder=0
            )
            ax_c.set_aspect("equal")
            ax_c.axis("off")

            ring = Circle((0, 0), 0.98, facecolor="none", edgecolor="k", linewidth=1.2, zorder=3)
            ring.set_clip_on(False)
            ax_c.add_patch(ring)

            # Cardinal labels (down/right/up/left)
            ax_c.text(0.00, -1.12, "0°", ha="center", va="top", fontsize=9, color="k")
            ax_c.text(1.12, 0.00, "90°", ha="left", va="center", fontsize=9, color="k")
            ax_c.text(0.00, 1.12, "180°", ha="center", va="bottom", fontsize=9, color="k")
            ax_c.text(-1.12, 0.00, "270°", ha="right", va="center", fontsize=9, color="k")

            # Scale arrow along +x, label centered above midpoint (white)
            scale_len = 0.85
            arrow_scale = FancyArrowPatch(
                (0.0, 0.0),
                (scale_len, 0.0),
                arrowstyle=ArrowStyle.Simple(head_length=10.0, head_width=6.0, tail_width=2.0),
                mutation_scale=1.0,
                linewidth=1.2,
                facecolor="k",
                edgecolor="k",
                zorder=4,
                shrinkA=0.0,
                shrinkB=0.0,
            )
            arrow_scale.set_clip_on(False)
            ax_c.add_patch(arrow_scale)

            mid_x, mid_y = scale_len / 2.0, 0.0
            ax_c.text(
                mid_x,
                mid_y + 0.14,
                f"{disp_cap_px:.2g} px",
                ha="center",
                va="bottom",
                fontsize=9,
                color="w",
            )

            # Crosshairs & generous limits to avoid clipping
            ax_c.plot([0, 0], [-0.9, 0.9], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
            ax_c.plot([-0.9, 0.9], [0, 0], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
            ax_c.set_xlim(-1.35, 1.35)
            ax_c.set_ylim(-1.25, 1.35)

        return fig, ax

    def plot_polarization_image(
        self,
        pol_vec: "Vector",
        *,
        pixel_size: int = 16,
        padding: int = 8,
        spacing: int = 2,
        subtract_median: bool = False,
        chroma_boost: float = 2.0,
        use_magnitude_lightness: bool = True,
        disp_color_max: float | None = None,
        phase_offset_deg: float = 180.0,  # red = down (your convention)
        phase_dir_flip: bool = False,  # flip global hue mapping if desired
        aggregator: str = "mean",  # 'mean' or 'maxmag'
        plot: bool = False,  # if True, draw with show_2d and legend
        returnfig: bool = False,  # if True (and plot=True) also return (fig, ax)
        show_colorbar: bool = True,
        figsize=(6, 6),
        **kwargs,
    ):
        """
        Build and return an RGB superpixel image indexed by integer (a,b), colored by
        the same JCh cyclic mapping used for polarization vectors.

        Returns
        -------
        img_rgb : (H,W,3) float in [0,1]
        (fig, ax) : optional, only when plot=True and returnfig=True
        """
        import numpy as np
        from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from quantem.core.visualization.visualization_utils import array_to_rgba
        # Requires the shared helper from the arrow script:
        # _compute_polar_color_mapping(dr, dc, subtract_median=..., use_magnitude_lightness=..., disp_color_max=...)

        # --- Extract data ---
        data = pol_vec.get_data(0)
        if isinstance(data, list) or data is None or data.size == 0:
            H = padding * 2 + pixel_size
            W = padding * 2 + pixel_size
            img_rgb = np.zeros((H, W, 3), dtype=float)
            if plot:
                fig, ax = show_2d(img_rgb, returnfig=True, figsize=figsize, **kwargs)
                ax.set_title(
                    "polarization image" + (" (median subtracted)" if subtract_median else "")
                )
                if returnfig:
                    return img_rgb, (fig, ax)
            return img_rgb

        # fields
        xA = pol_vec[0]["x"]
        yA = pol_vec[0]["y"]
        xR = pol_vec[0]["x_ref"]
        yR = pol_vec[0]["y_ref"]
        a_raw = pol_vec[0]["a"]
        b_raw = pol_vec[0]["b"]

        # displacements (rows/cols)
        dr_raw = (xA - xR).astype(float)  # down +
        dc_raw = (yA - yR).astype(float)  # right +

        # --- Unified color mapping (identical to arrow plot) ---
        dr, dc, amp, disp_cap_px = _compute_polar_color_mapping(
            dr_raw,
            dc_raw,
            subtract_median=subtract_median,
            use_magnitude_lightness=use_magnitude_lightness,
            disp_color_max=disp_color_max,
        )

        # Hue angles with your convention (down=0°, right=+90°, up=180°, left=-90°)
        ang = np.arctan2(dc, dr)
        if phase_dir_flip:
            ang = -ang
        ang += np.deg2rad(phase_offset_deg)

        # Per-sample RGB from JCh mapping
        rgba = array_to_rgba(amp, ang, chroma_boost=chroma_boost)
        colors = rgba.reshape(-1, 4)[:, :3] if rgba.ndim != 2 else rgba[:, :3]

        # Quantize to integer (a,b) tiles
        ai = np.rint(a_raw).astype(int)
        bi = np.rint(b_raw).astype(int)

        a_min, a_max = int(ai.min()), int(ai.max())
        b_min, b_max = int(bi.min()), int(bi.max())
        nrows = a_max - a_min + 1
        ncols = b_max - b_min + 1

        # Output canvas
        H = padding * 2 + nrows * pixel_size + (nrows - 1) * spacing
        W = padding * 2 + ncols * pixel_size + (ncols - 1) * spacing
        img_rgb = np.zeros((H, W, 3), dtype=float)

        # Group indices by (a,b)
        from collections import defaultdict

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (aa, bb) in enumerate(zip(ai, bi)):
            groups[(aa, bb)].append(idx)

        # Optional magnitude (after median subtraction) for 'maxmag' selection
        mag = np.hypot(dr, dc)

        # Fill tiles
        for (aa, bb), idx_list in groups.items():
            rr, cc = aa - a_min, bb - b_min
            r0 = padding + rr * (pixel_size + spacing)
            c0 = padding + cc * (pixel_size + spacing)

            if aggregator == "maxmag":
                j = idx_list[int(np.argmax(mag[idx_list]))]
                color = colors[j]
            else:  # 'mean'
                color = colors[idx_list].mean(axis=0)

            img_rgb[r0 : r0 + pixel_size, c0 : c0 + pixel_size, :] = color

        # --- Optional rendering with legend ---
        if plot:
            fig, ax = show_2d(img_rgb, returnfig=True, figsize=figsize, **kwargs)
            ax.set_title(
                "polarization image" + (" (median subtracted)" if subtract_median else "")
            )

            if show_colorbar:
                divider = make_axes_locatable(ax)
                ax_c = divider.append_axes("right", size="28%", pad="6%")

                N = 256
                yy = np.linspace(-1, 1, N)
                xx = np.linspace(-1, 1, N)
                YY, XX = np.meshgrid(yy, xx, indexing="ij")
                rr = np.sqrt(XX**2 + YY**2)
                disk = rr <= 1.0

                # Legend angle mapping identical to main mapping
                ang_grid = np.arctan2(XX, -YY)  # down=0 at bottom, right=+90° on +x
                if phase_dir_flip:
                    ang_grid = -ang_grid
                ang_grid += np.deg2rad(phase_offset_deg)

                amp_grid = np.clip(rr, 0, 1)
                rgba_grid = array_to_rgba(amp_grid, ang_grid, chroma_boost=chroma_boost)
                rgba_grid[~disk] = 0.0

                ax_c.imshow(
                    rgba_grid,
                    origin="lower",
                    extent=(-1, 1, -1, 1),
                    interpolation="nearest",
                    zorder=0,
                )
                ax_c.set_aspect("equal")
                ax_c.axis("off")

                # ring outline (no clipping so it isn't cut off)
                ring = Circle(
                    (0, 0), 0.98, facecolor="none", edgecolor="k", linewidth=1.2, zorder=3
                )
                ring.set_clip_on(False)
                ax_c.add_patch(ring)

                # angle labels (down/right/up/left)
                ax_c.text(0.00, -1.12, "0°", ha="center", va="top", fontsize=9, color="k")
                ax_c.text(1.12, 0.00, "90°", ha="left", va="center", fontsize=9, color="k")
                ax_c.text(0.00, 1.12, "180°", ha="center", va="bottom", fontsize=9, color="k")
                ax_c.text(-1.12, 0.00, "270°", ha="right", va="center", fontsize=9, color="k")

                # black arrow (scale) and white label centered above it
                scale_len = 0.85
                arrow = FancyArrowPatch(
                    (0.0, 0.0),
                    (scale_len, 0.0),
                    arrowstyle=ArrowStyle.Simple(head_length=10.0, head_width=6.0, tail_width=2.0),
                    mutation_scale=1.0,
                    linewidth=1.2,
                    facecolor="k",
                    edgecolor="k",
                    zorder=4,
                    shrinkA=0.0,
                    shrinkB=0.0,
                )
                arrow.set_clip_on(False)
                ax_c.add_patch(arrow)
                mid_x, mid_y = scale_len / 2.0, 0.0
                ax_c.text(
                    mid_x,
                    mid_y + 0.14,
                    f"{disp_cap_px:.2g} px",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="w",
                )

                # subtle crosshairs & generous limits to avoid clipping
                ax_c.plot([0, 0], [-0.9, 0.9], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
                ax_c.plot([-0.9, 0.9], [0, 0], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
                ax_c.set_xlim(-1.35, 1.35)
                ax_c.set_ylim(-1.25, 1.35)

            if returnfig:
                return img_rgb, (fig, ax)

        return img_rgb


# helper function for polar color mapping
def _compute_polar_color_mapping(
    dr: np.ndarray,
    dc: np.ndarray,
    *,
    subtract_median: bool,
    use_magnitude_lightness: bool,
    disp_color_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (dr_adj, dc_adj, amp, disp_cap_px):
      dr_adj, dc_adj  -> components after optional median subtraction
      amp             -> [0,1] lightness (or constant if not using magnitude lightness)
      disp_cap_px     -> saturation cap (px): user value or 95th percentile
    """
    dr = np.asarray(dr, float).copy()
    dc = np.asarray(dc, float).copy()

    if subtract_median and dr.size:
        dr -= np.median(dr)
        dc -= np.median(dc)

    mag = np.hypot(dr, dc)

    if use_magnitude_lightness:
        if disp_color_max is None:
            nz = mag[mag > 0]
            disp_cap_px = float(np.percentile(nz, 95)) if nz.size else 1.0
        else:
            disp_cap_px = max(float(disp_color_max), 1e-9)
        amp = np.clip(mag / disp_cap_px, 0.0, 1.0)
    else:
        disp_cap_px = float(disp_color_max) if disp_color_max is not None else 1.0
        amp = np.full_like(mag, 0.85, dtype=float)

    return dr, dc, amp, disp_cap_px


def site_colors(number: int) -> tuple[float, float, float]:
    """
    Map an integer 'number' to an RGB triple in [0,1].
    Starts with the requested seed palette and cycles thereafter.
    """
    palette = [
        (0.00, 0.00, 0.00),  # 0: black
        (1.00, 0.00, 0.00),  # 1: red
        (0.00, 0.70, 1.00),  # 2: light blue (cyan-ish)
        (0.00, 0.70, 0.00),  # 3: green
        (1.00, 0.00, 1.00),  # 4: magenta
        (1.00, 0.70, 0.00),  # 5: orange
        (0.00, 0.30, 1.00),  # 6: blue-ish
        # extras to improve variety when cycling:
        (0.60, 0.20, 0.80),
        (0.30, 0.75, 0.75),
        (0.80, 0.40, 0.00),
        (0.20, 0.60, 0.20),
        (0.70, 0.70, 0.00),
    ]
    idx = int(number) % len(palette)
    return palette[idx]
