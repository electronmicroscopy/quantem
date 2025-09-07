from typing import Union

import numpy as np
from numpy.typing import NDArray

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

            a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
            b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

            aa, bb = np.meshgrid(
                np.arange(a_min, a_max + 1),  # inclusive
                np.arange(b_min, b_max + 1),
                indexing="ij",
            )
            basis = np.vstack(
                (
                    np.ones(aa.size),
                    aa.ravel(),
                    bb.ravel(),
                )
            ).T  # (N,3)

            def bilinear_sum(im_: np.ndarray, xy: np.ndarray) -> float:
                """Sum of bilinearly interpolated intensities at (x,y) points."""
                x = xy[:, 0]
                y = xy[:, 1]
                # clamp so x0+1 <= H-1, y0+1 <= W-1
                x0 = np.clip(np.floor(x).astype(int), 0, im_.shape[0] - 2)
                y0 = np.clip(np.floor(y).astype(int), 0, im_.shape[1] - 2)
                dx = x - x0
                dy = y - y0

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

            def objective(theta: np.ndarray) -> float:
                # theta is 6-vector -> (3,2) matrix [[r0],[u],[v]]
                lat = theta.reshape(3, 2)
                xy = basis @ lat  # (N,2) with columns (x,y)
                # Negative: maximize intensity sum by minimizing its negative
                return -bilinear_sum(im, xy)

            theta0 = self._lat.astype(float).reshape(-1)
            res = minimize(
                objective,
                theta0,
                method="Powell",  # robust, derivative-free
                options={
                    "maxiter": int(refine_maxiter),
                    "xtol": 1e-3,
                    "ftol": 1e-3,
                    "disp": False,
                },
            )

            # Update lattice (even if not fully converged)
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
        **kwargs,
    ):
        self._positions_frac = np.array(positions_frac, dtype=float)
        self._num_sites = self._positions_frac.shape[0]
        if numbers is None:
            self._numbers = np.arange(1, self._num_sites + 1, dtype=int)
        else:
            self._numbers = np.array(numbers, dtype=int)

        # --- Image and lattice ---
        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape  # rows=x, cols=y
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)  # (x, y)

        # Determine integer a,b bounds from image corners
        corners = np.array(
            [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]],
            dtype=float,
        )
        A = np.column_stack((u, v))  # (2,2)
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        # Prepare ragged vector: one row per sublattice (site), ragged columns: [x,y,a,b,int_peak]
        self.atoms = Vector.from_shape(
            shape=(self._num_sites,),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )

        # Bilinear sampling helper (vectorized)
        def bilinear_sample(im_, xy_):
            """
            xy_: (N,2) with columns (x,y). Returns intensity (N,), and a valid mask
            requiring that the 2x2 neighborhood is fully inside the image.
            """
            x = xy_[:, 0]
            y = xy_[:, 1]
            # enforce neighborhood in-bounds for x0+1,y0+1 access
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            valid = (x0 >= 0) & (y0 >= 0) & (x0 <= im_.shape[0] - 2) & (y0 <= im_.shape[1] - 2)
            if not np.any(valid):
                return np.zeros_like(x), valid

            xv = x[valid]
            yv = y[valid]
            x0v = x0[valid]
            y0v = y0[valid]
            dx = xv - x0v
            dy = yv - y0v

            Ia = im_[x0v, y0v]
            Ib = im_[x0v + 1, y0v]
            Ic = im_[x0v, y0v + 1]
            Id = im_[x0v + 1, y0v + 1]

            intensity = (
                Ia * (1 - dx) * (1 - dy) + Ib * dx * (1 - dy) + Ic * (1 - dx) * dy + Id * dx * dy
            )

            out = np.zeros_like(x)
            out[valid] = intensity
            return out, valid

        # Build each sublattice
        for a0 in range(self._num_sites):
            da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]

            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),  # small margin is okay
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T  # (N,3)
            xy_cand = basis @ self._lat  # (N,2) in (x,y)

            # Sample intensities and filter
            int_peak, valid_nbhd = bilinear_sample(im, xy_cand)
            keep = valid_nbhd.copy()
            if intensity_min is not None:
                keep &= int_peak >= float(intensity_min)

            if np.any(keep):
                self.atoms[a0] = np.vstack(
                    (
                        xy_cand[keep, 0],  # x
                        xy_cand[keep, 1],  # y
                        basis[keep, 1],  # a
                        basis[keep, 2],  # b
                        int_peak[keep],  # intensity
                    )
                ).T
            else:
                # Store an empty (0,num_fields) row if nothing to keep
                self.atoms[a0] = np.zeros((0, 5), dtype=float)

        # --- Plotting ---
        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)  # image at bottom

            for a0 in range(self._num_sites):
                data = self.atoms[a0]  # (Ni,5)
                if data.size == 0:
                    continue
                # x=rows, y=cols => scatter(y, x)
                x = data[:, 0]
                y = data[:, 1]
                rgb = site_colors(int(self._numbers[a0]))
                ax.scatter(
                    y,
                    x,
                    s=18,
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=18,  # above lines, below origin/vectors
                )

            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self


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
