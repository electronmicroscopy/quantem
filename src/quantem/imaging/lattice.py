from typing import Union

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
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
    def from_data(cls, image: Union[Dataset2d, NDArray]) -> "Lattice":
        if isinstance(image, Dataset2d):
            ds2d = image
        else:
            arr = ensure_valid_array(image, ndim=2)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]
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
