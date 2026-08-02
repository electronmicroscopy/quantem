"""
lattice_visualization.py
--------------------------------
All visualization helpers for the Lattice class.  Nothing here modifies
Lattice state — functions only read from the instance they receive.

Registered plot names (callable via ``lattice.plot(kind=...)``)
---------------------------------------------------------------
  "lattice_vectors"         - image + lattice vectors/grid (after define_lattice_vectors)
"""

import numpy as np

from quantem.core.visualization import show_2d

# ---------------------------------------------------------------------------
# Registry used by Lattice.plot()
# ---------------------------------------------------------------------------
PLOT_REGISTRY: dict[str, callable] = {}  # type:ignore


def _register(name: str):
    def decorator(fn):
        PLOT_REGISTRY[name] = fn
        return fn

    return decorator


### --- Plotting Functions ---
@_register("lattice_vectors")
def plot_lattice_vectors(
    lattice, *, returnfig: bool = False, bound_num_vectors: int | None = None, **kwargs
) -> None | tuple:
    """
    Overlay fitted lattice vectors and grid lines on the image.
    Call after define_lattice_vectors().

    Parameters
    ----------
    bound_num_vectors : int | None
        Number of lattice vectors to draw in each direction.
        None clips lines to image edges.
    **kwargs forwarded to show_2d (e.g. cmap, title).
    """
    # Check if lattice vectors have been defined
    if not hasattr(lattice, "_lat"):
        raise ValueError(
            "Must define lattice vectors first. Call `Lattice.define_lattice_vectors()`"
        )
    # Adding a defualt figsize
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 10)
    fig, ax = show_2d(lattice._image.array, returnfig=True, **kwargs)
    if ax.images:
        ax.images[-1].set_zorder(0)
    H, W = lattice._image.shape
    r0, u, v = (np.asarray(x, dtype=float) for x in lattice._lat)

    ax.scatter(
        r0[1], r0[0], s=60, edgecolor=(0, 0, 0), facecolor=(0, 0.5, 0), marker="s", zorder=5
    )

    n_vec = int(np.ceil(bound_num_vectors)) if bound_num_vectors is not None else 1
    for k in range(1, n_vec + 1):
        tip = r0 + k * u
        ax.arrow(
            r0[1],
            r0[0],
            (tip - r0)[1],
            (tip - r0)[0],
            length_includes_head=True,
            head_width=4.0,
            head_length=6.0,
            linewidth=2.0,
            color="red",
            zorder=4,
        )
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
            zorder=4,
        )

    if bound_num_vectors is None:
        corners = np.array([[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]])
        x_lo, x_hi, y_lo, y_hi = 0.0, float(H), 0.0, float(W)
    else:
        n = float(bound_num_vectors)
        corners = np.array([r0 - n * u, r0 - n * v, r0 + n * u, r0 + n * v], dtype=float)
        x_lo = float(np.min(corners[:, 0]))
        x_hi = float(np.max(corners[:, 0]))
        y_lo = float(np.min(corners[:, 1]))
        y_hi = float(np.max(corners[:, 1]))

    A = np.column_stack((u, v))
    ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
    a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
    b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

    def clipped_segment(base, direction):
        x0, y0 = base
        dx, dy = direction
        t0, t1 = -np.inf, np.inf
        eps = 1e-12
        if abs(dx) < eps:
            if not (x_lo <= x0 <= x_hi):
                return None
        else:
            ts = sorted([(x_lo - x0) / dx, (x_hi - x0) / dx])
            t0, t1 = max(t0, ts[0]), min(t1, ts[1])
        if abs(dy) < eps:
            if not (y_lo <= y0 <= y_hi):
                return None
        else:
            ts = sorted([(y_lo - y0) / dy, (y_hi - y0) / dy])
            t0, t1 = max(t0, ts[0]), min(t1, ts[1])
        if t0 > t1:
            return None
        return base + t0 * direction, base + t1 * direction

    for a in range(a_min, a_max + 1):
        seg = clipped_segment(r0 + a * u, v)
        if seg:
            ax.plot(
                [seg[0][1], seg[1][1]],
                [seg[0][0], seg[1][0]],
                color=(0.0, 0.7, 1.0),
                lw=1,
                zorder=10,
            )
    for b in range(b_min, b_max + 1):
        seg = clipped_segment(r0 + b * v, u)
        if seg:
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="red", lw=1, zorder=10)

    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(x_hi, x_lo)
    if returnfig:
        return fig, ax
    else:
        return None
