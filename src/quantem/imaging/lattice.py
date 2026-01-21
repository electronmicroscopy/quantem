import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
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
        image: Dataset2d | NDArray,
        normalize_min: bool = True,
        normalize_max: bool = True,
    ) -> "Lattice":
        """
        Create a Lattice instance from a 2D image-like input.

        Parameters:
        - image: A 2D numpy array or a Dataset2d instance representing the image.
        - normalize_min: If True, shift the image so its minimum becomes 0.
        - normalize_max: If True, scale the image by its maximum after min-shift
          so values are in [0, 1]. If the maximum is 0 or non-finite (NaN/Inf),
          scaling is skipped to avoid invalid operations.

        Notes:
        - Non-2D inputs and empty arrays raise a ValueError.
        - Inputs with boolean dtype are safely converted to float before normalization.
        - NaN values are ignored when computing min/max (using nanmin/nanmax). If the
          data is all-NaN, normalization is skipped.
        """
        if isinstance(image, Dataset2d):
            ds2d = image
            # Ensure numeric operations are valid (e.g., for bool dtype)
            ds2d.array = np.asarray(ds2d.array, dtype=float)
            # Validate shape
            if ds2d.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if ds2d.array.size == 0:
                raise ValueError("Input image array must not be empty.")
        else:
            # Validate dimensionality and emptiness before any processing
            arr = np.asarray(image)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            # Convert to float for safe arithmetic (handles bool arrays)
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]

        # Normalization (robust to constant, NaN, and bool inputs)
        if normalize_min:
            # Use nanmin to ignore NaNs; if all-NaN, skip
            try:
                min_val = np.nanmin(ds2d.array)
                if np.isfinite(min_val):
                    ds2d.array = ds2d.array - min_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        if normalize_max:
            # Use nanmax to ignore NaNs; skip division if max <= 0 or not finite
            try:
                max_val = np.nanmax(ds2d.array)
                if np.isfinite(max_val) and max_val > 0.0:
                    ds2d.array = ds2d.array / max_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        return cls(image=ds2d, _token=cls._token)

    # --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Dataset2d | NDArray):
        if isinstance(value, Dataset2d):
            # Ensure numeric dtype to avoid boolean arithmetic issues downstream
            value.array = np.asarray(value.array, dtype=float)
            # Validate shape
            if value.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if value.array.size == 0:
                raise ValueError("Input image array must not be empty.")
            self._image = value
        else:
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            arr = arr.astype(float, copy=False)
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
        refine_lattice: bool = True,
        block_size: int | None = None,
        plot_lattice: bool = True,
        bound_num_vectors: int | None = None,
        refine_maxiter: int = 200,
        **kwargs,
    ) -> "Lattice":
        """
        Define the lattice for the image using the origin and the u and v vectors starting from the origin.
        The lattice is defined as r = r0 + nu + mv.

        Parameters
        ----------
        origin : NDArray[2] | Sequence[float]
            Start point (r0) to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
            Ideally a lattice point.
        u : NDArray[2] | Sequence[float]
            Basis vector u to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        v : NDArray[2] | Sequence[float]
            Basis vector v to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        refine_lattice : bool, default=True
            If True, refines the values of r0, u, and v by maximizing the bilinear intensity sum.
        block_size : int | None , default=None
            Fit the lattice points in steps of block_size * lattice_vectors(u, v).
            For example, if block_size = 5, then the lattice points will be fit in steps of
            (-5, 5)u * (-5, 5)v -> (-10, 10)u * (-10, 10)v -> ...
            block_size = None means the entire image will be fit at once.
        plot_lattice : bool, default=True
            If True, the lattice vectors and lines will be plotted overlaid on the image.
        bound_num_vectors : int | None, default=None
            The maximum number of lattice vectors to plot in each direction.
            For example, if bound_num_vectors = 5, lattice lines between (-5, 5)u * (-5, 5)v will be plotted.
            If None, the plotting bounds are set to the image edges.
        refine_maxiter : int, default=200
            Maximum number of iterations for the lattice refinement optimizer (Powell method).
        **kwargs
            Additional keyword arguments forwarded to the plotting function (show_2d), e.g., cmap, title, etc.

        Returns
        -------
        self : Lattice
            Returns the same object, modified in-place.
            The final values of r0, u, v are stored in self._lat.
        """
        # Lattice
        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )
        if not self._lat.shape == (3, 2):
            raise ValueError("origin, u, v must be in (row, col) format only.")

        # Refine lattice coordinates
        # Note that we currently assume corners are local maxima
        if refine_lattice:
            from scipy.optimize import minimize

            assert block_size is None or block_size > 0, "block_size must be positive or None."

            H, W = self._image.shape
            im = np.asarray(self._image.array, dtype=float)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

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

            # Getting the min and max values for the indices a, b from the corners
            a_min, a_max = int(np.floor(ab[0].min())), int(np.ceil(ab[0].max()))
            b_min, b_max = int(np.floor(ab[1].min())), int(np.ceil(ab[1].max()))

            max_ind = max(abs(a_min), a_max, abs(b_min), b_max)
            if not block_size:
                steps = [max_ind]
            else:
                steps = (
                    [*np.arange(0, max_ind + 1, block_size)[1:], max_ind]
                    if max_ind > 0
                    else [max_ind]
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
                """Function to be minimized"""
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

            H, W = self._image.shape
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

            # Origin marker (TOP of stack)
            ax.scatter(
                r0[1],
                r0[0],  # (y, x)
                s=60,
                edgecolor=(0, 0, 0),
                facecolor=(0, 0.5, 0),
                marker="s",
                zorder=30,
            )

            # Lattice vectors as arrows
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

            # Solve for a,b at plot corners (bounds)
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
            A = np.column_stack((u, v))
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]

            a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
            b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

            # Clipping rectangle (image or custom)
            if bound_num_vectors is None:
                x_lo, x_hi = 0.0, float(H)
                y_lo, y_hi = 0.0, float(W)
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

            # Lattice lines (zorder above image)
            # Using x=rows, y=cols: plot(y, x)

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
    ) -> "Lattice":
        """
        Add atoms for each lattice site by sampling all integer lattice translations that fall inside
        the image, measuring local intensity, and filtering candidates by bounds, edge distance,
        mask, and optional intensity/contrast thresholds. Optionally plots the detected atoms.

        Parameters
        ----------
        positions_frac : array-like, shape (S, 2)
            Fractional positions (a, b) of S lattice sites within the unit cell. These are offsets
            relative to the lattice origin r0 and basis vectors (u, v), and are used to tile the
            image with candidate atom centers at all visible integer translations.
        numbers : array-like of int, shape (S,), optional
            Identifier per site (e.g., species or label). If None, uses 1..S. Used only for plotting
            color coding; not used in detection logic.
        intensity_min : float, optional
            Minimum mean intensity inside the detection disk required to keep a candidate atom.
            If None, no intensity thresholding is applied.
        intensity_radius : float, optional
            Radius (in pixels) of the detection disk used to compute the mean intensity at each
            candidate center. If None, an automatic radius is estimated as half of the nearest-neighbor
            spacing in pixels (see Notes).
        plot_atoms : bool, default True
            If True, displays the image and overlays the detected atoms for each site.
        edge_min_dist_px : float, optional
            Minimum distance (in pixels) that candidate centers must maintain from the image borders.
            If a mask is provided and a distance transform can be computed, this same threshold is also
            used to enforce a minimum distance from masked boundaries.
        mask : array-like of bool, shape (H, W), optional
            Binary mask defining valid regions. If provided:
            - When a distance transform is available, candidates must be at least edge_min_dist_px away
            from masked boundaries.
            - Otherwise, candidates are kept only if the nearest integer-pixel location is True in the mask.
        contrast_min : float, optional
            Minimum contrast required to keep a candidate, defined as (disk mean) - (annulus mean).
            If None, no contrast thresholding is applied.
        annulus_radii : tuple of float, optional
            Inner and outer radii (in pixels) of the background annulus used for contrast estimation.
            If None, defaults to (1.5 * intensity_radius, 3.0 * intensity_radius).
        **kwargs
            Additional keyword arguments forwarded to the plotting helper (show_2d) when plot_atoms is True.

        Returns
        -------
        self
            The current object, with the following side effects:
            - self._positions_frac set from positions_frac
            - self._num_sites set to S
            - self._numbers set from numbers or default sequence
            - self.atoms populated with detected atom data per site

        Raises
        ------
        ValueError
            If a provided mask does not match the image shape (H, W).

        Side Effects
        ------------
        self.atoms : Vector
            shape=(S,), fields=("x", "y", "a", "b", "int_peak"), units=("px", "px", "ind", "ind", "counts").
            For each site index s, self.atoms[s] holds a table with one row per detected atom:
            - x, y: pixel coordinates of the atom center (x is row, y is column; origin at top-left)
            - a, b: fractional lattice indices for that atom (including the site's fractional offset plus integer translations)
            - int_peak: mean intensity inside the detection disk at (x, y)

        Notes
        -----
        Lattice and image geometry
            - The image array is of shape (H, W), where x indexes rows and y indexes columns.
            - Lattice parameters are taken from self._lat = [r0, u, v], with r0 the origin (in pixels)
            and u, v the lattice basis vectors (in pixels). Candidate centers are generated by tiling
            each site's fractional offset across all integer translations that map into the image bounds.
            - The visible range of integer translations (a, b) is determined by projecting the image corners
            through the inverse lattice transform.

        Automatic detection radius (when intensity_radius is None)
            - If there are at least two sites, the nearest-neighbor spacing is computed from fractional
            differences between site positions, accounting for periodic wrapping, and converted to pixels
            via the lattice matrix [u v]. The radius is set to half of this spacing.
            - If there is only one site, the spacing fallback is min(||u||, ||v||, ||u+v||, ||u-v||), and the
            radius is half of this value.
            - If the estimate is invalid or non-positive, a robust fallback of 0.5 * (0.5 * (||u|| + ||v||)) is used.

        Filtering
            - Candidates must lie fully within image bounds and satisfy the edge_min_dist_px constraint.
            - If mask is provided and a distance transform can be computed, candidates must also be at least
            edge_min_dist_px inside the masked region; otherwise, the mask must be True at the nearest integer pixel.
            - intensity_min filters by the disk mean; contrast_min filters by the difference between the disk mean
            and the annulus mean, where the annulus default is (1.5 * r, 3.0 * r).

        Plotting
            - When plot_atoms is True, the image is shown and detected atoms are rendered as semi-transparent
            colored markers per site. Colors are determined by site numbers. Axes are set to match image
            coordinates (x increasing downward).
        """
        if not hasattr(self, "_lat") or self._lat is None:
            raise ValueError(
                "Lattice vectors have not been fitted. Please call define_lattice() first."
            )
        # Handle empty positions early without creating a Vector of length 0
        positions_frac_arr = np.asarray(positions_frac, dtype=float)
        if positions_frac_arr.size == 0:
            # Bookkeeping for consistency
            self._positions_frac = np.empty((0, 2), dtype=float)
            self._num_sites = 0
            self._numbers = (
                np.array([], dtype=int)
                if numbers is None
                else np.atleast_1d(np.array(numbers, dtype=int))
            )
            # Do not construct an empty Vector with zero shape (causes error). Just return.
            return self

        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]
        self._numbers = (
            np.arange(0, self._num_sites, dtype=int)
            if numbers is None
            else np.atleast_1d(np.array(numbers, dtype=int))
        )

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        # Min and max values of a,b are calculated based on corners
        corners = np.array(
            [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]], dtype=float
        )
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        def _auto_radius_px() -> float:
            """
            Estimate a default disk radius in pixels as half the nearest-neighbor spacing
            (with periodic wrapping), or from lattice vectors if insufficient points.
            """
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
            """
            Compute the mean image intensity within a circular disk of radius r_px centered at (x, y),
            with boundary clipping and fallback to the center pixel if empty.
            """
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
            """
            Compute the mean and standard deviation of intensities within an annulus [rin, rout] centered at (x, y),
            with boundary clipping and fallback to the center pixel and zero std if empty.
            """
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
            fields=["x", "y", "a", "b", "int_peak"],
            units=["px", "px", "ind", "ind", "counts"],
        )

        for a0 in range(self._num_sites):
            da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]
            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T
            xy = basis @ self._lat  # (N,2)

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
    ) -> "Lattice":
        """
        Refine atom centers by local 2D Gaussian fitting around each previously detected atom.
        Updates atom positions and peak intensity and adds per-atom sigma and background fields.
        Optionally plots the refined atoms.
        Parameters
        ----------
        fit_radius : float, optional
            Radius (in pixels) of the circular fitting region around each atom's current center.
            If None, an automatic radius is estimated as half of the nearest-neighbor spacing
            between lattice sites in pixels. When there is only one site, the spacing fallback
            is min(||u||, ||v||, ||u+v||, ||u-v||) where u and v are lattice vectors. If this
            estimate is invalid or non-positive, a robust fallback is used.
        max_nfev : int, default 200
            Maximum number of function evaluations for the non-linear least-squares solver.
        max_move_px : float, optional
            Maximum allowed movement (in pixels) of the refined center from its initial position.
            If None, defaults to the fitting radius. Bounds also enforce staying within image limits.
        plot_atoms : bool, default False
            If True, displays the image and overlays the refined atom positions.
        **kwargs
            Additional keyword arguments forwarded to the plotting helper when plot_atoms is True.

        Returns
        -------
        self
            The current object, with self.atoms updated per site to refined values.

        Raises
        ------
        ValueError
            If no atoms are present to refine (call add_atoms() first).

        Side Effects
        ------------
        self.atoms : Vector
            For each site index s, the per-atom rows are updated:
            - x, y: pixel coordinates refined by local Gaussian fitting (x is row, y is column).
            - int_peak: updated to the fitted Gaussian amplitude at the center.
            - sigma: added or updated; the fitted Gaussian width (pixels).
            - int_bg: added or updated; the fitted local constant background level.
            If "sigma" and "int_bg" fields do not exist, they are added automatically.

        Notes
        -----
        Model and fitting
            - A circular patch of radius fit_radius is extracted around each atom's current center.
            - Within that patch, a 2D isotropic Gaussian plus constant background is fit:
            I(x, y) = amp * exp(-0.5 * r^2 / sigma^2) + bg, where r^2 is the squared distance
            to the fitted center (x_c, y_c).
            - Initial guesses:
            - Center starts at the current atom position.
            - amp starts from the central pixel value minus the local median background.
            - sigma starts at max(0.5 * fit_radius, 0.5).
            - bg starts at the median of the patch outside the circular mask (or full patch median).
            - Parameter bounds:
            - Center (x_c, y_c) limited to within max_move_px of the initial center and within
                image bounds.
            - amp in [0, max(pmax - pmin, 4 * amp0)], using local patch extrema and initial amp0.
            - sigma in [0.25, max(2 x fit_radius, 1.0)].
            - bg in [pmin * (pmax - pmin), pmax + (pmax - pmin)].
            - Optimization uses scipy.optimize.least_squares with "trf" method and "soft_l1" loss.

        Automatic fitting radius (when fit_radius is None)
            - If there are at least two sites, the nearest-neighbor spacing is computed from fractional
            differences between site positions (wrapped to [-0.5, 0.5]) and converted to pixels using
            the lattice matrix [u v]; the radius is set to half of this spacing.
            - If there is only one site, the spacing fallback is min(||u||, ||v||, ||u+v||, ||u-v||),
            and the radius is half of this value.
            - If the estimate is invalid or non-positive, a robust fallback is used based on the lattice
            vector norms to ensure a reasonable, non-zero radius.

        Plotting
            - When plot_atoms is True, the image is shown and refined atom centers are rendered as
            semi-transparent colored markers per site. Colors are determined by site numbers.
            - Axes are set to match image coordinates (x increasing downward).
        """

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
        measure_ind: int,
        reference_ind: int,
        reference_radius: float | None = None,
        min_neighbours: int | None = 2,
        max_neighbours: int | None = None,
        plot_polarization_vectors: bool = False,
        **plot_kwargs,
    ) -> "Vector":
        """
        Measure the polarization of atoms at one site with respect to atoms at another site.
        Polarization is computed as a fractional displacement (da, db) of each atom in the
        'measure' site relative to the expected position inferred from the nearest atoms
        in the 'reference' site and the current lattice vectors. The expected position is
        the mean of neighbor positions shifted by the lattice vector transform of the
        fractional index difference.

        Parameters
        ----------
        measure_ind : int
            Index of the site whose polarization is to be measured.
            This corresponds to the index in `positions_frac` used in `add_atoms()`.
        reference_ind : int
            Index of the reference site used to calculate polarization.
            This corresponds to the index in `positions_frac` used in `add_atoms()`.
        reference_radius : float | None, default=None
            If provided, neighbors are selected by radius search (in pixels) using a KD-tree.
            Must be at least 1 pixel. If None, neighbors are selected by k-nearest search.
        min_neighbours : int | None, default=2
            Minimum number of nearest neighbors used to calculate polarization. Must be >= 2
            when using k-nearest search (i.e., when `reference_radius` is None).
        max_neighbours : int | None, default=None
            Maximum number of nearest neighbors to use. Required when `reference_radius` is None.
        plot_polarization_vectors : bool, default=False
            If True, plots the polarization vectors using `self.plot_polarization_vectors(...)`.
        **plot_kwargs
            Additional keyword arguments forwarded to the plotting function.

        Returns
        -------
        out : quantem.core.datastructures.vector.Vector
            A Vector object containing the polarizations with:
            - shape=(1,)
            - fields=("x", "y", "a", "b", "da", "db")
            - units=("px", "px", "ind", "ind", "ind", "ind")
            Here, (x, y) are positions in pixels, (a, b) are fractional indices,
            and (da, db) are fractional displacements (polarization).

        Raises
        ------
        ValueError
            - If the lattice vectors are singular (cannot invert).
            - If neither `reference_radius` nor both `min_neighbours` and `max_neighbours` are specified.
            - If `reference_radius` < 1.
            - If radius-based search fails to find at least `min_neighbours` for any atom.
            - If k-nearest search is used and `min_neighbours` or `max_neighbours` is missing.
            - If k-nearest search is used with `min_neighbours` < 2 or `max_neighbours` < 2.
            - If `min_neighbours` > `max_neighbours`.
            - If no atoms have any neighbors identified (increase `reference_radius`).
        Warning
            If some atoms do not have any neighbors identified (suggests increasing `reference_radius`).

        Notes
        -----
        - Lattice vectors are taken from `self._lat` and are in pixel units.
        - Neighbor selection:
            - If `reference_radius` is provided, a radius search (KD-tree) is used and optionally
                truncated by `max_neighbours`.
            - If `reference_radius` is None, k-nearest neighbors are used with `k=max_neighbours`.
        - The expected position for each measured atom is computed as the mean over selected
        neighbors of: neighbor_position + L @ ([a - a_i, b - b_i]), where L = [u v], and
        (a, b) and (a_i, b_i) are the fractional indices of the measured atom and the neighbor,
        respectively. The polarization (da, db) is then obtained by transforming the
        Cartesian displacement back to fractional coordinates using L^{-1}.
        - If either the measure or reference site is empty, an empty Vector (with zero rows) is returned.
        """
        from scipy.spatial import cKDTree

        measure_ind = int(measure_ind)
        reference_ind = int(reference_ind)

        def is_empty(cell):
            if cell is None:
                return True
            if isinstance(cell, list):
                return len(cell) == 0
            if isinstance(cell, dict):
                x = cell.get("x", None)
                return x is None or np.size(x) == 0
            # Fallback to numpy-like objects
            if hasattr(cell, "size"):
                return cell.size == 0
            return False

        # Check for empty cells
        A_cell = self.atoms.get_data(measure_ind)
        B_cell = self.atoms.get_data(reference_ind)
        self._pol_meas_ref_ind = (measure_ind, reference_ind)

        # Prepare a Vector with structured dtype (even for empty data)
        fields = ["x", "y", "a", "b", "da", "db"]
        units = ["px", "px", "ind", "ind", "ind", "ind"]

        def empty_vector():
            out = Vector.from_shape(
                shape=(1,),
                fields=fields,
                units=units,
                name="polarization",
            )
            # Create empty array with shape (0, 6) to match expected format
            empty_data = np.zeros((0, 6), dtype=float)
            out.set_data(empty_data, 0)
            return out

        if is_empty(A_cell) or is_empty(B_cell):
            return empty_vector()

        # Extract site data
        Ax = self.atoms[measure_ind]["x"]
        Ay = self.atoms[measure_ind]["y"]
        Aa = self.atoms[measure_ind]["a"]
        Ab = self.atoms[measure_ind]["b"]
        Bx = self.atoms[reference_ind]["x"]
        By = self.atoms[reference_ind]["y"]
        Ba = self.atoms[reference_ind]["a"]
        Bb = self.atoms[reference_ind]["b"]

        if Ax.size == 0 or Bx.size == 0:
            return empty_vector()

        # Lattice vectors: r0, u, v
        lat = np.asarray(getattr(self, "_lat", None))
        if lat is None or lat.shape[0] < 3:
            raise ValueError("Lattice vectors (_lat) are missing or malformed.")
        _, u, v = lat[0], lat[1], lat[2]
        L = np.column_stack((u, v))
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            raise ValueError("Lattice vectors are singular and cannot be inverted.")

        query_coords = np.column_stack([Ax, Ay])
        ref_coords = np.column_stack([Bx, By])

        # Pre-allocate result array memory
        x_arr = Ax.copy().astype(float)
        y_arr = Ay.copy().astype(float)
        a_arr = Aa.copy().astype(float)
        b_arr = Ab.copy().astype(float)
        da_arr = np.zeros_like(x_arr, dtype=float)
        db_arr = np.zeros_like(x_arr, dtype=float)

        # KD-tree query
        tree = cKDTree(ref_coords)

        if max_neighbours is None and reference_radius is None:
            raise ValueError(
                "Either min_neighbours or max_neighbours or reference_radius must be passed."
            )

        # Initialize arrays for results
        dists = []
        idxs = []

        if reference_radius is not None:
            # Radius-based query
            if reference_radius < 1:
                raise ValueError(
                    f"reference_radius must be atleast 1 pixel. You have passed : {reference_radius}"
                )

            neighbor_lists = tree.query_ball_point(
                query_coords,
                r=reference_radius,
                workers=-1,
            )

            for i, neighbors in enumerate(neighbor_lists):
                if len(neighbors) == 0:
                    dists.append(np.array([]))
                    idxs.append(np.array([]))
                    continue

                # Distance calculation
                neighbor_coords = ref_coords[neighbors]
                query_point = query_coords[i]
                distances = np.linalg.norm(neighbor_coords - query_point, axis=1)

                # Sorting
                sort_idx = np.argsort(distances)
                sorted_distances = distances[sort_idx]
                sorted_indices = np.array(neighbors)[sort_idx]

                # Apply max_neighbours limit if specified
                if max_neighbours is not None and len(sorted_distances) > max_neighbours:
                    sorted_distances = sorted_distances[:max_neighbours]
                    sorted_indices = sorted_indices[:max_neighbours]

                dists.append(sorted_distances)
                idxs.append(sorted_indices)

            # Length checking
            lengths = np.array([len(row) for row in dists])
            if min_neighbours is not None and np.any(lengths < min_neighbours):
                raise ValueError(
                    "Failed to calculate enough nearest neighbours. Increase the reference_radius"
                )

        elif reference_radius is None:
            # K-nearest neighbors query
            if min_neighbours is None or max_neighbours is None:
                raise ValueError(
                    "min_neighbours and max_neighbours should be specified if reference_radius is None"
                )
            if min_neighbours < 2 or max_neighbours < 2:
                raise ValueError(
                    "Must use atleast 2 nearest neighbours to calculate the Polarization"
                )
            if min_neighbours > max_neighbours:
                raise ValueError("'min_neighbours' cannot be larger than 'max_neighbours'")

            dist_array, idx_array = tree.query(
                query_coords,
                k=max_neighbours,
                workers=-1,
            )

            # Processing of results
            finite_mask = np.isfinite(dist_array)
            for i in range(len(query_coords)):
                mask = finite_mask[i]
                dists.append(dist_array[i][mask])
                idxs.append(idx_array[i][mask])

        # Neighbor checking
        lengths = np.array([len(row) for row in dists])
        atoms_with_atleast_one_neighbour = lengths > 0

        if not np.any(atoms_with_atleast_one_neighbour):
            raise ValueError(
                "Failed to calculate nearest neighbours for all atoms. Increase reference_radius."
            )

        if not np.all(atoms_with_atleast_one_neighbour):
            missing_count = len(atoms_with_atleast_one_neighbour) - np.sum(
                atoms_with_atleast_one_neighbour
            )
            raise Warning(
                f"{missing_count} atoms do not have any neighbours identified. Try increasing reference_radius."
            )

        # Pre-allocate arrays for better performance
        da_arr = np.zeros(len(query_coords))
        db_arr = np.zeros(len(query_coords))

        # Calculate displacements with optimizations
        for i, (atom_dists, atom_idxs) in enumerate(zip(dists, idxs)):
            if len(atom_idxs) == 0:
                # Arrays already initialized to 0
                continue

            # Check if we have enough neighbors
            if min_neighbours is not None and len(atom_idxs) < min_neighbours:
                # Arrays already initialized to 0
                continue

            # Determine how many neighbors to use
            num_neighbors_to_use = len(atom_idxs)
            if max_neighbours is not None:
                num_neighbors_to_use = min(num_neighbors_to_use, max_neighbours)
            if min_neighbours is not None:
                num_neighbors_to_use = max(
                    num_neighbors_to_use, min(min_neighbours, len(atom_idxs))
                )

            # Select the neighbors to use
            if num_neighbors_to_use < len(atom_idxs):
                closest_order = np.argpartition(atom_dists, num_neighbors_to_use)[
                    :num_neighbors_to_use
                ]
                nbr_idx = atom_idxs[closest_order].astype(int)
            else:
                nbr_idx = atom_idxs.astype(int)

            # Get actual positions of the atoms
            actual_pos = np.array([x_arr[i], y_arr[i]])

            # Calculate the expected positions of the atoms using its n_neighbors
            a, b = a_arr[i], b_arr[i]
            ai, bi = Ba[nbr_idx], Bb[nbr_idx]
            xi, yi = Bx[nbr_idx], By[nbr_idx]

            fractional_diff = np.array([a - ai, b - bi])  # (2, n_neighbors)
            neighbor_positions = np.array([xi, yi])  # (2, n_neighbors)

            expected_positions = neighbor_positions + L @ fractional_diff  # (2, n_neighbors)

            # Taking the mean of the expected position calculated using each neighbor for better robustness.
            expected_position = np.mean(expected_positions, axis=1)  # (2,)

            # Difference between actual and expected positions gives us polarization.
            displacement_cartesian = actual_pos - expected_position
            displacement_fractional = L_inv @ displacement_cartesian

            da_arr[i] = displacement_fractional[0]
            db_arr[i] = displacement_fractional[1]

        out = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )

        # Create structured array if needed
        if len(x_arr) > 0:
            arr = np.column_stack([x_arr, y_arr, a_arr, b_arr, da_arr, db_arr])
        else:
            arr = np.zeros((0, 6), dtype=float)

        out.set_data(arr, 0)

        if plot_polarization_vectors:
            self.plot_polarization_vectors(out, **plot_kwargs)

        return out

    def calculate_order_parameter(
        self,
        polarization_vectors: Vector,
        num_phases: int = 2,
        phase_polarization_peak_array: NDArray | None = None,
        refine_means: bool = True,
        run_with_restarts: bool = False,
        num_restarts: int = 1,
        verbose: bool = False,
        plot_order_parameter: bool = True,
        plot_gmm_visualization: bool = True,
        torch_device: str = "cpu",
        **kwargs,
    ) -> "Lattice":
        """
        Estimate a multi-phase order parameter by fitting a Gaussian Mixture Model (GMM)
        to fractional polarization components (da, db). The order parameter for each site
        is defined as the posterior membership probabilities (responsibilities) of the
        fitted GMM components evaluated in the 2D polarization space.

        The method can optionally:
            - Use provided phase centers (polarization peaks) to initialize or fix the GMM means.
            - Visualize the mixture model in (da, db) space with KDE density, centers, and
            ~95% confidence ellipses.
            - Overlay the order parameter (probability-colored sites) on the original image grid.

        Parameters
        ----------
        polarization_vectors : Vector
            A collection holding polarization data. Only the first element
            polarization_vectors[0] is used and must provide the following keys:
                - 'x': NDArray of shape (N,), row coordinates for each site.
                - 'y': NDArray of shape (N,), column coordinates for each site.
                - 'da': NDArray of shape (N,), fractional polarization along a (e.g., du).
                - 'db': NDArray of shape (N,), fractional polarization along b (e.g., dv).
            All arrays must be one-dimensional, aligned, and of equal length N.

        num_phases : int, default=2
            Number of Gaussian components (phases) in the mixture. Must be >= 1.
            For num_phases=1, all sites belong to a single phase (probabilities are all 1).

        phase_polarization_peak_array : NDArray | None, default=None
            Optional array of shape (num_phases, 2) specifying phase centers (means)
            in (da, db) space:
                - If refine_means = True, these values initialize the GMM means.
                - If refine_means = False, the means are held fixed during fitting
                    and only covariances and weights are updated.

        refine_means : bool, default=True
            If False, requires phase_polarization_peak_array to be provided with shape
            (num_phases, 2). The GMM means are fixed to these values throughout EM.

        run_with_restarts : bool, default=False
            If True, runs the GMM fitting multiple times with different initializations
            and selects the best result based on classification certainty.

        num_restarts : int, default=1
            Number of random restarts when run_with_restarts=True. Must be >= 1.

        verbose : bool, default=False
            If True, prints diagnostic information including fitted means and error
            metrics for each restart.

        plot_order_parameter : bool, default=True
            If True, overlays sites on self._image.array and colors them by their full
            mixture probability distribution:
                - For 2 phases, adds a two-color probability bar.
                - For 3 phases, adds a ternary-style color triangle.
                - For other values, no legend is shown.

        plot_gmm_visualization : bool, default=True
            If True, shows a visualization in (da, db) space:
                - A Gaussian KDE density (scipy.stats.gaussian_kde) on a symmetric grid
                    spanning max(abs(da), abs(db)).
                - Scatter of points colored by mixture probabilities.
                - GMM centers (means) and ~95% confidence ellipses (2 standard deviations).

        torch_device : str, default='cpu'
            Torch device used by the TorchGMM backend. Examples: 'cpu', 'cuda',
            'cuda:0'. If a CUDA device is requested but unavailable, the underlying
            GMM implementation may raise an error.

        **kwargs : dict
            Additional keyword arguments controlling visualization.
            When plot_gmm_visualization=True, the following keys are supported and validated:

            contour_cmap : str, optional
                Matplotlib colormap name for the background contour;
                invalid names fall back to a preset ('gray_r') with a warning.

            gmm_center_colour : color specification, optional
                Color for GMM center markers; invalid values fall back to a preset with a warning.
                Presets depend on num_phases (2: lime; 3-4: Yellow; ≥5: Black).

            gmm_ellipse_colour : color specification, optional
                Color for GMM covariance ellipses; invalid values fall back to a preset with a warning.
                Presets depend on num_phases (2: lime; 3-4: Yellow; ≥5: White).

            scatter_colours : callable, array, or list, optional
                Colors used to map phase probabilities for scatter points
                (and the order-parameter map). Accepted forms:
                    • callable f(i) -> RGB(A) (first 3 components used),
                    • numpy array of shape (num_phases, 3) with RGB in [0, 1],
                    • list/tuple of valid color names/values of length num_phases,
                    • single valid color (applied to all phases; prints a warning).
                Invalid inputs fall back to a preset (site_colors) with a warning.
                When plot_order_parameter=True,
                scatter_colours is used to color points by phase probabilities.

        Returns
        -------
        self : Lattice
            The same object, modified in-place.

        Side Effects
        ------------
        Sets the following attributes on self:
            - self._polarization_means : NDArray of shape (num_phases, 2),
                the fitted (or fixed) means in (da, db) space.
            - self._order_parameter_probabilities : NDArray of shape (N, num_phases),
                posterior probabilities per site.

        Produces plots if plot_gmm_visualization or plot_order_parameter is True.

        Notes
        -----
        - The GMM uses full covariance matrices (covariance_type='full') and an EM
        implementation backed by TorchGMM (PyTorch).
        - The KDE contour limits are symmetric around the origin and set by
        max(abs(da), abs(db)).
        - In the order-parameter overlay, coordinates are plotted as:
        x-axis: 'y' (column), y-axis: 'x' (row).
        - Helper functions expected to exist:
            - create_colors_from_probabilities(probabilities, num_phases, colors)
            - add_2phase_colorbar(ax, colors)
            - add_3phase_color_triangle(fig, ax, colors)
            - show_2d(image, ...)
        - Requires self._image.array to be present for the order-parameter overlay.

        Raises
        ------
        ValueError
            If phase_polarization_peak_array is provided with incorrect shape
            (must be (num_phases, 2)).
        ValueError
            If refine_means=False and phase_polarization_peak_array is None.
        AttributeError
            If plot_order_parameter=True but self._image or self._image.array is missing.
        ImportError
            If required plotting/scientific packages (matplotlib, scipy) are unavailable.
        RuntimeError or ValueError
            From TorchGMM if the torch device is invalid or unavailable.

        Examples
        --------
        Fit a 2-phase GMM and show both visualizations:

        >>> lattice.calculate_order_parameter(
        ...     polarization_vectors,
        ...     num_phases=2,
        ...     plot_gmm_visualization=True,
        ...     plot_order_parameter=True
        ... )

        Use fixed phase peaks:

        >>> peaks = np.array([[0.10, -0.05],
        ...                   [0.30,  0.07]], dtype=float)
        >>> lattice.calculate_order_parameter(
        ...     polarization_vectors,
        ...     num_phases=2,
        ...     phase_polarization_peak_array=peaks,
        ...     refine_means=True
        ... )

        Run on GPU (if available):

        >>> lattice.calculate_order_parameter(
        ...     polarization_vectors,
        ...     num_phases=3,
        ...     torch_device='cuda:0'
        ... )
        """
        # Imports
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from scipy.stats import gaussian_kde

        # Validate inputs
        if run_with_restarts:
            assert isinstance(num_restarts, int) and num_restarts > 0, (
                "num_restarts must be positive when run_with_restarts is True"
            )
        else:
            assert num_restarts == 1, "num_restarts must be 1 when run_with_restarts is False"
        assert isinstance(num_phases, int) and num_phases >= 1, (
            "num_phases must be an integer >= 1"
        )

        # Functions
        def plot_gaussian_ellipse(ax, mean, cov, n_std=2, clip_path=None, **kwargs):
            """
            Plot confidence ellipse for a 2D Gaussian

            Parameters:
            -----------
            ax : matplotlib axis
            mean : array-like, shape (2,)
                Mean of the Gaussian
            cov : array-like, shape (2, 2)
                Covariance matrix
            n_std : float
                Number of standard deviations (2 = ~95% confidence)
            clip_path : matplotlib.path.Path, optional
                Path to use for clipping the ellipse
            """
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)

            # Create ellipse
            ellipse = Ellipse(mean, width, height, angle=angle, fill=False, **kwargs)

            if clip_path is not None:
                ellipse.set_clip_path(clip_path, transform=ax.transData)

            ax.add_patch(ellipse)

            return ellipse

        def to_percent(x, pos):
            """Format axis labels as percentages"""
            return f"{x * 100:.1f}%"

        # Function to validate colormap
        def is_valid_cmap(cmap_name):
            """Check if a colormap name is valid in matplotlib"""
            try:
                plt.get_cmap(cmap_name)
                return True
            except (ValueError, TypeError):
                return False

        # Function to validate color
        def is_valid_color(color):
            """Check if a color is valid in matplotlib"""
            try:
                mcolors.to_rgba(color)
                return True
            except (ValueError, TypeError):
                return False

        # Function to convert color names to RGB for scatter_cmap
        def convert_colors_to_rgb(colors, num_phases):
            """
            Convert colors to RGB array format.
            Args:
                colors: either a callable function, array of colors, or list of color names
                num_phases: number of phases/clusters
            Returns:
                numpy array of shape (num_phases, 3) with RGB values
            """
            # If it's a function (like site_colors), call it for each index
            if callable(colors):
                rgb_array = np.array([colors(i)[:3] for i in range(num_phases)])
                return rgb_array

            # If it's already an array, validate dimensions
            if isinstance(colors, np.ndarray):
                if colors.shape == (num_phases, 3):
                    return colors
                else:
                    return None

            # If it's a list/tuple of color names or values
            if isinstance(colors, (list, tuple)):
                try:
                    rgb_array = np.array([mcolors.to_rgb(c) for c in colors])
                    if rgb_array.shape == (num_phases, 3):
                        return rgb_array
                    else:
                        return None
                except (ValueError, TypeError):
                    return None

            return None

        class FixedMeansGMM(TorchGMM):
            """
            GMM variant with fixed component means.
            Means are set via fixed_means at init and held constant during EM;
            only weights and covariances are updated.
            """

            def __init__(self, fixed_means, **kwargs):
                fixed_means = np.asarray(fixed_means, dtype=np.float32)
                super().__init__(n_components=len(fixed_means), means_init=fixed_means, **kwargs)
                self.fixed_means = fixed_means

            def _m_step(self, X, r):
                """
                M-step with fixed means:
                update mixture weights and covariances from responsibilities,
                keeping means unchanged.
                """
                # Override to keep means fixed while updating weights and covariances
                N, D = X.shape
                K = self.n_components
                Nk = r.sum(dim=0) + 1e-12
                self._weights = (Nk / (N + 1e-12)).clamp_min(1e-12)

                # Keep means fixed
                self._means = self._to_tensor(self.fixed_means).clone()

                # Update covariances with fixed means
                covs = []
                for k in range(K):
                    diff = X - self._means[k]
                    cov_k = (r[:, k][:, None] * diff).T @ diff
                    cov_k = cov_k / (Nk[k] + 1e-12)
                    cov_k = cov_k + self.reg_covar * torch.eye(
                        D, device=self.device, dtype=self.dtype
                    )
                    covs.append(cov_k)
                self._covariances = torch.stack(covs, dim=0)

        x_arr = polarization_vectors[0]["x"]
        y_arr = polarization_vectors[0]["y"]

        da_arr = polarization_vectors[0]["da"]
        db_arr = polarization_vectors[0]["db"]

        d_frac_arr = np.vstack([da_arr, db_arr])
        data = np.column_stack([da_arr, db_arr])

        # Important validations and error handling
        # Handle empty polarization vectors early
        if len(da_arr) == 0:
            # Set empty attributes and return early
            self._polarization_means = np.empty((num_phases, 2), dtype=float)
            self._order_parameter_probabilities = np.empty((0, num_phases), dtype=float)
            if hasattr(self, "_polarization_labels"):
                self._polarization_labels = np.array([], dtype=int)
            return self

        # Check for minimum number of samples for GMM
        n_samples = len(da_arr)
        if n_samples < num_phases:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= num_phases ({num_phases}) "
                f"for Gaussian Mixture Model fitting."
            )

        # For KDE visualization, need at least 2 samples
        if plot_gmm_visualization and n_samples < 2:
            import warnings

            warnings.warn(
                f"Cannot plot KDE with only {n_samples} sample(s). "
                "Disabling GMM visualization plot.",
                UserWarning,
            )
            plot_gmm_visualization = False

        # Fit GMM with N Gaussians
        if phase_polarization_peak_array is None:
            gmm = TorchGMM(n_components=num_phases, covariance_type="full", device=torch_device)
        else:
            # Basic checks
            if phase_polarization_peak_array.shape != (num_phases, 2):
                raise ValueError(
                    f"phase_polarization_peak_array should have dimensions ({num_phases}, 2). You have input : {phase_polarization_peak_array.shape}"
                )
            if not refine_means:
                gmm = FixedMeansGMM(
                    covariance_type="full",
                    fixed_means=phase_polarization_peak_array,
                    device=torch_device,
                )
            else:
                gmm = TorchGMM(
                    n_components=num_phases,
                    covariance_type="full",
                    means_init=phase_polarization_peak_array,
                    device=torch_device,
                )

        # Intialize best fit tracking variables if run_with_restarts
        if run_with_restarts:
            best_error = np.inf
            best_means = None
            best_probabilities = None
            best_cov = None

        for i in range(num_restarts):
            gmm.fit(data)

            # Calculate score between 0 and 1 for each point
            # Get probabilities for each Gaussian
            probabilities = gmm.predict_proba(data)  # Shape: (n_points, num_phases)

            # Measure error as 1 - (mean of (probabilities of best fit))
            error = 1 - probabilities.max(axis=1).mean()

            # Calculate means
            means = gmm.means_

            if verbose:
                print(f"Restart {i + 1}/{num_restarts}:")
                print(f"    Means: \n{means}")
                print(f"    Error: {error:.4f}")
            if run_with_restarts:
                if error < best_error:
                    best_error = error
                    best_means = means
                    best_probabilities = probabilities
                    best_cov = gmm.covariances_

        if run_with_restarts and verbose:
            print("Best results after restarts:")
            print(f"    Means: \n{best_means}")
            print(f"    Error: {best_error:.4f}")
        elif verbose:
            print("GMM fitting results:")
            print(f"    Means: \n{gmm.means_}")
            print(f"    Error: {i - probabilities.max(axis=1).mean():.4f}")

        # Create grid for contour - use max_bound to cover entire plot area
        max_bound = max(abs(da_arr).max(), abs(db_arr).max())

        x_grid = np.linspace(-max_bound, max_bound, 100)
        y_grid = np.linspace(-max_bound, max_bound, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = gaussian_kde(d_frac_arr)(positions).reshape(X.shape)

        # Save GMM data
        if run_with_restarts:
            self._polarization_means = best_means
            self._order_parameter_probabilities = best_probabilities
        else:
            self._polarization_means = gmm.means_
            self._order_parameter_probabilities = probabilities
            best_means = gmm.means_
            best_probabilities = probabilities
            best_cov = gmm.covariances_

        num_components = num_phases

        # --- Combined Plot: Scatter overlaid on Contour ---
        if plot_gmm_visualization:
            from matplotlib.path import Path
            from matplotlib.ticker import FuncFormatter

            # Define preset colors based on num_phases
            preset_contour_cmap = "gray_r"
            if num_phases == 2:
                preset_gmm_center_colour = (0, 0.7, 0)
                preset_gmm_ellipse_colour = (0, 0.7, 0)
            elif num_phases < 5:
                preset_gmm_center_colour = "Yellow"
                preset_gmm_ellipse_colour = "Yellow"
            else:
                preset_gmm_center_colour = "Black"
                preset_gmm_ellipse_colour = "White"

            preset_scatter_colours = site_colors

            # Check and assign contour_cmap
            if "contour_cmap" in kwargs:
                if is_valid_cmap(kwargs["contour_cmap"]):
                    contour_cmap = kwargs["contour_cmap"]
                else:
                    print(
                        f"Warning: '{kwargs['contour_cmap']}' is not a valid colormap, using preset"
                    )
                    contour_cmap = preset_contour_cmap
            else:
                contour_cmap = preset_contour_cmap

            # Check and assign gmm_center_colour
            if "gmm_center_colour" in kwargs:
                if is_valid_color(kwargs["gmm_center_colour"]):
                    gmm_center_colour = kwargs["gmm_center_colour"]
                else:
                    print(
                        f"Warning: '{kwargs['gmm_center_colour']}' is not a valid color, using preset"
                    )
                    gmm_center_colour = preset_gmm_center_colour
            else:
                gmm_center_colour = preset_gmm_center_colour

            # Check and assign gmm_ellipse_colour
            if "gmm_ellipse_colour" in kwargs:
                if is_valid_color(kwargs["gmm_ellipse_colour"]):
                    gmm_ellipse_colour = kwargs["gmm_ellipse_colour"]
                else:
                    print(
                        f"Warning: '{kwargs['gmm_ellipse_colour']}' is not a valid color, using preset"
                    )
                    gmm_ellipse_colour = preset_gmm_ellipse_colour
            else:
                gmm_ellipse_colour = preset_gmm_ellipse_colour

            # Check and assign scatter_colours (with special handling)
            if "scatter_colours" in kwargs:
                scatter_colours_input = kwargs["scatter_colours"]

                # Try to convert to RGB format
                scatter_colours_rgb = convert_colors_to_rgb(scatter_colours_input, num_phases)

                if scatter_colours_rgb is not None:
                    # Successfully converted to (num_phases, 3) RGB array
                    scatter_colours = scatter_colours_rgb
                else:
                    # Check if it's a single valid color
                    if is_valid_color(scatter_colours_input):
                        # Convert single color to repeated array for indexing
                        single_color_rgb = mcolors.to_rgb(scatter_colours_input)
                        scatter_colours = np.tile(single_color_rgb, (num_phases, 1))
                        print(
                            f"Warning: Using single color '{scatter_colours_input}' for all {num_phases} phases"
                        )
                    else:
                        print(
                            "Warning: scatter_colours invalid (must be (num_phases, 3) array, list of valid colors, or callable), using preset"
                        )
                        scatter_colours = convert_colors_to_rgb(preset_scatter_colours, num_phases)
            else:
                scatter_colours = convert_colors_to_rgb(preset_scatter_colours, num_phases)

            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111)

            # Set symmetric limits centered at origin
            ax.set_xlim(-max_bound, max_bound)
            ax.set_ylim(-max_bound, max_bound)

            # Format axes as percentages
            percent_formatter = FuncFormatter(to_percent)
            ax.xaxis.set_major_formatter(percent_formatter)
            ax.yaxis.set_major_formatter(percent_formatter)

            # First: Plot contour in the background with distinct colormap
            ax.contourf(X, Y, Z, levels=15, cmap=contour_cmap, alpha=0.9)
            ax.contour(X, Y, Z, levels=15, cmap=contour_cmap, linewidths=0.5, alpha=0.9)

            # Second: Overlay scatter points with classification colors
            point_colors = create_colors_from_probabilities(
                best_probabilities, num_components, scatter_colours
            )
            ax.scatter(
                da_arr,
                db_arr,
                c=point_colors,
                alpha=0.7,
                s=20,
                edgecolors="black",
                linewidths=0.3,
                zorder=7,
            )

            # Create a clip path from the contour
            contour_path = None
            for collection in ax.collections:
                if isinstance(collection, plt.matplotlib.collections.LineCollection):
                    for path in collection.get_paths():
                        if contour_path is None:
                            contour_path = path
                        else:
                            contour_path = Path.make_compound_path(contour_path, path)

            # Plot GMM centers and ellipses using validated kwargs colors
            gmm_color = [gmm_center_colour, gmm_ellipse_colour]

            ax.scatter(
                best_means[:, 0],
                best_means[:, 1],
                c=gmm_color[0],
                s=300,
                marker="x",
                linewidths=4,
                alpha=0.8,
                label="GMM Centers",
                zorder=10,
            )

            for i in range(num_components):
                plot_gaussian_ellipse(
                    ax,
                    best_means[i],
                    best_cov[i],
                    n_std=2,
                    edgecolor=gmm_color[1],
                    linewidth=1.5,
                    linestyle="-",
                    alpha=0.6,
                    zorder=8,
                    clip_path=contour_path,
                )

            # Add x and y axes through origin
            ax.axhline(y=0, color="black", linewidth=1.5, linestyle="-", alpha=0.7, zorder=1)
            ax.axvline(x=0, color="black", linewidth=1.5, linestyle="-", alpha=0.7, zorder=1)

            ax.set_xlabel("du")
            ax.set_ylabel("dv")
            ax.set_title("Classification & Contour Overlay")

            # Add colorbar for contour (density)
            # plt.colorbar(contour, ax=ax, label="Density")

            # Add appropriate color reference based on number of phases
            if num_phases == 2:
                add_2phase_colorbar(ax, scatter_colours)
            elif num_phases == 3:
                add_3phase_color_triangle(fig, ax, scatter_colours)
            # For num_phases > 3 or == 1, don't add any color reference

            ax.legend(loc="best")
            # plt.tight_layout()
            plt.show()

        if plot_order_parameter:
            # Create colors from full probability distribution with custom scatter_colours
            colors = create_colors_from_probabilities(
                best_probabilities, num_phases, scatter_colours
            )

            fig, ax = show_2d(
                self._image.array,
                axsize=(8, 7),
                cmap="gray",
            )

            # Plot points with colormap
            ax.scatter(
                y_arr,  # col (x-axis)
                x_arr,  # row (y-axis)
                c=colors,  # color by probabilities
                s=50,  # point size
                alpha=0.8,  # slight transparency
                edgecolors="black",  # edge for visibility
                linewidth=1,
            )

            ax.set_title("Spatial phase probability map")

            # Add appropriate color reference based on number of phases
            if num_phases == 2:
                add_2phase_colorbar(ax, scatter_colours)
            elif num_phases == 3:
                add_3phase_color_triangle(fig, ax, scatter_colours)
            # For num_phases > 3 or == 1, don't add any color reference

            ax.axis("off")
            fig.tight_layout()
            fig.show()

        return self

    # --- Plotting Functions ---
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
        da = pol_vec[0]["da"]
        db = pol_vec[0]["db"]

        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        L = np.column_stack((u, v))
        dr = L @ np.vstack((da, db))

        # Displacements (rows, cols)
        dr_raw = dr[0].astype(float)
        dc_raw = dr[1].astype(float)

        xR = xA - dr_raw
        yR = yA - dc_raw

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
        phase_offset_deg: float = 180.0,  # red = down
        phase_dir_flip: bool = False,  # flip global hue mapping if desired
        aggregator: str = "mean",  # 'mean' or 'maxmag'
        square_tiles: bool = False,  # if True, use square pixels; if False, use rectangles
        plot: bool = False,  # if True, draw with show_2d and legend
        returnfig: bool = False,  # if True (and plot=True) also return (fig, ax)
        show_colorbar: bool = True,
        figsize=(6, 6),
        **kwargs,
    ):
        """
        Build and return an RGB superpixel image indexed by integer (a,b), where each
        pixel is colored according to the direction and magnitude of polarization vectors
        using a perceptually uniform polar color mapping.

        The hue encodes the displacement direction, while lightness and chroma encode the
        magnitude. This provides a consistent visual representation across both arrow and
        image-based polarization visualizations.

        Parameters
        ----------
        square_tiles : bool, default False
            If True, use square pixels (original method).
            If False, use rectangular pixels proportional to lattice vectors u and v,
            with area close to pixel_size^2.

        Returns
        -------
        img_rgb : (H,W,3) float in [0,1]
        (fig, ax) : optional, only when plot=True and returnfig=True
        """
        import numpy as np
        from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from quantem.core.visualization.visualization_utils import array_to_rgba

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
        a_raw = pol_vec[0]["a"]
        b_raw = pol_vec[0]["b"]
        da = pol_vec[0]["da"]  # fractional displacement in a direction
        db = pol_vec[0]["db"]  # fractional displacement in b direction

        # Convert fractional displacements to Cartesian displacements
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        L = np.column_stack((u, v))
        displacement_fractional = np.vstack((da, db))
        displacement_cartesian = L @ displacement_fractional

        # Extract Cartesian displacements
        dr_raw = displacement_cartesian[0].astype(float)  # down +
        dc_raw = displacement_cartesian[1].astype(float)  # right +

        # --- Calculate pixel sizes ---
        if square_tiles:
            # Square pixels
            pixel_size_a = pixel_size
            pixel_size_b = pixel_size
        else:
            # Rectangular pixels proportional to u and v
            # We want pixel_size_a * pixel_size_b ≈ pixel_size^2
            # and pixel_size_a / pixel_size_b = |u| / |v|

            # Get lattice vector magnitudes
            u_mag = np.linalg.norm(u)
            v_mag = np.linalg.norm(v)

            # Calculate aspect ratio
            aspect_ratio = u_mag / v_mag

            # Solve for pixel dimensions:
            # pixel_size_a = aspect_ratio * pixel_size_b
            # pixel_size_a * pixel_size_b = pixel_size^2
            # => aspect_ratio * pixel_size_b^2 = pixel_size^2
            # => pixel_size_b = pixel_size / sqrt(aspect_ratio)
            # => pixel_size_a = pixel_size * sqrt(aspect_ratio)

            pixel_size_b = max(1, round(pixel_size / np.sqrt(aspect_ratio)))
            pixel_size_a = max(1, round(pixel_size * np.sqrt(aspect_ratio)))

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

        # Per-sample RGB from perceptually uniform polar color mapping
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
        H = padding * 2 + nrows * pixel_size_a + (nrows - 1) * spacing
        W = padding * 2 + ncols * pixel_size_b + (ncols - 1) * spacing
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
            r0 = padding + rr * (pixel_size_a + spacing)
            c0 = padding + cc * (pixel_size_b + spacing)

            if aggregator == "maxmag":
                j = idx_list[int(np.argmax(mag[idx_list]))]
                color = colors[j]
            else:  # 'mean'
                color = colors[idx_list].mean(axis=0)

            img_rgb[r0 : r0 + pixel_size_a, c0 : c0 + pixel_size_b, :] = color

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


# Implementing GMM using Torch (don't want skimage as a dependency)
class TorchGMM:
    """
    PyTorch Gaussian Mixture Model with full covariances optimized via EM.
    Only 'full' covariance is supported.
    Allows custom means initialization, cov regularization, and device/dtype control.
    After fit, exposes means_, covariances_, and weights_; use predict_proba for responsibilities.
    """

    def __init__(
        self,
        n_components,
        covariance_type="full",
        means_init=None,
        tol=1e-4,
        max_iter=200,
        reg_covar=1e-6,
        device=None,
        dtype=torch.float32,
    ):
        if covariance_type != "full":
            raise NotImplementedError("Only 'full' covariance_type is supported as of now.")

        # Store parameters - handle edge cases gracefully
        self.n_components = int(n_components)

        # Convert negative max_iter to 0 (or absolute value)
        self.max_iter = abs(int(max_iter))

        self.covariance_type = covariance_type
        self.means_init = None if means_init is None else np.asarray(means_init, dtype=np.float32)
        self.tol = abs(float(tol))  # Also handle negative tolerance
        self.reg_covar = float(reg_covar)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Fitted attributes (NumPy for external access)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

        # Internal torch parameters
        self._means = None  # [K, D]
        self._covariances = None  # [K, D, D]
        self._weights = None  # [K]

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(x, dtype=self.dtype, device=self.device)

    def _kmeans_plusplus_init(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """Initialize means using k-means++ algorithm for better spread."""
        N, D = X.shape

        # Work on CPU for deterministic behavior
        X_cpu = X.cpu()

        # First center: random choice
        indices = [torch.randint(0, N, (1,), device="cpu").item()]

        # Remaining centers: choose based on distance to existing centers
        for _ in range(1, K):
            # Compute distances to nearest existing center
            centers = X_cpu[indices]
            dists = torch.cdist(X_cpu, centers)  # [N, num_centers]
            min_dists = dists.min(dim=1)[0]  # [N]

            # Square distances for probability weighting
            probs = min_dists**2
            probs_sum = probs.sum()

            # Handle case where all points are identical (probs_sum == 0)
            if probs_sum > 1e-10:
                probs = probs / probs_sum
                # Sample next center
                next_idx = torch.multinomial(probs, 1).item()
            else:
                # All points are very close, just pick randomly
                next_idx = torch.randint(0, N, (1,), device="cpu").item()

            indices.append(next_idx)

        return X_cpu[indices].to(device=self.device, dtype=self.dtype)

    def _init_params(self, X: torch.Tensor) -> None:
        N, D = X.shape
        K = self.n_components

        if self.means_init is not None:
            if self.means_init.shape != (K, D):
                raise ValueError(
                    f"means_init must have shape ({K}, {D}), got {self.means_init.shape}"
                )
            self._means = self._to_tensor(self.means_init).clone()
        else:
            # Initialize means using k-means++ for better separation
            if N > 0 and K > 0:
                if N >= K:
                    self._means = self._kmeans_plusplus_init(X, K)
                else:
                    # Sample with replacement if not enough samples
                    X_cpu = X.cpu()
                    indices = torch.randint(0, N, (K,), device="cpu")
                    self._means = X_cpu[indices].clone().to(device=self.device, dtype=self.dtype)
            else:
                self._means = torch.zeros((K, D), device=self.device, dtype=self.dtype)

        # Initialize covariances with global covariance for stability
        if N > 1:
            X_centered = X - X.mean(dim=0, keepdim=True)
            global_cov = (X_centered.T @ X_centered) / (N - 1)
            # Add strong regularization for near-singular cases
            global_cov = global_cov + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )
        else:
            global_cov = self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

        # Ensure minimum eigenvalue for numerical stability
        eigenvalues = torch.linalg.eigvalsh(global_cov)
        if eigenvalues.min() < self.reg_covar:
            global_cov = global_cov + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                D, device=self.device, dtype=self.dtype
            )

        self._covariances = global_cov.unsqueeze(0).repeat(K, 1, 1).clone()

        # Initialize weights uniformly - handle K=0 case
        self._weights = torch.full(
            (K,), 1.0 / K if K > 0 else 1.0, device=self.device, dtype=self.dtype
        )

    def _log_gaussians(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, D], means: [K, D], covs: [K, D, D]
        N, D = X.shape
        K = self.n_components

        # Compute log probabilities for each component
        log_probs = []
        for k in range(K):
            # Ensure covariance is positive definite
            cov_k = self._covariances[k]

            # Check if covariance needs additional regularization
            try:
                # Try with current covariance
                dist = torch.distributions.MultivariateNormal(
                    loc=self._means[k], covariance_matrix=cov_k, validate_args=False
                )
                log_prob = dist.log_prob(X)
            except (RuntimeError, ValueError):
                # Add stronger regularization if needed
                cov_reg = cov_k + 1e-3 * torch.eye(D, device=self.device, dtype=self.dtype)
                dist = torch.distributions.MultivariateNormal(
                    loc=self._means[k], covariance_matrix=cov_reg, validate_args=False
                )
                log_prob = dist.log_prob(X)

            log_probs.append(log_prob)  # [N]

        log_comp = torch.stack(log_probs, dim=1)  # [N, K]
        return log_comp

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_comp = self._log_gaussians(X)  # [N, K]
        log_weights = torch.log(self._weights.clamp_min(1e-12))  # [K]
        log_post = log_comp + log_weights[None, :]  # [N, K]
        r = torch.softmax(log_post, dim=1)  # responsibilities [N, K]
        return r, log_post

    def _m_step(self, X: torch.Tensor, r: torch.Tensor) -> None:
        N, D = X.shape
        K = self.n_components
        Nk = r.sum(dim=0).clamp_min(1e-12)  # [K]
        self._weights = (Nk / N).clamp_min(1e-12)

        # Means
        self._means = (r.T @ X) / Nk[:, None]

        # Covariances (full)
        covs = []
        for k in range(K):
            diff = X - self._means[k]  # [N, D]
            cov_k = (r[:, k][:, None] * diff).T @ diff
            cov_k = cov_k / Nk[k]

            # Add regularization
            cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

            # Ensure positive definiteness
            eigenvalues = torch.linalg.eigvalsh(cov_k)
            if eigenvalues.min() < self.reg_covar:
                cov_k = cov_k + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                    D, device=self.device, dtype=self.dtype
                )

            covs.append(cov_k)
        self._covariances = torch.stack(covs, dim=0)  # [K, D, D]

    def fit(self, data) -> "TorchGMM":
        X = self._to_tensor(data)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D with shape (N, D)")

        self._init_params(X)

        prev_ll = torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)

        for iteration in range(self.max_iter):
            r, _ = self._e_step(X)
            self._m_step(X, r)

            # Compute average log-likelihood of data under mixture
            log_comp = self._log_gaussians(X)
            log_weighted = log_comp + torch.log(self._weights)[None, :]
            ll = torch.logsumexp(log_weighted, dim=1).mean()

            # Check convergence
            if iteration > 0 and torch.isfinite(prev_ll) and torch.isfinite(ll):
                improvement = (ll - prev_ll).abs()
                if improvement < self.tol:
                    break
            prev_ll = ll

        # Store NumPy copies for external use (decoupled from internal tensors)
        self.means_ = self._means.detach().clone().cpu().numpy()
        self.covariances_ = self._covariances.detach().clone().cpu().numpy()
        self.weights_ = self._weights.detach().clone().cpu().numpy()
        return self

    def predict_proba(self, data) -> np.ndarray:
        X = self._to_tensor(data)
        r, _ = self._e_step(X)
        return r.detach().cpu().numpy()


# helper functions for plotting
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


def site_colors(number):
    """
    Map an integer 'number' to an RGB triple in [0,1].
    If 'number' is a list, array, or tuple, returns an array of RGB triples.
    Starts with the requested seed palette and cycles thereafter.
    """

    palette = [
        (1.00, 0.00, 0.00),  # 0: red
        (0.00, 0.70, 1.00),  # 1: lighter blue
        (0.00, 0.70, 0.00),  # 2: green with lower perceptual brightness
        (1.00, 0.00, 1.00),  # 3: magenta
        (1.00, 0.70, 0.00),  # 4: orange
        (0.00, 0.00, 1.00),  # 5: full blue
        # extras to improve variety when cycling:
        (0.60, 0.20, 0.80),
        (0.30, 0.75, 0.75),
        (0.80, 0.40, 0.00),
        (0.20, 0.60, 0.20),
        (0.70, 0.70, 0.00),
        (0.00, 0.00, 0.00),  # -1: black
        # ENSURE BLACK IS ALWAYS LAST IF ADDING NEW COLORS
    ]

    # Check if input is a list, tuple, or array
    if isinstance(number, int):
        # Original behavior for single integer
        idx = int(number) % len(palette)
        return palette[idx]
    else:
        # Convert to numpy array for vectorized operations
        numbers = np.asarray(number, dtype=int)
        indices = numbers % len(palette)
        # Return array of RGB tuples
        return np.array([palette[idx] for idx in indices.flat]).reshape(numbers.shape + (3,))


def create_colors_from_probabilities(probabilities, num_phases, category_colors=None):
    """
    Create colors from probability distribution with a smooth transition to white for uncertainty.
    Smoothing is applied only when num_phases = 3.

    Parameters:
    -----------
    probabilities : array of shape (N, n_categories)
        Probabilities for each category (rows should sum to 1)
    num_phases : int
        Number of phases/categories
    category_colors : array of shape (num_phases, 3), optional
        Custom RGB colors for each category. If None, uses site_colors.

    Returns:
    --------
    colors : array of shape (N, 3)
        RGB colors for each point
    """
    import matplotlib.colors as mcolors

    # Get base colors for each category (0-1 range)
    if category_colors is None:
        category_colors = np.array([site_colors(i) for i in range(num_phases)])

    # Mix colors based on probabilities
    mixed_colors = probabilities @ category_colors

    if num_phases == 3:
        # Apply smoothing for 3-phase system
        # Calculate certainty (max probability)
        certainty = np.max(probabilities, axis=1)

        # Create a smooth transition function
        def smooth_transition(x):
            return 3 * x**2 - 2 * x**3

        # Apply smooth transition to certainty
        smooth_certainty = smooth_transition(certainty)

        # Blend with white: uncertain -> white, certain -> category color
        white = np.array([1.0, 1.0, 1.0])
        final_colors = (
            smooth_certainty[:, np.newaxis] * mixed_colors
            + (1 - smooth_certainty[:, np.newaxis]) * white
        )

        # Ensure colors are in valid range [0, 1] BEFORE HSV conversion
        final_colors = np.clip(final_colors, 0, 1)

        # Convert to HSV for final adjustments
        hsv_colors = mcolors.rgb_to_hsv(final_colors)

        # Adjust saturation based on certainty
        hsv_colors[:, 1] *= smooth_certainty

        # Convert back to RGB
        final_colors = mcolors.hsv_to_rgb(hsv_colors)
    else:
        # For 2-phase system, use the original method
        # Calculate certainty (inverse of entropy)
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        max_entropy = np.log(num_phases)

        # Certainty: 0 (uncertain) to 1 (certain)
        certainty = 1 - (entropy / max_entropy)

        # Blend with white: uncertain -> white, certain -> category color
        white = np.array([1.0, 1.0, 1.0])
        final_colors = (
            certainty[:, np.newaxis] * mixed_colors + (1 - certainty[:, np.newaxis]) * white
        )

    # Ensure final colors are in valid range [0, 1]
    final_colors = np.clip(final_colors, 0, 1)

    return final_colors


def add_2phase_colorbar(ax, scatter_colours):
    """
    Add a 1D colorbar for 2-phase system
    Creates a colormap that goes: color0 -> white (center) -> color1

    Parameters:
    -----------
    ax : matplotlib axes
        The main plot axes
    scatter_colours : array of shape (2, 3)
        RGB colors for the two phases
    """
    from matplotlib.colors import LinearSegmentedColormap

    fig = ax.get_figure()

    # Find the rightmost edge of all existing axes
    max_right = ax.get_position().x1
    for fig_ax in fig.get_axes():
        if fig_ax != ax:
            max_right = max(max_right, fig_ax.get_position().x1)

    # Calculate the position for the colorbar
    ax_pos = ax.get_position()
    cbar_width = 0.035
    cbar_pad = 0.05
    cbar_left = max_right + cbar_pad
    cbar_bottom = ax_pos.y0
    cbar_height = ax_pos.height

    # Create new axes for colorbar
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    # Get the two phase colors from scatter_colours
    color0 = scatter_colours[0]
    color1 = scatter_colours[1]

    # Create a colormap that goes: color0 -> white (center) -> color1
    colors_list = [color0, (1, 1, 1), color1]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("two_phase", colors_list, N=n_bins)
    # Create gradient
    gradient = np.linspace(0, 1, 256).reshape(256, 1)

    # Display the colorbar
    cax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")

    # Configure ticks and labels
    cax.set_xticks([])
    cax.set_yticks([0, 128, 255])
    cax.set_yticklabels(["Phase 0", "Uncertain", "Phase 1"])
    cax.yaxis.tick_right()

    return cax


def add_3phase_color_triangle(fig, ax, scatter_colours):
    """
    Add a ternary color triangle for 3-phase system

    Parameters:
    -----------
    fig : matplotlib figure
        The figure object
    ax : matplotlib axes
        The main plot axes
    scatter_colours : array of shape (3, 3)
        RGB colors for the three phases
    """

    # Check if there are existing colorbars/triangles attached to the figure
    box = ax.get_position()
    existing_elements = []

    # Find all axes that might be colorbars or previous triangles
    for fig_ax in fig.get_axes():
        if fig_ax != ax:
            pos = fig_ax.get_position()
            # Check if it's positioned to the right of the main axes
            if pos.x0 >= box.x1:
                existing_elements.append(fig_ax)

    # Calculate horizontal offset based on existing elements
    if existing_elements:
        # Find the rightmost existing element
        rightmost_x = max(elem.get_position().x1 for elem in existing_elements)
        x_offset = rightmost_x + 0.02  # Add spacing after the rightmost element
    else:
        x_offset = box.x1 + 0.02

    # Create a new axes for the triangle
    # Adjust position to account for existing colorbars
    triangle_width = box.height * 0.8
    triangle_ax = fig.add_axes([x_offset, box.y0, triangle_width, box.height * 0.8])

    # Get the three phase colors from scatter_colours
    color0 = scatter_colours[0]
    color1 = scatter_colours[1]
    color2 = scatter_colours[2]

    # Create ternary color grid
    resolution = 100
    positions = []
    probabilities_list = []

    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j

            # Probabilities (barycentric coordinates)
            p0, p1, p2 = i / resolution, j / resolution, k / resolution
            probabilities_list.append([p0, p1, p2])

            # Convert to Cartesian coordinates for ternary plot
            x = 0.5 * (2 * p1 + p2)
            y = (np.sqrt(3) / 2) * p2
            positions.append([x, y])

    positions = np.array(positions)
    probabilities_array = np.array(probabilities_list)

    # Get colors with custom scatter_colours
    colors = create_colors_from_probabilities(probabilities_array, 3, scatter_colours)

    # Plot the triangle
    triangle_ax.scatter(
        positions[:, 0], positions[:, 1], c=colors, s=20, marker="s", edgecolors="none"
    )

    # Draw triangle edges
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    triangle_ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], "k-", linewidth=2)

    # Add vertex markers and labels
    vertex_size = 150

    # Vertex 0 (bottom left) - Phase 0
    triangle_ax.scatter(
        0, 0, s=vertex_size, c=[color0], edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(0, -0.1, "Phase 0", ha="center", va="top", fontsize=10, fontweight="bold")

    # Vertex 1 (bottom right) - Phase 1
    triangle_ax.scatter(
        1, 0, s=vertex_size, c=[color1], edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(1, -0.1, "Phase 1", ha="center", va="top", fontsize=10, fontweight="bold")

    # Vertex 2 (top) - Phase 2
    triangle_ax.scatter(
        0.5, np.sqrt(3) / 2, s=vertex_size, c=[color2], edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(
        0.5,
        np.sqrt(3) / 2 + 0.1,
        "Phase 2",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    # Mark center (maximum uncertainty) - white
    triangle_ax.scatter(
        0.5, np.sqrt(3) / 6, s=vertex_size, c="white", edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(
        0.65,
        np.sqrt(3) / 6,
        "Uncertain\n(Equal)",
        ha="left",
        va="center",
        fontsize=8,
        style="italic",
    )

    # Set limits and styling
    triangle_ax.set_xlim(-0.15, 1.15)
    triangle_ax.set_ylim(-0.2, np.sqrt(3) / 2 + 0.15)
    triangle_ax.set_aspect("equal")
    triangle_ax.axis("off")
    triangle_ax.set_title("Probability Map", fontsize=11, pad=10)

    return triangle_ax
