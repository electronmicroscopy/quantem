import inspect

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.imaging.lattice_visualization import PLOT_REGISTRY


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

    ### --- Constructors ---
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

    ### --- Properties ---
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

    ### --- Functions ---
    def define_lattice_vectors(
        self,
        origin,
        u,
        v,
        refine_lattice: bool = True,
        block_size: int | None = None,
        refine_maxiter: int = 200,
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
        refine_maxiter : int, default=200
            Maximum number of iterations for the lattice refinement optimizer (Powell method).

        Returns
        -------
        self : Lattice
            Returns the same object, modified in-place.
            The final values of r0, u, v are stored in self._lat.

        Side Effects
        ------------
            Creates self._lat with rows corresponding to r0, u and v
            Sets self.default_plot to "lattice_vectors"
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
        if not (
            0 <= origin[0] < self.image.array.shape[0]
            and 0 <= origin[1] < self.image.array.shape[1]
        ):
            raise ValueError("origin must be within the image bounds.")
        try:
            L = self._lat[1:]
            _ = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            raise ValueError("u, v must be invertible.")

        # Refine lattice coordinates
        # Note that we currently assume corners are local maxima
        if refine_lattice:
            from scipy.optimize import minimize

            if block_size is not None and block_size < 0:
                raise ValueError("block_size must be positive or None.")

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

        self.default_plot = "lattice_vectors"

        return self

    def add_atoms(
        self,
        positions_frac,
        numbers=None,
        intensity_min=None,
        intensity_radius=None,
        refine_atoms: bool = False,
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
        mask, and optional intensity/contrast thresholds.

        Parameters
        ----------
        positions_frac : array-like, shape (S, 2)
            Fractional positions (a, b) of S lattice sites within the unit cell. These are offsets
            relative to the lattice origin r0 and basis vectors (u, v), and are used to tile the
            image with candidate atom centers at all visible integer translations.
        numbers : array-like of int, shape (S,), optional
            Identifier per site (e.g., species or label). If None, uses 0,1..,(S-1). Used only for plotting
            color coding; not used in detection logic.
        intensity_min : float, optional
            Minimum mean intensity inside the detection disk required to keep a candidate atom.
            If None, no intensity thresholding is applied.
        intensity_radius : float, optional
            Radius (in pixels) of the detection disk used to compute the mean intensity at each
            candidate center. If None, an automatic radius is estimated as half of the nearest-neighbor
            spacing in pixels (see Notes).
        refine_atoms : bool, default False
            If True, calls self.refine_atoms() after detecting atoms.
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
        Sets self.default_plot to "atoms".

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
        """

        # VALIDATION: Check that lattice vectors have been defined
        if not hasattr(self, "_lat") or self._lat is None:
            raise ValueError(
                "Lattice vectors have not been fitted. Please call define_lattice_vectors() first."
            )

        # Initialize fractional positions and metadata

        # Handle the case where positions_frac is empty (no sites to detect)
        positions_frac_arr = np.asarray(positions_frac, dtype=float)
        if positions_frac_arr.size == 0:
            # Bookkeeping for consistency: set empty arrays and return early
            self._positions_frac = np.empty((0, 2), dtype=float)
            self._num_sites = 0
            self._numbers = (
                np.array([], dtype=int)
                if numbers is None
                else np.atleast_1d(np.array(numbers, dtype=int))
            )
            # Do not construct an empty Vector with zero shape (causes error). Just return.
            return self

        # Store fractional positions and count the number of sites
        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]

        # Assign site identifiers (default: 0 to S-1, or use provided numbers)
        self._numbers = (
            np.arange(0, self._num_sites, dtype=int)
            if numbers is None
            else np.atleast_1d(np.array(numbers, dtype=int))
        )

        # Initialization and Setup

        # Extract image data and dimensions
        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape  # H = number of rows, W = number of columns

        # Extract lattice parameters: origin (r0) and basis vectors (u, v)
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

        # Lattice transformation matrix: [u | v] converts fractional coords to pixel coords
        A = np.column_stack((u, v))

        # DETERMINE RANGE OF LATTICE INDICES
        # Approximate the range of lattice indices (a, b) by
        # estimating the indices at the corners of the image

        corners = np.array(
            [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]], dtype=float
        )
        # Solve A @ [a, b]^T = corner - r0 to find fractional lattice indices at corners
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]

        # Round to integers to get the bounding box of visible lattice indices
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        # AUTOMATIC DETECTION RADIUS ESTIMATION
        def _auto_radius_px() -> float:
            """
            Estimate a default disk radius in pixels as half the nearest-neighbor spacing
            (with periodic wrapping), or from lattice vectors if insufficient points.

            Returns
            -------
            float
                Estimated disk radius in pixels.
            """
            S = self._positions_frac

            # If multiple sites exist, compute nearest-neighbor spacing in fractional coords
            if S.shape[0] >= 2:
                # Compute pairwise differences (with periodic wrapping)
                d = S[:, None, :] - S[None, :, :]  # shape: (S, S, 2)
                d = d - np.round(d)  # Account for periodicity

                # Find pairs that are effectively the same site (within numerical tolerance)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)

                # Convert fractional distances to pixel distances
                dpix = d @ A.T  # shape: (S, S, 2)
                dist = np.linalg.norm(dpix, axis=2)  # Euclidean distance

                # Ignore self-pairs (set to infinity)
                dist[same] = np.inf

                # Use the nearest-neighbor distance
                nn = float(np.min(dist))
            else:
                # Fallback for single site: use minimum distance among lattice vectors
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))

            # Safety check: ensure a valid positive estimate
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))

            # Return half the nearest-neighbor spacing as the disk radius
            return 0.5 * nn  # type:ignore

        # SETUP DETECTION PARAMETERS

        # Disk radius for intensity measurement (in pixels)
        r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()

        print(f"r_px : {r_px}")

        # Annulus radii for background contrast measurement (in pixels)
        rin, rout = (1.0 * r_px, 1.5 * r_px) if annulus_radii is None else annulus_radii

        # Precompute integer pixel ranges for disk and annulus (used in neighbor lookups)
        R_disk = int(np.ceil(r_px))
        R_ring = int(np.ceil(rout))

        # Edge distance threshold (in pixels)
        edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

        # PREPARE MASK AND DISTANCE TRANSFORM (if provided)

        DT: None = None  # Distance transform of the mask (computed if mask is provided)
        if mask is not None:
            m = np.asarray(mask).astype(bool)

            # Validate mask dimensions
            if m.shape != (H, W):
                raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")

            # Attempt to compute distance transform for fast edge distance checking
            try:
                from scipy.ndimage import distance_transform_edt

                DT = distance_transform_edt(m)
            except Exception as e:
                # If distance_transform fails, fall back to pixel-level mask checking
                import warnings

                warnings.warn(
                    f"distance_tranform failed with Exception:{e}. Defaulting to pixel-level mask checking."
                )
                DT = None

        # HELPER FUNCTIONS: Intensity Measurement

        def mean_disk(x: float, y: float) -> float:
            """
            Compute the mean image intensity within a circular disk of radius r_px centered at (x, y),
            with boundary clipping and fallback to the center pixel if empty.

            Parameters
            ----------
            x : float
                Row coordinate (0-indexed from top)
            y : float
                Column coordinate (0-indexed from left)

            Returns
            -------
            float
                Mean intensity within the disk.
            """
            # Determine the bounding box of the disk in image coordinates
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
            j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)

            # Create a grid of integer pixel indices within the bounding box
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]

            # Compute distances from the center point
            dx, dy = ii - x, jj - y

            # Create circular mask
            mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)

            # Extract and average intensities within the disk
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]

            # If no pixels fall within the disk, return the center pixel intensity
            if vals.size == 0:
                return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])

            return float(vals.mean())

        def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
            """
            Compute the mean and standard deviation of intensities within an annulus [rin, rout] centered at (x, y),
            with boundary clipping and fallback to the center pixel and zero std if empty.

            Parameters
            ----------
            x : float
                Row coordinate (0-indexed from top)
            y : float
                Column coordinate (0-indexed from left)

            Returns
            -------
            tuple[float, float]
                (mean intensity, standard deviation) within the annulus.
            """
            # Determine the bounding box for the annulus
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
            j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)

            # Create a grid of integer pixel indices within the bounding box
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]

            # Compute squared distances from the center point
            dx, dy = ii - x, jj - y
            r2 = dx * dx + dy * dy

            # Create annulus mask: include pixels with radius in [rin, rout]
            mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)

            # Extract intensities within the annulus
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]

            # If no pixels fall within the annulus, return center pixel and zero std
            if vals.size == 0:
                val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                return val, 0.0

            return float(vals.mean()), float(vals.std(ddof=0))

        # ATOM DETECTION LOOP

        # Create a Vector object to store detected atoms per site
        self.atoms = Vector.from_shape(
            shape=(self._num_sites,),
            fields=["x", "y", "a", "b", "int_peak"],
            units=["px", "px", "ind", "ind", "counts"],
        )

        # Iterate over each site in the unit cell
        for a0 in range(self._num_sites):
            # Get the fractional offset of this site within the unit cell
            da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]

            # Generate candidate lattice indices by adding integer translations to the site offset
            # This creates a grid of all possible atom positions for this site type
            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )

            # Create a basis matrix for converting fractional coords to pixel coords
            # Each row: [1, a_i, b_i] to multiply by [r0, u, v]
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T

            # Transform from fractional lattice coords to pixel coords (x, y)
            xy = basis @ self._lat  # shape: (N, 2), columns are (x, y)

            # Extract pixel coordinates
            x, y = xy[:, 0], xy[:, 1]

            # FILTERING STAGE 1: Image Bounds and Edge Distance

            # Check if candidates fall within image bounds
            in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)

            # Check edge distance: candidates must be edge_thresh pixels away from borders
            border_ok = (
                (x - edge_thresh >= 0.0)
                & (x + edge_thresh <= H - 1)
                & (y - edge_thresh >= 0.0)
                & (y + edge_thresh <= W - 1)
            )

            # FILTERING STAGE 2: Mask Checking

            if mask is not None:
                if DT is not None:
                    # Use distance transform: candidate must be at least edge_thresh away from mask boundary
                    ii = np.clip(np.round(x).astype(int), 0, H - 1)
                    jj = np.clip(np.round(y).astype(int), 0, W - 1)
                    mask_ok = DT[ii, jj] >= edge_thresh  # type:ignore
                else:
                    # Fallback: check that nearest integer pixel is True in the mask
                    m = np.asarray(mask).astype(bool)
                    mask_ok = m[
                        np.clip(np.round(x).astype(int), 0, H - 1),
                        np.clip(np.round(y).astype(int), 0, W - 1),
                    ]
            else:
                # No mask: all candidates are mask-okay
                mask_ok = np.ones_like(in_bounds, dtype=bool)

            # FILTERING STAGE 3: Intensity and Contrast Thresholds

            # Start with candidates that pass bounds, border, and mask checks
            keep = in_bounds & border_ok & mask_ok

            # INTENSITY MEASUREMENT

            # Compute mean intensity in the detection disk for all candidates
            int_center = np.empty(xy.shape[0], dtype=float)
            for i in range(xy.shape[0]):
                if keep[i]:
                    int_center[i] = mean_disk(x[i], y[i])
                else:
                    int_center[i] = 0.0

            # Apply intensity minimum threshold (if provided)
            if intensity_min is not None:
                keep &= int_center >= float(intensity_min)

            # Apply contrast minimum threshold (if provided)
            if contrast_min is not None:
                # Compute background intensity in the annulus for all candidates
                bg_mean = np.empty(xy.shape[0], dtype=float)
                for i in range(xy.shape[0]):
                    if keep[i]:
                        bg_mean[i], _ = mean_std_annulus(x[i], y[i])
                    else:
                        bg_mean[i] = np.inf

                # Keep candidates where (disk mean - annulus mean) >= contrast_min
                keep &= (int_center - bg_mean) >= float(contrast_min)

            # STORE DETECTED ATOMS FOR THIS SITE

            # Compile the results for this site into a structured array
            if np.any(keep):
                # Stack the kept candidates: (x, y, a, b, int_peak)
                arr = np.vstack(
                    (x[keep], y[keep], basis[keep, 1], basis[keep, 2], int_center[keep])
                ).T
            else:
                # No candidates passed all filters: create an empty array with correct shape
                arr = np.zeros((0, 5), dtype=float)

            # Store in the atoms container for this site
            self.atoms[a0] = arr

        # POST-PROCESSING AND RETURN
        # Set the default plot type to show atoms
        self.default_plot = "atoms"

        # Optionally refine atom positions using a separate refinement algorithm
        if refine_atoms:
            # Extract only parameters that refine_atoms accepts
            refine_kwargs = {
                k: v for k, v in kwargs.items() if k in ["fit_radius", "max_nfev", "max_move_px"]
            }
            self.refine_atoms(**refine_kwargs)

        return self

    def refine_atoms(
        self,
        fit_radius=None,
        max_nfev: int = 200,
        max_move_px: float | None = None,
    ) -> "Lattice":
        """
        Refine atom centers by local 2D Gaussian fitting around each previously detected atom.
        Updates atom positions and peak intensity and adds per-atom sigma and background fields.

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
        Sets self.default_plot to "atoms".

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
        """

        # VALIDATION: Ensure atoms exist before attempting refinement
        if not hasattr(self, "atoms"):
            raise ValueError("No atoms to refine. Call add_atoms() first.")

        # Load image as float array and get dimensions
        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape  # H = rows, W = columns

        # Extract lattice origin and basis vectors
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

        # Lattice transformation matrix: [u | v] converts fractional coords to pixel coords
        A = np.column_stack((u, v))

        # AUTOMATIC FITTING RADIUS ESTIMATION
        def _auto_radius_px() -> float:
            """
            Estimate a default fitting radius as half the nearest-neighbor spacing between sites,
            or from lattice vectors if insufficient sites.

            Returns
            -------
            float
                Estimated fitting radius in pixels.
            """
            # Get stored fractional site positions; default to origin if not available
            S = np.asarray(getattr(self, "_positions_frac", [[0.0, 0.0]]), dtype=float)

            # For multiple sites: compute nearest-neighbor spacing with periodic wrapping
            if S.shape[0] >= 2:
                # Pairwise differences between all sites
                d = S[:, None, :] - S[None, :, :]  # shape: (S, S, 2)

                # Wrap to [-0.5, 0.5] to account for periodic boundary conditions
                d = d - np.round(d)

                # Identify self-pairs (within numerical tolerance)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)

                # Convert fractional distances to pixel distances
                dpix = d @ A.T  # shape: (S, S, 2)
                dist = np.linalg.norm(dpix, axis=2)  # Euclidean distance

                # Ignore self-pairs by setting to infinity
                dist[same] = np.inf

                # Use minimum nearest-neighbor distance
                nn = float(np.min(dist))
            else:
                # Single site fallback: use minimum lattice vector magnitude
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))

            # Safety check for invalid or non-positive estimates
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))

            # Return half the spacing as the fitting radius
            return 0.5 * nn  # type:ignore

        # SETUP FITTING PARAMETERS

        # Fitting radius: auto-estimate or user-provided
        r_fit = float(fit_radius) if fit_radius is not None else _auto_radius_px()

        # Integer pixel range for patch extraction (used in neighborhood lookups)
        R = int(np.ceil(r_fit))

        # Maximum allowed movement: defaults to fitting radius
        max_move = float(max_move_px) if max_move_px is not None else r_fit

        # ENSURE ADDITIONAL FIELDS EXIST

        # Check which fields need to be added (sigma and int_bg for refined results)
        needed = [f for f in ("sigma", "int_bg") if f not in self.atoms.fields]

        # Add missing fields to all sites if needed
        if needed:
            self.atoms.add_fields(needed)

        # PRECOMPUTE COLUMN INDICES FOR EFFICIENT UPDATES
        # Store column indices to avoid repeated lookups during refinement loop
        idx_x = self.atoms.fields.index("x")
        idx_y = self.atoms.fields.index("y")
        idx_amp = self.atoms.fields.index("int_peak")
        idx_sigma = self.atoms.fields.index("sigma")
        idx_bg = self.atoms.fields.index("int_bg")

        # MAIN REFINEMENT LOOP: Iterate over all sites and atoms

        for s in range(self._num_sites):
            # Get the atom data for this site
            row = self.atoms[s].array

            # Skip if no atoms exist for this site
            if isinstance(row, list) or row is None or row.size == 0:
                continue

            # Extract x and y coordinates for all atoms at this site
            x_arr = self.atoms[s].select_fields("x").array[:, 0]
            y_arr = self.atoms[s].select_fields("y").array[:, 0]

            # Create a copy of the atom data to accumulate refinements
            updated = row.copy()

            # REFINE EACH INDIVIDUAL ATOM
            for i in range(row.shape[0]):
                # Current atom position (center of fitting region)
                x0, y0 = float(x_arr[i]), float(y_arr[i])

                # EXTRACT CIRCULAR PATCH AROUND ATOM

                # Determine bounding box of the patch
                ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
                i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
                j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)

                # Skip if patch is too small or out of bounds
                if i1 <= i0 or j1 <= j0:
                    continue

                # Extract the image patch
                patch = im[i0 : i1 + 1, j0 : j1 + 1]

                # Create coordinate grids for the patch
                ii = np.arange(i0, i1 + 1)[:, None]
                jj = np.arange(j0, j1 + 1)[None, :]
                II = np.broadcast_to(ii, patch.shape)
                JJ = np.broadcast_to(jj, patch.shape)

                # CREATE CIRCULAR MASK AND EXTRACT PIXELS

                # Squared distances from atom center to all patch pixels
                r2 = (II - x0) ** 2 + (JJ - y0) ** 2

                # Circular mask: pixels within r_fit of center
                mask = r2 <= (r_fit * r_fit)

                # Skip if no pixels fall within the circular region
                if not np.any(mask):
                    continue

                # Extract intensities within the circular region
                vals = patch[mask].astype(float).ravel()

                # INITIAL GUESS FOR GAUSSIAN PARAMETERS

                # Patch intensity extrema (used for bounds and initial guess)
                pmin, pmax = float(vals.min()), float(vals.max())

                # Background: median of pixels outside the circular mask (or full patch if none)
                bg0 = float(np.median(patch[~mask])) if np.any(~mask) else float(np.median(patch))

                # Amplitude: central pixel value minus background (with safety floor)
                amp0 = max(float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6)

                # Gaussian width: half the fitting radius (with safety floor)
                sig0 = max(r_fit * 0.5, 0.5)

                # Extract coordinates of pixels in the mask
                x_coords = II[mask].astype(float).ravel()
                y_coords = JJ[mask].astype(float).ravel()

                # DEFINE RESIDUAL FUNCTION FOR GAUSSIAN FIT
                def residual(theta):
                    """
                    Compute residuals between observed intensities and 2D Gaussian model.

                    Parameters
                    ----------
                    theta : array of length 5
                        [x_c, y_c, amp, sigma, bg] parameters

                    Returns
                    -------
                    ndarray
                        Residuals at each masked pixel.
                    """
                    x_c, y_c, amp, sig, bg = theta

                    # Avoid division by zero in the Gaussian exponent
                    sig2 = max(sig, 1e-6) ** 2

                    # Squared distance from fitted center to each pixel
                    rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2

                    # Isotropic 2D Gaussian + constant background
                    model = amp * np.exp(-0.5 * rr / sig2) + bg

                    # Return residuals (model - observed)
                    return model - vals

                # PARAMETER BOUNDS

                # Center bounds: limited by max_move and image bounds
                x_lb = max(x0 - max_move, 0.0)
                x_ub = min(x0 + max_move, H - 1.0)
                y_lb = max(y0 - max_move, 0.0)
                y_ub = min(y0 + max_move, W - 1.0)

                # Lower and upper bounds for all five parameters
                lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
                ub = [
                    x_ub,
                    y_ub,
                    max(pmax - pmin, amp0 * 4.0),
                    max(2.0 * r_fit, 1.0),
                    pmax + (pmax - pmin),
                ]

                # Initial guess for optimization
                theta0 = [x0, y0, amp0, sig0, bg0]

                # NON-LINEAR LEAST-SQUARES OPTIMIZATION
                res = least_squares(
                    residual,
                    theta0,
                    bounds=(lb, ub),
                    method="trf",  # Trust Region Reflective (handles bounds well)
                    loss="soft_l1",  # Soft L1 loss (robust to outliers)
                    max_nfev=int(max_nfev),
                    xtol=1e-6,  # Tolerance for parameter changes
                    ftol=1e-6,  # Tolerance for residual changes
                    gtol=1e-6,  # Tolerance for gradient
                )

                # UPDATE ATOM PARAMETERS WITH REFINED VALUES
                # Extract optimized parameters
                x_c, y_c, amp, sig, bg = res.x

                # Update the row with refined values
                updated[i, idx_x] = x_c
                updated[i, idx_y] = y_c
                updated[i, idx_amp] = amp
                updated[i, idx_sigma] = sig
                updated[i, idx_bg] = bg

            # STORE UPDATED ATOMS FOR THIS SITE
            # Write the refined atoms back to the atoms container
            self.atoms[s] = updated

        # POST-PROCESSING AND RETURN
        self.default_plot = "atoms"

        return self

    ### --- Plot dispatcher ---
    def plot(self, kind: str | None = None, show_docstring: bool = False, **kwargs):
        """
        Dispatch to a registered visualization function.

        The function is selected by ``kind`` if given, otherwise by
        ``self.default_plot``.  Passing ``kind`` here does not mutate the instance.

        Parameters
        ----------
        kind : str | None
            Name of the plot.  Registered names:

            "image" | "dataset"
                Shows the image using the default show_2d with no overlays.
                Default if default_plot is not set.

            "lattice_vectors"
                Lattice vectors + grid lines overlaid on the image.
                Call after define_lattice_vectors().

            "atoms"
                Atom positions overlaid on the image.
                Call after add_atoms() or refine_atoms().

        show_docstring : bool, default False
            If True, return formatted signature and docstring of the plotting
            function instead of calling it. If False, call the function and
            return its result.

        **kwargs
            Forwarded verbatim to the selected plotting function.

        Returns
        -------
        str or function return
            If ``show_docstring`` is True, returns a formatted string with the
            function signature and docstring. Otherwise, returns whatever the
            selected function returns (typically ``(fig, ax)``).

        Raises
        ------
        ValueError
            If ``kind`` or ``self.default_plot`` is not in PLOT_REGISTRY.

        Examples
        --------
        ::

            lat.plot(kind="lattice_vectors")
            lat.plot(kind="atoms")
            lat.plot(kind="atoms", show_docstring=True)
        """
        if not hasattr(self, "default_plot") or kind in ["image", "dataset"]:
            from quantem.core.visualization import show_2d

            return show_2d(self.image, **kwargs)  # type:ignore

        plot_name = kind if kind is not None else self.default_plot
        if plot_name not in PLOT_REGISTRY:
            raise ValueError(
                f"Unknown plot kind {plot_name!r}. Available: {sorted(PLOT_REGISTRY)}"
            )

        plot_func = PLOT_REGISTRY[plot_name]
        if show_docstring:
            plot_func = PLOT_REGISTRY[plot_name]
            sig = inspect.signature(plot_func)
            doc = inspect.getdoc(plot_func)

            if kind is None:
                print(f"Current default plot: {self.default_plot}")

            print("\nSignature:")
            params = []
            for param_name, param in sig.parameters.items():
                params.append(f"    {param}")

            param_str = ",\n".join(params)
            return_annotation = (
                f" -> {sig.return_annotation}"
                if sig.return_annotation != inspect.Signature.empty
                else ""
            )

            print(f"def {plot_func.__name__}(\n{param_str}\n){return_annotation}:")

            print("\nDocstring:")
            if doc:
                print(doc)
            else:
                print("[No docstring]")

            return None

        return plot_func(self, **kwargs)
