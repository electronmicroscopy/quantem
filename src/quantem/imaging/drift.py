from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fft import fft2, fftfreq, ifft2
import warnings
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.core.config import validate_device

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.utils.imaging_utils import (
    bilinear_kde,
    cross_correlation_shift,
    fourier_cropping,
)
from quantem.imaging.drift_utils import (
    bilinear_kde_batch_torch,
    cross_corr_batch_torch,
    dft_upsample_torch,
    transform_coordinates_torch,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d


class DriftCorrection(AutoSerialize):
    """
    DriftCorrection provides translation, affine, and non-rigid drift correction for
    sequential 2D images using scan direction metadata and flexible spatial interpolation.

    This class supports input data as numpy arrays, Dataset2d, or Dataset3d instances,
    with various padding strategies and configurable spline interpolation of scanline
    trajectories via Bézier knot control.

    Features
    --------
    - Load data from arrays or files
    - Apply initial scanline resampling using Bézier curves
    - Align images using translation, affine, or non-rigid optimization
    - Visualize intermediate and final results with optional knot overlays
    - Serialize state with `.save()` and restore with `.load()`

    Parameters (via `from_data` or `from_file`)
    -------------------------------------------
    images : list of 2D arrays, Dataset2d, Dataset3d, or file names, or a 3D numpy array
        The image stack to correct for drift.
    scan_direction_degrees : list of float
        The scan direction angle (in degrees) for each image, measured relative to vertical.
    pad_fraction : float, default 0.25
        Fraction of padding to add around each image during interpolation.
    pad_value : str, float, or list of float, default 'median'
        How to pad outside the image area during warping. Can be:
        - One of: 'median', 'mean', 'min', 'max'
        - A float quantile value (e.g., 0.25)
        - A list of per-image float values
    number_knots : int, default 1
        Number of knots to use for Bézier interpolation of scanline trajectories.
        We strongly recommend using `number_knots = 1` unless the fast scan direction is
        expected to vary within the image.

    Example
    -------
    Instantiate the DriftCorrection class, run preprocessing and alignment, and save/load results:

    >>> drift = DriftCorrection.from_data(
    ...     images=[
    ...         image0,  # 2D numpy array or Dataset2d
    ...         image1,
    ...     ],
    ...     scan_direction_degrees=[0, 90],
    ... ).preprocess(
    ...     pad_fraction=0.25,
    ...     pad_value='median',
    ...     number_knots=1,
    ... )

    >>> drift.align_affine()
    >>> drift.align_nonrigid()
    >>> drift.plot_merged_images()
    >>> image_corr = drift.generate_corrected_image()

    >>> drift.save("drift_result.zip")
    >>> drift_reloaded = quantem.io.load("drift_result.zip")

    >>> image_corr.save("image_corrected.zip")
    >>> image_corr_reloaded = quantem.io.load("image_corrected.zip")

    Notes
    -----
    - Use `align_translation()` for rigid shifts, `align_affine()` for scan-shear or uniform drift,
      and `align_nonrigid()` for flexible per-row or per-image correction.
    - The class stores resampled images in `self.images_warped` and the control knots in `self.knots`.
    - Interactive visualization is supported through `plot_merged_images()` and `plot_transformed_images()`.
    """

    _token = object()

    def __init__(
        self,
        images: list[Dataset2d],
        scan_direction_degrees: NDArray,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )

        self._images = images
        self.scan_direction_degrees = scan_direction_degrees

        # BSBL: detect best available device once at construction —
        # all GPU methods (align_affine, align_nonrigid) use self._device.
        # float32 for portability (works on CUDA/MPS/CPU) and halves VRAM.
        device, _ = validate_device(None)
        self._device = device
        self._dtype = torch.float32

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Sequence[float] | NDArray,
        file_type: str | None = None,
    ) -> "DriftCorrection":
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(
            image_list,
            scan_direction_degrees,
        )

    @classmethod
    def from_data(
        cls,
        images: list[Dataset2d] | list[NDArray] | Dataset3d | NDArray,
        scan_direction_degrees: list[float] | NDArray,
    ) -> "DriftCorrection":
        validated_images = validate_list_of_dataset2d(images)

        return cls(
            images=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

    # --- Properties ---

    @property
    def device(self) -> str:
        # BSBL: lazy init for backward compat — deserialization bypasses __init__
        if not hasattr(self, "_device"):
            self._device, _ = validate_device(None)
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        if not hasattr(self, "_dtype"):
            self._dtype = torch.float32
        return self._dtype

    @property
    def images(self) -> list[Dataset2d]:
        return self._images

    @images.setter
    def images(self, value: list[Dataset2d] | list[NDArray] | Dataset3d | NDArray):
        self._images = validate_list_of_dataset2d(value)
        self.pad_value = self.pad_value
        # Invalidate GPU cache — must call preprocess() again to rebuild
        self._images_gpu = None
        self._scan_fast_gpu = None

    @property
    def pad_value(self) -> list[float]:
        return self._pad_value

    @pad_value.setter
    def pad_value(self, value: float | str | list[float]):
        self._pad_value = validate_pad_value(value, self.images)

    @property
    def scan_direction_degrees(self) -> NDArray:
        return self._scan_direction_degrees

    @scan_direction_degrees.setter
    def scan_direction_degrees(self, value: list[float] | NDArray):
        self._scan_direction_degrees = ensure_valid_array(value, ndim=1)

    @property
    def pad_fraction(self) -> float:
        return self._pad_fraction

    @pad_fraction.setter
    def pad_fraction(self, value: float):
        self._pad_fraction = float(value)

    @property
    def kde_sigma(self) -> float:
        return self._kde_sigma

    @kde_sigma.setter
    def kde_sigma(self, value: float):
        self._kde_sigma = float(value)

    @property
    def number_knots(self) -> int:
        return self._number_knots

    @number_knots.setter
    def number_knots(self, value: float):
        self._number_knots = int(value)

    def preprocess(
        self,
        pad_fraction: float = 0.25,
        pad_value: float | str | list[float] = "median",
        kde_sigma: float = 0.5,
        number_knots: int = 1,
        show_merged: bool = False,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        # Validators
        validated_pad_value = validate_pad_value(pad_value, self._images)

        # Input data
        self.pad_fraction = pad_fraction
        self._pad_value = validated_pad_value
        self.kde_sigma = kde_sigma
        self.number_knots = number_knots

        # Derived data
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [
                np.sin(-self.scan_direction),
                np.cos(-self.scan_direction),
            ],
            axis=1,
        )
        self.scan_slow = np.stack(
            [
                np.cos(-self.scan_direction),
                -np.sin(-self.scan_direction),
            ],
            axis=1,
        )
        self.shape = (
            len(self.images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )

        # Initialize Bezier knots and scan vectors for scanlines
        self.knots = []
        for a0 in range(self.shape[0]):
            shape = self.images[a0].shape

            v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
            u_fast = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots)

            xa = (
                (self.shape[1] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 0]
                + v_slow[:, None] * self.scan_slow[a0, 0]
            )
            ya = (
                (self.shape[2] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 1]
                + v_slow[:, None] * self.scan_slow[a0, 1]
            )

            self.knots.append(np.stack([xa, ya], axis=0))

        # Precompute the interpolator for all images
        self.interpolator = []
        for a0 in range(self.shape[0]):
            self.interpolator.append(
                DriftInterpolator(
                    input_shape=self.images[a0].shape,
                    output_shape=self.shape[1:],
                    scan_fast=self.scan_fast[a0],
                    scan_slow=self.scan_slow[a0],
                    pad_value=self.pad_value[a0],
                    kde_sigma=self.kde_sigma,
                )
            )

        # Cache source images and scan vectors on GPU — avoids repeated CPU→GPU
        # transfers in align_affine (saves ~90ms per pair at 2048x2048).
        # These don't change after preprocess, so they're created once and reused
        # by _affine_grid_search_batch and _warp_and_translate_torch.
        device = self.device
        dtype = self.dtype
        self._images_gpu = [
            torch.tensor(self.images[i].array, dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]
        self._scan_fast_gpu = [
            torch.tensor(self.scan_fast[i], dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]

        # Generate initial resampled images on GPU
        self.images_warped = Dataset3d.from_shape(self.shape)
        self.weights_warped = Dataset3d.from_shape(self.shape)
        canvas_shape = (self.shape[1], self.shape[2])
        num_images = self.shape[0]
        warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
        for ind in range(num_images):
            knots_t = torch.tensor(self.knots[ind], dtype=dtype, device=device)
            row_t, col_t = transform_coordinates_torch(
                knots_t, self._scan_fast_gpu[ind], self.images[ind].shape)
            w, wt = bilinear_kde_batch_torch(
                row_t[None], col_t[None], self._images_gpu[ind], canvas_shape,
                self.kde_sigma, self.pad_value[ind])
            warped_t[ind] = w[0]
            self.images_warped.array[ind] = w[0].cpu().numpy()
            self.weights_warped.array[ind] = wt[0].cpu().numpy()

        # Error tracking on GPU tensor (avoids re-reading from CPU)
        self.calculate_error(0, _warped_t=warped_t)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: initial", **kwargs)
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: initial" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # Translation alignment
    def align_translation(
        self,
        upsample_factor: int = 8,
        min_image_shift: float | None = None,
        max_image_shift: float = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Solve for the translation between all images in DriftCorrection.images_warped
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        # init
        dxy = np.zeros((self.shape[0], 2))

        # loop over images
        F_ref = np.fft.fft2(self.images_warped.array[0])
        for ind in range(1, self.shape[0]):
            shifts, image_shift = cross_correlation_shift(
                F_ref,
                np.fft.fft2(self.images_warped.array[ind]),
                upsample_factor=upsample_factor,
                max_shift=max_image_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )

            dxy[ind, :] = shifts
            F_ref = F_ref * ind / (ind + 1) + image_shift / (ind + 1)

        # Normalize dxy
        dxy -= np.mean(dxy, axis=0)

        # Minimum image shift
        if min_image_shift is not None:
            if np.linalg.norm(dxy[ind]) < min_image_shift:
                dxy[ind] = 0.0

        # Apply shifts to knots
        for ind in range(self.shape[0]):
            self.knots[ind][0] += dxy[ind, 0]
            self.knots[ind][1] += dxy[ind, 1]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: translation", **kwargs)
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: translation" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # Affine alignment
    def align_affine(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """Correct affine drift between scan pairs using a GPU grid search.

        Builds a grid of candidate linear-drift vectors, warps both images
        for each candidate, and picks the one with the lowest cross-correlation
        cost. An optional refinement pass subdivides the winning cell for
        sub-step accuracy. Translation alignment is applied afterwards.

        Parameters
        ----------
        step : float
            Search resolution in pixels per scan line. The grid search
            tests drift rates from ``-step * num_tests/2`` to
            ``+step * num_tests/2`` px/line. For example, ``step=0.02``
            with ``num_tests=11`` searches drifts from -0.10 to +0.10
            px/line. Smaller values detect subtler drift but test more
            candidates.
        num_tests : int
            Number of drift rates to test along each axis. Must be odd
            so the grid is centered on zero drift. Total candidates
            ≈ ``π/4 * num_tests²``: ``num_tests=5`` → 21,
            ``num_tests=9`` → 61, ``num_tests=11`` → 97.
        refine : bool
            If True, run a second pass at ``step / (num_tests - 1)``
            resolution, centered on the coarse winner.
        upsample_factor : int
            Sub-pixel precision for measuring the translational shift
            between warped image pairs. 8 means 1/8-pixel precision.
            Higher values are more accurate but slower.
        max_image_shift : float or None
            Maximum allowed translational shift in pixels. Cross-correlation
            peaks beyond this radius are masked to reject spurious matches
            from noise or periodic artifacts. Set to None to allow any shift.
        show_merged : bool
            Display the merged (averaged) image after alignment.
        show_images : bool
            Display each individual warped image after alignment.
        show_knots : bool
            Overlay knot positions on the displayed images.
        **kwargs
            Additional keyword arguments passed to the plotting functions.

        Returns
        -------
        DriftCorrection
            Self, for method chaining.

        Examples
        --------
        >>> drift = DriftCorrection.from_data(
        ...     images=[im0, im1], scan_direction_degrees=[0, 90])
        >>> drift.preprocess().align_affine(step=0.02, num_tests=11)
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if self.shape[0] < 2:
            raise ValueError(
                f"align_affine requires at least 2 images (got {self.shape[0]}). "
                f"Affine drift correction cross-correlates image pairs with "
                f"different scan directions to estimate the drift vector."
            )

        if num_tests % 2 == 0:
            raise ValueError(
                f"You passed num_tests={num_tests}, but num_tests must be odd "
                f"so the grid search is centered on zero drift. "
                f"Try num_tests={num_tests + 1}."
            )

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        row_grid, col_grid = np.meshgrid(vec, vec, indexing="ij")
        # BSBL: circular mask eliminates corner candidates that are farther
        # from zero drift than any axis-aligned candidate — reduces search
        # space by ~21% without missing useful directions
        keep = row_grid**2 + col_grid**2 <= (num_tests / 2) ** 2
        drift_rc = (
            np.vstack(
                (
                    row_grid[keep],
                    col_grid[keep],
                )
            ).T
            * step
        )

        # Coarse grid search
        ind = self._affine_grid_search_batch(drift_rc, upsample_factor, max_image_shift)
        drift_total = drift_rc[ind].copy()
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += drift_rc[ind, 0] * u[:, None]
            self.knots[a0][1] += drift_rc[ind, 1] * u[:, None]

        warped_t = self._warp_and_translate_torch(max_image_shift, upsample_factor)
        self.calculate_error(1, _warped_t=warped_t)

        # Refinement pass
        if refine:
            drift_rc /= num_tests - 1
            ind = self._affine_grid_search_batch(drift_rc, upsample_factor, max_image_shift)
            drift_total += drift_rc[ind]
            for a0 in range(self.shape[0]):
                u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
                self.knots[a0][0] += drift_rc[ind, 0] * u[:, None]
                self.knots[a0][1] += drift_rc[ind, 1] * u[:, None]

        warped_t = self._warp_and_translate_torch(max_image_shift, upsample_factor)
        self.calculate_error(1, _warped_t=warped_t)

        # Summary — helps users assess drift severity on the microscope
        num_rows = self.images[0].shape[0]
        drift_rate = np.sqrt(drift_total[0] ** 2 + drift_total[1] ** 2)
        total_shift = drift_rate * num_rows
        drift_angle_deg = np.degrees(np.arctan2(drift_total[1], drift_total[0]))

        # Cardinal direction from drift vector (row=down, col=right in image space)
        row_dir = "down" if drift_total[0] > 0 else "up"
        col_dir = "right" if drift_total[1] > 0 else "left"
        abs_row = abs(drift_total[0])
        abs_col = abs(drift_total[1])
        if abs_row < 0.001 and abs_col < 0.001:
            direction = "no direction"
        elif abs_row > abs_col * 3:
            direction = row_dir
        elif abs_col > abs_row * 3:
            direction = col_dir
        else:
            direction = f"{row_dir}-{col_dir}"

        msg = (
            f"Estimated affine drift: {direction} "
            f"({drift_rate:.4f} px/line, "
            f"{total_shift:.1f} px total over {num_rows} lines)"
        )
        if self.images[0].sampling is not None:
            px_size = self.images[0].sampling[0]
            unit = self.images[0].units[0] if self.images[0].units else "px"
            msg += f" = {total_shift * px_size:.2f} {unit}"
        print(msg)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    def _affine_grid_search_batch(self, drift_rc, upsample_factor, max_image_shift):
        """Evaluate all candidate drift vectors in parallel on GPU.

        Parameters
        ----------
        drift_rc : ndarray, shape (N, 2)
            Candidate drift vectors to test, columns are (row, col).
        upsample_factor : int
            Subpixel cross-correlation upsampling factor.
        max_image_shift : float or None
            Maximum allowed shift for cross-correlation peak search.

        Returns
        -------
        int
            Index of the best candidate in drift_rc.
        """
        device = self.device
        dtype = self.dtype
        N = drift_rc.shape[0]
        drift_rc_t = torch.tensor(drift_rc, dtype=dtype, device=device)
        canvas_shape = (self.shape[1], self.shape[2])

        # Precompute base coordinates and image tensors on GPU once,
        # since they are shared across all candidate chunks
        # BSBL: only precompute images 0 and 1 — the grid search scores
        # candidates by cross-correlating this pair, other images unused
        base_data = []
        for img_idx in range(min(2, self.shape[0])):
            knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
            row_base, col_base = transform_coordinates_torch(
                knots_t, self._scan_fast_gpu[img_idx], self.images[img_idx].shape)
            num_rows = self.knots[img_idx].shape[1]
            u = (torch.arange(num_rows, dtype=dtype, device=device)
                 - (num_rows - 1) / 2)
            base_data.append((self._images_gpu[img_idx], row_base, col_base, u))

        # Precompute max_shift mask and frequency grids once — these are the
        # same for every candidate and every chunk, so computing them here
        # avoids redundant fftfreq calls inside cross_corr_batch_torch.
        shift_mask = None
        if max_image_shift is not None:
            canvas_rows, canvas_cols = canvas_shape
            freq_row = fftfreq(
                canvas_rows, 1.0 / canvas_rows, device=device, dtype=dtype)
            freq_col = fftfreq(
                canvas_cols, 1.0 / canvas_cols, device=device, dtype=dtype)
            shift_mask = freq_row[:, None] ** 2 + freq_col[None, :] ** 2 >= max_image_shift ** 2

        freq_grids = (
            fftfreq(canvas_shape[0], device=device, dtype=dtype)[:, None],
            fftfreq(canvas_shape[1], device=device, dtype=dtype)[None, :],
        )

        # Chunk candidates to fit in GPU memory. Each candidate needs ~6 canvas-sized
        # float32 tensors (coordinates, scatter buffers, warped image). On CUDA we
        # query free VRAM directly; on CPU/MPS we use a conservative default since
        # there's no portable way to query available memory without psutil.
        elem_bytes = 4 if dtype == torch.float32 else 8
        bytes_per_candidate = 6 * elem_bytes * canvas_shape[0] * canvas_shape[1]
        if "cuda" in device:
            free_mem = torch.cuda.mem_get_info(device)[0]
            chunk_size = max(1, min(N, int(0.7 * free_mem / bytes_per_candidate)))
        else:
            chunk_size = min(N, 8)

        num_chunks = (N + chunk_size - 1) // chunk_size
        if num_chunks > 1:
            warnings.warn(
                f"align_affine is testing {N} drift angles to find the best "
                f"correction, but your GPU can only fit {chunk_size} at a time "
                f"({num_chunks} passes needed instead of 1). "
                f"This is ~{num_chunks}x slower than single-pass. "
                f"To speed up, close other GPU applications or use a GPU with "
                f"more memory (need ~{N * bytes_per_candidate / 1e9:.1f} GB free).",
                stacklevel=3,
            )

        all_costs = []
        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            drift_chunk = drift_rc_t[chunk_start:chunk_end]

            warped_stack = []
            for img_idx in range(min(2, self.shape[0])):
                image_t, row_base, col_base, u = base_data[img_idx]
                row_all = row_base[None] + drift_chunk[:, 0, None, None] * u[None, :, None]
                col_all = col_base[None] + drift_chunk[:, 1, None, None] * u[None, :, None]
                warped, _ = bilinear_kde_batch_torch(
                    row_all, col_all, image_t,
                    canvas_shape, self.kde_sigma,
                    self.pad_value[img_idx])
                warped_stack.append(warped)

            all_costs.append(cross_corr_batch_torch(
                warped_stack[0], warped_stack[1],
                upsample_factor, max_image_shift,
                max_shift_mask=shift_mask,
                freq_grids=freq_grids))

        all_costs = torch.cat(all_costs)
        return torch.argmin(all_costs).item()

    def _warp_and_translate_torch(self, max_image_shift, upsample_factor=8):
        """Regenerate warped images and solve translation on GPU.

        Replaces the NumPy warp_image + align_translation combo with torch
        equivalents for use inside align_affine. Returns the GPU tensor
        so callers can pass it to calculate_error without a round-trip.
        """
        device = self.device
        dtype = self.dtype
        num_images = self.shape[0]
        canvas_shape = (self.shape[1], self.shape[2])

        # First warp pass — for translation cross-correlation
        warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
        weights_t = torch.zeros_like(warped_t)
        for img_idx in range(num_images):
            knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
            row_t, col_t = transform_coordinates_torch(
                knots_t, self._scan_fast_gpu[img_idx], self.images[img_idx].shape)
            w, wt = bilinear_kde_batch_torch(
                row_t[None], col_t[None], self._images_gpu[img_idx], canvas_shape,
                self.kde_sigma, self.pad_value[img_idx])
            warped_t[img_idx] = w[0]
            weights_t[img_idx] = wt[0]

        # Translation alignment — fully on GPU, no .item() calls.
        # Pairwise cross-correlation with running Fourier-domain average.
        num_rows, num_cols = canvas_shape
        shifts_t = torch.zeros(num_images, 2, dtype=dtype, device=device)
        F_ref = fft2(warped_t[0])

        # Precompute max_shift mask and frequency grids once
        shift_mask = None
        if max_image_shift is not None:
            fr = fftfreq(num_rows, 1.0 / num_rows, device=device, dtype=dtype)
            fc = fftfreq(num_cols, 1.0 / num_cols, device=device, dtype=dtype)
            shift_mask = fr[:, None] ** 2 + fc[None, :] ** 2 >= max_image_shift ** 2
        kr = fftfreq(num_rows, device=device, dtype=dtype)[:, None]
        kc = fftfreq(num_cols, device=device, dtype=dtype)[None, :]

        for img_idx in range(1, num_images):
            F_im = fft2(warped_t[img_idx])
            cc = F_ref * F_im.conj()
            cc_real = ifft2(cc).real

            if shift_mask is not None:
                cc_real[shift_mask] = 0.0

            # Coarse peak — stay on GPU tensors
            peak_flat = cc_real.reshape(-1).argmax()
            r0 = peak_flat // num_cols
            c0 = peak_flat % num_cols

            # Parabolic refinement — all tensor ops, no .item()
            vr = cc_real[torch.stack([(r0 - 1) % num_rows, r0, (r0 + 1) % num_rows]), c0]
            vc = cc_real[r0, torch.stack([(c0 - 1) % num_cols, c0, (c0 + 1) % num_cols])]
            dr_denom = 4 * vr[1] - 2 * vr[2] - 2 * vr[0]
            dc_denom = 4 * vc[1] - 2 * vc[2] - 2 * vc[0]
            dr = torch.where(dr_denom != 0, (vr[2] - vr[0]) / dr_denom, torch.zeros(1, device=device, dtype=dtype))
            dc = torch.where(dc_denom != 0, (vc[2] - vc[0]) / dc_denom, torch.zeros(1, device=device, dtype=dtype))
            r0_para = (r0.to(dtype) + dr) % num_rows
            c0_para = (c0.to(dtype) + dc) % num_cols

            # DFT upsample — pass tensor shifts directly, no .item() sync
            local = dft_upsample_torch(cc, upsample_factor, (r0_para, c0_para))

            # Upsampled peak — back on GPU
            pk_flat = local.reshape(-1).argmax()
            lr = pk_flat // local.shape[1]
            lc = pk_flat % local.shape[1]
            patch_size = local.shape[0]

            # Parabolic on upsampled patch — clamp to avoid OOB
            can_refine = (lr >= 1) & (lr < patch_size - 1) & (lc >= 1) & (lc < patch_size - 1)
            lr_safe = lr.clamp(1, patch_size - 2)
            lc_safe = lc.clamp(1, patch_size - 2)

            vr_up = local[torch.stack([lr_safe - 1, lr_safe, lr_safe + 1]), lc_safe]
            vc_up = local[lr_safe, torch.stack([lc_safe - 1, lc_safe, lc_safe + 1])]
            dr2 = 4 * vr_up[1] - 2 * vr_up[2] - 2 * vr_up[0]
            dc2 = 4 * vc_up[1] - 2 * vc_up[2] - 2 * vc_up[0]
            d_row_fine = torch.where(can_refine & (dr2 != 0), (vr_up[2] - vr_up[0]) / dr2, torch.zeros(1, device=device, dtype=dtype))
            d_col_fine = torch.where(can_refine & (dc2 != 0), (vc_up[2] - vc_up[0]) / dc2, torch.zeros(1, device=device, dtype=dtype))

            # Convert upsampled-patch coords to image-pixel shifts
            shift_r = r0_para + (lr.to(dtype) - upsample_factor) / upsample_factor + d_row_fine / upsample_factor
            shift_c = c0_para + (lc.to(dtype) - upsample_factor) / upsample_factor + d_col_fine / upsample_factor
            shift_r = ((shift_r + num_rows / 2) % num_rows) - num_rows / 2
            shift_c = ((shift_c + num_cols / 2) % num_cols) - num_cols / 2
            shifts_t[img_idx, 0] = shift_r
            shifts_t[img_idx, 1] = shift_c

            # Fourier shift and running average
            phase = torch.exp(-2j * torch.pi * (kr * shift_r + kc * shift_c))
            F_shifted = F_im * phase
            F_ref = F_ref * img_idx / (img_idx + 1) + F_shifted / (img_idx + 1)

        # Zero-mean normalization — single CPU sync at the end
        shifts_t -= shifts_t.mean(dim=0)
        shifts_np = shifts_t.cpu().numpy()

        for img_idx in range(num_images):
            self.knots[img_idx][0] += shifts_np[img_idx, 0]
            self.knots[img_idx][1] += shifts_np[img_idx, 1]

        # Second warp pass with corrected knots
        for img_idx in range(num_images):
            knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
            row_t, col_t = transform_coordinates_torch(
                knots_t, self._scan_fast_gpu[img_idx], self.images[img_idx].shape)
            w, wt = bilinear_kde_batch_torch(
                row_t[None], col_t[None], self._images_gpu[img_idx], canvas_shape,
                self.kde_sigma, self.pad_value[img_idx])
            warped_t[img_idx] = w[0]
            weights_t[img_idx] = wt[0]

        # Single CPU sync at the end (for plots, serialization, downstream methods)
        self.images_warped.array[:] = warped_t.cpu().numpy()
        self.weights_warped.array[:] = weights_t.cpu().numpy()

        return warped_t

    # non-rigid alignment
    def align_nonrigid(
        self,
        backend: str = "pytorch",
        # Shared parameters
        num_iterations: int = 8,
        regularization_sigma_px: float = 16.0,
        regularization_update_step_size: float | None = 0.8,
        min_image_shift: float | None = None,
        max_image_shift: float | None = 32.0,
        # PyTorch parameters
        adam_steps: int = 50,
        lr: float = 0.02,
        # SciPy parameters
        max_optimize_iterations: int = 10,
        regularization_poly_order: int = 1,
        regularization_max_image_shift_px: float | None = None,
        solve_individual_rows: bool = True,
        # Display parameters
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Non-rigid drift correction using PyTorch (default) or SciPy backend.

        Parameters
        ----------
        backend : str, default "pytorch"
            Optimization backend. "pytorch" uses GPU-accelerated Adam optimizer
            with batched optimization (faster). "scipy" uses L-BFGS row-by-row.

        Shared Parameters
        -----------------
        num_iterations : int, default 8
            Number of outer iterations for alternating optimization.
        regularization_sigma_px : float, default 16.0
            Gaussian smoothing sigma for knot regularization.
        regularization_update_step_size : float, default 0.8
            Step size for knot updates (0-1, lower = more conservative).
        min_image_shift : float, optional
            Minimum shift for translation alignment between iterations.
        max_image_shift : float, default 32.0
            Maximum shift for translation alignment between iterations.

        PyTorch Parameters (ignored if backend="scipy")
        -----------------------------------------------
        adam_steps : int, default 5
            Number of Adam optimization steps per image.
        lr : float, default 0.02
            Learning rate for Adam optimizer.

        SciPy Parameters (ignored if backend="pytorch")
        -----------------------------------------------
        max_optimize_iterations : int, default 10
            Maximum L-BFGS iterations per row.
        regularization_poly_order : int, default 1
            Polynomial order for trend removal in regularization.
        regularization_max_image_shift_px : float, optional
            Maximum allowed shift per iteration.
        solve_individual_rows : bool, default True
            If True, optimize each row independently.

        Display Parameters
        ------------------
        show_merged : bool, default True
            Show merged image after alignment.
        show_images : bool, default False
            Show individual aligned images.
        show_knots : bool, default True
            Overlay knot positions on visualizations.
        """
        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()
        # Main optimization loop
        for _ in tqdm(range(num_iterations), desc=f"Solving nonrigid drift ({backend})"):
            for ind in range(self.shape[0]):
                image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(axis=0)
                knots_init = self.knots[ind]
                # Optimize knots
                if backend == "pytorch":
                    knots_updated = self._optimize_knots_pytorch(
                        ind, image_ref, knots_init, adam_steps=adam_steps, lr=lr)
                else:
                    knots_updated = self._optimize_knots_scipy(
                        ind, image_ref, knots_init,
                        max_optimize_iterations=max_optimize_iterations,
                        solve_individual_rows=solve_individual_rows)
                # Max shift regularization
                if regularization_max_image_shift_px is not None:
                    knots_shift = knots_updated - self.knots[ind]
                    knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                    sub = knots_dist > regularization_max_image_shift_px
                    knots_updated[0][sub] = (self.knots[ind][0][sub]
                        + knots_shift[0][sub] * regularization_max_image_shift_px / knots_dist[sub])
                    knots_updated[1][sub] = (self.knots[ind][1][sub]
                        + knots_shift[1][sub] * regularization_max_image_shift_px / knots_dist[sub])
                # Smoothness regularization
                if regularization_sigma_px is not None and regularization_sigma_px > 0:
                    knots_smoothed = knots_updated.copy()
                    for dim in range(knots_updated.shape[0]):
                        x = np.arange(knots_updated.shape[1])
                        for knot_ind in range(knots_updated.shape[2]):
                            y = knots_updated[dim, :, knot_ind]
                            coefs = np.polyfit(x, y, deg=regularization_poly_order)
                            trend = np.polyval(coefs, x)
                            residual = y - trend
                            residual_smooth = gaussian_filter(residual, sigma=regularization_sigma_px)
                            knots_smoothed[dim, :, knot_ind] = residual_smooth + trend
                    knots_updated = knots_smoothed
                # Step size
                if regularization_update_step_size is not None:
                    knots_updated = (self.knots[ind]
                        + (knots_updated - self.knots[ind]) * regularization_update_step_size)
                self.knots[ind] = knots_updated
            # Update warped images
            for ind in range(self.shape[0]):
                self.images_warped.array[ind], self.weights_warped.array[ind] = (
                    self.interpolator[ind].warp_image(self.images[ind].array, self.knots[ind]))
            # Translation alignment
            self.align_translation(
                min_image_shift=min_image_shift, max_image_shift=max_image_shift,
                show_images=False, show_merged=False, show_knots=False)
            self.calculate_error(2)

        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: non-rigid",
                **kwargs,
            )

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    def _optimize_knots_pytorch(
        self, idx: int, image_ref: np.ndarray, knots_init: np.ndarray,
        adam_steps: int = 5, lr: float = 0.02,
    ) -> np.ndarray:
        """PyTorch Adam batched optimization for one image (single knot only)."""
        # TODO: support multiple knots (requires differentiable spline interpolation)
        if knots_init.shape[2] != 1:
            raise NotImplementedError(
                f"PyTorch backend only supports single knot (got {knots_init.shape[2]}). "
                "Use backend='scipy' for multiple knots.")
        device = self.device
        num_rows, num_cols = self.images[idx].array.shape
        # Convert to tensors
        ref_image = torch.tensor(image_ref, dtype=torch.float32, device=device)
        target_image = torch.tensor(self.images[idx].array, dtype=torch.float32, device=device)
        row_position = torch.tensor(self.interpolator[idx].u, dtype=torch.float32, device=device)
        scan_fast = self.interpolator[idx].scan_fast
        scale_row = scan_fast[0] * (num_rows - 1)
        scale_col = scan_fast[1] * (num_cols - 1)
        # Initialize knots as trainable tensor: shape (2, num_rows)
        knots = torch.tensor(knots_init[:, :, 0], dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([knots], lr=lr)
        # Adam optimization (batched over all rows)
        for _ in range(adam_steps):
            optimizer.zero_grad()
            # Transform: single knot = shift along scan direction
            row_coords = knots[0, :, None] + row_position[None, :] * scale_row
            col_coords = knots[1, :, None] + row_position[None, :] * scale_col
            # Bilinear interpolation (boundary clamp critical for lower RMSE than scipy's L-BFGS)
            row_c = row_coords.clamp(0, num_rows - 1.001)
            col_c = col_coords.clamp(0, num_cols - 1.001)
            row_f = row_c.floor().long().clamp(0, num_rows - 2)
            col_f = col_c.floor().long().clamp(0, num_cols - 2)
            d_row, d_col = row_c - row_f.float(), col_c - col_f.float()
            warped = (ref_image[row_f, col_f] * (1 - d_row) * (1 - d_col)
                      + ref_image[row_f + 1, col_f] * d_row * (1 - d_col)
                      + ref_image[row_f, col_f + 1] * (1 - d_row) * d_col
                      + ref_image[row_f + 1, col_f + 1] * d_row * d_col)
            loss = ((warped - target_image) ** 2).mean()
            loss.backward()
            optimizer.step()
        return knots.detach().cpu().numpy()[:, :, None]

    def _optimize_knots_scipy(
        self, idx: int, image_ref: np.ndarray, knots_init: np.ndarray,
        max_optimize_iterations: int = 10, solve_individual_rows: bool = True,
    ) -> np.ndarray:
        """SciPy L-BFGS optimization for one image."""
        shape_knots = knots_init.shape
        options = {"maxiter": max_optimize_iterations} if max_optimize_iterations else {}
        if solve_individual_rows:
            knots_updated = np.zeros_like(knots_init)
            for row_ind in range(knots_init.shape[1]):
                x0 = knots_init[:, row_ind, :].ravel()
                def cost_function(x):
                    knots_row = x.reshape(shape_knots[0], shape_knots[2])
                    xa, ya = self.interpolator[idx].transform_rows(knots_row)
                    xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                    yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                    dx, dy = xa - xf, ya - yf
                    warped = (image_ref[xf, yf] * (1 - dx) * (1 - dy)
                              + image_ref[xf + 1, yf] * dx * (1 - dy)
                              + image_ref[xf, yf + 1] * (1 - dx) * dy
                              + image_ref[xf + 1, yf + 1] * dx * dy)
                    return np.sum((warped - self.images[idx].array[row_ind, :]) ** 2)
                result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
                knots_updated[:, row_ind, :] = result.x.reshape((2, -1))
        else:
            x0 = knots_init.ravel()
            def cost_function(x):
                knots = x.reshape(shape_knots)
                xa, ya = self.interpolator[idx].transform_coordinates(knots)
                xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                dx, dy = xa - xf, ya - yf
                warped = (image_ref[xf, yf] * (1 - dx) * (1 - dy)
                          + image_ref[xf + 1, yf] * dx * (1 - dy)
                          + image_ref[xf, yf + 1] * (1 - dx) * dy
                          + image_ref[xf + 1, yf + 1] * dx * dy)
                return np.sum((warped - self.images[idx].array) ** 2)
            result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
            knots_updated = result.x.reshape(shape_knots)
        return knots_updated
    def generate_corrected_image(
        self,
        upsample_factor: int = 2,
        output_original_shape: bool = True,
        mask_output: bool = True,
        mask_edge_blend: float = 8.0,
        fourier_filter: bool = True,
        filter_midpoint: float = 0.5,
        kde_sigma: float = 0.5,
        weight_thresh=0.1,
        show_image: bool = True,
        **kwargs,
    ):
        """
        Generate the final drift-corrected image after aligning a stack of input images.

        Parameters
        ----------
        upsample_factor : int, default 2
            Factor to upsample the output image for enhanced interpolation accuracy.
        output_original_shape : bool, default True
            If True, crop the output image back to the original input dimensions after processing.
        mask_output : bool, default True
            If true, mask the output using the probe position weights
        mask_edge_blend : float, default 8.0
            Value in pixels to blend from the edge of the mask (where we have data)
        fourier_filter : bool, default True
            Whether to apply Fourier-based directional filtering to merge corrected images.
        filter_midpoint : float, default 0.5
            Midpoint for the sigmoid-based Fourier weighting filter, determining transition smoothness.
            Setting this to a low value close to 0 will include more signal but also more slow scan artifacts.
            If using 2 images at 0 and 90 degrees scan angles, any value >0.75 will be unstable.
            Only use larger values (close to 1.0) if multiple images covering many scan angles are used.
        kde_sigma : float, default 0.5
            Standard deviation for kernel density estimation used during image interpolation. Defaults
            to the object's stored kde_sigma if set to None.
        weight_thresh: float, default 0.1
            This value sets the threshold for masking the outputs.
            For very large jitter artifacts this value can be lowered.
        show_image : bool, default True
            Whether to display the final corrected image after processing.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function when displaying the image.

        Returns
        -------
        image_corr : Dataset2d
            The final drift-corrected output image encapsulated in a Dataset2d object.

        Notes
        -----
        - The function applies per-frame warping using knot-based interpolation and optionally
          performs directional Fourier filtering to blend multiple warped images.
        - The Fourier filter suppresses directional artifacts by weighting image contributions based
          on their scan angles, utilizing a bounded sine sigmoid for smooth transition.
        - Upsampling enhances interpolation precision but may increase computational cost.
        """

        # init
        stack_corr = np.zeros(
            (
                self.shape[0],
                np.round(self.shape[1] * upsample_factor).astype("int"),
                np.round(self.shape[2] * upsample_factor).astype("int"),
            )
        )
        weight_corr = np.zeros(
            (
                self.shape[0],
                np.round(self.shape[1] * upsample_factor).astype("int"),
                np.round(self.shape[2] * upsample_factor).astype("int"),
            )
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        # Update images
        for ind in range(self.shape[0]):
            stack_corr[ind], weight_corr[ind] = self.interpolator[ind].warp_image(
                self.images[ind].array,
                self.knots[ind],
                kde_sigma=kde_sigma,
                upsample_factor=upsample_factor,
            )

        if fourier_filter:
            # Apply fourier filtering
            kx = np.fft.fftfreq(stack_corr.shape[1])[:, None]
            ky = np.fft.fftfreq(stack_corr.shape[2])[None, :]
            kt = np.arctan2(ky, kx)

            stack_fft = np.fft.fft2(stack_corr)
            weights = np.zeros_like(stack_corr)

            for ind in range(stack_corr.shape[0]):
                # Calculate weights as a function of angle
                weights[ind] = np.abs(
                    np.mod((kt - self.scan_direction[ind]) / np.pi + 0.5, 1.0) - 0.5
                ) / (1 / 2)
                weights[ind][0, 0] = 1.0

                # Apply sigmoid to weighting function
                weights[ind] = bounded_sine_sigmoid(
                    weights[ind],
                    midpoint=filter_midpoint,
                )

                # Weight the fourier transformed images
                stack_fft[ind] *= weights[ind]

            weights_sum = np.sum(weights, axis=0)
            image_corr_fft = np.divide(
                np.sum(stack_fft, axis=0),
                weights_sum,
                where=weights_sum > 0.0,
            )

        else:
            image_corr_fft = np.fft.fft2(np.mean(stack_corr, axis=0))

        if mask_output:
            # Note that we compute 2 boolean masks to round off the corners of image blending

            # calculate mask from product of individual image masks
            # scale weights by upsample factor to normalize to mean value of 1.0
            mask_edge = np.prod(weight_corr >= (weight_thresh / upsample_factor**2), axis=0)

            # Set outermost pixels to False to define the boundary for edge blending
            mask_edge[:, 0] = False
            mask_edge[:, -1] = False
            mask_edge[0, :] = False
            mask_edge[-1, :] = False

            # Find inner boundary mask
            mask_inner = distance_transform_edt(mask_edge) <= mask_edge_blend

            # compute mask using edge blending value
            mask = (
                np.cos(
                    (np.pi / 2)
                    * np.clip(distance_transform_edt(mask_inner) / mask_edge_blend, 0.0, 1.0)
                )
                ** 2
            )

            # Mean pad value
            pad_value_mean = np.mean([ind.pad_value for ind in self.interpolator])

            # apply mask
            image_corr_fft = np.fft.fft2(
                np.fft.ifft2(image_corr_fft) * mask + pad_value_mean * (1 - mask)
            )

        if output_original_shape:
            image_corr_fft = fourier_cropping(image_corr_fft, self.shape[-2:]) / upsample_factor**2

        # TODO - adjust origin / sampling if output sampling is different from input
        # i.e. if output_original_shape is False, and upsample_factor > 1
        image_corr = Dataset2d.from_array(
            np.real(np.fft.ifft2(image_corr_fft)),
            name="drift corrected image",
            origin=self.images[0].origin,
            sampling=self.images[0].sampling,
            units=self.images[0].units,
        )

        if show_image:
            fig, ax = show_2d(image_corr.array, **kwargs)

            # Force a render whether we're drawing into a provided Axes or a fresh Figure
            ax_to_draw = kwargs.get("ax", ax)
            try:
                ax_to_draw.figure.canvas.draw_idle()
                # If we're not drawing into a caller-provided Axes, also pop the window
                if "ax" not in kwargs:
                    plt.show()
            except Exception:
                # Fallback: if backend is odd, try a blocking show
                plt.show()

        # if show_image:
        #     fig, ax = image_corr.show(**kwargs)

        return image_corr

    def calculate_error(
        self,
        mode,
        _warped_t=None,
    ):
        # Estimate current error — use GPU tensor if provided, else NumPy
        if _warped_t is not None:
            images_mean = _warped_t.mean(dim=0)
            sig_diff = torch.mean(
                torch.abs(_warped_t - images_mean[None]), dim=(1, 2)
            ).cpu().numpy()
        else:
            images_mean = np.mean(self.images_warped.array, axis=0)
            sig_diff = np.mean(
                np.abs(self.images_warped.array - images_mean[None, :, :]), axis=(1, 2)
            )

        # Error vector
        error_current = np.hstack((mode, np.mean(sig_diff), sig_diff))

        # Initialize or append to error tracking array
        if not hasattr(self, "error_track"):
            self.error_track = error_current[None, :]  # initialize with first row
        else:
            self.error_track = np.vstack((self.error_track, error_current))

    def plot_transformed_images(self, show_knots: bool = True, **kwargs):
        fig, ax = show_2d(
            list(self.images_warped.array),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax[a0].plot(
                    y,
                    x,
                    color="r",
                )

    def plot_convergence(
        self,
        figsize=(8, 3),
        **kwargs,
    ):
        """
        Plot the convergence of the drift correction.
        """
        sub = np.abs(self.error_track[:, 0] - 2) < 0.1
        error = self.error_track[:, 1]
        it = np.arange(error.shape[0])

        from matplotlib.ticker import FormatStrFormatter, MaxNLocator

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        color = (1, 0, 0)  # red

        # Plot Affine
        if np.any(~sub):
            ax[0].plot(
                it[~sub],
                100 * error[~sub],
                marker="o",
                color=color,
                linestyle="-",
                label="Affine",
                **kwargs,
            )
            ax[0].set_xlabel("Affine Iterations")
            ax[0].set_ylabel("Mean Error [%]")
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        else:
            ax[0].axis("off")

        # Plot Non-Rigid
        if np.any(sub):
            first_true = np.argmax(sub)
            if first_true > 0:
                sub[first_true - 1] = True

            ax[1].plot(
                it[sub],
                100 * error[sub],
                marker="o",
                color=color,
                linestyle="-",
                label="Non-Rigid",
                **kwargs,
            )
            ax[1].set_xlabel("Non-Rigid Iterations")
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        else:
            ax[1].axis("off")

        plt.tight_layout()

        return self

    def plot_merged_images(self, show_knots: bool = True, **kwargs):
        """
        Plot the current transformed images, with knot overlays.
        """
        fig, ax = show_2d(
            self.images_warped.array.mean(0),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax.plot(
                    y,
                    x,
                )


class DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        scan_slow,
        pad_value,
        kde_sigma,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.scan_slow = scan_slow
        self.pad_value = pad_value
        self.kde_sigma = kde_sigma

        self.rows_input = np.arange(input_shape[0])
        self.cols_input = np.arange(input_shape[1])
        self.u = np.linspace(0, 1, input_shape[1])

    def transform_rows(
        self,
        knots_row: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya

    def transform_coordinates(
        self,
        knots: NDArray,
    ):
        num_knots = knots.shape[-1]

        if num_knots == 1:
            # vectorized version for speed
            xa, ya = self.transform_rows(knots)
        else:
            xa = np.zeros(self.input_shape)
            ya = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                xa[i], ya[i] = self.transform_rows(knots[:, i])

        return xa, ya

    def warp_image(
        self,
        image: NDArray,
        knots: NDArray,  # shape: (2, rows, num_knots)
        kde_sigma=None,
        output_shape=None,
        pad_value=None,
        upsample_factor=None,
    ) -> NDArray:
        xa, ya = self.transform_coordinates(
            knots,
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        if output_shape is None:
            output_shape = self.output_shape

        if pad_value is None:
            pad_value = self.pad_value

        if upsample_factor is None:
            upsample_factor = 1.0

        image_interp, weight_interp = bilinear_kde(
            xa=xa * upsample_factor,  # rows
            ya=ya * upsample_factor,  # cols
            values=image,
            output_shape=np.round(np.array(output_shape) * upsample_factor).astype("int"),
            kde_sigma=kde_sigma * upsample_factor,
            pad_value=pad_value,
            return_pix_count=True,
        )

        return image_interp, weight_interp


def bounded_sine_sigmoid(x, midpoint=0.5, width=1.0):
    """
    Piecewise bounded sigmoid: zero, raised sine squared, one.

    Parameters
    ----------
    x : array-like, shape (...,)
        Input values in [0, 1].
    midpoint : float
        Center of the sigmoid transition.
    width : float
        Width of the sigmoid (range over which it ramps from 0 to 1).
    Returns
    -------
    y : array-like
        Output in [0, 1], same shape as x.
    """
    x = np.asarray(x)
    # Truncate width if midpoint too close to edge
    left_max = midpoint - width / 2
    right_min = midpoint + width / 2
    if left_max < 0:
        warnings.warn(
            f"width={width} is too large for midpoint={midpoint}, "
            f"clamping width to {2 * midpoint}.",
            RuntimeWarning,
        )
        width = 2 * midpoint

    if right_min > 1:
        warnings.warn(
            f"width={width} is too large for midpoint={midpoint}, "
            f"clamping width to {2 * (1 - midpoint)}.",
            RuntimeWarning,
        )
        width = 2 * (1 - midpoint)
    # Recalculate edges
    left = midpoint - width / 2
    right = midpoint + width / 2

    y = np.zeros_like(x, dtype=float)
    in_band = (x >= left) & (x <= right)
    # Map [left, right] to [0, pi/2]
    t = (x[in_band] - left) / width  # goes from 0 to 1
    y[in_band] = np.sin(t * np.pi / 2) ** 2
    y[x > right] = 1.0
    return y
