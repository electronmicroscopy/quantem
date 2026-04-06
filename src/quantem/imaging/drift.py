from collections.abc import Sequence
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fft import fftfreq
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
    bilinear_kde_batch,
    cross_corr_batch,
    gaussian_smooth_1d,
    transform_coordinates,
    translate_align,
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
    - Visualization is supported through `plot_merged_images()` and `plot_transformed_images()`.

    Performance
    -----------
    ``align_affine`` uses PyTorch to run all heavy operations on GPU
    (works on CUDA, MPS, and CPU). The key optimizations are:

    - **Batched grid search**: all ~97 candidate drift vectors are warped
      and scored in parallel, instead of one-at-a-time in a Python loop.
    - **Batched bilinear KDE** (``drift_utils.bilinear_kde_batch``):
      scatter-based image warping via ``scatter_add_`` with int32 indices.
    - **Batched FFT cross-correlation** (``drift_utils.cross_corr_batch``):
      sub-pixel alignment using DFT upsampling across all candidates at once.
    - **Zero CPU round-trips**: coordinate transforms, Gaussian smoothing,
      translation alignment, and error computation all stay on GPU until
      the final sync.

    This gives ~300× speedup over the original NumPy implementation
    (e.g. 436 s → 1.5 s on 2048×2048 image pairs).

    Memory is automatically chunked when the full batch doesn't fit.
    Approximate memory per candidate at common sizes:

    ========== =========== ================
    Input size Canvas size Mem / candidate
    ========== =========== ================
    1024×1024  1280×1280     85 MB
    2048×2048  2560×2560    341 MB
    4096×4096  5120×5120   1.36 GB
    ========== =========== ================
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

        self.images = images
        self.scan_direction_degrees = ensure_valid_array(scan_direction_degrees, ndim=1)

        device, _ = validate_device(None)
        self._device = device
        self._dtype = torch.float32

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Sequence[float] | NDArray,
        file_type: str | None = None,
    ) -> Self:
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
    ) -> Self:
        validated_images = validate_list_of_dataset2d(images)

        return cls(
            images=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

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
        """Prepare images for drift correction by building the scanline model.

        Computes scan direction vectors, initializes Bezier knots that map
        each scanline onto a padded canvas, and generates the initial warped
        images. This must be called before any alignment step.

        Without preprocessing, there is no spatial model connecting the raw
        images to the shared canvas — alignment methods would have no
        coordinates to optimize.

        Parameters
        ----------
        pad_fraction : float
            Fraction of the image size to add as padding around the canvas.
            Larger values give more room for drift but use more memory.
            ``pad_fraction=0.25`` adds 25% on each side.
        pad_value : float, str, or list[float]
            Fill value for pixels outside the image footprint. Can be
            ``'median'``, ``'mean'``, ``'min'``, ``'max'``, a quantile
            (e.g. ``0.25``), or a per-image list of floats.
        kde_sigma : float
            Gaussian smoothing sigma (in pixels) applied after bilinear
            scatter. Smooths the warped images to reduce scatter noise.
        number_knots : int
            Number of Bezier knots per scanline. Use ``1`` (recommended)
            for linear drift correction. Higher values allow per-scanline
            curvature but are slower and rarely needed.
        show_merged : bool
            Display the merged (averaged) warped images after preprocessing.
        show_images : bool
            Display each individual warped image after preprocessing.
        show_knots : bool
            Overlay knot positions on displayed images.
        **kwargs
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        Self
            For method chaining: ``drift.preprocess().align_affine()``.

        Examples
        --------
        >>> drift = DriftCorrection.from_data(
        ...     images=[im0, im1], scan_direction_degrees=[0, 90])
        >>> drift.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1)
        """
        self.pad_fraction = float(pad_fraction)
        self.pad_value = validate_pad_value(pad_value, self.images)
        self.kde_sigma = float(kde_sigma)
        self.number_knots = int(number_knots)
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [np.sin(-self.scan_direction), np.cos(-self.scan_direction)], axis=1)
        self.scan_slow = np.stack(
            [np.cos(-self.scan_direction), -np.sin(-self.scan_direction)], axis=1)
        self.shape = (
            len(self.images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )
        # Initialize knots — each image's scanlines mapped to the padded canvas
        self.knots = []
        for img_idx in range(self.shape[0]):
            shape = self.images[img_idx].shape
            v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
            u_fast = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots)
            row_knots = ((self.shape[1] - 1) / 2
                         + u_fast[None, :] * self.scan_fast[img_idx, 0]
                         + v_slow[:, None] * self.scan_slow[img_idx, 0])
            col_knots = ((self.shape[2] - 1) / 2
                         + u_fast[None, :] * self.scan_fast[img_idx, 1]
                         + v_slow[:, None] * self.scan_slow[img_idx, 1])
            self.knots.append(np.stack([row_knots, col_knots], axis=0))
        self.interpolator = [
            DriftInterpolator(
                input_shape=self.images[i].shape,
                output_shape=self.shape[1:],
                scan_fast=self.scan_fast[i],
                scan_slow=self.scan_slow[i],
                pad_value=self.pad_value[i],
                kde_sigma=self.kde_sigma,
            )
            for i in range(self.shape[0])
        ]
        # Cache source data on GPU and generate initial warped images
        device = self._device
        dtype = self._dtype
        self.images_t = [
            torch.tensor(self.images[i].array, dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]
        self.scan_fast_t = [
            torch.tensor(self.scan_fast[i], dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]
        self.images_warped = Dataset3d.from_shape(self.shape)
        self.weights_warped = Dataset3d.from_shape(self.shape)
        canvas_shape = (self.shape[1], self.shape[2])
        warped_t = torch.zeros(self.shape[0], *canvas_shape, dtype=dtype, device=device)
        for img_idx in range(self.shape[0]):
            knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
            row_t, col_t = transform_coordinates(
                knots_t, self.scan_fast_t[img_idx], self.images[img_idx].shape)
            warped, weights = bilinear_kde_batch(
                row_t[None], col_t[None], self.images_t[img_idx], canvas_shape,
                self.kde_sigma, self.pad_value[img_idx])
            warped_t[img_idx] = warped[0]
            self.images_warped.array[img_idx] = warped[0].cpu().numpy()
            self.weights_warped.array[img_idx] = weights[0].cpu().numpy()
        self.calculate_error(0, _warped_t=warped_t)
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
        dxy = np.zeros((self.shape[0], 2))
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
        dxy -= np.mean(dxy, axis=0)
        if min_image_shift is not None:
            if np.linalg.norm(dxy[ind]) < min_image_shift:
                dxy[ind] = 0.0
        for ind in range(self.shape[0]):
            self.knots[ind][0] += dxy[ind, 0]
            self.knots[ind][1] += dxy[ind, 1]
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )
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
        chunk_size: int | None = None,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """Correct affine drift between scan pairs using a batched grid search.

        Builds a grid of candidate linear-drift vectors, warps both images
        for each candidate, and picks the one with the lowest cross-correlation
        cost. An optional refinement pass subdivides the winning cell for
        sub-step accuracy. Without affine correction, per-scanline drift
        causes shear distortion that translation alignment alone cannot fix.

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
        chunk_size : int or None
            Number of candidates per pass. If None, all candidates at once.
            Set to a smaller value if you run out of memory.
        show_merged : bool
            Display the merged (averaged) image after alignment.
        show_images : bool
            Display each individual warped image after alignment.
        show_knots : bool
            Overlay knot positions on the displayed images.
        verbose : bool
            If True, print the top 5 candidate drift vectors with their
            cost and direction after the grid search. Useful for
            diagnosing ambiguous alignments or verifying the winning
            candidate has a clear margin over runner-ups.
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
        if self.shape[0] < 2:
            raise ValueError(
                f"align_affine requires at least 2 images (got {self.shape[0]}). "
                f"Provide image pairs with different scan directions."
            )
        if num_tests % 2 == 0:
            raise ValueError(
                f"num_tests must be odd (got {num_tests}). Try {num_tests + 1}."
            )
        # Build candidate grid with circular mask (~21% fewer than square)
        grid_axis = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        row_grid, col_grid = np.meshgrid(grid_axis, grid_axis, indexing="ij")
        circular_mask = row_grid**2 + col_grid**2 <= (num_tests / 2) ** 2
        drift_vectors = np.vstack((row_grid[circular_mask], col_grid[circular_mask])).T * step

        def _print_top_candidates(label, candidates, costs_tensor):
            costs_np = costs_tensor.cpu().numpy()
            ranked = np.argsort(costs_np)
            best_cost = costs_np[ranked[0]]
            print(f"  {label} — top 5 candidates:")
            for rank in range(min(5, len(ranked))):
                idx = ranked[rank]
                drift_row, drift_col = candidates[idx]
                magnitude = np.sqrt(drift_row**2 + drift_col**2)
                gap = (costs_np[idx] - best_cost) / best_cost * 100 if rank > 0 else 0
                print(f"    drift=({drift_row:+.4f}, {drift_col:+.4f}) px/line "
                      f"({magnitude:.4f} magnitude), cost={costs_np[idx]:.4f}"
                      f"{f' (+{gap:.1f}%)' if rank > 0 else ' (best)'}")

        def _apply_drift(drift_vec):
            for img_idx in range(self.shape[0]):
                scanline_offset = np.arange(self.knots[img_idx].shape[1]) - (self.knots[img_idx].shape[1] - 1) / 2
                self.knots[img_idx][0] += drift_vec[0] * scanline_offset[:, None]
                self.knots[img_idx][1] += drift_vec[1] * scanline_offset[:, None]

        def _search_and_apply(candidates, label):
            best_idx, costs = self._affine_grid_search_batch(candidates, upsample_factor, max_image_shift, chunk_size)
            _apply_drift(candidates[best_idx])
            if verbose:
                _print_top_candidates(label, candidates, costs)
            warped_t = self._warp_and_translate_torch(max_image_shift, upsample_factor)
            self.calculate_error(1, _warped_t=warped_t)
            return candidates[best_idx]

        drift_total = _search_and_apply(drift_vectors, "Coarse search")
        if refine:
            drift_fine = drift_vectors / (num_tests - 1)
            drift_total = drift_total + _search_and_apply(drift_fine, "Refine search")
        if verbose:
            num_rows = self.images[0].shape[0]
            drift_rate = np.sqrt(drift_total[0] ** 2 + drift_total[1] ** 2)
            total_shift = drift_rate * num_rows
            angle_deg = np.degrees(np.arctan2(drift_total[1], drift_total[0]))
            print(f"align_affine: step={step}, num_tests={num_tests} "
                  f"({len(drift_vectors)} candidates), refine={refine}, "
                  f"max_image_shift={max_image_shift}")
            msg = (f"Drift: ({drift_total[0]:+.4f}, {drift_total[1]:+.4f}) px/line, "
                   f"{drift_rate:.4f} magnitude, {angle_deg:.1f} deg, "
                   f"{total_shift:.1f} px total over {num_rows} lines")
            if self.images[0].sampling is not None:
                px_size = self.images[0].sampling[0]
                unit = self.images[0].units[0] if self.images[0].units else "px"
                msg += f" = {total_shift * px_size:.2f} {unit}"
            print(msg)
            err = self.error_track
            print(f"Error: {err[0, 1]:.2f} -> {err[-1, 1]:.2f} "
                  f"({(err[0, 1] - err[-1, 1]) / err[0, 1] * 100:+.1f}%)")

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

    @torch.inference_mode()
    def _affine_grid_search_batch(self, drift_vectors, upsample_factor, max_image_shift, chunk_size=None):
        """Evaluate all candidate drift vectors in parallel.

        Warps both images for each candidate using ``bilinear_kde_batch``
        and scores alignment quality via ``cross_corr_batch``. Without
        batching, each candidate would be a separate Python iteration — this
        is the key operation that enables the 300x speedup.

        Parameters
        ----------
        drift_vectors : ndarray, shape (N, 2)
            Candidate drift vectors to test, columns are (row, col).
        upsample_factor : int
            Subpixel cross-correlation upsampling factor.
        max_image_shift : float or None
            Maximum allowed shift for cross-correlation peak search.
        chunk_size : int or None
            Number of candidates per pass. If None, all at once.

        Returns
        -------
        tuple[int, torch.Tensor]
            Index of the best candidate in ``drift_vectors``, and the full
            cost tensor of shape ``(N,)`` for all candidates (used by
            verbose mode to rank runner-ups).
        """
        device = self._device
        dtype = self._dtype
        num_candidates = drift_vectors.shape[0]
        drift_vectors_t = torch.tensor(drift_vectors, dtype=dtype, device=device)
        canvas_shape = (self.shape[1], self.shape[2])
        # Base coordinates shared across all candidates
        base_data = []
        for img_idx in range(2):
            knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
            row_base, col_base = transform_coordinates(
                knots_t, self.scan_fast_t[img_idx], self.images[img_idx].shape)
            num_rows = self.knots[img_idx].shape[1]
            scanline_offset = (torch.arange(num_rows, dtype=dtype, device=device)
                               - (num_rows - 1) / 2)
            base_data.append((self.images_t[img_idx], row_base, col_base, scanline_offset))
        # Precompute shift mask and frequency grids (shared across chunks)
        shift_mask = None
        if max_image_shift is not None:
            canvas_rows, canvas_cols = canvas_shape
            freq_row = fftfreq(canvas_rows, 1.0 / canvas_rows, device=device, dtype=dtype)
            freq_col = fftfreq(canvas_cols, 1.0 / canvas_cols, device=device, dtype=dtype)
            shift_mask = freq_row[:, None] ** 2 + freq_col[None, :] ** 2 >= max_image_shift ** 2
        freq_grids = (
            fftfreq(canvas_shape[0], device=device, dtype=dtype)[:, None],
            fftfreq(canvas_shape[1], device=device, dtype=dtype)[None, :],
        )
        if chunk_size is None:
            chunk_size = num_candidates
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        if num_chunks > 1:
            warnings.warn(
                f"Processing {num_candidates} candidates in {num_chunks} chunks of {chunk_size}. "
                f"Increase chunk_size for fewer passes.",
                stacklevel=3,
            )
        all_costs = []
        for chunk_start in range(0, num_candidates, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_candidates)
            drift_chunk = drift_vectors_t[chunk_start:chunk_end]
            warped_pair = []
            for img_idx in range(2):
                image_t, row_base, col_base, scanline_offset = base_data[img_idx]
                row_candidates = row_base[None] + drift_chunk[:, 0, None, None] * scanline_offset[None, :, None]
                col_candidates = col_base[None] + drift_chunk[:, 1, None, None] * scanline_offset[None, :, None]
                warped, _ = bilinear_kde_batch(
                    row_candidates, col_candidates, image_t,
                    canvas_shape, self.kde_sigma,
                    self.pad_value[img_idx])
                warped_pair.append(warped)
            all_costs.append(cross_corr_batch(
                warped_pair[0], warped_pair[1],
                upsample_factor,
                max_shift_mask=shift_mask,
                freq_grids=freq_grids))
        all_costs = torch.cat(all_costs)
        return torch.argmin(all_costs).item(), all_costs

    @torch.inference_mode()
    def _warp_and_translate_torch(
        self,
        max_image_shift: float | None,
        upsample_factor: int = 8,
        knots_t: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Regenerate warped images and solve translation on GPU.

        Three phases: warp → solve translation → re-warp. When ``knots_t``
        is provided, reads/writes torch tensors directly (zero numpy
        crossings). Without it, falls back to ``self.knots`` (numpy) for
        compatibility with ``align_affine``.

        Parameters
        ----------
        max_image_shift : float or None
            Maximum allowed translational shift in pixels.
        upsample_factor : int
            Sub-pixel precision for cross-correlation (1/N pixel).
        knots_t : list of torch.Tensor or None
            If provided, list of ``(2, num_rows)`` torch tensors on GPU.
            Translation shifts are applied in-place. Skips numpy sync.

        Returns
        -------
        torch.Tensor
            Warped images on GPU, shape ``(num_images, H, W)``.
        """
        device = self._device
        dtype = self._dtype
        num_images = self.shape[0]
        canvas_shape = (self.shape[1], self.shape[2])

        def _warp_all(warped_t, weights_t):
            """Warp all images onto the canvas using current knots."""
            for img_idx in range(num_images):
                if knots_t is not None:
                    # Torch path: add trailing dim for transform_coordinates (2, N) → (2, N, 1)
                    k = knots_t[img_idx].detach()[:, :, None]
                else:
                    k = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
                row_t, col_t = transform_coordinates(
                    k, self.scan_fast_t[img_idx], self.images[img_idx].shape)
                warped, weights = bilinear_kde_batch(
                    row_t[None], col_t[None], self.images_t[img_idx], canvas_shape,
                    self.kde_sigma, self.pad_value[img_idx])
                warped_t[img_idx] = warped[0]
                weights_t[img_idx] = weights[0]

        # Phase 1: warp all images onto canvas using current knots
        warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
        weights_t = torch.zeros_like(warped_t)
        _warp_all(warped_t, weights_t)
        # Phase 2: solve translation shifts and apply to knots
        shifts_t = translate_align(warped_t, upsample_factor, max_image_shift)
        if knots_t is not None:
            for img_idx in range(num_images):
                knots_t[img_idx].data[0] += shifts_t[img_idx, 0]
                knots_t[img_idx].data[1] += shifts_t[img_idx, 1]
        else:
            shifts_np = shifts_t.cpu().numpy()
            for img_idx in range(num_images):
                self.knots[img_idx][0] += shifts_np[img_idx, 0]
                self.knots[img_idx][1] += shifts_np[img_idx, 1]
        # Phase 3: re-warp with corrected knots and sync to CPU
        _warp_all(warped_t, weights_t)
        if knots_t is None:
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
        adam_steps : int, default 50
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
            raise RuntimeError(
                "No knots found. Call .preprocess() before running alignment."
            )
        # Main optimization loop
        if backend == "pytorch":
            device = self._device
            dtype = self._dtype
            num_images = self.shape[0]
            canvas_shape = (self.shape[1], self.shape[2])
            # Setup once: knots, targets, optimizers on GPU
            knots_t = [
                torch.tensor(self.knots[i][:, :, 0], dtype=dtype, device=device, requires_grad=True)
                for i in range(num_images)
            ]
            # Precompute per-image scan geometry (constant across iterations)
            row_positions_t = [
                torch.tensor(self.interpolator[i].u, dtype=dtype, device=device)
                for i in range(num_images)
            ]
            scale_rows = [self.interpolator[i].scan_fast[0] * (self.images[i].shape[0] - 1) for i in range(num_images)]
            scale_cols = [self.interpolator[i].scan_fast[1] * (self.images[i].shape[1] - 1) for i in range(num_images)]
            optimizers = [torch.optim.Adam([k], lr=lr) for k in knots_t]
            # Precompute Vandermonde matrix for polynomial detrending (constant across iterations)
            if regularization_sigma_px is not None and regularization_sigma_px > 0:
                num_rows_knot = knots_t[0].shape[1]
                x_knot = torch.arange(num_rows_knot, dtype=dtype, device=device)
                x_norm = (x_knot - x_knot.mean()) / x_knot.std()
                vander = torch.stack([x_norm ** p for p in range(regularization_poly_order + 1)], dim=1)
            warped_t = self._warp_and_translate_torch(
                max_image_shift, upsample_factor=8, knots_t=knots_t)
            for _ in tqdm(range(num_iterations), desc="Solving nonrigid drift (pytorch)"):
                warped_sum = warped_t.sum(0)
                for ind in range(num_images):
                    ref_image_t = ((warped_sum - warped_t[ind]) / (num_images - 1))[None, None]
                    knots_prev = knots_t[ind].detach().clone()
                    optimizers[ind].state.clear()
                    self._optimize_knots_pytorch(
                        ref_image_t, self.images_t[ind], knots_t[ind],
                        row_positions_t[ind], scale_rows[ind], scale_cols[ind],
                        canvas_shape[0], canvas_shape[1],
                        optimizers[ind], adam_steps)
                    with torch.no_grad():
                        if regularization_max_image_shift_px is not None:
                            shift = knots_t[ind] - knots_prev
                            dist = torch.norm(shift, dim=0, keepdim=True)
                            scale_factor = torch.clamp(regularization_max_image_shift_px / dist.clamp(min=1e-8), max=1.0)
                            knots_t[ind].data.copy_(knots_prev + shift * scale_factor)
                        if regularization_sigma_px is not None and regularization_sigma_px > 0:
                            for axis in range(2):
                                y = knots_t[ind][axis]
                                coefs, _, _, _ = torch.linalg.lstsq(vander, y)
                                trend = vander @ coefs
                                residual = y - trend
                                smoothed = gaussian_smooth_1d(residual[None], regularization_sigma_px)[0]
                                knots_t[ind].data[axis] = smoothed + trend
                        if regularization_update_step_size is not None:
                            knots_t[ind].data.copy_(
                                knots_prev + (knots_t[ind] - knots_prev) * regularization_update_step_size)
                warped_t = self._warp_and_translate_torch(
                    max_image_shift, upsample_factor=8, knots_t=knots_t)
                self.calculate_error(2, _warped_t=warped_t)
            # Sync torch knots back to numpy once, then final warp to populate images_warped
            for ind in range(num_images):
                self.knots[ind][:, :, 0] = knots_t[ind].detach().cpu().numpy()
            self._warp_and_translate_torch(max_image_shift, upsample_factor=8)
        else:
            for _ in tqdm(range(num_iterations), desc="Solving nonrigid drift (scipy)"):
                for ind in range(self.shape[0]):
                    image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(axis=0)
                    knots_updated = self._optimize_knots_scipy(
                        ind, image_ref, self.knots[ind],
                        max_optimize_iterations=max_optimize_iterations,
                        solve_individual_rows=solve_individual_rows)
                    if regularization_max_image_shift_px is not None:
                        knots_shift = knots_updated - self.knots[ind]
                        knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                        sub = knots_dist > regularization_max_image_shift_px
                        knots_updated[0][sub] = (self.knots[ind][0][sub]
                            + knots_shift[0][sub] * regularization_max_image_shift_px / knots_dist[sub])
                        knots_updated[1][sub] = (self.knots[ind][1][sub]
                            + knots_shift[1][sub] * regularization_max_image_shift_px / knots_dist[sub])
                    if regularization_sigma_px is not None and regularization_sigma_px > 0:
                        knots_smoothed = knots_updated.copy()
                        for dim in range(2):
                            x = np.arange(knots_updated.shape[1])
                            for knot_ind in range(knots_updated.shape[2]):
                                y = knots_updated[dim, :, knot_ind]
                                coefs = np.polyfit(x, y, deg=regularization_poly_order)
                                trend = np.polyval(coefs, x)
                                residual = y - trend
                                residual_smooth = gaussian_filter(residual, sigma=regularization_sigma_px)
                                knots_smoothed[dim, :, knot_ind] = residual_smooth + trend
                        knots_updated = knots_smoothed
                    if regularization_update_step_size is not None:
                        knots_updated = (self.knots[ind]
                            + (knots_updated - self.knots[ind]) * regularization_update_step_size)
                    self.knots[ind] = knots_updated
                warped_t = self._warp_and_translate_torch(max_image_shift, upsample_factor=8)
                self.calculate_error(2, _warped_t=warped_t)

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
        self, ref_image_t, target_image_t, knots_t,
        row_position_t, scale_row, scale_col,
        num_rows_canvas, num_cols_canvas,
        optimizer, adam_steps,
    ):
        """Run Adam steps on knots to minimize MSE against reference.

        All inputs and outputs are torch tensors on GPU — no numpy
        crossings. The optimizer is created once and reused across
        outer iterations to avoid construction overhead.
        """
        for _ in range(adam_steps):
            optimizer.zero_grad()
            row_coords = knots_t[0, :, None] + row_position_t[None, :] * scale_row
            col_coords = knots_t[1, :, None] + row_position_t[None, :] * scale_col
            grid_col = 2.0 * col_coords / (num_cols_canvas - 1) - 1.0
            grid_row = 2.0 * row_coords / (num_rows_canvas - 1) - 1.0
            grid = torch.stack([grid_col, grid_row], dim=-1)[None]
            warped = torch.nn.functional.grid_sample(
                ref_image_t, grid, mode='bilinear', align_corners=True, padding_mode='border')[0, 0]
            loss = ((warped - target_image_t) ** 2).mean()
            loss.backward()
            optimizer.step()

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
            image_corr_fft = np.zeros_like(weights_sum, dtype=complex)
            np.divide(
                np.sum(stack_fft, axis=0),
                weights_sum,
                where=weights_sum > 0.0,
                out=image_corr_fft,
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
        return image_corr

    def calculate_error(
        self,
        mode: int,
        _warped_t: torch.Tensor | None = None,
    ):
        """Compute per-image MAE against the mean and append to error history.

        Measures how well the warped images agree by computing the mean
        absolute difference of each image from the stack mean. Without
        error tracking, there is no way to verify that alignment steps
        are actually improving the result.

        Parameters
        ----------
        mode : int
            Stage identifier (0=preprocess, 1=affine, 2=nonrigid).
        _warped_t : torch.Tensor or None
            If provided, compute error from this tensor directly,
            avoiding a GPU-to-CPU round-trip.
        """
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
