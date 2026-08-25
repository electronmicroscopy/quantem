"""Apply a solved drift field and return corrected scientific data."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import binary_closing as ndi_binary_closing
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.imaging.drift.core import knots as drift_knots
from quantem.imaging.drift.core.warping import (
    backward_warp,
    ensure_warped_images,
    reference_scan_stack,
    warp_and_translate,
)


def dataset_info(dataset) -> dict[str, object]:
    """Copy the calibration and metadata needed for corrected output."""
    if not hasattr(dataset, "array"):
        return {}
    info = {}
    for name in ("name", "origin", "sampling", "units", "signal_units"):
        if hasattr(dataset, name):
            info[name] = deepcopy(getattr(dataset, name))
    metadata = getattr(dataset, "metadata", None)
    if isinstance(metadata, dict):
        info["metadata"] = deepcopy(metadata)
    return info


def _corrected_dataset(array: np.ndarray, info: dict[str, object]):
    """Construct corrected data without discarding source calibration."""
    dataset_class = Dataset2d if array.ndim == 2 else Dataset3d
    if array.ndim == 4:
        from quantem.core.datastructures.dataset4d import Dataset4d

        dataset_class = Dataset4d
    kwargs = {
        name: deepcopy(info[name])
        for name in ("name", "origin", "sampling", "units", "signal_units")
        if name in info
    }
    result = dataset_class.from_array(array, **kwargs)
    if metadata := info.get("metadata"):
        result.metadata.update(deepcopy(metadata))
    return result


def padding_offset(
    canvas_shape: tuple[int, int],
    scan_shape: tuple[int, int],
    *,
    integer: bool = False,
) -> tuple[float, float] | tuple[int, int]:
    """Return the ``(row, col)`` offset of a scan in its padded canvas."""
    canvas_h, canvas_w = canvas_shape
    scan_h, scan_w = scan_shape
    if integer:
        return (canvas_h - scan_h) // 2, (canvas_w - scan_w) // 2
    return (canvas_h - scan_h) / 2.0, (canvas_w - scan_w) / 2.0


def fourier_crop_torch(
    fft_array: torch.Tensor,
    crop_shape: tuple[int, int],
) -> torch.Tensor:
    """Crop a corner-centered FFT tensor to its lowest frequencies."""
    crop_h, crop_w = crop_shape
    h1 = crop_h // 2
    h2 = crop_h - h1
    w1 = crop_w // 2
    w2 = crop_w - w1
    result = torch.zeros(crop_shape, dtype=fft_array.dtype, device=fft_array.device)
    result[:h1, :w1] = fft_array[:h1, :w1]
    result[:h1, -w2:] = fft_array[:h1, -w2:]
    result[-h2:, :w1] = fft_array[-h2:, :w1]
    result[-h2:, -w2:] = fft_array[-h2:, -w2:]
    return result


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest axis-aligned all-true rectangle in ``mask``."""
    bin_factor = max(1, int(np.ceil(max(mask.shape) / 512)))
    if bin_factor > 1:
        height = mask.shape[0] // bin_factor
        width = mask.shape[1] // bin_factor
        small = (
            mask[: height * bin_factor, : width * bin_factor]
            .reshape(height, bin_factor, width, bin_factor)
            .all(axis=(1, 3))
        )
    else:
        small = mask
    heights = np.zeros(small.shape[1], dtype=int)
    best = (0, 0, 0, 0, 0)
    for row in range(small.shape[0]):
        heights = np.where(small[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        extended = np.append(heights, 0)
        for col in range(len(extended)):
            start = col
            while stack and stack[-1][1] >= extended[col]:
                stack_start, height = stack.pop()
                if height * (col - stack_start) > best[0]:
                    best = (
                        height * (col - stack_start),
                        row - height + 1,
                        row + 1,
                        stack_start,
                        col,
                    )
                start = stack_start
            stack.append((start, extended[col]))
    _, row_0, row_1, col_0, col_1 = best
    row_0, row_1, col_0, col_1 = (
        row_0 * bin_factor,
        row_1 * bin_factor,
        col_0 * bin_factor,
        col_1 * bin_factor,
    )
    row_1 = min(row_1, mask.shape[0])
    col_1 = min(col_1, mask.shape[1])
    while row_0 < row_1 and not mask[row_0, col_0:col_1].all():
        row_0 += 1
    while row_1 > row_0 and not mask[row_1 - 1, col_0:col_1].all():
        row_1 -= 1
    while col_0 < col_1 and not mask[row_0:row_1, col_0].all():
        col_0 += 1
    while col_1 > col_0 and not mask[row_0:row_1, col_1 - 1].all():
        col_1 -= 1
    return row_0, row_1, col_0, col_1


def corrected(
    self,
    *,
    upsample_factor: int = 2,
    output_original_shape: bool | None = None,
    strip_padding: bool = False,
    smoothing_sigma: float | None = 0.5,
    stage: str | None = None,
    merge: bool = True,
    verbose: bool = True,
):
    """Return corrected scientific data in the natural type for the acquisition.

    A single endpoint keeps HAADF, spectrum-image, and 4D-STEM workflows from
    requiring separate output APIs while preserving each dataset's calibration
    and axis order.

    Parameters
    ----------
    upsample_factor : int, default 2
        Sampling multiplier for the corrected image.
    output_original_shape : bool or None, default None
        ``None`` keeps the natural frame: the padded solver canvas for paired
        2-D scans and the native input frame for reference datasets. ``True``
        returns the original image shape; ``False`` keeps the solver canvas.
    strip_padding : bool, default False
        Remove pixels that do not share measured coverage across scans.
    smoothing_sigma : float or None, default 0.5
        Gaussian smoothing applied during scanline interpolation.
    stage : {"initial", "affine", "strip", None}, optional
        Saved correction stage to apply. ``None`` uses the current solution.
    merge : bool, default True
        Average corrected image pairs. ``False`` returns each scan separately.
    verbose : bool, default True
        Show progress for a multi-channel reference dataset.

    Returns
    -------
    Dataset2d, Dataset3d, or list of Dataset2d
        Corrected data with the source calibration and metadata.

    Examples
    --------
    >>> corrected = drift.corrected()
    >>> scans = drift.corrected(merge=False)
    """
    automatic_output_frame = output_original_shape is None
    output_original_shape = (
        self._reference_mode
        if automatic_output_frame
        else bool(output_original_shape)
    )

    if self._reference_mode:
        if (
            upsample_factor != 2
            or not output_original_shape
            or strip_padding
            or smoothing_sigma != 0.5
            or not merge
        ):
            raise ValueError(
                "Reference-mode corrected() returns the corrected source dataset "
                "at its native shape. Use apply_correction() for interpolation "
                "controls and crop() to remove unsupported borders."
            )
        drifted = self._datasets[1]
        corrected = apply_correction(
            self,
            drifted,
            image_index=1,
            stage=stage,
            verbose=verbose,
        )
        if isinstance(corrected, torch.Tensor):
            corrected = corrected.cpu().numpy()
        corrected = corrected.astype(np.float32, copy=False)
        return _corrected_dataset(
            corrected,
            getattr(self, "_reference_dataset_info", {}),
        )
    if getattr(self, "_datasets", None) is not None:
        raise RuntimeError(
            "4D-STEM collection correction uses the explicit "
            "corrected_4dstem() API. Use "
            "DriftCorrection.from_4dstem(data_0, data_1, ...)"
            ".preprocess().correct_affine().corrected_4dstem()."
        )

    if not merge:
        if (
            upsample_factor != 2
            or (not automatic_output_frame and not output_original_shape)
            or strip_padding
            or smoothing_sigma != 0.5
        ):
            raise ValueError(
                "corrected(merge=False) returns the individual scans on the "
                "solver canvas. Resampling, output-shape, smoothing, and crop "
                "controls apply only to the merged image."
            )
        panels = comparison_panels(self, stage)
        return [
            Dataset2d.from_array(
                np.asarray(array),
                name=f"drift corrected scan {index}",
                origin=self.imgs[0].origin,
                sampling=self.imgs[0].sampling,
                units=self.imgs[0].units,
            )
            for index, array in enumerate(panels["corrected_scans"])
        ]

    device = self._device
    dtype = self._dtype
    up_h = round(self.shape[1] * upsample_factor)
    up_w = round(self.shape[2] * upsample_factor)
    canvas_up = (up_h, up_w)
    if smoothing_sigma is None:
        smoothing_sigma = self.kde_sigma

    stack_corr = torch.zeros(self.shape[0], up_h, up_w, dtype=dtype, device=device)
    knots = drift_knots.stage_knots(self, stage)
    for image_index in range(self.shape[0]):
        warped, _ = drift_knots.interpolator(
            self, image_index, knots[image_index]
        ).warp_to_canvas(
            self.imgs_t[image_index],
            canvas_up,
            smoothing_sigma * upsample_factor,
            self.pad_value[image_index],
            upsample_factor=upsample_factor,
        )
        stack_corr[image_index] = warped
    image_corr_fft = torch.fft.fft2(stack_corr.mean(0))

    output_shape = (
        tuple(int(value) for value in self.imgs[0].shape[:2])
        if output_original_shape
        else tuple(int(value) for value in self.shape[-2:])
    )
    image_corr_fft = fourier_crop_torch(
        image_corr_fft,
        output_shape,
    ) / upsample_factor**2
    corrected_array = torch.fft.ifft2(image_corr_fft).real.cpu().numpy()
    if strip_padding and output_original_shape:
        scan_h, scan_w = self.imgs[0].shape[:2]
        pad_h, pad_w = padding_offset(corrected_array.shape[:2], (scan_h, scan_w), integer=True)
        corrected_array = corrected_array[pad_h : pad_h + scan_h, pad_w : pad_w + scan_w]

    image_corr = Dataset2d.from_array(
        corrected_array,
        name="drift corrected image",
        origin=self.imgs[0].origin,
        sampling=self.imgs[0].sampling,
        units=self.imgs[0].units,
    )
    image_corr.metadata.update(
        {
            "downsample": int(getattr(self, "downsample", 1)),
            "downsample_method": getattr(self, "downsample_method", "none"),
            "downsample_metadata": getattr(
                self,
                "downsample_metadata",
                {"factor": 1, "method": "none"},
            ),
            "downsample_sampling": np.asarray(self.imgs[0].sampling, dtype=float).tolist(),
            "downsample_units": list(self.imgs[0].units),
        }
    )
    return image_corr


def apply_correction_to_dataset(
    correction,
    ds_4d: torch.Tensor | np.ndarray | None = None,
    *,
    image_index: int = -1,
    mode: str = "bilinear",
    chunk_size: int | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
    output: np.ndarray | None = None,
    verbose: bool = False,
    progress_desc: str = "Applying drift correction",
    stage: str | None = None,
) -> torch.Tensor | np.ndarray:
    """Apply drift correction to a ≥3-D dataset with scan axes leading.

    Internal worker for the 4D-STEM / spectral path of
    :meth:`DriftCorrection.apply_correction`. It selects single-shot or chunked
    processing from available device memory and supports preallocated
    ``output=`` for zero-copy memmap workflows.
    """
    # Resolve dataset from stored data when not provided explicitly
    if ds_4d is None:
        datasets = correction._datasets
        if datasets is None:
            raise ValueError(
                "No dataset provided and none stored. Pass ds_4d "
                "explicitly, or build this instance with "
                "DriftCorrection(ds_a, ds_b, ...)."
            )
        if image_index < 0:
            image_index = len(datasets) + image_index
        if image_index < 0 or image_index >= len(datasets):
            raise IndexError(
                f"image_index={image_index} out of range for "
                f"{len(datasets)} stored datasets"
            )
        ds_4d = datasets[image_index]

    is_numpy = isinstance(ds_4d, np.ndarray)
    original_shape = ds_4d.shape if is_numpy else tuple(ds_4d.shape)
    input_np_dtype = ds_4d.dtype if is_numpy else None
    use_external_output = output is not None

    if use_external_output:
        if not isinstance(output, np.ndarray):
            raise TypeError(
                "output must be a numpy ndarray (or np.memmap), "
                f"got {type(output).__name__}"
            )
        if tuple(output.shape) != tuple(original_shape):
            raise ValueError(
                f"output shape {output.shape} does not match "
                f"ds_4d shape {original_shape}"
            )

    ndim = len(original_shape)
    if ndim < 3:
        raise ValueError(
            f"ds_4d must be at least 3D, got shape {original_shape}"
        )

    scan_h, scan_w = original_shape[0], original_shape[1]
    n_channels = 1
    for d in range(2, ndim):
        n_channels *= original_shape[d]

    device = torch.device(correction._device)

    # ── Drift from knots (canvas → raw frame) ──
    idx = image_index % len(correction.knots)
    # Validates preprocess+align ran.
    knots = drift_knots.stage_knots(correction, stage)[idx]
    knot_h = knots.shape[1]
    if knot_h != scan_h:
        raise ValueError(
            f"Drift grid has {knot_h} rows but ds_4d has "
            f"{scan_h} scan rows. Ensure reference image and ds_4d "
            f"have matching scan dimensions (check padding / resize).")

    drift = drift_knots.interpolator(correction, idx, knots).drift_raw(
        correction._initial_knots[idx]
    ).to(
        device=device, dtype=torch.float32
    )
    # K=1 → drift is (2, H), broadcast across columns.
    # K>=2 → drift is (2, H, W), varies along fast axis.
    if drift.ndim == 2:
        drift_row = drift[0][:, None]
        drift_col = drift[1][:, None]
    else:
        drift_row = drift[0]
        drift_col = drift[1]
    row_coords = torch.arange(scan_h, device=device, dtype=torch.float32)
    col_coords = torch.arange(scan_w, device=device, dtype=torch.float32)
    sample_row = row_coords[:, None].expand(scan_h, scan_w) - drift_row
    sample_col = col_coords[None, :].expand(scan_h, scan_w) - drift_col
    # ── Pre-compute warp grid ONCE (tiny: 1×H×W×2 f32) ──
    warp_grid = torch.stack([
        2.0 * sample_col / (scan_w - 1) - 1.0,
        2.0 * sample_row / (scan_h - 1) - 1.0,
    ], dim=-1)[None]                                       # (1, H, W, 2)

    # ── Flatten input to (H, W, C) view ──
    flat = (
        torch.from_numpy(ds_4d.reshape(scan_h, scan_w, n_channels))
        if is_numpy
        else ds_4d.reshape(scan_h, scan_w, n_channels)
    )

    # Output dtype for device intermediates.
    out_dt = torch.float32
    if output_dtype == "same":
        if is_numpy and input_np_dtype is not None:
            out_dt = torch.from_numpy(
                np.empty(0, dtype=input_np_dtype)
            ).dtype
        elif not is_numpy:
            out_dt = ds_4d.dtype
    elif isinstance(output_dtype, torch.dtype):
        out_dt = output_dtype

    # ── Target device ──
    # Default the output to the input's device so the pipeline stays
    # in place; explicit ``output_device`` overrides.
    if use_external_output:
        target = torch.device("cpu")
    elif output_device is not None:
        target = torch.device(output_device)
        if target.type == "cuda":
            target = device
    elif (
        isinstance(ds_4d, torch.Tensor)
        and (ds_4d.is_cuda or ds_4d.device.type == "mps")
    ):
        target = device
    else:
        target = torch.device("cpu")
    return_numpy = (
        use_external_output
        or (is_numpy and output_device is None)
    )

    # Choose a chunk size from available device memory.
    if chunk_size is None:
        bytes_per_ch = scan_h * scan_w * 4
        if device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
            except RuntimeError:
                free_bytes = 0
        else:
            free_bytes = 0
        if target.type == "cuda" and not use_external_output:
            out_elem = torch.tensor([], dtype=out_dt).element_size()
            free_bytes = max(
                0,
                free_bytes - n_channels * scan_h * scan_w * out_elem,
            )
        if device.type == "mps":
            chunk_size = min(n_channels, 64)
        elif device.type == "cuda":
            chunk_size = min(
                n_channels,
                max(1, int(free_bytes * 0.7 / (bytes_per_ch * 2))),
            )
        else:
            chunk_size = min(n_channels, 64)

    # ── Allocate output ──
    if use_external_output:
        out_flat = output.reshape(scan_h, scan_w, n_channels)
    else:
        internal_output = torch.empty(
            scan_h, scan_w, n_channels, dtype=out_dt, device=target,
        )

    # ── Numpy dtype for external output conversion ──
    if use_external_output:
        _out_np_dtype = output.dtype

    # ── Vectorized grid_sample with pre-computed grid ──
    chunks = range(0, n_channels, chunk_size)
    num_chunks = len(chunks)
    correction_progress = tqdm(
        total=n_channels,
        desc=progress_desc,
        unit="channel",
        disable=not verbose or num_chunks <= 1,
    )
    for start in chunks:
        end = min(start + chunk_size, n_channels)
        warped = F.grid_sample(
            flat[:, :, start:end].permute(2, 0, 1).contiguous()
            .to(device=device, dtype=torch.float32)[None],
            warp_grid,
            mode=mode, align_corners=True, padding_mode="border",
        )[0].permute(1, 2, 0)

        # Integer casts truncate. Round first or low-count detector pixels
        # collapse to zero after interpolation.
        is_int = isinstance(out_dt, torch.dtype) and not out_dt.is_floating_point
        if is_int:
            warped_cast = warped.round().clamp_(
                torch.iinfo(out_dt).min, torch.iinfo(out_dt).max)
        else:
            warped_cast = warped
        if use_external_output:
            out_flat[:, :, start:end] = (
                warped_cast.cpu().numpy().astype(_out_np_dtype)
            )
        else:
            internal_output[:, :, start:end] = warped_cast.to(
                device=target, dtype=out_dt,
            )
        correction_progress.update(end - start)
    correction_progress.close()

    if use_external_output:
        return output

    result = internal_output.reshape(original_shape)
    if return_numpy:
        return result.detach().cpu().numpy()
    return result


def apply_correction(
    self,
    data: Dataset2d | Dataset3d | torch.Tensor | np.ndarray | None = None,
    image_index: int = -1,
    *,
    stage: str | None = None,
    mode: str = "bilinear",
    chunk_size: int | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
    output: np.ndarray | None = None,
    verbose: bool = True,
) -> torch.Tensor | np.ndarray:
    """Apply the learned drift correction to an image or dataset.

    Image collections use the last two axes as scan coordinates. Reference
    and 4D-STEM datasets use the first two axes, so spectra and diffraction
    patterns remain attached to their corrected probe positions. Large
    datasets are processed in chunks and may be written directly into a
    preallocated array.

    Parameters
    ----------
    data : Dataset2d, Dataset3d, ndarray, or torch.Tensor, optional
        Data to correct. If omitted, use the stored alignment image or
        4D-STEM dataset.
    image_index : int, optional
        Scan trajectory to apply. Default is the last scan.
    stage : {"initial", "affine", "strip", None}, optional
        Saved correction stage to apply. ``None`` uses the current solution.
    mode : str, optional
        Interpolation kernel, either ``"bilinear"`` or ``"bicubic"``.
        Default is ``"bilinear"``.
    chunk_size : int, optional
        Number of detector or spectral channels corrected together. The
        default selects a size that fits the available device memory.
    output_dtype : torch.dtype, numpy dtype, or str, optional
        Output precision for dataset correction. By default, preserve the
        input precision.
    output_device : str or torch.device, optional
        Device for the returned tensor. NumPy inputs return NumPy arrays by
        default.
    output : ndarray, optional
        Preallocated destination, such as a memory-mapped array.
    verbose : bool, optional
        Show progress for multi-chunk datasets. Default is ``True``.

    Returns
    -------
    Dataset2d, Dataset3d, torch.Tensor, or ndarray
        Corrected data with the same shape and axis order as the input.
        QuantEM datasets retain calibration, units, and metadata.

    Examples
    --------
    >>> dc = DriftCorrection(reference, moving, scan_direction_degrees=(0, 90))
    >>> dc.correct_affine(show_combined=False)
    >>> corrected = dc.apply_correction(image_index=1)

    """
    if isinstance(data, (Dataset2d, Dataset3d)):
        source = data
        corrected = apply_correction(
            self,
            source.array,
            image_index=image_index,
            stage=stage,
            mode=mode,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
            output_device=output_device,
            output=output,
            verbose=verbose,
        )
        if isinstance(corrected, torch.Tensor):
            corrected = corrected.detach().cpu().numpy()
        return _corrected_dataset(
            np.asarray(corrected),
            dataset_info(source),
        )

    if not hasattr(self, "knots") or not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "apply_correction() requires preprocess() and correct_affine() "
            "first. Run dc.preprocess().correct_affine() (and optionally "
            ".correct_nonrigid()) before apply_correction()."
        )
    valid_modes = {"bilinear", "bicubic"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
    index = image_index % len(self.knots)
    drift_knots.knot_delta_canvas(self, index)
    is_4dstem_mode = getattr(self, "_datasets", None) is not None and not self._reference_mode
    scan_h = self.imgs[index].shape[0]
    scan_w = self.imgs[index].shape[1]
    dataset_layout = data is None and is_4dstem_mode
    if data is None:
        if (
            getattr(self, "_built_from_datasets", False)
            and getattr(self, "_datasets", None) is None
        ):
            raise ValueError(
                "apply_correction() has no data: this instance was built "
                "from a 4D-STEM / reference dataset, but save() dropped the "
                "dataset (too large to serialize) and it was not re-attached "
                "after load. Pass the dataset explicitly, e.g. "
                "dc.apply_correction(data=my_4dstem_array), or re-attach it "
                "before calling apply_correction()."
            )
        if self._reference_mode:
            data = self._datasets[1]
    if data is not None:
        ndim = data.ndim
        shape = tuple(data.shape)
        dataset_layout = ndim >= 4
        if ndim == 3:
            cube_layout = shape[0] == scan_h and shape[1] == scan_w
            batch_layout = shape[-2] == scan_h and shape[-1] == scan_w and not cube_layout
            dataset_layout = cube_layout and (not batch_layout or is_4dstem_mode)

    if dataset_layout:
        return apply_correction_to_dataset(
            self,
            data,
            image_index=image_index,
            stage=stage,
            mode=mode,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
            output_device=output_device,
            output=output,
            verbose=verbose,
        )

    if data is None:
        data_t = self.imgs_t[index]
    elif isinstance(data, np.ndarray):
        data_t = torch.tensor(data, dtype=self._dtype, device=self._device)
    else:
        data_t = data.to(device=self._device, dtype=self._dtype)
    image_height = data_t.shape[-2]
    knots = drift_knots.stage_knots(self, stage)[index]
    knot_height = knots.shape[1]
    if image_height != knot_height:
        raise ValueError(
            f"Input scan-row axis ({image_height}) does not match knot grid "
            f"height ({knot_height}). For 4D-STEM mode the leading axis is "
            "the scan row; for image collection mode the trailing-2 axes "
            "are scan."
        )
    drift = drift_knots.interpolator(self, index, knots).drift_raw(
        self._initial_knots[index]
    )
    return backward_warp(data_t, drift=drift, mode=mode)


def crop(self, image: NDArray, *, shape: str = "square") -> NDArray:
    """Crop an image to the field measured by every corrected scan.

    Parameters
    ----------
    image : numpy.ndarray
        Corrected image or scan-axis-leading dataset.
    shape : {"square", "rectangle"}, default "square"
        Keep the largest centered square or the full common rectangle.

    Returns
    -------
    numpy.ndarray
        Cropped data with trailing channel or detector axes unchanged.

    Examples
    --------
    >>> cropped = drift.crop(corrected.array, shape="rectangle")
    """
    rows, cols = crop_slices(self)
    if shape == "square":
        height = rows.stop - rows.start
        width = cols.stop - cols.start
        side = min(height, width)
        row_start = rows.start + (height - side) // 2
        col_start = cols.start + (width - side) // 2
        rows = slice(row_start, row_start + side)
        cols = slice(col_start, col_start + side)
    elif shape != "rectangle":
        raise ValueError(f'shape must be "rectangle" or "square", got {shape!r}')
    # Scan axes are the LEADING two: an EDS cube is (row, col, channel), so
    # trailing-axis indexing would slice width and channels instead of the
    # scan field. 4D-STEM mode shares the same (row, col, ...) layout.
    return np.asarray(image)[rows, cols, ...]


def crop_slices(self) -> tuple[slice, slice]:
    """Return row and column slices for the common measured field of view.

    Use these slices when several related arrays must receive exactly the same
    crop as the corrected image.

    Returns
    -------
    tuple of slice
        Row and column slices in ``(row, col)`` order.

    Examples
    --------
    >>> rows, cols = drift.crop_slices()
    >>> cropped_spectrum = spectrum[rows, cols, :]
    """
    if getattr(self, "_reference_mode", False) and hasattr(self, "_initial_knots"):
        scan_h, scan_w = np.asarray(self.imgs[0].array).shape[:2]
        pad = 4
        field = self.drift_field(1).detach().cpu().numpy()
        row_drift = field[0].ravel()
        col_drift = field[1].ravel()
        top = max(0.0, float(row_drift.max()))
        bottom = max(0.0, float(-row_drift.min()))
        left = max(0.0, float(col_drift.max()))
        right = max(0.0, float(-col_drift.min()))
        row = slice(
            int(np.ceil(top)) + pad,
            scan_h - int(np.ceil(bottom)) - pad,
        )
        col = slice(
            int(np.ceil(left)) + pad,
            scan_w - int(np.ceil(right)) - pad,
        )
        return row, col
    mask = coverage_mask(self)
    scan_h, scan_w = mask.shape
    pad = 4
    row_0, row_1, col_0, col_1 = largest_rectangle(mask)
    row = slice(min(row_0 + pad, scan_h), max(row_1 - pad, 0))
    col = slice(min(col_0 + pad, scan_w), max(col_1 - pad, 0))
    return row, col


def coverage_mask(self) -> np.ndarray:
    """Identify pixels supported by every corrected scan.

    The mask separates measured overlap from padded canvas pixels, making NCC
    and other comparisons use the same physical field of view.

    Returns
    -------
    numpy.ndarray
        Boolean mask in the original scan frame.

    Examples
    --------
    >>> common_pixels = drift.coverage_mask()
    >>> ncc = compare(first[common_pixels], second[common_pixels])
    """
    canvas_h, canvas_w = self.imgs_warped.array.shape[-2:]
    knot_counts = {int(knots.shape[2]) for knots in self.knots}
    if knot_counts != {1}:
        knots = torch.stack(
            [value.detach() for value in self.knots]
        ).to(device=self._device, dtype=self._dtype)
        _, weights = warp_and_translate(
            self,
            max_image_shift=None,
            knots_batch=knots,
            solve_translation=False,
            return_weights=True,
        )
        common = (weights >= 1e-3).all(dim=0).cpu().numpy()
        scan_h, scan_w = self.imgs[0].shape[:2]
        offset_row = (canvas_h - scan_h) // 2
        offset_col = (canvas_w - scan_w) // 2
        return common[
            offset_row : offset_row + scan_h,
            offset_col : offset_col + scan_w,
        ]

    common = np.ones((canvas_h, canvas_w), dtype=bool)
    for index in range(len(self.knots)):
        knots_full = self.knots[index].detach().cpu().numpy()
        knots = knots_full[:, :, 0]
        fast = np.asarray(self.scan_fast[index], dtype=float)
        width = int(self.imgs[index].shape[1])
        position = np.arange(width, dtype=float)
        rows = np.round(knots[0][:, None] + fast[0] * position[None, :]).astype(int)
        cols = np.round(knots[1][:, None] + fast[1] * position[None, :]).astype(int)
        footprint = np.zeros((canvas_h, canvas_w), dtype=bool)
        inside = (rows >= 0) & (rows < canvas_h) & (cols >= 0) & (cols < canvas_w)
        footprint[rows[inside], cols[inside]] = True
        footprint = ndi_binary_closing(footprint, structure=np.ones((3, 3), dtype=bool))
        common &= footprint
    scan_h = int(self.imgs[0].shape[0])
    scan_w = int(self.imgs[0].shape[1])
    offset_row = (canvas_h - scan_h) // 2
    offset_col = (canvas_w - scan_w) // 2
    return common[
        offset_row : offset_row + scan_h,
        offset_col : offset_col + scan_w,
    ]


def warped_stack(correction, stage: str | None = None) -> np.ndarray:
    """Return aligned scans at one checkpoint without changing the solved object.

    This is the shared data path behind stage-aware figures, reports, and
    ``corrected(merge=False)``. Historical checkpoints are rendered directly
    from their saved knots, so asking for an affine result after non-rigid
    refinement cannot alter the final correction or its cache.
    """
    if stage in (None, "nonrigid", "non-rigid"):
        ensure_warped_images(correction)
        return np.asarray(correction.imgs_warped.array, dtype=np.float32)

    knots = drift_knots.stage_knots(correction, stage)
    if not correction._reference_mode:
        warped = warp_and_translate(
            correction,
            max_image_shift=None,
            upsample_factor=8,
            knots_batch=torch.stack([k.detach() for k in knots]),
            solve_translation=False,
        )
        return warped.detach().cpu().numpy().astype(np.float32, copy=False)

    stack = np.stack(reference_scan_stack(correction, knots)).astype(
        np.float32, copy=False
    )
    canvas = np.empty((2, *correction.shape[1:]), dtype=np.float32)
    row = (correction.shape[1] - stack.shape[1]) // 2
    col = (correction.shape[2] - stack.shape[2]) // 2
    for index in range(2):
        canvas[index].fill(float(correction.pad_value[index]))
        canvas[
            index,
            row : row + stack.shape[1],
            col : col + stack.shape[2],
        ] = stack[index]
    return canvas


def comparison_panels(correction, stage: str | None = None) -> dict:
    """Compare raw and corrected scans in the first acquisition's frame."""
    num_scans = correction.shape[0]
    angles = np.asarray(correction.scan_direction_degrees, dtype=float)
    quarter = [int(round(-angle / 90.0)) % 4 for angle in angles]
    reference = quarter[0]
    raw = [
        np.rot90(
            np.asarray(correction.imgs[index].array),
            (quarter[index] - reference) % 4,
        )
        for index in range(num_scans)
    ]
    corrected = np.rot90(
        warped_stack(correction, stage),
        (-reference) % 4,
        axes=(1, 2),
    )
    raw_combined = sum(image.astype(np.float32) for image in raw) / num_scans
    relative_angles = [
        ((float(angle) - float(angles[0]) + 180.0) % 360.0) - 180.0
        for angle in angles
    ]
    names = ["0deg"] + [
        f"{int(round(relative_angles[index]))}deg" for index in range(1, num_scans)
    ]
    raw_labels = [f"{names[0]} scan"] + [
        f"{names[index]} scan -> 0deg frame" for index in range(1, num_scans)
    ]
    images = raw + [raw_combined] + list(corrected) + [corrected.mean(0)]
    labels = (
        raw_labels
        + ["combined scan"]
        + [f"corrected {name}" for name in names]
        + ["corrected combined scan"]
    )
    sampling = np.asarray(correction.imgs[0].sampling, dtype=float)
    unit = correction.imgs[0].units[0] if getattr(correction.imgs[0], "units", None) else "pixels"
    return {
        "images": images,
        "labels": labels,
        "ncols": num_scans + 1,
        "raw_scans": raw,
        "raw_combined": raw_combined,
        "corrected_scans": list(corrected),
        "corrected_combined": corrected.mean(0),
        "pixel_size": float(sampling[0]),
        "pixel_unit": unit,
    }
