"""Virtual-detector and paired-dataset products for 4D-STEM correction."""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.imaging.drift.apply import (
    apply_correction_to_dataset,
    crop_slices,
    padding_offset,
)
from quantem.imaging.drift.core import knots as drift_knots


@dataclass
class CorrectionResult:
    """Container returned by 0/90 4D-STEM collection correction.

    This result represents the scan-derived corrected coordinate system:
    both input 4D-STEM datasets are treated as drifted scans and corrected toward a
    shared consensus frame before optional diffraction-pattern-level merge.

    Attributes
    ----------
    corrected_4dstem_0, corrected_4dstem_1 : np.ndarray | torch.Tensor
        Per-side drift-corrected 4D-STEM datasets, scan-axis-leading layout.
        Dataset 1 has already been oriented into dataset 0's display frame.
    corrected_4dstem : np.ndarray | torch.Tensor | None
        Diffraction-pattern-level average of ``corrected_4dstem_0`` and the
        oriented ``corrected_4dstem_1``. ``None`` when ``merge=False``.
    scalar_corrected_vdf : np.ndarray | None
        Scan-derived corrected VDF computed by correcting the raw alignment
        VDFs with the same operator used for 4D-STEM channels. This is not an
        external ground truth; it is the scalar virtual-image result implied
        by the learned scan drift fields.
    """
    corrected_4dstem_0: np.ndarray | torch.Tensor
    corrected_4dstem_1: np.ndarray | torch.Tensor
    corrected_4dstem: np.ndarray | torch.Tensor | None = None
    scalar_corrected_vdf: np.ndarray | None = None


def _rot90_to_image0_frame(
    dc, image_index: int = 1, reference_index: int = 0
) -> int:
    """Return the scan-axis rot90 count to show ``image_index`` like the reference.

    Two 0/90 scans differ by a 90-degree scan-axis rotation, so displaying one in
    the other's frame is a ``rot90`` whose count is the signed angle difference in
    quarter turns. ``reference_index`` defaults to image 0 (the consensus frame);
    a caller comparing against a different reference passes its index.
    """
    delta = float(
        dc.scan_direction_degrees[image_index] - dc.scan_direction_degrees[reference_index]
    )
    return (-int(round(delta / 90.0))) % 4


def integrate_virtual_detector(
    dataset,
    detector_mask: np.ndarray | torch.Tensor | None = None,
    *,
    reduce: str = "mean",
) -> np.ndarray:
    """Integrate a virtual image from a scan-axis-leading dataset.

    Sums (or averages) the selected trailing detector/channel pixels for every
    scan position, producing a 2-D scan image. With ``detector_mask=None`` it
    integrates the *whole* detector (a full/total virtual image, bright-field
    dominated for thin samples); pass a disk mask for bright field or an annulus
    for dark field. This is the single canonical virtual-image reduction for the
    drift package; every other virtual-image helper delegates here.

    QuantEM's detector backend selects the resident NumPy, Torch, CuPy, CUDA,
    or MPS reduction path. Three-dimensional spectrum images are treated as a
    one-row detector so the same backend also integrates their channel axis.

    Parameters
    ----------
    dataset : ndarray or torch.Tensor, shape ``(H, W, ...channels)``
        3-D/4-D dataset with scan axes first. numpy may be a ``np.memmap``.
    detector_mask : ndarray or torch.Tensor, optional
        Boolean mask over the trailing detector / channel axes. ``None``
        selects every channel.
    reduce : {"mean", "sum"}
        Average or sum the selected detector pixels.
    Returns
    -------
    numpy.ndarray, shape ``(H, W)``, dtype float32

    Examples
    --------
    >>> adf = integrate_virtual_detector(data, detector_mask=annulus)
    """
    try:
        from quantem.gpu.detector import masked_sum
    except ModuleNotFoundError as exc:
        if exc.name not in {"quantem.gpu", "quantem.gpu.detector"}:
            raise
        masked_sum = None

    if reduce not in {"mean", "sum"}:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")
    scan_shape = tuple(int(value) for value in dataset.shape[:2])
    detector_shape = tuple(int(value) for value in dataset.shape[2:])
    if len(detector_shape) == 1:
        detector_shape = (1, detector_shape[0])
        dataset = dataset.reshape(-1, *detector_shape)
    elif len(detector_shape) != 2:
        raise ValueError(
            "integrate_virtual_detector expects (row, col, channel) or "
            f"(row, col, detector_row, detector_col), got {tuple(dataset.shape)}"
        )
    mask = (
        np.ones(detector_shape, dtype=bool)
        if detector_mask is None
        else to_numpy(detector_mask, dtype=bool).reshape(detector_shape)
    )
    num_selected = int(mask.sum())
    if num_selected == 0:
        raise ValueError("detector_mask selects zero detector pixels")
    if isinstance(dataset, np.ndarray) and np.issubdtype(
        dataset.dtype,
        np.floating,
    ):
        flat = dataset.reshape(*scan_shape, -1)
        image = flat[..., mask.reshape(-1)].sum(axis=-1, dtype=np.float32)
    elif masked_sum is not None:
        image = masked_sum(dataset, mask).reshape(scan_shape)
    elif isinstance(dataset, torch.Tensor):
        mask_t = torch.as_tensor(mask, device=dataset.device)
        flat = dataset.reshape(*scan_shape, -1).to(torch.float32)
        image = flat[..., mask_t.reshape(-1)].sum(-1).detach().cpu().numpy()
    else:
        array = dataset.get() if hasattr(dataset, "get") else np.asarray(dataset)
        flat = array.reshape(*scan_shape, -1)
        image = flat[..., mask.reshape(-1)].sum(
            axis=-1,
            dtype=np.float32,
        )
    return image / num_selected if reduce == "mean" else image


def drift_field(self, idx: int) -> torch.Tensor:
    """Return fitted raw-frame drift for one scan in ``(row, col)`` order.

    A one-knot model returns ``(2, scan_rows)`` because its displacement is
    constant along each scanline. Multi-knot models return
    ``(2, scan_rows, scan_columns)``.

    Parameters
    ----------
    idx : int
        Scan index in the correction pair.

    Returns
    -------
    torch.Tensor
        Row and column displacement in raw scan coordinates.

    Examples
    --------
    >>> field_90 = drift.drift_field(1)
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "drift_field() requires preprocess() and correct_affine() first. "
            "Run dc.preprocess().correct_affine() (and optionally "
            ".correct_nonrigid()) before drift_field()."
        )
    return drift_knots.interpolator(self, idx).drift_raw(
        self._initial_knots[idx]
    )


def probe_positions(
    self,
    image_index: int = 0,
    *,
    corrected: bool = True,
    strip_padding: bool = True,
    plot: bool = True,
    stride: int = 16,
) -> np.ndarray:
    """Return nominal or drift-updated probe positions for one scan image.

    Positions retain the raw diffraction-pattern indexing and use ``(row,
    col)`` coordinates in the shared corrected frame. This lets iterative
    ptychography consume corrected positions without interpolating detector
    data.

    Parameters
    ----------
    image_index : int, default 0
        Scan image or 4D-STEM acquisition to describe.
    corrected : bool, default True
        Return fitted positions instead of the nominal scan grid.
    strip_padding : bool, default True
        Express positions in the original image-0 frame instead of the padded
        solver canvas.
    plot : bool, default True
        Draw the nominal and corrected positions for inspection.
    stride : int, default 16
        Subsampling used only by the plot.

    Returns
    -------
    numpy.ndarray
        Position array with shape ``(scan_rows, scan_columns, 2)``.

    Examples
    --------
    >>> positions = drift.probe_positions(image_index=0, plot=False)
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "probe_positions() requires preprocess() first. Run "
            "dc.preprocess() before exporting nominal or corrected "
            "probe positions."
        )
    index = image_index % len(self.imgs)
    knots = (
        self.knots[index]
        if corrected
        else self._initial_knots[index]
    )
    row, column = drift_knots.interpolator(self, index, knots).to_canvas()
    positions = torch.stack([row, column], dim=-1)
    if strip_padding:
        pad_row, pad_column = padding_offset(
            (self.shape[1], self.shape[2]),
            self.imgs[0].shape[:2],
        )
        positions -= torch.tensor(
            [pad_row, pad_column],
            device=positions.device,
            dtype=positions.dtype,
        )
    result = to_numpy(positions, dtype=np.float32)
    if plot:
        self.plot_probe_positions(
            image_index=index,
            strip_padding=strip_padding,
            stride=stride,
        )
    return result


@torch.inference_mode()
def corrected_virtual_images(
    self,
    image_0,
    image_1,
    *,
    output_frame: str = "scan",
) -> dict[str, np.ndarray]:
    """Correct two scalar virtual images like matching 4D-STEM channels.

    Each scalar image is treated as a one-channel dataset with scan axes first,
    corrected with the same ``grid_sample`` operator used for every diffraction
    pixel, and image 1 is oriented into image 0's display frame before the
    average. The returned ``corrected_image`` should therefore match integrating
    the same virtual detector from ``corrected_4dstem()`` output,
    up to output quantization.

    Parameters
    ----------
    image_0, image_1 : array-like
        Scalar virtual images from the two 4D-STEM acquisitions. Their scan
        shapes must match the images used to solve the correction.
    output_frame : {"scan", "canvas"}, default "scan"
        Return each corrected image in image 0's scan frame or on the shared
        padded correction canvas. Canvas output also includes per-scan and
        combined coverage arrays and averages only scans that cover each
        output pixel.

    Returns
    -------
    dict[str, np.ndarray]
        The merged ``corrected_image`` and the separately corrected
        ``corrected_image_0`` and ``corrected_image_1``, all in image 0's scan
        frame by default. Canvas output also contains ``coverage_image``,
        ``coverage_image_0``, and ``coverage_image_1``.

    Raises
    ------
    ValueError
        If ``output_frame`` is not ``"scan"`` or ``"canvas"``.

    Examples
    --------
    >>> images = drift.corrected_virtual_images(vdf_0, vdf_90)
    >>> corrected_vdf = images["corrected_image"]
    >>> canvas = drift.corrected_virtual_images(
    ...     vdf_0, vdf_90, output_frame="canvas"
    ... )
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "corrected_virtual_images() requires preprocess() and "
            "correct_affine() first."
        )
    if len(self.imgs) != 2:
        raise ValueError(
            "corrected_virtual_images() expects exactly two scan images"
        )
    if output_frame not in {"scan", "canvas"}:
        raise ValueError(
            "output_frame must be 'scan' or 'canvas'; "
            f"got {output_frame!r}"
        )
    images = [
        np.asarray(image_0, dtype=np.float32),
        np.asarray(image_1, dtype=np.float32),
    ]
    if images[0].shape != self.imgs[0].shape or images[1].shape != self.imgs[1].shape:
        raise ValueError(
            "virtual image shapes must match the raw scan images used for drift "
            f"alignment: got {images[0].shape}, {images[1].shape}; expected "
            f"{self.imgs[0].shape}, {self.imgs[1].shape}"
        )

    if output_frame == "canvas":
        canvas_shape = tuple(int(value) for value in self.shape[-2:])
        components = []
        coverages = []
        for image_index, image in enumerate(images):
            image_t = torch.as_tensor(
                image,
                device=self._device,
                dtype=torch.float32,
            )
            corrected, coverage = drift_knots.interpolator(
                self,
                image_index,
            ).warp_to_canvas(
                image_t,
                canvas_shape,
                self.kde_sigma,
                0.0,
            )
            components.append(corrected)
            coverages.append(coverage)

        valid = [coverage >= 1e-3 for coverage in coverages]
        contribution_count = valid[0].to(torch.float32) + valid[1].to(
            torch.float32
        )
        merged = torch.where(
            contribution_count > 0,
            (
                components[0] * valid[0]
                + components[1] * valid[1]
            )
            / contribution_count.clamp_min(1),
            0.0,
        )
        return {
            "corrected_image": to_numpy(merged, dtype=np.float32),
            "corrected_image_0": to_numpy(components[0], dtype=np.float32),
            "corrected_image_1": to_numpy(components[1], dtype=np.float32),
            "coverage_image": to_numpy(
                torch.maximum(coverages[0], coverages[1]),
                dtype=np.float32,
            ),
            "coverage_image_0": to_numpy(coverages[0], dtype=np.float32),
            "coverage_image_1": to_numpy(coverages[1], dtype=np.float32),
        }

    components = []
    for image_index, image in enumerate(images):
        image_t = torch.as_tensor(
            image[..., None],
            device=self._device,
            dtype=torch.float32,
        )
        corrected = apply_correction_to_dataset(
            self,
            image_t,
            image_index=image_index,
            mode="bilinear",
            chunk_size=1,
            output_dtype=torch.float32,
            output_device=self._device,
        )[..., 0]
        if image_index == 1:
            rot_k = _rot90_to_image0_frame(self, image_index=1)
            if rot_k:
                corrected = torch.rot90(corrected, k=rot_k, dims=(0, 1))
        components.append(corrected)

    merged = (components[0] + components[1]) * 0.5

    return {
        "corrected_image": to_numpy(merged, dtype=np.float32),
        "corrected_image_0": to_numpy(components[0], dtype=np.float32),
        "corrected_image_1": to_numpy(components[1], dtype=np.float32),
    }


def regional_diffraction_patterns(
    self,
    regions: Mapping[str, tuple[float, float]],
    *,
    radius_px: float = 4.0,
    datasets=None,
    stages: tuple[str, ...] = ("initial", "corrected"),
) -> dict[str, object]:
    """Average diffraction patterns from named specimen regions.

    Region membership is evaluated in the shared scan frame using either the
    nominal or drift-corrected probe positions. Detector pixels are never
    interpolated: the method averages the original diffraction patterns whose
    probe positions fall inside each circular region. This makes before/after
    comparisons test spatial indexing without changing diffraction detail.

    Parameters
    ----------
    regions : mapping of str to tuple of float
        Named region centers in shared ``(row, column)`` scan pixels.
    radius_px : float, default 4.0
        Circular region radius in scan pixels.
    datasets : sequence of two arrays, optional
        Raw scan-axis-leading 4D-STEM datasets. When omitted, use the datasets
        retained by ``DriftCorrection.from_4dstem``. Pass this explicitly when
        working from a saved correction because serialization intentionally
        excludes multi-gigabyte diffraction cubes.
    stages : tuple of str, default ("initial", "corrected")
        Probe-position stages to compare. Each entry must be ``"initial"`` or
        ``"corrected"``.

    Returns
    -------
    dict[str, object]
        ``patterns`` has shape ``(stage, region, scan, detector_row,
        detector_column)``. The result also includes ``sample_counts``,
        ``region_names``, ``region_centers_px``, ``radius_px``, ``stages``, and
        ``scan_direction_degrees``.

    Examples
    --------
    >>> regions = {"Au": (121, 220), "support region": (25, 109)}
    >>> comparison = drift.regional_diffraction_patterns(regions, radius_px=4)
    >>> comparison["patterns"].shape
    (2, 2, 2, 192, 192)
    """
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("regions must be a non-empty name-to-(row, column) mapping")
    region_names = tuple(regions)
    if any(not isinstance(name, str) or not name for name in region_names):
        raise ValueError("every region name must be a non-empty string")
    region_centers = np.asarray(list(regions.values()), dtype=np.float32)
    if region_centers.shape != (len(region_names), 2) or not np.isfinite(
        region_centers
    ).all():
        raise ValueError(
            "region centers must be finite (row, column) pairs; "
            f"got shape {region_centers.shape}"
        )
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius_px must be positive and finite, got {radius_px!r}")

    requested_stages = tuple(stages)
    valid_stages = {"initial", "corrected"}
    invalid_stages = [stage for stage in requested_stages if stage not in valid_stages]
    if not requested_stages or invalid_stages:
        raise ValueError(
            "stages must contain 'initial' and/or 'corrected'; "
            f"got {requested_stages!r}"
        )
    if len(set(requested_stages)) != len(requested_stages):
        raise ValueError(f"stages must not contain duplicates, got {requested_stages!r}")

    source_datasets = getattr(self, "_datasets", None) if datasets is None else datasets
    if source_datasets is None:
        raise RuntimeError(
            "regional_diffraction_patterns() needs the two raw 4D-STEM datasets. "
            "Pass datasets=(scan_0, scan_90) when using a saved correction."
        )
    source_datasets = tuple(source_datasets)
    if len(source_datasets) != 2 or any(dataset is None for dataset in source_datasets):
        raise ValueError(
            "regional_diffraction_patterns() expects exactly two raw 4D-STEM "
            f"datasets, got {len(source_datasets)}"
        )
    scan_shapes = [tuple(int(value) for value in data.shape[:2]) for data in source_datasets]
    expected_shapes = [tuple(int(value) for value in image.shape) for image in self.imgs]
    detector_shapes = [tuple(int(value) for value in data.shape[2:]) for data in source_datasets]
    if scan_shapes != expected_shapes:
        raise ValueError(
            "dataset scan shapes must match the images used for correction: "
            f"got {scan_shapes}, expected {expected_shapes}"
        )
    if len(detector_shapes[0]) != 2 or detector_shapes[0] != detector_shapes[1]:
        raise ValueError(
            "datasets must share one 2-D detector shape, got "
            f"{detector_shapes}"
        )

    patterns = np.empty(
        (
            len(requested_stages),
            len(region_names),
            2,
            *detector_shapes[0],
        ),
        dtype=np.float32,
    )
    sample_counts = np.empty(
        (len(requested_stages), len(region_names), 2),
        dtype=np.int32,
    )
    radius_squared = radius**2
    for stage_index, stage in enumerate(requested_stages):
        corrected = stage == "corrected"
        for scan_index, dataset in enumerate(source_datasets):
            positions = self.probe_positions(
                scan_index,
                corrected=corrected,
                strip_padding=True,
                plot=False,
            )
            for region_index, (name, center) in enumerate(
                zip(region_names, region_centers, strict=True)
            ):
                mask = (
                    (positions[..., 0] - center[0]) ** 2
                    + (positions[..., 1] - center[1]) ** 2
                    <= radius_squared
                )
                count = int(mask.sum())
                if count == 0:
                    raise ValueError(
                        f"region {name!r} at ({center[0]:g}, {center[1]:g}) with "
                        f"radius_px={radius:g} selects no samples for {stage!r} "
                        f"scan {scan_index}"
                    )
                coordinates = np.argwhere(mask)
                row_start, column_start = coordinates.min(axis=0)
                row_stop, column_stop = coordinates.max(axis=0) + 1
                local_mask = mask[
                    row_start:row_stop,
                    column_start:column_stop,
                ]
                block = dataset[
                    int(row_start):int(row_stop),
                    int(column_start):int(column_stop),
                ]
                if isinstance(block, torch.Tensor):
                    # CUDA cannot boolean-index uint16 tensors. Convert only
                    # this small region, never the full diffraction cube.
                    local_mask_t = torch.as_tensor(local_mask, device=block.device)
                    pattern = block.to(torch.float32)[local_mask_t].mean(dim=0)
                    pattern = to_numpy(pattern, dtype=np.float32)
                else:
                    if hasattr(block, "get"):
                        block = block.get()
                    pattern = np.asarray(block, dtype=np.float32)[local_mask].mean(
                        axis=0,
                        dtype=np.float32,
                    )
                patterns[stage_index, region_index, scan_index] = pattern
                sample_counts[stage_index, region_index, scan_index] = count

    return {
        "patterns": patterns,
        "sample_counts": sample_counts,
        "region_names": region_names,
        "region_centers_px": region_centers,
        "radius_px": radius,
        "stages": requested_stages,
        "scan_direction_degrees": np.asarray(
            self.scan_direction_degrees,
            dtype=np.float32,
        ),
    }


def corrected_4dstem_views(correction, *, det_bin: int = 1) -> list[np.ndarray]:
    """Prepare a compact raw-to-corrected 4D-STEM comparison.

    Detector binning before correction preserves the correction field while
    avoiding work on detector detail that the interactive viewer will discard.
    The returned stages share image 0's scan frame and the solved crop.
    """
    def detector_bin(cube):
        if det_bin == 1:
            return cube
        detector_rows, detector_columns = cube.shape[-2:]
        shape = cube.shape
        reshaped = cube.reshape(
            shape[0],
            shape[1],
            detector_rows // det_bin,
            det_bin,
            detector_columns // det_bin,
            det_bin,
        )
        if isinstance(reshaped, torch.Tensor):
            dtype = reshaped.dtype if reshaped.is_floating_point() else torch.int32
        else:
            dtype = (
                reshaped.dtype
                if np.issubdtype(reshaped.dtype, np.floating)
                else np.int32
            )
        return reshaped.sum((3, 5), dtype=dtype)

    raw_0, raw_1 = (detector_bin(dataset) for dataset in correction._datasets)
    corrected_0 = correction.apply_correction(
        raw_0,
        image_index=0,
        output_dtype="same",
        output_device=correction.device,
        verbose=False,
    )
    corrected_1 = correction.apply_correction(
        raw_1,
        image_index=1,
        output_dtype="same",
        output_device=correction.device,
        verbose=False,
    )
    quarter_turns = _rot90_to_image0_frame(correction)
    if quarter_turns:
        corrected_1 = (
            torch.rot90(corrected_1, quarter_turns, dims=(0, 1))
            if isinstance(corrected_1, torch.Tensor)
            else np.rot90(corrected_1, quarter_turns, axes=(0, 1)).copy()
        )
    if isinstance(corrected_0, torch.Tensor):
        if corrected_0.is_floating_point():
            merged = (corrected_0 + corrected_1) * 0.5
        else:
            merged = (
                (corrected_0.to(torch.int64) + corrected_1.to(torch.int64)) >> 1
            ).to(corrected_0.dtype)
    elif np.issubdtype(corrected_0.dtype, np.floating):
        merged = (corrected_0 + corrected_1) * 0.5
    else:
        merged = (
            (corrected_0.astype(np.int64) + corrected_1.astype(np.int64)) >> 1
        ).astype(corrected_0.dtype)
    rows, columns = crop_slices(correction)
    return [to_numpy(cube[rows, columns]) for cube in (raw_0, corrected_0, merged)]


def corrected_4dstem(
    self,
    *,
    mode: str = "bilinear",
    chunk_size: int | None = None,
    merge: bool = True,
    verbose: bool = True,
    output_0: np.ndarray | None = None,
    output_1: np.ndarray | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
) -> CorrectionResult:
    """Correct and optionally merge a 0/90 4D-STEM acquisition pair.

    Each acquisition receives its learned scan drift before the second scan is
    rotated into the first scan's frame. Preallocated NumPy or memmap outputs
    keep large detector datasets from requiring another full-size allocation.

    Parameters
    ----------
    mode : str, default "bilinear"
        Interpolation used along the scan axes.
    chunk_size : int or None, default None
        Detector channels corrected per batch. ``None`` selects automatically.
    merge : bool, default True
        Average the two corrected acquisitions in their shared frame.
    verbose : bool, default True
        Show progress for chunked correction and merging.
    output_0, output_1 : numpy.ndarray or None, default None
        Preallocated outputs for the two corrected acquisitions.
    output_dtype : torch.dtype, numpy dtype, str, or None, default None
        Output numeric type. Use ``"same"`` to preserve the input type.
    output_device : str, torch.device, or None, default None
        Device holding returned arrays when no preallocated output is supplied.

    Returns
    -------
    CorrectionResult
        Corrected acquisitions and their optional merged dataset.

    Examples
    --------
    >>> result = drift.corrected_4dstem(chunk_size=64)
    >>> merged = result.corrected_4dstem
    """
    if getattr(self, "_datasets", None) is None or self._reference_mode:
        raise RuntimeError(
            "corrected_4dstem() requires DriftCorrection.from_4dstem(data_0, "
            "data_1, ...). For reference-mode EDS/EELS/4D-STEM, use corrected()."
        )
    datasets = self._datasets
    if self._datasets_consumed:
        raise RuntimeError(
            "Raw datasets were already released to free device memory "
            "during a prior corrected call. Construct a new "
            "DriftCorrection to re-correct."
        )
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 datasets for scan collection correction, "
            f"got {len(datasets)}"
        )

    # When inputs are device-resident, release each raw dataset as soon as
    # its corrected output exists; otherwise we hold four full datasets
    # simultaneously, which exceeds device memory for multi-GB scan collections.
    inputs_on_device = (
        isinstance(datasets[0], torch.Tensor) and datasets[0].is_cuda
        and isinstance(datasets[1], torch.Tensor) and datasets[1].is_cuda
    )

    corrected_4dstem_0 = apply_correction_to_dataset(
        self, None, image_index=0, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_0, verbose=verbose,
        progress_desc="Correcting scan 1/2",
    )
    if inputs_on_device:
        self._datasets[0] = None
        torch.cuda.empty_cache()
    corrected_4dstem_1 = apply_correction_to_dataset(
        self, None, image_index=1, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_1, verbose=verbose,
        progress_desc="Correcting scan 2/2",
    )
    if inputs_on_device:
        self._datasets[1] = None
        self._datasets_consumed = True
        torch.cuda.empty_cache()

    rot_k = _rot90_to_image0_frame(self, image_index=1)
    if rot_k:
        if isinstance(corrected_4dstem_1, torch.Tensor):
            corrected_4dstem_1 = torch.rot90(
                corrected_4dstem_1, k=rot_k, dims=(0, 1),
            )
        else:
            corrected_4dstem_1 = np.rot90(
                corrected_4dstem_1, k=rot_k, axes=(0, 1),
            ).copy()

    corrected_4dstem = None
    if merge:
        if corrected_4dstem_0.shape != corrected_4dstem_1.shape:
            raise ValueError(
                f"Cannot merge: corrected_4dstem_0 shape {corrected_4dstem_0.shape} "
                f"!= corrected_4dstem_1 shape {corrected_4dstem_1.shape}. "
                f"Scan collection must have compatible scan dimensions "
                f"after correction and scan-angle rotation."
            )
        Hm = corrected_4dstem_0.shape[0]
        row_block = max(1, min(32, Hm))
        row_starts = range(0, Hm, row_block)
        merge_progress = tqdm(
            total=Hm,
            desc="Merging corrected scans",
            unit="row",
            disable=not verbose or len(row_starts) <= 1,
        )
        if isinstance(corrected_4dstem_0, torch.Tensor):
            merge_dtype = (
                corrected_4dstem_0.dtype
                if corrected_4dstem_0.is_floating_point()
                else torch.float32
            )
            corrected_4dstem = torch.empty(
                corrected_4dstem_0.shape,
                dtype=merge_dtype,
                device=corrected_4dstem_0.device,
            )
            for r0 in row_starts:
                r1 = min(r0 + row_block, Hm)
                corrected_4dstem[r0:r1] = (
                    corrected_4dstem_0[r0:r1].to(merge_dtype)
                    + corrected_4dstem_1[r0:r1].to(merge_dtype)
                ) * 0.5
                merge_progress.update(r1 - r0)
        else:
            corrected_4dstem = np.empty_like(corrected_4dstem_0, dtype=np.float32)
            for r0 in row_starts:
                r1 = min(r0 + row_block, Hm)
                np.add(
                    corrected_4dstem_0[r0:r1],
                    corrected_4dstem_1[r0:r1],
                    out=corrected_4dstem[r0:r1],
                    dtype=np.float32,
                )
                corrected_4dstem[r0:r1] *= 0.5
                merge_progress.update(r1 - r0)
        merge_progress.close()

    # Extract raw VDFs from the stored alignment images. The scan collection
    # reference is the scalar channel correction implied by the learned scan
    # drift fields, not an external ground truth.
    alignment_vdf_0 = np.asarray(self.imgs[0].array)
    alignment_vdf_1 = np.asarray(self.imgs[1].array)
    scalar_corrected_vdf = corrected_virtual_images(
        self,
        alignment_vdf_0,
        alignment_vdf_1,
    )["corrected_image"]

    return CorrectionResult(
        corrected_4dstem=corrected_4dstem,
        corrected_4dstem_0=corrected_4dstem_0,
        corrected_4dstem_1=corrected_4dstem_1,
        scalar_corrected_vdf=scalar_corrected_vdf,
    )


def to_numpy(array, *, dtype=None):
    """Convert a device or host array to NumPy with optional dtype conversion."""
    if isinstance(array, torch.Tensor):
        result = (
            array.detach().cpu().numpy()
            if array.is_cuda or array.device.type == "mps"
            else array.detach().numpy()
        )
    elif hasattr(array, "get"):  # CuPy ndarray
        result = array.get()
    else:
        result = np.asarray(array)
    return result.astype(dtype) if dtype is not None else result
