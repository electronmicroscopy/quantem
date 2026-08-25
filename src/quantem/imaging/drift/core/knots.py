"""Scanline-knot geometry and forward warping.

:class:`DriftKnot` keeps the single- and multi-knot coordinate models consistent.
It provides per-pixel canvas coordinates (:meth:`to_canvas`), raw-frame drift
(:meth:`drift_raw`), in-place affine slope (:meth:`apply_affine_shift`),
and forward scatter onto the canvas (:meth:`warp_to_canvas`).

The forward-warp kernels live here too because forward scatter is what
the knot *does* to a source image:

* :func:`bilinear_kde_batch` — batched bilinear forward scatter.
* :func:`gaussian_smooth_batch` / :func:`gaussian_smooth_1d` — separable
  Gaussian smoothing reused by KDE normalization and (1-D) by the
  knot regularizer in :mod:`nonrigid`.
* :func:`initialize_scanline_knots` — initial knot grid for ``preprocess``.
* Private :func:`_transform_coordinates_single_knot` /
  :func:`_transform_coordinates_multi_knot` — the K-specific coordinate
  formulas the class dispatches between.

Backward warping, cross-correlation, and translation alignment are a
separate concerns and live in :mod:`warping`.
"""
import numpy as np
import torch
from numpy.typing import NDArray


def interpolator(correction, image_index: int, knots: torch.Tensor | None = None):
    """Map one scan's knot positions into its padded correction canvas.

    The knot geometry is shared by affine fitting, non-rigid warping, corrected
    products, and probe-position recovery. Keeping that geometry here gives
    every stage the same row/column coordinate transformation.
    """
    if knots is None:
        knots = correction.knots[image_index]
    return DriftKnot(
        knots,
        correction.scan_fast_t[image_index],
        correction.scan_slow_t[image_index],
        correction.imgs[image_index].shape,
    )


def knot_delta_canvas(correction, image_index: int) -> torch.Tensor:
    """Return one scan's fitted knot displacement on the correction canvas."""
    if not hasattr(correction, "_initial_knots"):
        raise RuntimeError(
            "apply_correction() requires preprocess() and correct_affine() "
            "first. Run dc.preprocess().correct_affine() (and optionally "
            ".correct_nonrigid()) before apply_correction()."
        )
    return correction.knots[image_index] - correction._initial_knots[image_index]


def stage_knots(correction, stage: str | None) -> list[torch.Tensor]:
    """Select the saved knot field for a correction stage comparison."""
    if stage in (None, "nonrigid", "non-rigid"):
        return correction.knots
    attribute = {
        "initial": "_initial_knots",
        "raw": "_initial_knots",
        "affine": "_knots_after_affine",
        "strip": "_knots_after_strip",
    }.get(stage)
    if attribute is None:
        raise ValueError(
            f"Unknown correction stage {stage!r}. Choose initial, affine, "
            "strip, nonrigid, or None for the current result."
        )
    knots = getattr(correction, attribute, None)
    if knots is None:
        required = "preprocess" if stage in ("initial", "raw") else f"correct_{stage}"
        raise ValueError(f"stage={stage!r} needs a prior {required} call.")
    return knots


def initialize_scanline_knots(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    scan_fast: NDArray,
    scan_slow: NDArray,
    number_knots: int,
) -> NDArray:
    """Build the initial knot grid used by ``DriftCorrection.preprocess``.

    The knot anchors define where each scanline starts on the padded canvas
    before any affine or non-rigid optimization. For ``number_knots == 1``,
    this is a vertical line of anchors at the fast-scan start edge of the
    centered footprint. The full scanline width is then added later by
    ``_transform_coordinates_single_knot``.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Raw image shape ``(num_rows, num_cols)``.
    output_shape : tuple[int, int]
        Padded canvas shape ``(num_rows, num_cols)``.
    scan_fast : NDArray
        Unit vector of the fast scan direction in ``(row, col)`` order.
    scan_slow : NDArray
        Unit vector of the slow scan direction in ``(row, col)`` order.
    number_knots : int
        Number of control knots per scanline.

    Returns
    -------
    NDArray
        Initial knot array with shape ``(2, input_rows, number_knots)``.
    """
    v_slow = np.linspace(-(input_shape[0] - 1) / 2, (input_shape[0] - 1) / 2, input_shape[0])
    u_fast = np.linspace(-(input_shape[1] - 1) / 2, (input_shape[1] - 1) / 2, number_knots)
    row_knots = ((output_shape[0] - 1) / 2
                 + u_fast[None, :] * scan_fast[0]
                 + v_slow[:, None] * scan_slow[0])
    col_knots = ((output_shape[1] - 1) / 2
                 + u_fast[None, :] * scan_fast[1]
                 + v_slow[:, None] * scan_slow[1])
    return np.stack([row_knots, col_knots], axis=0)


def resize_scanline_knots(correction, num_knots: int):
    """Change fast-scan knot density without changing the fitted drift field

    Affine and strip correction establish a displacement field before a
    scientist decides how much fast-scan flexibility the non-rigid stage
    needs. Resampling the displacement at a new knot density lets
    ``correct_nonrigid(num_knots=...)`` retain that corrected geometry instead
    of repeating affine correction or discarding its result.

    Parameters
    ----------
    correction : DriftCorrection
        Prepared correction containing the current and initial knot fields.
    num_knots : int
        New number of knots along every fast-scan line.

    Returns
    -------
    DriftCorrection
        The same correction with every saved checkpoint represented at the
        requested knot density.
    """
    count = int(num_knots)
    if count < 1:
        raise ValueError(f"num_knots must be >= 1, got {num_knots!r}.")
    current = {int(value.shape[2]) for value in correction.knots}
    if current == {count}:
        return correction
    if len(current) != 1:
        raise ValueError(
            "All scans must use the same knot count before resizing; "
            f"got {sorted(current)}."
        )

    old_initial = correction._initial_knots
    new_initial = [
        torch.as_tensor(
            initialize_scanline_knots(
                input_shape=correction.imgs[index].shape,
                output_shape=correction.shape[1:],
                scan_fast=correction.scan_fast[index],
                scan_slow=correction.scan_slow[index],
                number_knots=count,
            ),
            dtype=correction._dtype,
            device=correction._device,
        )
        for index in range(correction.shape[0])
    ]

    def resize_checkpoint(checkpoint):
        resized = []
        for value, initial, target in zip(
            checkpoint,
            old_initial,
            new_initial,
            strict=True,
        ):
            displacement = value - initial
            if displacement.shape[2] == 1:
                displacement = displacement.expand(-1, -1, count)
            else:
                rows = displacement.shape[1]
                displacement = torch.nn.functional.interpolate(
                    displacement.reshape(1, 2 * rows, -1),
                    size=count,
                    mode="linear",
                    align_corners=True,
                ).reshape(2, rows, count)
            resized.append(target + displacement)
        return resized

    checkpoints = {
        name: resize_checkpoint(getattr(correction, name))
        for name in ("knots", "_knots_after_affine", "_knots_after_strip")
        if hasattr(correction, name)
    }
    correction._initial_knots = new_initial
    for name, values in checkpoints.items():
        setattr(correction, name, values)
    correction.number_knots = count
    correction.preprocess_info["num_knots"] = count
    correction._images_warped_stale = True
    return correction


def _transform_coordinates_single_knot(
    knots: torch.Tensor,
    scan_fast: torch.Tensor,
    input_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-knot fast path: map source pixels to canvas coordinates.

    **Single-knot only.** Each scanline has exactly one (row, col) anchor;
    the fast-scan-direction position is filled in by linear interpolation
    along the scanline.  Multi-knot input is handled by
    :func:`_transform_coordinates_multi_knot`; :class:`DriftKnot` dispatches
    on ``knots.shape[2]``.

    Each input row maps to a line on the canvas:
    ``row = knot_row + fraction * scan_fast[0] * (num_rows - 1)``
    ``col = knot_col + fraction * scan_fast[1] * (num_cols - 1)``
    where row and col dimensions scale independently for non-square images.

    Parameters
    ----------
    knots : torch.Tensor
        Knot positions, shape ``(2, num_rows, 1)``. First dim is (row, col).
    scan_fast : torch.Tensor
        Fast scan direction vector, shape ``(2,)``.
    input_shape : tuple[int, int]
        Original image shape ``(num_rows, num_cols)``.

    Returns
    -------
    row_coords : torch.Tensor
        Row coordinates on canvas, shape ``(num_rows, num_cols)``.
    col_coords : torch.Tensor
        Column coordinates on canvas, shape ``(num_rows, num_cols)``.
    """
    num_rows, num_cols = input_shape
    fast_fraction = torch.linspace(0, 1, num_cols, dtype=knots.dtype, device=knots.device)
    row_coords = knots[0, :, 0:1] + fast_fraction[None, :] * scan_fast[0] * (num_rows - 1)
    col_coords = knots[1, :, 0:1] + fast_fraction[None, :] * scan_fast[1] * (num_cols - 1)
    return row_coords, col_coords


def _transform_coordinates_multi_knot(
    knots: torch.Tensor,
    input_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-knot path: linearly interpolate K knot anchors per scanline.

    For ``K`` knots per scanline (``K >= 2``), knot ``k`` sits at fast-axis
    fraction ``k / (K - 1)`` of the scanline.  Per-pixel canvas coordinates
    are obtained by piecewise-linear interpolation between the two adjacent
    knots.  The K=1 case is intentionally not handled here — it has no
    end-knot to interpolate toward, so the caller dispatches to
    :func:`_transform_coordinates_single_knot` (which walks along
    ``scan_fast`` instead).

    With ``K = 2`` and the initial knot grid (knots at the start and end
    of each scanline along ``scan_fast``), this reduces to the same
    per-pixel canvas positions as the K=1 path — verified in tests.

    Parameters
    ----------
    knots : torch.Tensor
        Knot positions, shape ``(2, num_rows, K)`` with ``K >= 2``.
        First axis is ``(row, col)``.
    input_shape : tuple[int, int]
        Original image shape ``(num_rows, num_cols)``.  Only ``num_cols``
        is consulted (``num_rows`` is implicit in ``knots.shape[1]``).

    Returns
    -------
    row_coords : torch.Tensor
        Row coordinates on canvas, shape ``(num_rows, num_cols)``.
    col_coords : torch.Tensor
        Column coordinates on canvas, shape ``(num_rows, num_cols)``.
    """
    _, num_cols = input_shape
    K = knots.shape[2]
    t = torch.linspace(0, 1, num_cols, dtype=knots.dtype, device=knots.device) * (K - 1)
    seg = torch.clamp(t.long(), max=K - 2)
    frac = (t - seg.to(knots.dtype))[None, :]
    row_lo = knots[0, :, seg]
    row_hi = knots[0, :, seg + 1]
    col_lo = knots[1, :, seg]
    col_hi = knots[1, :, seg + 1]
    row_coords = row_lo + (row_hi - row_lo) * frac
    col_coords = col_lo + (col_hi - col_lo) * frac
    return row_coords, col_coords


class DriftKnot:
    """Maps K knot anchors per scanline to canvas geometry, drift, and warps.

    This is the single dispatch point for K=1 versus K>=2 geometry.
    K=1 walks along ``scan_fast`` (one anchor per row, scanline geometry
    implicit); K>=2 piecewise-linearly interpolates the K anchors along
    the fast axis. :func:`interpolator` builds the geometry for each scan.

    Attributes
    ----------
    knots : torch.Tensor
        Knot anchors, shape ``(2, H, K)`` (row/col × scanline × knot).
    scan_fast, scan_slow : torch.Tensor
        Fast / slow scan unit vectors ``(2,)``.  ``scan_fast`` is only
        consulted when ``K == 1`` (walk); ``scan_slow`` enters
        :meth:`drift_raw` to invert the canvas Jacobian.
    input_shape : tuple[int, int]
        Source image shape ``(H, W)``.
    K : int
        Number of knots per scanline (cached from ``knots.shape[2]``).
    """

    def __init__(
        self,
        knots: torch.Tensor,
        scan_fast: torch.Tensor,
        scan_slow: torch.Tensor,
        input_shape: tuple[int, int],
    ):
        self.knots = knots
        self.scan_fast = scan_fast
        self.scan_slow = scan_slow
        self.input_shape = input_shape
        self.K = knots.shape[2]

    def to_canvas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel canvas (row, col) coordinates for warping.

        Returns ``(row_coords, col_coords)`` each ``(H, W)``.  K=1 uses the
        ``scan_fast`` walk fast path; K>=2 uses the multi-knot lerp.
        """
        if self.K == 1:
            return _transform_coordinates_single_knot(
                self.knots, self.scan_fast, self.input_shape)
        return _transform_coordinates_multi_knot(self.knots, self.input_shape)

    def _drift_canvas(self, initial_knots: torch.Tensor) -> torch.Tensor:
        """Drift in canvas coordinates relative to ``initial_knots``.

        Internal step used by :meth:`drift_raw`.  Returns ``(2, H)`` for
        K=1 (per-row shift) or ``(2, H, W)`` for K>=2 (per-pixel via lerp).
        Callers want raw-frame drift, not canvas-frame, so this method
        stays private.
        """
        delta = self.knots - initial_knots
        if self.K == 1:
            return delta[:, :, 0]
        _, num_cols = self.input_shape
        t = torch.linspace(0, 1, num_cols, dtype=delta.dtype, device=delta.device) * (self.K - 1)
        seg = torch.clamp(t.long(), max=self.K - 2)
        frac = (t - seg.to(delta.dtype))[None, :]
        delta_row = delta[0, :, seg] + (delta[0, :, seg + 1] - delta[0, :, seg]) * frac
        delta_col = delta[1, :, seg] + (delta[1, :, seg + 1] - delta[1, :, seg]) * frac
        return torch.stack([delta_row, delta_col])

    def drift_raw(self, initial_knots: torch.Tensor) -> torch.Tensor:
        """Drift relative to ``initial_knots`` in raw-frame coordinates.

        Returns ``(2, H)`` for K=1 (per-row shift) or ``(2, H, W)`` for K>=2
        (per-pixel).  Inverts the canvas Jacobian so callers feed the
        result straight into ``backward_warp``.  For square images this
        reduces to a rotation by the scan angle; the ``alpha`` factor
        handles non-square scans.
        """
        delta_canvas = self._drift_canvas(initial_knots)
        scan_h, scan_w = self.input_shape
        alpha = float(scan_h - 1) / float(scan_w - 1) if scan_w > 1 else 1.0
        det = (self.scan_slow[0] * self.scan_fast[1]
               - self.scan_fast[0] * alpha * self.scan_slow[1])
        drift_row = (
            self.scan_fast[1] * delta_canvas[0]
            - self.scan_fast[0] * alpha * delta_canvas[1]
        ) / det
        drift_col = (
            -self.scan_slow[1] * delta_canvas[0]
            + self.scan_slow[0] * delta_canvas[1]
        ) / det
        return torch.stack([drift_row, drift_col])

    def affine_candidate_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-pixel canvas coords plus the scanline-centered offset axis.

        Returns ``(row_base, col_base, scanline_offset)``.  The first two
        are :meth:`to_canvas`'s output; the third is
        ``arange(H) - (H-1)/2`` shaped ``(H,)`` so callers can broadcast a
        candidate ``drift_vec`` over scanlines as
        ``row_base + drift_vec[0] * scanline_offset[:, None]``.

        Used by ``affine.grid_search_batch`` so candidate-broadcast geometry
        construction lives on the geometry class instead of the orchestrator.
        """
        row_base, col_base = self.to_canvas()
        H = self.knots.shape[1]
        scanline_offset = (
            torch.arange(H, dtype=self.knots.dtype, device=self.knots.device)
            - (H - 1) / 2
        )
        return row_base, col_base, scanline_offset

    def apply_affine_shift(self, drift_vec: torch.Tensor) -> None:
        """Add an affine drift slope to every knot, in place.

        For each scanline ``i``, shifts the knots by
        ``drift_vec * (i - (H - 1) / 2)`` so the slow-axis-centered slope
        accumulates linearly across the image.  Used by ``correct_affine``
        to bake a per-row drift candidate into the knot grid.
        """
        H = self.knots.shape[1]
        scanline_offset = (
            torch.arange(H, dtype=self.knots.dtype, device=self.knots.device)
            - (H - 1) / 2
        )[:, None]
        self.knots[0] += drift_vec[0] * scanline_offset
        self.knots[1] += drift_vec[1] * scanline_offset

    def warp_to_canvas(
        self,
        source_image: torch.Tensor,
        canvas_shape: tuple[int, int],
        kde_sigma: float,
        pad_value,
        upsample_factor: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-scatter ``source_image`` onto the canvas using the knot grid.

        Wraps :meth:`to_canvas` + :func:`bilinear_kde_batch` so callers
        don't repeat the (knots → coords → scatter) idiom.  Returns
        ``(warped, weights)``, each ``canvas_shape``.

        ``upsample_factor`` scales the canvas coordinates and KDE sigma in
        lockstep, so ``corrected`` can scatter at a finer grid
        without recomputing the interpolation.
        """
        row_t, col_t = self.to_canvas()
        if upsample_factor != 1:
            row_t = row_t * upsample_factor
            col_t = col_t * upsample_factor
        warped, weights = bilinear_kde_batch(
            row_t[None], col_t[None], source_image, canvas_shape,
            kde_sigma, pad_value)
        return warped[0], weights[0]


# ---------------------------------------------------------------------------
# Forward-warp kernels: scatter source pixels onto the canvas, smooth, normalize.
# Used by :meth:`DriftKnot.warp_to_canvas` (and ``affine.grid_search_batch``'s
# candidate broadcast which calls bilinear_kde_batch directly).
# ---------------------------------------------------------------------------


def bilinear_kde_batch(
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    source_image: torch.Tensor,
    output_shape: tuple[int, int],
    kde_sigma: float,
    pad_value: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched bilinear KDE: scatter N source images onto an output canvas.

    Each pixel scatters its value to its 4 nearest grid neighbors with
    bilinear weights ``(1-dr)·(1-dc)``, ``dr·(1-dc)``, ``(1-dr)·dc``,
    ``dr·dc`` where ``dr, dc`` are fractional row/col distances.
    Accumulated counts and values are Gaussian-smoothed, then normalized:
    ``output = pad_value·(1-coverage) + coverage·(values/counts)``.

    Used by both the affine grid search (N = candidate drift vectors,
    single source image broadcast across drifts) and the nonrigid loop
    (N = stacked source images, one per drift).

    Parameters
    ----------
    row_coords : torch.Tensor
        Row coordinates of input pixels, shape ``(N, rows, cols)``.
    col_coords : torch.Tensor
        Column coordinates of input pixels, shape ``(N, rows, cols)``.
    source_image : torch.Tensor
        Pixel values to scatter. Either ``(rows, cols)`` (same image used
        for all N drifts - affine grid search) or ``(N, rows, cols)``
        (different image per drift - multi-image batched warping).
    output_shape : tuple[int, int]
        Canvas size ``(num_rows, num_cols)`` for the output images.
    kde_sigma : float
        Gaussian smoothing sigma in pixels.
    pad_value : float or torch.Tensor
        Fill value where pixel coverage is below threshold. If a tensor of
        shape ``(N,)``, applies a different pad value per drift.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(warped_images, sum_weights)`` - warped images and smoothed pixel coverage,
        both shape ``(N, num_rows, num_cols)``.
    """
    num_test_drifts = row_coords.shape[0]
    num_rows, num_cols = output_shape
    coverage_threshold = 1e-3
    # Flatten spatial dims - scatter_add_ works on 1D buffers
    row_flat = row_coords.flatten(1)
    col_flat = col_coords.flatten(1)
    # Stay in float for fractional distance, convert to int only for scatter indices
    row_floor = row_flat.floor()
    col_floor = col_flat.floor()
    frac_row = row_flat - row_floor
    frac_col = col_flat - col_floor
    row_floor = row_floor.int()
    col_floor = col_floor.int()
    if source_image.dim() == 3:
        # Per-drift source images: each drift scatters its own pixel values
        source_values_flat = source_image.flatten()
    else:
        source_values_flat = source_image.flatten().repeat(num_test_drifts)
    # All N batch entries scatter into one flat buffer - offset separates them
    batch_offsets = (
        torch.arange(num_test_drifts, device=row_coords.device, dtype=torch.int32)
        * num_rows * num_cols
    )[:, None]
    # Float32 accumulators - scatter_add_ requires source dtype to match,
    # so all input tensors must be float32 (raises on float64).
    sum_weights = torch.zeros(
        num_test_drifts * num_rows * num_cols, dtype=torch.float32, device=row_coords.device
    )
    sum_values = torch.zeros_like(sum_weights)
    # Periodic wrapping so pixels near edges scatter to the opposite side
    row_wrapped = row_floor % num_rows
    col_wrapped = col_floor % num_cols
    row_next = (row_wrapped + 1) % num_rows
    col_next = (col_wrapped + 1) % num_cols
    # Each pixel distributes its value to the 4 nearest grid neighbors
    # weighted by bilinear distance: (1-dr)(1-dc), dr(1-dc), (1-dr)dc, dr·dc
    for corner_row, corner_col, corner_weight in [
        (row_wrapped, col_wrapped, ((1 - frac_row) * (1 - frac_col)).flatten()),
        (row_next, col_wrapped, (frac_row * (1 - frac_col)).flatten()),
        (row_wrapped, col_next, ((1 - frac_row) * frac_col).flatten()),
        (row_next, col_next, (frac_row * frac_col).flatten()),
    ]:
        flat_indices = (corner_row * num_cols + corner_col + batch_offsets).flatten()
        sum_weights.scatter_add_(0, flat_indices, corner_weight)
        sum_values.scatter_add_(0, flat_indices, corner_weight * source_values_flat)
    sum_weights = sum_weights.reshape(num_test_drifts, num_rows, num_cols)
    sum_values = sum_values.reshape(num_test_drifts, num_rows, num_cols)
    # Smooth the scattered counts and values to fill gaps between pixels
    sum_weights = gaussian_smooth_batch(sum_weights, kde_sigma)
    sum_values = gaussian_smooth_batch(sum_values, kde_sigma)
    # Blend between pad_value (uncovered) and normalized values (covered),
    # ramping linearly with coverage to avoid hard edges at the boundary
    coverage_weight = torch.clamp(sum_weights / coverage_threshold, max=1.0)
    if isinstance(pad_value, torch.Tensor) and pad_value.dim() == 1:
        # Per-drift pad value: reshape (N,) → (N, 1, 1) for broadcasting
        pad_value = pad_value[:, None, None]
    warped_images = pad_value * (1 - coverage_weight) + coverage_weight * (
        sum_values / torch.clamp(sum_weights, min=1e-8)
    )
    return warped_images, sum_weights
def gaussian_smooth_batch(
    field_stack: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Batched 2D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Used by ``bilinear_kde_batch`` to smooth scattered counts and
    values before normalization. Without smoothing, the warped images
    have salt-and-pepper artifacts from the scatter step.

    Parameters
    ----------
    field_stack : torch.Tensor
        Input tensor of shape ``(N, num_rows, num_cols)``.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, num_rows, num_cols)``.

    """
    kernel, radius = _gaussian_kernel_1d(sigma, field_stack.dtype, field_stack.device)
    # Separable kernel: column pass then row pass to halve FLOPs vs full 2D conv
    kernel_col = kernel[None, None, None, :]
    kernel_row = kernel[None, None, :, None]
    field_stack = field_stack[:, None]
    field_stack = torch.nn.functional.conv2d(_symmetric_pad(field_stack, 0, radius), kernel_col)
    field_stack = torch.nn.functional.conv2d(_symmetric_pad(field_stack, radius, 0), kernel_row)
    return field_stack[:, 0]


def gaussian_smooth_1d(
    signal: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """1D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Smooths each row of the input independently using a separable 1D kernel.
    Used for regularizing knot displacement vectors in the nonrigid loop,
    where the signal is 1D (one value per scan line).

    Parameters
    ----------
    signal : torch.Tensor
        Input tensor of shape ``(N, L)`` - N channels, L samples.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, L)``.
    """
    kernel, radius = _gaussian_kernel_1d(sigma, signal.dtype, signal.device)
    signal_padded = _symmetric_pad_1d(signal[:, None], radius)
    return torch.nn.functional.conv1d(signal_padded, kernel[None, None, :])[:, 0]
def _symmetric_pad_1d(signal: torch.Tensor, pad: int) -> torch.Tensor:
    """Symmetric 1D padding matching scipy's reflect mode.

    Same edge-repeat semantics as ``_symmetric_pad`` but for 1D signals.
    Used by ``gaussian_smooth_1d`` for regularization of knot vectors.
    """
    if pad <= 0:
        # signal[:, :, -0:] is the whole signal, not an empty slice, so a naive
        # tail slice would double the length. No padding needed when pad == 0.
        return signal
    left = signal[:, :, :pad].flip(-1)
    right = signal[:, :, -pad:].flip(-1)
    return torch.cat([left, signal, right], dim=-1)


def _symmetric_pad(
    field_stack: torch.Tensor,
    pad_rows: int,
    pad_cols: int,
) -> torch.Tensor:
    """Symmetric padding matching scipy's reflect mode for parity.

    Without this, the torch and numpy Gaussian smoothing paths produce
    different results near canvas edges, breaking numerical parity.

    Scipy's ``mode='reflect'`` repeats the edge pixel
    (``[1,2,3]`` → ``[2,1,1,2,3,3,2]``), but PyTorch's
    ``F.pad(mode='reflect')`` does not (``[1,2,3]`` → ``[3,2,1,2,3,2,1]``).

    Parameters
    ----------
    field_stack : torch.Tensor
        Input tensor of shape ``(N, C, num_rows, num_cols)``.
    pad_rows : int
        Number of rows to pad on top and bottom.
    pad_cols : int
        Number of columns to pad on left and right.

    Returns
    -------
    torch.Tensor
        Padded tensor.

    Examples
    --------
    >>> t = torch.tensor([[[[1., 2., 3.]]]])
    >>> _symmetric_pad(t, 0, 2)
    tensor([[[[2., 1., 1., 2., 3., 3., 2.]]]])
    """
    if pad_cols > 0:
        left = field_stack[:, :, :, :pad_cols].flip(-1)
        right = field_stack[:, :, :, -pad_cols:].flip(-1)
        field_stack = torch.cat([left, field_stack, right], dim=-1)
    if pad_rows > 0:
        top = field_stack[:, :, :pad_rows, :].flip(-2)
        bottom = field_stack[:, :, -pad_rows:, :].flip(-2)
        field_stack = torch.cat([top, field_stack, bottom], dim=-2)
    return field_stack


def _gaussian_kernel_1d(sigma: float, dtype: torch.dtype, device: torch.device, _cache: dict = {}) -> torch.Tensor:
    """Normalized 1D Gaussian ``exp(-0.5*(x/sigma)^2)``, radius ``4*sigma``.

    Cached via mutable default arg - the grid search calls this ~800 times
    with the same sigma, saving ~44ms of redundant kernel construction.
    """
    key = (sigma, dtype, device)
    if key not in _cache:
        radius = int(4 * sigma + 0.5)
        offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        _cache[key] = (kernel / kernel.sum(), radius)
    return _cache[key]
