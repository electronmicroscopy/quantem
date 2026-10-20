"""Measuring how far each tilt is shifted, on the detector and in the scan.

MAPED aligns each tilt on two small images: its mean diffraction pattern
(detector shift) and a corresponding real-space image (scan shift). Both use the same
registration: the peak of the Fourier cross-correlation, refined to a fraction
of a pixel by evaluating the correlation on a finer grid around it
(Guizar-Sicairos et al., Opt. Lett. 33, 156 (2008)).
"""

from math import ceil, pi

import torch.nn.functional as F
from scipy.signal.windows import tukey
from torch import (
    Tensor,
    arange,
    cat,
    exp,
    get_default_dtype,
    hann_window,
    int32,
    sqrt,
    stack,
    tensor,
    where,
    zeros,
    zeros_like,
)
from torch.fft import fft2, fftfreq, ifft2, ifftshift
from torchvision.transforms import GaussianBlur


def beam_centers(patterns: list[Tensor], blur: float | None) -> Tensor:
    """Brightest pixel of each mean diffraction pattern, as (num_tilts, 2) int32 (row, col).

    Blurring first (standard deviation ``blur`` pixels) keeps one noisy pixel
    from outshining the direct beam.
    """
    stacked = stack(patterns)
    if blur:
        stacked = gaussian_blur(stacked, blur)
    brightest = stacked.flatten(1).argmax(dim=1)
    num_cols = stacked.shape[-1]
    return stack([brightest // num_cols, brightest % num_cols], dim=-1).to(int32)


def diffraction_shifts(patterns: list[Tensor], border_taper: float, precision: float) -> Tensor:
    """Detector shift of each tilt, (num_tilts, 2) (row, col) in pixels, relative to their mean.

    Tilt ``t`` is matched against the average of the tilts before it, then
    joins that average, moved by the Fourier phase ramp
    ``exp(-2 pi i (k_row s_row + k_col s_col))``, which shifts exactly. A Tukey
    window fades each pattern over ``border_taper`` pixels so the detector edge
    is not matched.
    """
    num_rows, num_cols = patterns[0].shape
    upsample_factor = round(1 / precision)
    with patterns[0].device:
        window = tukey_window_2d((num_rows, num_cols), border_taper)
        freq_row = fftfreq(num_rows)[:, None]
        freq_col = fftfreq(num_cols)
        shifts = zeros((len(patterns), 2))
        reference_ft = fft2(window * patterns[0])
        for index in range(1, len(patterns)):
            moving_ft = fft2(window * patterns[index])
            shift = cross_correlation_shift(reference_ft, moving_ft[None], upsample_factor)[0]
            shifts[index] = shift
            moved_ft = moving_ft * exp(-2j * pi * (freq_row * shift[0] + freq_col * shift[1]))
            reference_ft = reference_ft * (index / (index + 1)) + moved_ft / (index + 1)
    return shifts - shifts.mean(dim=0)


def scan_shifts(
    images: list[Tensor],
    iterations: int,
    max_shift: float,
    edge_blur: float | None,
    hann: bool,
    precision: float,
) -> Tensor:
    """Scan shift of each tilt, (num_tilts, 2) (row, col) in pixels, relative to their mean.

    Each iteration moves every image by its current shift, averages them, and
    corrects each shift by the image's offset from that average; tilt 0 holds
    the frame still meanwhile. The images are padded by ``max_shift + 4``
    pixels so a shifted feature does not wrap around. With ``edge_blur`` they
    are replaced by their edge strength, so slow brightness changes do not pull
    the shift; ``hann`` fades them out at the scan border so the frame is not
    matched.
    """
    num_rows, num_cols = images[0].shape
    padding = ceil(max_shift) + 4
    inside = (slice(padding, padding + num_rows), slice(padding, padding + num_cols))
    upsample_factor = round(1 / precision)
    with images[0].device:
        window = zeros((num_rows + 2 * padding, num_cols + 2 * padding))
        window[inside] = hann_window(num_rows)[:, None] * hann_window(num_cols) if hann else 1.0
        signals = zeros((len(images), *window.shape))
        for signal, image in zip(signals, images):
            signal[inside] = image if edge_blur is None else edge_strength(image, edge_blur)
        shifts = zeros((len(images), 2))
        for _ in range(iterations):
            shifted = shift_images(signals, shifts)
            means = (shifted * window).sum(dim=(1, 2)) / window.sum()
            images_ft = fft2((shifted - means[:, None, None]) * window)
            reference_ft = images_ft.mean(dim=0)
            shifts[1:] += cross_correlation_shift(reference_ft, images_ft[1:], upsample_factor)
            shifts = shifts - shifts[0]
    return shifts - shifts.mean(dim=0)


def cross_correlation_shift(
    reference_ft: Tensor, moving_ft: Tensor, upsample_factor: int
) -> Tensor:
    """Sub-pixel (row, col) shift that moves each image onto its reference, (num_pairs, 2).

    The cross-correlation ``ifft(F_ref * conj(F_mov))`` peaks at the shift
    between the images. Its integer peak, refined by a parabola, is trusted to
    half a pixel. The correlation is then evaluated on a 1.5 px patch
    ``upsample_factor`` times finer around it, as two small matrix products
    instead of an FFT ``upsample_factor`` times larger, which resolves
    ``1 / upsample_factor`` px. Inputs are Fourier transforms; the reference
    broadcasts against ``(num_pairs, num_rows, num_cols)``. Shifts wrap into
    ``[-n/2, n/2)``.
    """
    correlation_ft = reference_ft * moving_ft.conj()
    num_pairs, num_rows, num_cols = correlation_ft.shape
    with correlation_ft.device:
        peak, offset = _peak(ifft2(correlation_ft).real)
        shift = ((peak + offset) * 2).round() / 2
        if upsample_factor > 2:
            # a patch of `size` fine samples per axis, `center` of them before the estimate
            shift = (shift * upsample_factor).round() / upsample_factor
            size = ceil(1.5 * upsample_factor)
            center = size // 2
            origin = center - upsample_factor * shift
            row_coords = arange(float(size)) - origin[:, 0:1]
            col_coords = arange(float(size)) - origin[:, 1:2]
            row_freq = ifftshift(arange(num_rows)) - num_rows // 2
            col_freq = ifftshift(arange(num_cols)) - num_cols // 2
            row_phase = -2j * pi / (num_rows * upsample_factor)
            col_phase = -2j * pi / (num_cols * upsample_factor)
            row_kernel = exp(row_phase * (row_coords[:, :, None] * row_freq))
            col_kernel = exp(col_phase * (col_freq[:, None] * col_coords[:, None, :]))
            # one small product per pair: a batched complex matmul rounds differently by
            # 1e-7 relative, which 80 alignment iterations amplify to 3e-3 px
            conjugate_ft = correlation_ft.conj()
            patch = cat(
                [
                    row_kernel[index : index + 1]
                    @ conjugate_ft[index : index + 1]
                    @ col_kernel[index : index + 1]
                    for index in range(num_pairs)
                ]
            ).real
            fine_peak, fine_offset = _peak(patch)
            shift = shift + (fine_peak - center + fine_offset) / upsample_factor
    size = shift.new_tensor([num_rows, num_cols])
    return (shift + size / 2) % size - size / 2


def shift_images(images: Tensor, shifts: Tensor) -> Tensor:
    """Move each image of ``(num_images, num_rows, num_cols)`` by its (row, col) shift, bilinearly.

    Output pixel ``(r, c)`` samples the input at ``(r - shift_row, c - shift_col)``;
    samples outside the image are zero.
    """
    grid = translation_grid(shifts, images.shape[-2:])
    return F.grid_sample(images[:, None], grid, align_corners=True)[:, 0]


def translation_grid(shifts: Tensor, shape: tuple[int, int]) -> Tensor:
    """``grid_sample`` grid that moves each image by its (row, col) shift, ``(num_images, rows, cols, 2)``.

    With ``align_corners=True`` the ``n`` pixel centers sit at ``-1 ... 1``, so
    one pixel is ``2 / (n - 1)``; dividing by ``n`` instead would move by
    ``shift * (n - 1) / n`` and bias every measured shift.
    """
    num_rows, num_cols = shape
    theta = shifts.new_zeros((len(shifts), 2, 3))
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * shifts[:, 1] / (num_cols - 1)
    theta[:, 1, 2] = -2.0 * shifts[:, 0] / (num_rows - 1)
    return F.affine_grid(theta, [len(shifts), 1, num_rows, num_cols], align_corners=True)


def blend_shifted_images(images: Tensor, shifts: Tensor, edge_width: float) -> Tensor:
    """Average of shifted images whose borders fade out, on a canvas large enough for every shift.

    ``sum(window * shifted image) / sum(window)``, each image faded over
    ``edge_width`` pixels at its border so the seams between tilts do not show;
    a plain mean would show each tilt's scan frame as a sharp step. Positions
    no tilt covers take the median intensity.
    """
    num_images, num_rows, num_cols = images.shape
    padding = ceil(float(shifts.abs().max()) + edge_width) + 2
    inside = (slice(None), slice(padding, padding + num_rows), slice(padding, padding + num_cols))
    with images.device:
        window = tukey_window_2d((num_rows, num_cols), edge_width)
        weighted = zeros((num_images, num_rows + 2 * padding, num_cols + 2 * padding))
    weights = zeros_like(weighted)
    weighted[inside] = images * window
    weights[inside] = window
    coverage = shift_images(weights, shifts).clamp(0, 1).sum(dim=0)
    uncovered = (1 - coverage).clamp(0, 1)
    return (shift_images(weighted, shifts).sum(dim=0) + uncovered * images.median()) / (
        coverage + uncovered
    )


def edge_strength(image: Tensor, blur: float) -> Tensor:
    """Gradient magnitude ``sqrt(G_row**2 + G_col**2)`` from 3 x 3 Sobel derivatives, each blurred by ``blur`` px.

    Bright-field intensity drifts slowly across a scan and differs between
    tilts; edges do not, so aligning on them keeps those changes from pulling
    the shift.
    """
    sobel_row = image.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    padded = F.pad(image[None, None], pad=(1, 1, 1, 1), mode="reflect")
    gradient_row = gaussian_blur(F.conv2d(padded, sobel_row[None, None])[0], blur)
    gradient_col = gaussian_blur(F.conv2d(padded, sobel_row.T[None, None])[0], blur)
    return sqrt(gradient_row * gradient_row + gradient_col * gradient_col)


def gaussian_blur(images: Tensor, sigma: float) -> Tensor:
    """Gaussian blur of standard deviation ``sigma`` px over the last two axes, kernel ``4 sigma + 1`` wide."""
    kernel_size = 2 * int(2 * sigma) + 1
    return GaussianBlur(kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])(images)


def tukey_window_2d(shape: tuple[int, int], border_width: float) -> Tensor:
    """1 inside, a cosine fade over ``border_width`` px at each edge, on the current device.

    Correlating images that end in a sharp border adds a strong false feature,
    the frame itself; fading only the border removes it and keeps the interior
    at full weight.
    """
    row_window, col_window = (
        tensor(tukey(length, 2 * border_width / length), dtype=get_default_dtype())
        for length in shape
    )
    return row_window[:, None] * col_window


def _peak(surface: Tensor) -> tuple[Tensor, Tensor]:
    """Why: the correlation and its finer patch both peak between their samples.

    Returns the integer (row, col) of each surface's maximum, ``(num_pairs, 2)``,
    and its offset from a parabola through the maximum and its two neighbors
    on each axis, wrapping at the edges:
    ``(after - before) / (4 peak - 2 after - 2 before)``. Three collinear
    samples give zero, so a flat top stays put.
    """
    num_pairs, num_rows, num_cols = surface.shape
    index = surface.flatten(1).argmax(dim=1)
    row, col = index // num_cols, index % num_cols
    pairs = arange(num_pairs)
    peak = surface[pairs, row, col][:, None]
    before = stack(
        [surface[pairs, (row - 1) % num_rows, col], surface[pairs, row, (col - 1) % num_cols]], -1
    )
    after = stack(
        [surface[pairs, (row + 1) % num_rows, col], surface[pairs, row, (col + 1) % num_cols]], -1
    )
    denominator = 4 * peak - 2 * after - 2 * before
    offset = where(denominator != 0, (after - before) / denominator, 0.0)
    return stack([row, col], dim=-1).to(surface.dtype), offset
