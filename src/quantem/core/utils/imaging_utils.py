# Utilities for processing images

import math
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates

from quantem.core.utils.utils import generate_batches


def _parabolic_peak(v) -> float:
    denom = 4.0 * v[1] - 2.0 * v[2] - 2.0 * v[0]
    if denom == 0:
        return 0.0
    return float((v[2] - v[0]) / denom)


def dft_upsample(
    F: NDArray,
    up: int,
    shift: Tuple[float, float],
):
    """
    Matrix multiplication DFT, from:

    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel
    image registration algorithms," Opt. Lett. 33, 156-158 (2008).
    http://www.sciencedirect.com/science/article/pii/S0045790612000778
    """
    M, N = F.shape
    pixel_radius = 1.5
    num_row = int(math.ceil(pixel_radius * up))
    num_col = num_row

    col_freq = np.fft.ifftshift(np.arange(N)) - math.floor(N / 2)
    row_freq = np.fft.ifftshift(np.arange(M)) - math.floor(M / 2)

    row_coords = np.arange(num_row, dtype=float) - float(shift[0])
    col_coords = np.arange(num_col, dtype=float) - float(shift[1])

    factor_row = -2j * math.pi / (M * float(up))
    factor_col = -2j * math.pi / (N * float(up))

    row_kern = np.exp(factor_row * (row_coords[:, None] * row_freq[None, :])).astype(F.dtype)
    col_kern = np.exp(factor_col * (col_freq[:, None] * col_coords[None, :])).astype(F.dtype)

    return (row_kern @ F @ col_kern).real


def _upsampled_correlation_numpy(
    imageCorr: NDArray,
    upsampleFactor: int,
    xyShift: NDArray,
) -> NDArray:
    xyShift = np.round(xyShift * float(upsampleFactor)) / float(upsampleFactor)
    globalShift = math.floor(math.ceil(upsampleFactor * 1.5) / 2.0)
    upsampleCenter = float(globalShift) - (float(upsampleFactor) * xyShift)

    im_up = dft_upsample(np.conj(imageCorr), upsampleFactor, (float(upsampleCenter[0]), float(upsampleCenter[1])))
    imageCorrUpsample = np.conj(im_up)

    flat_idx = int(np.argmax(imageCorrUpsample.real))
    r = flat_idx // imageCorrUpsample.shape[1]
    c = flat_idx % imageCorrUpsample.shape[1]

    dx = 0.0
    dy = 0.0
    patch = imageCorrUpsample.real[r - 1 : r + 2, c - 1 : c + 2]
    if patch.shape == (3, 3):
        dx = _parabolic_peak(patch[:, 1])
        dy = _parabolic_peak(patch[1, :])

    xySubShift = np.array([float(r), float(c)], dtype=float) - float(globalShift)
    xyShift = xyShift + (xySubShift + np.array([dx, dy], dtype=float)) / float(upsampleFactor)

    return xyShift


def cross_correlation_shift(
    im_ref,
    im,
    upsample_factor: int = 1,
    max_shift=None,
    return_shifted_image: bool = False,
    fft_input: bool = False,
    fft_output: bool = False,
):
    """
    Estimate subpixel shift between two 2D images using Fourier cross-correlation.

    Parameters
    ----------
    im_ref : ndarray
        Reference image or its FFT if fft_input=True
    im : ndarray
        Image to align or its FFT if fft_input=True
    upsample_factor : int
        Subpixel upsampling factor (torch-equivalent behavior):
        - <= 2 : half-pixel refinement (parabolic, then rounded to nearest 0.5 px)
        - > 2  : additional DFT upsample refinement
    max_shift : float or None
        Optional radial cutoff in pixel-shift units (keeps only shifts with |shift| <= max_shift)
    return_shifted_image : bool
        If True, return the shifted version of `im` aligned to `im_ref`
    fft_input : bool
        If True, assumes im_ref and im are already in Fourier space
    fft_output : bool
        If True and return_shifted_image=True, return the shifted image in Fourier space

    Returns
    -------
    shifts : tuple of float
        (row_shift, col_shift) to align `im` to `im_ref`
    image_shifted : ndarray (optional)
        Shifted image in real space (or Fourier space if fft_output=True)
    """
    F_ref = np.asarray(im_ref) if fft_input else np.fft.fft2(np.asarray(im_ref))
    F_im = np.asarray(im) if fft_input else np.fft.fft2(np.asarray(im))

    cc = F_ref * np.conj(F_im)
    cc_real = np.fft.ifft2(cc).real

    M, N = cc_real.shape

    if max_shift is not None:
        x = np.fft.fftfreq(M) * M
        y = np.fft.fftfreq(N) * N
        mask = x[:, None] ** 2 + y[None, :] ** 2 > float(max_shift) ** 2
        cc_real = cc_real.copy()
        cc_real[mask] = -np.inf

    flat_idx = int(np.argmax(cc_real))
    x0 = flat_idx // N
    y0 = flat_idx % N

    x_inds = [((x0 + dx) % M) for dx in (-1, 0, 1)]
    y_inds = [((y0 + dy) % N) for dy in (-1, 0, 1)]

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    dx = _parabolic_peak(vx)
    dy = _parabolic_peak(vy)

    x0 = np.round((float(x0) + float(dx)) * 2.0) / 2.0
    y0 = np.round((float(y0) + float(dy)) * 2.0) / 2.0

    xy_shift = np.array([x0, y0], dtype=float)

    if upsample_factor > 2:
        xy_shift = _upsampled_correlation_numpy(cc, int(upsample_factor), xy_shift)

    shifts = np.empty(2, dtype=float)
    shifts[0] = ((xy_shift[0] + M / 2) % M) - M / 2
    shifts[1] = ((xy_shift[1] + N / 2) % N) - N / 2
    shifts = (float(shifts[0]), float(shifts[1]))

    if not return_shifted_image:
        return shifts

    kx = np.fft.fftfreq(F_im.shape[0])[:, None]
    ky = np.fft.fftfreq(F_im.shape[1])[None, :]
    phase_ramp = np.exp(-2j * np.pi * (kx * shifts[0] + ky * shifts[1]))
    F_im_shifted = F_im * phase_ramp

    if fft_output:
        image_shifted = F_im_shifted
    else:
        image_shifted = np.fft.ifft2(F_im_shifted).real

    return shifts, image_shifted


def cross_correlation_shift_torch(
    im_ref: torch.Tensor, im: torch.Tensor, upsample_factor: int = 2
) -> torch.Tensor:
    """
    Align two real images using Fourier cross-correlation and DFT upsampling.
    Returns dx, dy in pixel units (signed shifts).
    """
    G1 = torch.fft.fft2(im_ref)
    G2 = torch.fft.fft2(im)

    xy_shift = align_images_fourier_torch(G1, G2, upsample_factor)

    M, N = im_ref.shape
    dx = ((xy_shift[0] + M / 2) % M) - M / 2
    dy = ((xy_shift[1] + N / 2) % N) - N / 2

    return torch.tensor([dx, dy], device=G1.device)


def align_images_fourier_torch(
    G1: torch.Tensor,
    G2: torch.Tensor,
    upsample_factor: int,
) -> torch.Tensor:
    """
    Alignment using DFT upsampling of cross correlation.
    G1, G2: torch tensors representing FTs of images (complex)
    Returns: xy_shift (tensor length 2)
    """
    device = G1.device
    cc = G1 * G2.conj()
    cc_real = torch.fft.ifft2(cc).real

    flat_idx = torch.argmax(cc_real)
    x0 = (flat_idx // cc_real.shape[1]).to(torch.long).item()
    y0 = (flat_idx % cc_real.shape[1]).to(torch.long).item()

    M, N = cc_real.shape
    x_inds = [((x0 + dx) % M) for dx in (-1, 0, 1)]
    y_inds = [((y0 + dy) % N) for dy in (-1, 0, 1)]

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    denom_x = 4.0 * vx[1] - 2.0 * vx[2] - 2.0 * vx[0]
    denom_y = 4.0 * vy[1] - 2.0 * vy[2] - 2.0 * vy[0]
    dx = (vx[2] - vx[0]) / denom_x if denom_x != 0 else torch.tensor(0.0, device=device)
    dy = (vy[2] - vy[0]) / denom_y if denom_y != 0 else torch.tensor(0.0, device=device)

    x0 = torch.round((x0 + dx) * 2.0) / 2.0
    y0 = torch.round((y0 + dy) * 2.0) / 2.0

    xy_shift = torch.tensor([x0, y0])

    if upsample_factor > 2:
        xy_shift = upsampled_correlation_torch(cc, upsample_factor, xy_shift)

    return xy_shift


def upsampled_correlation_torch(
    imageCorr: torch.Tensor,
    upsampleFactor: int,
    xyShift: torch.Tensor,
) -> torch.Tensor:
    """
    Refine the correlation peak of imageCorr around xyShift by DFT upsampling.

    imageCorr: complex-valued FT-domain cross-correlation (G1 * conj(G2))
    upsampleFactor: integer > 2
    xyShift: 2-element tensor (x,y) in image coords; must be half-pixel precision as described.
    Returns refined xyShift (tensor length 2).
    """
    assert upsampleFactor > 2

    xyShift = torch.round(xyShift * float(upsampleFactor)) / float(upsampleFactor)
    globalShift = torch.floor(torch.ceil(torch.tensor(upsampleFactor * 1.5)) / 2.0)
    upsampleCenter = globalShift - (upsampleFactor * xyShift)

    conj_input = imageCorr.conj()
    im_up = dftUpsample_torch(conj_input, upsampleFactor, upsampleCenter)
    imageCorrUpsample = im_up.conj()

    flat_idx = torch.argmax(imageCorrUpsample.real)
    xySubShift0 = (flat_idx // imageCorrUpsample.shape[1]).to(torch.long)
    xySubShift1 = (flat_idx % imageCorrUpsample.shape[1]).to(torch.long)
    xySubShift = torch.tensor([xySubShift0.item(), xySubShift1.item()])

    dx = 0.0
    dy = 0.0
    try:
        r = xySubShift[0].item()
        c = xySubShift[1].item()
        patch = imageCorrUpsample.real[r - 1 : r + 2, c - 1 : c + 2]
        if patch.shape == (3, 3):
            icc = patch
            dx = (icc[2, 1] - icc[0, 1]) / (
                4.0 * icc[1, 1] - 2.0 * icc[2, 1] - 2.0 * icc[0, 1]
            )
            dy = (icc[1, 2] - icc[1, 0]) / (
                4.0 * icc[1, 1] - 2.0 * icc[1, 2] - 2.0 * icc[1, 0]
            )
            dx = dx.item()
            dy = dy.item()
        else:
            dx, dy = 0.0, 0.0
    except Exception:
        dx, dy = 0.0, 0.0

    xySubShift = xySubShift.to(dtype=torch.get_default_dtype())
    xySubShift = xySubShift - globalShift.to(xySubShift.dtype)

    xyShift = xyShift + (xySubShift + torch.tensor([dx, dy])) / float(upsampleFactor)

    return xyShift


def dftUpsample_torch(
    imageCorr: torch.Tensor,
    upsampleFactor: int,
    xyShift: torch.Tensor,
) -> torch.Tensor:
    """
    Corrected matrix-multiply DFT upsampling (matches the original numpy dftups).
    Returns the real-valued upsampled correlation patch.

    imageCorr: (M, N) complex tensor (FT-domain cross-correlation)
    upsampleFactor: int > 2
    xyShift: 2-element tensor [x0, y0] giving the (half-pixel-rounded) peak location
             in the UPSAMPLED grid (same convention used elsewhere).
    """
    device = imageCorr.device
    M, N = imageCorr.shape
    pixelRadius = 1.5
    numRow = int(math.ceil(pixelRadius * upsampleFactor))
    numCol = numRow

    col_freq = torch.fft.ifftshift(torch.arange(N, device=device)) - math.floor(N / 2)
    row_freq = torch.fft.ifftshift(torch.arange(M, device=device)) - math.floor(M / 2)

    col_coords = torch.arange(numCol, device=device, dtype=torch.get_default_dtype()) - float(
        xyShift[1]
    )
    row_coords = torch.arange(numRow, device=device, dtype=torch.get_default_dtype()) - float(
        xyShift[0]
    )

    factor_col = -2j * math.pi / (N * float(upsampleFactor))
    colKern = torch.exp(factor_col * (col_freq.unsqueeze(1) * col_coords.unsqueeze(0))).to(
        imageCorr.dtype
    )

    factor_row = -2j * math.pi / (M * float(upsampleFactor))
    rowKern = torch.exp(factor_row * (row_coords.unsqueeze(1) * row_freq.unsqueeze(0))).to(
        imageCorr.dtype
    )

    imageUpsample = rowKern @ imageCorr @ colKern

    return imageUpsample.real


def weighted_cross_correlation_shift(
    im_ref=None,
    im=None,
    *,
    cc=None,
    weight_real=None,
    upsample_factor: int = 1,
    max_shift=None,
    fft_input: bool = False,
    fft_output: bool = False,
    return_shifted_image: bool = False,
):
    """
    Weighted peak selection + DFT subpixel refinement for Fourier cross-correlation.

    Provide either:
      - im_ref and im (real-space images, or Fourier-domain if fft_input=True), OR
      - cc (the Fourier-domain cross-spectrum), where cc = F_ref * conj(F_im)

    The weight is applied ONLY in real-space correlation to choose the peak location,
    but the subpixel refinement uses the true (unweighted) cross-spectrum `cc`.

    Returns
    -------
    shift_rc : tuple[float, float]
        (d_row, d_col) shift to apply to `im` to align it to `im_ref`.
    shifted : ndarray (optional)
        If return_shifted=True: shifted image. If fft_output=True returns FFT (corner-centered),
        else returns real-space image.
    """
    if cc is None:
        if im_ref is None or im is None:
            raise ValueError("Provide either `cc` or both `im_ref` and `im`.")
        F_ref = np.asarray(im_ref) if fft_input else np.fft.fft2(np.asarray(im_ref))
        F_im = np.asarray(im) if fft_input else np.fft.fft2(np.asarray(im))
        cc = F_ref * np.conj(F_im)
    else:
        cc = np.asarray(cc)
        F_im = None

    cc_real = np.fft.ifft2(cc).real
    M, N = cc_real.shape

    if weight_real is not None:
        w = np.asarray(weight_real)
        if w.shape != cc_real.shape:
            raise ValueError(f"weight_real.shape={w.shape} must match correlation shape {cc_real.shape}.")
        cc_pick = cc_real * w
    else:
        cc_pick = cc_real

    if max_shift is not None:
        fr = np.fft.fftfreq(M) * M
        fc = np.fft.fftfreq(N) * N
        mask = fr[:, None] ** 2 + fc[None, :] ** 2 > float(max_shift) ** 2
        cc_pick = cc_pick.copy()
        cc_pick[mask] = -np.inf

    flat_idx = int(np.argmax(cc_pick))
    x0 = flat_idx // N
    y0 = flat_idx % N

    x_inds = [((x0 + dx) % M) for dx in (-1, 0, 1)]
    y_inds = [((y0 + dy) % N) for dy in (-1, 0, 1)]
    vx = cc_pick[x_inds, y0]
    vy = cc_pick[x0, y_inds]

    dx = _parabolic_peak(vx)
    dy = _parabolic_peak(vy)

    x0 = np.round((float(x0) + float(dx)) * 2.0) / 2.0
    y0 = np.round((float(y0) + float(dy)) * 2.0) / 2.0
    xy_shift = np.array([x0, y0], dtype=float)

    if upsample_factor > 2:
        xy_shift = _upsampled_correlation_numpy(cc, int(upsample_factor), xy_shift)

    dr = ((xy_shift[0] + M / 2) % M) - M / 2
    dc = ((xy_shift[1] + N / 2) % N) - N / 2
    shift_rc = (float(dr), float(dc))

    if not return_shifted_image:
        return shift_rc

    if im is None:
        raise ValueError("return_shifted_image=True requires `im` (or its FFT via fft_input=True).")

    if F_im is None:
        F_im = np.asarray(im) if fft_input else np.fft.fft2(np.asarray(im))

    kr = np.fft.fftfreq(M)[:, None]
    kc = np.fft.fftfreq(N)[None, :]
    phase_ramp = np.exp(-2j * np.pi * (kr * shift_rc[0] + kc * shift_rc[1]))
    F_im_shifted = F_im * phase_ramp

    if fft_output:
        return shift_rc, F_im_shifted
    return shift_rc, np.fft.ifft2(F_im_shifted).real


def bilinear_kde(
    xa: NDArray,
    ya: NDArray,
    values: NDArray,
    output_shape: Tuple[int, int],
    kde_sigma: float,
    pad_value: float = 0.0,
    threshold: float = 1e-3,
    lowpass_filter: bool = False,
    max_batch_size: Optional[int] = None,
    return_pix_count: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """
    Compute a bilinear kernel density estimate (KDE) with smooth threshold masking.
    """
    rows, cols = output_shape
    xF = np.floor(xa.ravel()).astype(int)
    yF = np.floor(ya.ravel()).astype(int)
    dx = xa.ravel() - xF
    dy = ya.ravel() - yF
    w = values.ravel()

    pix_count = np.zeros(rows * cols, dtype=np.float32)
    pix_output = np.zeros(rows * cols, dtype=np.float32)

    if max_batch_size is None:
        max_batch_size = xF.shape[0]

    for start, end in generate_batches(xF.shape[0], max_batch=max_batch_size):
        for dx_off, dy_off, weights in [
            (0, 0, (1 - dx[start:end]) * (1 - dy[start:end])),
            (1, 0, dx[start:end] * (1 - dy[start:end])),
            (0, 1, (1 - dx[start:end]) * dy[start:end]),
            (1, 1, dx[start:end] * dy[start:end]),
        ]:
            inds = [xF[start:end] + dx_off, yF[start:end] + dy_off]
            inds_1D = np.ravel_multi_index(inds, dims=output_shape, mode="wrap")

            pix_count += np.bincount(inds_1D, weights=weights, minlength=rows * cols)
            pix_output += np.bincount(
                inds_1D, weights=weights * w[start:end], minlength=rows * cols
            )

    pix_count = pix_count.reshape(output_shape)
    pix_output = pix_output.reshape(output_shape)

    pix_count = gaussian_filter(pix_count, kde_sigma)
    pix_output = gaussian_filter(pix_output, kde_sigma)

    weight = np.minimum(pix_count / threshold, 1.0)
    image = pad_value * (1.0 - weight) + weight * (pix_output / np.maximum(pix_count, 1e-8))

    if lowpass_filter:
        f_img = np.fft.fft2(image)
        fx = np.fft.fftfreq(rows)
        fy = np.fft.fftfreq(cols)
        f_img /= np.sinc(fx)[:, None]  # type: ignore
        f_img /= np.sinc(fy)[None, :]  # type: ignore
        image = np.real(np.fft.ifft2(f_img))

        if return_pix_count:
            f_img = np.fft.fft2(pix_count)
            f_img /= np.sinc(fx)[:, None]  # type: ignore
            f_img /= np.sinc(fy)[None, :]  # type: ignore
            pix_count = np.real(np.fft.ifft2(f_img))

    if return_pix_count:
        return image, pix_count
    else:
        return image


def bilinear_array_interpolation(
    image: NDArray,
    xa: NDArray,
    ya: NDArray,
    max_batch_size=None,
) -> NDArray:
    """
    Bilinear sampling of values from an array and pixel positions.
    """
    xF = np.floor(xa.ravel()).astype("int")
    yF = np.floor(ya.ravel()).astype("int")
    dx = xa.ravel() - xF
    dy = ya.ravel() - yF

    raveled_image = image.ravel()
    values = np.zeros(xF.shape, dtype=image.dtype)

    output_shape = image.shape

    if max_batch_size is None:
        max_batch_size = xF.shape[0]

    for start, end in generate_batches(xF.shape[0], max_batch=max_batch_size):
        for dx_off, dy_off, weights in [
            (0, 0, (1 - dx[start:end]) * (1 - dy[start:end])),
            (1, 0, dx[start:end] * (1 - dy[start:end])),
            (0, 1, (1 - dx[start:end]) * dy[start:end]),
            (1, 1, dx[start:end] * dy[start:end]),
        ]:
            inds = [xF[start:end] + dx_off, yF[start:end] + dy_off]
            inds_1D = np.ravel_multi_index(inds, dims=output_shape, mode="wrap")

            values[start:end] += raveled_image[inds_1D] * weights

    values = np.reshape(values, xa.shape)

    return values


def fourier_cropping(
    corner_centered_array: NDArray,
    crop_shape: Tuple[int, int],
):
    """
    Crops a corner-centered FFT array to retain only the lowest frequencies,
    equivalent to a center crop on the fftshifted version.
    """
    H, W = corner_centered_array.shape
    crop_h, crop_w = crop_shape

    h1 = crop_h // 2
    h2 = crop_h - h1
    w1 = crop_w // 2
    w2 = crop_w - w1

    result = np.zeros(crop_shape, dtype=corner_centered_array.dtype)

    result[:h1, :w1] = corner_centered_array[:h1, :w1]
    result[:h1, -w2:] = corner_centered_array[:h1, -w2:]
    result[-h2:, :w1] = corner_centered_array[-h2:, :w1]
    result[-h2:, -w2:] = corner_centered_array[-h2:, -w2:]

    return result


def compute_fsc_from_halfsets(
    halfset_recons: list[torch.Tensor],
    sampling: tuple[float, float],
    epsilon: float = 1e-12,
):
    """
    Compute radially averaged Fourier Shell Correlation (FSC)
    from two half-set reconstructions.
    """
    r1, r2 = halfset_recons

    F1 = torch.fft.fft2(r1)
    F2 = torch.fft.fft2(r2)

    cross = (F1 * F2.conj()).real
    p1 = F1.abs().square()
    p2 = F2.abs().square()

    device = F1.device
    nx, ny = F1.shape
    sx, sy = sampling

    kx = torch.fft.fftfreq(nx, d=sx, device=device)
    ky = torch.fft.fftfreq(ny, d=sy, device=device)
    k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).reshape(-1)

    bin_size = kx[1] - kx[0]
    max_k = k.max()
    num_bins = int(torch.floor(max_k / bin_size).item()) + 2

    inds = k / bin_size
    inds_f = torch.floor(inds).long()
    d_ind = inds - inds_f

    w0 = 1.0 - d_ind
    w1 = d_ind

    cross = cross.reshape(-1)
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)

    cross_b = torch.bincount(inds_f, weights=cross * w0, minlength=num_bins) + torch.bincount(
        inds_f + 1, weights=cross * w1, minlength=num_bins
    )

    p1_b = torch.bincount(inds_f, weights=p1 * w0, minlength=num_bins) + torch.bincount(
        inds_f + 1, weights=p1 * w1, minlength=num_bins
    )

    p2_b = torch.bincount(inds_f, weights=p2 * w0, minlength=num_bins) + torch.bincount(
        inds_f + 1, weights=p2 * w1, minlength=num_bins
    )

    denom = torch.sqrt(p1_b * p2_b).clamp_min(epsilon)
    fsc = cross_b / denom

    k_bins = torch.arange(num_bins, device=device, dtype=torch.float32) * bin_size
    valid = k_bins <= kx.abs().max()

    return k_bins[valid].cpu().numpy(), fsc[valid].cpu().numpy()


def compute_spectral_snr_from_halfsets(
    halfset_recons: list[torch.Tensor],
    sampling: tuple[float, float],
    total_dose: float,
    epsilon: float = 1e-12,
):
    """
    Compute spectral SNR from two half-set reconstructions using symmetric/antisymmetric decomposition.
    """
    halfset_1, halfset_2 = halfset_recons
    F1 = torch.fft.fft2(halfset_1)
    F2 = torch.fft.fft2(halfset_2)

    symmetric = (F1 + F2) / 2
    antisymmetric = (F1 - F2) / 2

    noise_power = antisymmetric.abs()
    total_power = symmetric.abs()
    signal_power = (total_power - noise_power).clamp_min(0)

    device = F1.device
    nx, ny = F1.shape
    sx, sy = sampling

    kx = torch.fft.fftfreq(nx, d=sx, device=device)
    ky = torch.fft.fftfreq(ny, d=sy, device=device)
    k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).reshape(-1)

    bin_size = kx[1] - kx[0]
    max_k = k.max()
    num_bins = int(torch.floor(max_k / bin_size).item()) + 2

    inds = k / bin_size
    inds_f = torch.floor(inds).long()
    d_ind = inds - inds_f

    w0 = 1.0 - d_ind
    w1 = d_ind

    signal = signal_power.reshape(-1)
    noise = noise_power.reshape(-1)

    signal_b = torch.bincount(inds_f, weights=signal * w0, minlength=num_bins) + torch.bincount(
        inds_f + 1, weights=signal * w1, minlength=num_bins
    )

    noise_b = torch.bincount(inds_f, weights=noise * w0, minlength=num_bins) + torch.bincount(
        inds_f + 1, weights=noise * w1, minlength=num_bins
    )

    ssnr = torch.sqrt(signal_b / noise_b.clamp_min(epsilon)) / (math.sqrt(total_dose) / 2)

    k_bins = torch.arange(num_bins, device=device, dtype=torch.float32) * bin_size
    valid = k_bins <= kx.abs().max()

    return k_bins[valid].cpu().numpy(), ssnr[valid].cpu().numpy()


def radially_average_fourier_array(
    corner_centered_array: torch.Tensor,
    sampling: tuple[float, float],
):
    """
    Radially average a corner-centered Fourier array.
    """
    device = corner_centered_array.device
    nx, ny = corner_centered_array.shape
    sx, sy = sampling

    kx = torch.fft.fftfreq(nx, d=sx, device=device)
    ky = torch.fft.fftfreq(ny, d=sy, device=device)
    k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).reshape(-1)

    bin_size = kx[1] - kx[0]
    max_k = k.max()
    num_bins = int(torch.floor(max_k / bin_size).item()) + 2

    inds = k / bin_size
    inds_f = torch.floor(inds).long()
    d_ind = inds - inds_f

    w0 = 1.0 - d_ind
    w1 = d_ind

    array = corner_centered_array.reshape(-1)

    array_b = torch.bincount(inds_f, weights=array * w0, minlength=num_bins) + torch.bincount(
        inds_f + 1, weights=array * w1, minlength=num_bins
    )

    counts_b = (
        torch.bincount(inds_f, weights=w0, minlength=num_bins)
        + torch.bincount(inds_f + 1, weights=w1, minlength=num_bins)
    ).clamp_min(1)

    array_b = array_b / counts_b

    k_bins = torch.arange(num_bins, device=device, dtype=torch.float32) * bin_size
    valid = k_bins <= kx.abs().max()

    return k_bins[valid].cpu().numpy(), array_b[valid].cpu().numpy()


def _wrap_to_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _find_wrap(a, b):
    d = a - b
    return torch.where(d > math.pi, -1, torch.where(d < -math.pi, 1, 0))


def _pixel_reliability(phi, mask=None):
    """
    phi: (H, W) wrapped phase (CPU tensor)
    mask: optional boolean mask
    """
    c = phi
    left = torch.roll(c, 1, 1)
    right = torch.roll(c, -1, 1)
    up = torch.roll(c, 1, 0)
    down = torch.roll(c, -1, 0)

    ul = torch.roll(left, 1, 0)
    dr = torch.roll(right, -1, 0)
    ur = torch.roll(right, 1, 0)
    dl = torch.roll(left, -1, 0)

    Hterm = _wrap_to_pi(left - c) - _wrap_to_pi(c - right)
    Vterm = _wrap_to_pi(up - c) - _wrap_to_pi(c - down)
    D1term = _wrap_to_pi(ul - c) - _wrap_to_pi(c - dr)
    D2term = _wrap_to_pi(ur - c) - _wrap_to_pi(c - dl)

    R = Hterm**2 + Vterm**2 + D1term**2 + D2term**2

    if mask is not None:
        R = torch.where(mask, R, torch.full_like(R, float("inf")))

    return R


def _build_edges(phi, reliability, mask=None, wrap_around=True):
    """
    Returns edges as CPU tensors:
        i1, i2, inc sorted by reliability
    """
    H, W = phi.shape
    N = H * W

    idx = torch.arange(N).reshape(H, W)
    edges = []

    phi_f = phi.flatten()
    rel_f = reliability.flatten()
    mask_f = mask.flatten() if mask is not None else None

    def add_edges(i1, i2):
        if mask_f is not None:
            valid = mask_f[i1] & mask_f[i2]
            i1, i2 = i1[valid], i2[valid]

        inc = _find_wrap(phi_f[i1], phi_f[i2])
        rel = rel_f[i1] + rel_f[i2]

        edges.append(torch.stack([i1, i2, rel, inc], dim=1))

    if wrap_around:
        add_edges(idx.flatten(), torch.roll(idx, -1, 1).flatten())
        add_edges(idx.flatten(), torch.roll(idx, -1, 0).flatten())
    else:
        add_edges(idx[:, :-1].flatten(), idx[:, 1:].flatten())
        add_edges(idx[:-1, :].flatten(), idx[1:, :].flatten())

    edges = torch.cat(edges, dim=0)
    edges = edges[edges[:, 2].argsort()]

    return (
        edges[:, 0].long(),
        edges[:, 1].long(),
        edges[:, 3].long(),
    )


class UnionFindPhase:
    def __init__(self, n):
        self.parent = torch.arange(n)
        self.rank = torch.zeros(n, dtype=torch.int32)
        self.offset = torch.zeros(n)

    def find_root_and_offset(self, x):
        root = x
        total = 0.0
        while self.parent[root] != root:
            total += self.offset[root]
            root = self.parent[root]
        return root, total

    def union(self, x, y, inc_xy):
        rx, ox = self.find_root_and_offset(x)
        ry, oy = self.find_root_and_offset(y)

        if rx == ry:
            return

        delta = ox - oy - inc_xy

        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
            self.offset[rx] = -delta
        else:
            self.parent[ry] = rx
            self.offset[ry] = delta
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1


def _final_offsets(uf):
    """
    Single-pass offset computation (no path compression).
    """
    N = uf.parent.numel()
    incs = torch.zeros(N)

    for i in range(N):
        root = i
        total = 0.0
        while uf.parent[root] != root:
            total += uf.offset[root]
            root = uf.parent[root]
        incs[i] = total

    return incs


def _unwrap_phase_2d_torch_reliability_sorting(
    phi,
    mask=None,
    wrap_around=True,
):
    """
    Herráez 2D phase unwrapping.
    Runs on CPU by design.
    """
    with torch.no_grad():
        orig_device = phi.device
        phi = phi.detach().cpu()
        if mask is not None:
            mask = mask.detach().cpu().to(torch.bool)

        H, W = phi.shape
        N = H * W

        reliability = _pixel_reliability(phi, mask)

        i1, i2, inc = _build_edges(
            phi,
            reliability,
            mask,
            wrap_around=wrap_around,
        )

        uf = UnionFindPhase(N)

        for k in range(i1.numel()):
            uf.union(i1[k].item(), i2[k].item(), inc[k].item())

        incs = _final_offsets(uf)

        out = (phi.flatten() + 2 * math.pi * incs).reshape(H, W)
        out -= out.mean()
        return out.to(orig_device)


def _unwrap_phase_2d_torch_poisson(
    phi_wrapped,
    mask=None,
    wrap_around=True,
    regularization_lambda=None,
):
    """
    Least-squares / Poisson phase unwrapping with optional mask.
    """
    device = phi_wrapped.device
    dtype = phi_wrapped.dtype
    H, W = phi_wrapped.shape

    if not wrap_around:
        raise NotImplementedError()

    if mask is not None:
        mask = mask.to(device=device, dtype=torch.bool)

    dx = torch.roll(phi_wrapped, -1, dims=1) - phi_wrapped
    dy = torch.roll(phi_wrapped, -1, dims=0) - phi_wrapped

    dx = (dx + math.pi) % (2 * math.pi) - math.pi
    dy = (dy + math.pi) % (2 * math.pi) - math.pi

    if mask is not None:
        mask_x = mask & torch.roll(mask, -1, dims=1)
        mask_y = mask & torch.roll(mask, -1, dims=0)

        dx = torch.where(mask_x, dx, torch.zeros_like(dx))
        dy = torch.where(mask_y, dy, torch.zeros_like(dy))

    div = dx - torch.roll(dx, 1, dims=1) + dy - torch.roll(dy, 1, dims=0)

    if mask is not None:
        div = torch.where(mask, div, torch.zeros_like(div))

    div_hat = torch.fft.fftn(div)

    ky = torch.fft.fftfreq(H, device=device, dtype=dtype) * 2 * math.pi
    kx = torch.fft.fftfreq(W, device=device, dtype=dtype) * 2 * math.pi
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")

    if regularization_lambda is not None:
        denom = kx**2 + ky**2 + regularization_lambda
    else:
        denom = kx**2 + ky**2
    denom[0, 0] = 1.0

    phi_hat = -div_hat / denom
    phi_hat[0, 0] = 0.0

    phi = torch.fft.ifftn(phi_hat).real

    if mask is not None:
        phi = torch.where(mask, phi, torch.zeros_like(phi))

    return phi


def unwrap_phase_2d_torch(
    phi_wrapped,
    method="reliability-sorting",
    mask=None,
    wrap_around=True,
    regularization_lambda=None,
):
    if method == "reliability-sorting":
        return _unwrap_phase_2d_torch_reliability_sorting(
            phi_wrapped, mask, wrap_around=wrap_around
        )
    elif method == "poisson":
        return _unwrap_phase_2d_torch_poisson(
            phi_wrapped,
            mask,
            wrap_around=wrap_around,
            regularization_lambda=regularization_lambda,
        )
    else:
        raise ValueError(
            f'`method` must be one of {{"reliability-sorting", "poisson"}}, got {method!r}'
        )


def rotate_image(
    im,
    rotation_deg: float,
    origin: tuple[float, float] | None = None,
    clockwise: bool = True,
    interpolation: str = "bilinear",
    mode: str = "constant",
    cval: float = 0.0,
):
    """Rotate an array about a pixel origin using bilinear/bicubic interpolation."""
    im = np.asarray(im)
    if im.ndim < 2:
        raise ValueError("im must have at least 2 dimensions")

    H, W = im.shape[-2], im.shape[-1]
    if origin is None:
        r0 = float(H // 2)
        c0 = float(W // 2)
    else:
        r0 = float(origin[0])
        c0 = float(origin[1])

    interp = str(interpolation).lower()
    if interp in {"bilinear", "linear"}:
        order = 1
    elif interp in {"bicubic", "cubic"}:
        order = 3
    else:
        raise ValueError("interpolation must be 'bilinear' or 'bicubic'")

    theta = float(np.deg2rad(rotation_deg))
    if not clockwise:
        theta = -theta

    ct = float(np.cos(theta))
    st = float(np.sin(theta))

    r_out, c_out = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    c_rel = c_out - c0
    r_rel = r_out - r0

    c_in = ct * c_rel + st * r_rel + c0
    r_in = -st * c_rel + ct * r_rel + r0

    coords = np.vstack((r_in.ravel(), c_in.ravel()))

    if im.ndim == 2:
        out = map_coordinates(im, coords, order=order, mode=mode, cval=cval)
        return out.reshape(H, W)

    prefix = im.shape[:-2]
    n = int(np.prod(prefix)) if prefix else 1
    im_flat = im.reshape(n, H, W)
    out_flat = np.empty((n, H * W), dtype=np.result_type(im_flat.dtype, np.float64))
    for i in range(n):
        out_flat[i] = map_coordinates(im_flat[i], coords, order=order, mode=mode, cval=cval)
    return out_flat.reshape(*prefix, H, W)
