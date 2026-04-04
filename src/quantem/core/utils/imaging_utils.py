# Utilities for processing images

import math
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from quantem.core.utils.utils import generate_batches


def dft_upsample(
    F: NDArray,
    up: int,
    shift: Tuple[float, float],
    device: str = "cpu",
):
    """
    Matrix multiplication DFT, from:

    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel
    image registration algorithms," Opt. Lett. 33, 156-158 (2008).
    http://www.sciencedirect.com/science/article/pii/S0045790612000778
    """
    if device == "gpu":
        import cupy as cp  # type: ignore

        xp = cp
    else:
        xp = np

    M, N = F.shape
    du = np.ceil(1.5 * up).astype(int)
    row = np.arange(-du, du + 1)
    col = np.arange(-du, du + 1)
    r_shift = shift[0] - M // 2
    c_shift = shift[1] - N // 2

    kern_row = np.exp(
        -2j * np.pi / (M * up) * np.outer(row, xp.fft.ifftshift(xp.arange(M)) - M // 2 + r_shift)
    )
    kern_col = np.exp(
        -2j * np.pi / (N * up) * np.outer(xp.fft.ifftshift(xp.arange(N)) - N // 2 + c_shift, col)
    )
    return xp.real(kern_row @ F @ kern_col)


def cross_correlation_shift(
    im_ref,
    im,
    upsample_factor: int = 1,
    max_shift=None,
    return_shifted_image: bool = False,
    fft_input: bool = False,
    fft_output: bool = False,
    device: str = "cpu",
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
        Subpixel upsampling factor (must be > 1 for subpixel accuracy)
    fft_input : bool
        If True, assumes im_ref and im are already in Fourier space
    return_shifted_image : bool
        If True, return the shifted version of `im` aligned to `im_ref`
    device : str
        'cpu' or 'gpu' (requires CuPy)

    Returns
    -------
    shifts : tuple of float
        (row_shift, col_shift) to align `im` to `im_ref`
    image_shifted : ndarray (optional)
        Shifted image in real space, only returned if return_shifted_image=True
    """
    if device == "gpu":
        import cupy as cp  # type: ignore

        xp = cp
    else:
        xp = np

    # Fourier transforms
    F_ref = im_ref if fft_input else xp.fft.fft2(im_ref)
    F_im = im if fft_input else xp.fft.fft2(im)

    # Correlation
    cc = F_ref * xp.conj(F_im)
    cc_real = xp.real(xp.fft.ifft2(cc))

    if max_shift is not None:
        x = np.fft.fftfreq(cc.shape[0], 1 / cc.shape[0])
        y = np.fft.fftfreq(cc.shape[1], 1 / cc.shape[1])
        mask = x[:, None] ** 2 + y[None, :] ** 2 >= max_shift**2
        cc_real[mask] = 0.0

    # Coarse peak
    peak = xp.unravel_index(xp.argmax(cc_real), cc_real.shape)
    x0, y0 = peak

    # Parabolic refinement
    x_inds = xp.mod(x0 + xp.arange(-1, 2), cc.shape[0]).astype(int)
    y_inds = xp.mod(y0 + xp.arange(-1, 2), cc.shape[1]).astype(int)

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    def parabolic_peak(v):
        return (v[2] - v[0]) / (4 * v[1] - 2 * v[2] - 2 * v[0])

    dx = parabolic_peak(vx)
    dy = parabolic_peak(vy)

    x0 = (x0 + dx) % cc.shape[0]
    y0 = (y0 + dy) % cc.shape[1]

    if upsample_factor <= 1:
        shifts = (x0, y0)
    else:
        # Local DFT upsampling

        local = dft_upsample(cc, upsample_factor, (x0, y0), device=device)
        peak = np.unravel_index(xp.argmax(local), local.shape)

        try:
            lx, ly = peak
            icc = local[lx - 1 : lx + 2, ly - 1 : ly + 2]
            if icc.shape == (3, 3):
                dxf = parabolic_peak(icc[:, 1])
                dyf = parabolic_peak(icc[1, :])
            else:
                raise ValueError("Subarray too close to edge")
        except (IndexError, ValueError):
            dxf = dyf = 0.0

        shifts = np.array([x0, y0]) + (np.array(peak) - upsample_factor) / upsample_factor
        shifts += np.array([dxf, dyf]) / upsample_factor

    shifts = (shifts + 0.5 * np.array(cc.shape)) % cc.shape - 0.5 * np.array(cc.shape)

    if not return_shifted_image:
        return shifts

    # Fourier shift image (F_im assumed to be FFT)
    kx = xp.fft.fftfreq(F_im.shape[0])[:, None]
    ky = xp.fft.fftfreq(F_im.shape[1])[None, :]
    phase_ramp = xp.exp(-2j * np.pi * (kx * shifts[0] + ky * shifts[1]))
    F_im_shifted = F_im * phase_ramp
    if fft_output:
        image_shifted = F_im_shifted
    else:
        image_shifted = xp.real(xp.fft.ifft2(F_im_shifted))

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

    # convert to centered signed shifts as original code
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

    # local max (integer)
    flat_idx = torch.argmax(cc_real)
    x0 = (flat_idx // cc_real.shape[1]).to(torch.long).item()
    y0 = (flat_idx % cc_real.shape[1]).to(torch.long).item()

    # half pixel shifts: pick ±1 indices with wrap (mod)
    M, N = cc_real.shape
    x_inds = [((x0 + dx) % M) for dx in (-1, 0, 1)]
    y_inds = [((y0 + dy) % N) for dy in (-1, 0, 1)]

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    # parabolic half-pixel refine
    # dx = (vx[2] - vx[0]) / (4*vx[1] - 2*vx[2] - 2*vx[0])
    denom_x = 4.0 * vx[1] - 2.0 * vx[2] - 2.0 * vx[0]
    denom_y = 4.0 * vy[1] - 2.0 * vy[2] - 2.0 * vy[0]
    dx = (vx[2] - vx[0]) / denom_x if denom_x != 0 else torch.tensor(0.0, device=device)
    dy = (vy[2] - vy[0]) / denom_y if denom_y != 0 else torch.tensor(0.0, device=device)

    # round to nearest half-pixel
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

    # find maximum
    # flatten argmax -> unravel to 2D
    flat_idx = torch.argmax(imageCorrUpsample.real)
    # unravel_index
    xySubShift0 = (flat_idx // imageCorrUpsample.shape[1]).to(torch.long)
    xySubShift1 = (flat_idx % imageCorrUpsample.shape[1]).to(torch.long)
    xySubShift = torch.tensor([xySubShift0.item(), xySubShift1.item()])

    # parabolic subpixel refinement
    dx = 0.0
    dy = 0.0
    try:
        # extract 3x3 patch around found peak
        r = xySubShift[0].item()
        c = xySubShift[1].item()
        patch = imageCorrUpsample.real[r - 1 : r + 2, c - 1 : c + 2]
        # if patch is incomplete (near edge) this will raise / have wrong shape -> except
        if patch.shape == (3, 3):
            icc = patch
            # dx corresponds to row direction (vertical axis) as in original code:
            dx = (icc[2, 1] - icc[0, 1]) / (4.0 * icc[1, 1] - 2.0 * icc[2, 1] - 2.0 * icc[0, 1])
            dy = (icc[1, 2] - icc[1, 0]) / (4.0 * icc[1, 1] - 2.0 * icc[1, 2] - 2.0 * icc[1, 0])
            dx = dx.item()
            dy = dy.item()
        else:
            dx, dy = 0.0, 0.0
    except Exception:
        dx, dy = 0.0, 0.0

    # convert xySubShift to zero-centered by subtracting globalShift
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

    # prepare the vectors exactly like the numpy version
    # col: frequency indices (centered) for N
    col_freq = torch.fft.ifftshift(torch.arange(N, device=device)) - math.floor(N / 2)
    # row: frequency indices (centered) for M
    row_freq = torch.fft.ifftshift(torch.arange(M, device=device)) - math.floor(M / 2)

    # small upsample grid coordinates (integer positions in the UPSAMPLED GRID)
    col_coords = torch.arange(numCol, device=device, dtype=torch.get_default_dtype()) - float(
        xyShift[1]
    )
    row_coords = torch.arange(numRow, device=device, dtype=torch.get_default_dtype()) - float(
        xyShift[0]
    )

    # build kernels: note factor signs and denominators match original numpy code
    # colKern: shape (N, numCol)
    factor_col = -2j * math.pi / (N * float(upsampleFactor))
    # outer(col_freq, col_coords) -> shape (N, numCol)
    colKern = torch.exp(factor_col * (col_freq.unsqueeze(1) * col_coords.unsqueeze(0))).to(
        imageCorr.dtype
    )

    # rowKern: shape (numRow, M)
    factor_row = -2j * math.pi / (M * float(upsampleFactor))
    # outer(row_coords, row_freq) -> shape (numRow, M)
    rowKern = torch.exp(factor_row * (row_coords.unsqueeze(1) * row_freq.unsqueeze(0))).to(
        imageCorr.dtype
    )

    # perform the small-matrix DFT: (numRow, M) @ (M, N) @ (N, numCol) -> (numRow, numCol)
    imageUpsample = rowKern @ imageCorr @ colKern

    # original code took xp.real(...) before returning
    return imageUpsample.real


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

    Parameters
    ----------
    xa : NDArray
        Vertical (row) coordinates of input points.
    ya : NDArray
        Horizontal (col) coordinates of input points.
    values : NDArray
        Weights for each (xa, ya) point.
    output_shape : tuple of int
        Output image shape (rows, cols).
    kde_sigma : float
        Standard deviation of Gaussian KDE smoothing.
    pad_value : float, default = 1.0
        Value to return when KDE support is too low.
    threshold : float, default = 1e-3
        Minimum counts_KDE value for trusting the output signal.
    lowpass_filter : bool, optional
        If True, apply sinc-based inverse filtering to deconvolve the kernel.
    max_batch_size : int or None, optional
        Max number of points to process in one batch.

    Returns
    -------
    NDArray
        The estimated KDE image with threshold-masked output.
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

    # Reshape to 2D and apply Gaussian KDE
    pix_count = pix_count.reshape(output_shape)
    pix_output = pix_output.reshape(output_shape)

    pix_count = gaussian_filter(pix_count, kde_sigma)
    pix_output = gaussian_filter(pix_output, kde_sigma)

    # Final image
    weight = np.minimum(pix_count / threshold, 1.0)
    image = pad_value * (1.0 - weight) + weight * (pix_output / np.maximum(pix_count, 1e-8))

    if lowpass_filter:
        f_img = np.fft.fft2(image)
        fx = np.fft.fftfreq(rows)
        fy = np.fft.fftfreq(cols)
        f_img /= np.sinc(fx)[:, None]
        f_img /= np.sinc(fy)[None, :]
        image = np.real(np.fft.ifft2(f_img))

        if return_pix_count:
            f_img = np.fft.fft2(pix_count)
            f_img /= np.sinc(fx)[:, None]
            f_img /= np.sinc(fy)[None, :]
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

    Parameters
    ----------
    image: np.ndarray
        Image array to sample from
    xa: np.ndarray
        Vertical interpolation sampling positions of image array in pixels
    ya: np.ndarray
        Horizontal interpolation sampling positions of image array in pixels

    Returns
    -------
    values: np.ndarray
        Bilinear interpolation values of array at (xa,ya) positions

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

    values = np.reshape(
        values,
        xa.shape,
    )

    return values


def fourier_cropping(
    corner_centered_array: NDArray,
    crop_shape: Tuple[int, int],
):
    """
    Crops a corner-centered FFT array to retain only the lowest frequencies,
    equivalent to a center crop on the fftshifted version.

    Parameters:
    -----------
    corner_centered_array : ndarray
        2D array (typically result of np.fft.fft2) with corner-centered DC
    crop_shape : tuple of int
        (height, width) of the desired cropped array (could be odd or even depending on arr.shape)

    Returns:
    --------
    cropped : ndarray
        Cropped array containing only the lowest frequencies, still corner-centered.
    """

    H, W = corner_centered_array.shape
    crop_h, crop_w = crop_shape

    h1 = crop_h // 2
    h2 = crop_h - h1
    w1 = crop_w // 2
    w2 = crop_w - w1

    result = np.zeros(crop_shape, dtype=corner_centered_array.dtype)

    # Top-left
    result[:h1, :w1] = corner_centered_array[:h1, :w1]
    # Top-right
    result[:h1, -w2:] = corner_centered_array[:h1, -w2:]
    # Bottom-left
    result[-h2:, :w1] = corner_centered_array[-h2:, :w1]
    # Bottom-right
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

    Parameters
    ----------
    halfset_recons : list[torch.Tensor]
        Two statistically-independent reconstructions, using half the dataset.
    sampling: tuple[float,float]
        Reconstruction sampling in Angstroms.
    epsilon: float, optional
        Small number to avoid dividing by zero

    Returns
    -------
    q_bins: NDarray
        Spatial frequency bins
    fsc : NDarray
        Fourier shell correlation as function of spatial frequency
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

    # Flatten arrays
    cross = cross.reshape(-1)
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)

    # Accumulate
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

    The method decomposes the Fourier transforms into:
    - Symmetric: (F₁ + F₂)/2  → signal + correlated noise
    - Antisymmetric: (F₁ - F₂)/2  → uncorrelated noise only

    SSNR(q) = sqrt(signal_power / noise_power)

    where:
    - signal_power = (|symmetric|² - |antisymmetric|²)₊
    - noise_power = |antisymmetric|²

    Parameters
    ----------
    halfset_recons : list[torch.Tensor]
        Two statistically-independent reconstructions, using half the dataset.
    sampling: tuple[float,float]
        Reconstruction sampling in Angstroms.
    total_dose: float
        Total _normalized_ electron dose, e.g. in DirectPtychography this is ~self.num_bf
    epsilon: float, optional
        Small number to avoid dividing by zero

    Returns
    -------
    q_bins: NDarray
        Spatial frequency bins
    ssnr : NDarray
        Radially averaged spectral SNR as function of spatial frequency
    """
    # Compute Fourier transforms
    halfset_1, halfset_2 = halfset_recons
    F1 = torch.fft.fft2(halfset_1)
    F2 = torch.fft.fft2(halfset_2)

    # Symmetric and antisymmetric decomposition
    symmetric = (F1 + F2) / 2
    antisymmetric = (F1 - F2) / 2

    # Power spectra
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

    # Flatten arrays
    signal = signal_power.reshape(-1)
    noise = noise_power.reshape(-1)

    # Accumulate
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

    Parameters
    ----------
    corner_centered_array : list[torch.Tensor]
        Fourier array to average radially.
    sampling: tuple[float,float]
        Reconstruction sampling in Angstroms.

    Returns
    -------
    q_bins: NDarray
        Spatial frequency bins
    array_1d : NDarray
        Radially averaged Fourier array as function of spatial frequency
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

    # Flatten arrays
    array = corner_centered_array.reshape(-1)

    # Accumulate
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

        edges.append(  # ty:ignore[possibly-missing-attribute]
            torch.stack([i1, i2, rel, inc], dim=1)
        )

    if wrap_around:
        add_edges(idx.flatten(), torch.roll(idx, -1, 1).flatten())
        add_edges(idx.flatten(), torch.roll(idx, -1, 0).flatten())
    else:
        add_edges(idx[:, :-1].flatten(), idx[:, 1:].flatten())
        add_edges(idx[:-1, :].flatten(), idx[1:, :].flatten())

    edges = torch.cat(edges, dim=0)
    edges = edges[edges[:, 2].argsort()]

    # return integer tensors only (CPU)
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

        # phase(y) + oy + inc = phase(x) + ox
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

    Parameters
    ----------
    phi_wrapped : (H, W) tensor
        Wrapped phase in (-pi, pi], any device
    mask : (H, W) bool tensor, optional
        True = valid pixel

    Returns
    -------
    phi_unwrapped : (H, W) tensor
        Unwrapped phase (same device as input)
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
    denom[0, 0] = 1.0  # avoid divide by zero

    phi_hat = -div_hat / denom
    phi_hat[0, 0] = 0.0  # fix piston

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


def radially_project_fourier_tensor(
    corner_centered_array: torch.Tensor,
    sampling: Tuple[float, float],
    q_bins: torch.Tensor | None = None,
):
    """
    Radially project a corner-centered Fourier array onto radial bins.

    Supports:
    - single array: (kx, ky)
    - batched arrays: (n, kx, ky)
    - implicit bins (from grid) or explicit external bins

    Parameters
    ----------
    corner_centered_array : torch.Tensor
        Shape (kx, ky) or (n, kx, ky)
    sampling : (float, float)
        Real-space sampling (sx, sy)
    q_bins : torch.Tensor, optional
        1D tensor of radial bin centers

    Returns
    -------
    q_bins_out : torch.Tensor
        Radial bin centers
    array_1d : torch.Tensor
        Shape (n, nq) or (nq,)
    """

    if corner_centered_array.is_complex():
        q, real_part = radially_project_fourier_tensor(
            corner_centered_array.real, sampling, q_bins
        )
        # zero by symmetry
        # _, imag_part = radially_project_fourier_tensor(
        #     corner_centered_array.imag, sampling, q_bins
        # )
        return q, real_part  # + 1j * imag_part

    device = corner_centered_array.device
    sx, sy = sampling

    # --- normalize shape to (batch, kx, ky)
    if corner_centered_array.ndim == 2:
        array = corner_centered_array[None, ...]
        squeeze_output = True
    elif corner_centered_array.ndim == 3:
        array = corner_centered_array
        squeeze_output = False
    else:
        raise ValueError("Input must be 2D or 3D tensor")

    n_batch, nkx, nky = array.shape

    # --- build k-grid
    kx = torch.fft.fftfreq(nkx, d=sx, device=device)
    ky = torch.fft.fftfreq(nky, d=sy, device=device)
    k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).reshape(-1)

    # --- determine radial bins
    k_max = min(0.5 / sx, 0.5 / sy)

    if q_bins is None:
        dk = kx[1] - kx[0]
        num_bins = int(torch.floor(k_max / dk).item()) + 1
        dq = dk
        q_bins_out = torch.arange(num_bins, device=device, dtype=k.dtype) * dk
    else:
        q_bins = q_bins.to(device)
        dq = q_bins[1] - q_bins[0]
        q_bins_out = q_bins[q_bins <= k_max]
        num_bins = q_bins_out.numel()

    # ---- INTERNAL padding (key change)
    num_bins_internal = num_bins + 2

    # --- map k -> bin indices (NO CLIPPING)
    inds = k / dq
    inds_f = torch.floor(inds).long()
    inds_f = torch.clamp(inds_f, 0, num_bins_internal - 2)

    d_ind = inds - inds_f
    w0 = 1.0 - d_ind
    w1 = d_ind

    # --- flatten spatial dims
    array_f = array.reshape(n_batch, -1)

    # --- accumulate per batch
    out = []
    for b in range(n_batch):
        a = array_f[b]

        num = torch.bincount(inds_f, weights=a * w0, minlength=num_bins_internal) + torch.bincount(
            inds_f + 1, weights=a * w1, minlength=num_bins_internal
        )

        den = (
            torch.bincount(inds_f, weights=w0, minlength=num_bins_internal)
            + torch.bincount(inds_f + 1, weights=w1, minlength=num_bins_internal)
        ).clamp_min(1)

        out.append(num / den)

    array_1d = torch.stack(out, dim=0)

    # ---- truncate to physical bins (key change)
    array_1d = array_1d[..., :num_bins]
    q_bins_out = q_bins_out[:num_bins]

    if squeeze_output:
        array_1d = array_1d[0]

    return q_bins_out, array_1d
