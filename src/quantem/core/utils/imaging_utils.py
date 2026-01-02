# Utilities for processing images

import math
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates

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
) -> NDArray:
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
