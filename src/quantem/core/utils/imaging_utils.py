# Utilities for processing images

from typing import Optional, Tuple

import numpy as np
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
