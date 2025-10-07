# Utilities for processing images
from __future__ import annotations
from typing import Optional, Tuple, Union, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.special import comb
from matplotlib import cm

from quantem.core.utils.utils import generate_batches
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.visualization import show_2d

ArrayOrDS = Union[NDArray, Dataset2d]


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


def _as_array(x: NDArray | Dataset2d) -> NDArray:
    return x.array if isinstance(x, Dataset2d) else np.asarray(x)


def _bernstein_basis_1d(n: int, t: NDArray) -> NDArray:
    k = np.arange(n + 1, dtype=int)
    return comb(n, k)[None, :] * (t[:, None] ** k[None, :]) * ((1.0 - t)[:, None] ** (n - k)[None, :])


def _build_basis_matrix(im_shape: Tuple[int, int], order: Tuple[int, int]) -> NDArray:
    H, W = im_shape
    ou, ov = int(order[0]), int(order[1])
    u = np.linspace(0.0, 1.0, H)
    v = np.linspace(0.0, 1.0, W)
    Bu = _bernstein_basis_1d(ou, u)   # (H, ou+1)
    Bv = _bernstein_basis_1d(ov, v)   # (W, ov+1)
    basis_cube = np.einsum("ik,jl->ijkl", Bu, Bv)  # (H, W, ou+1, ov+1)
    return basis_cube.reshape(H * W, (ou + 1) * (ov + 1))


# --- Overloads: NDArray input ---
@overload
def background_subtract(
    image: NDArray,
    mask: Optional[NDArray] = ...,
    thresh_bg: Optional[float] = ...,
    order: Tuple[int, int] = ...,
    sigma: Optional[float] = ...,
    num_iter: int = ...,
    plot_result: bool = ...,
    axsize: Tuple[int, int] = ...,
    cmap: str = ...,
    return_background_and_mask: Literal[False] = ...,
    **show_kwargs,
) -> NDArray: ...
@overload
def background_subtract(
    image: NDArray,
    mask: Optional[NDArray] = ...,
    thresh_bg: Optional[float] = ...,
    order: Tuple[int, int] = ...,
    sigma: Optional[float] = ...,
    num_iter: int = ...,
    plot_result: bool = ...,
    axsize: Tuple[int, int] = ...,
    cmap: str = ...,
    return_background_and_mask: Literal[True] = ...,
    **show_kwargs,
) -> Tuple[NDArray, NDArray, NDArray]: ...

# --- Overloads: Dataset2d input (mask stays NDArray) ---
@overload
def background_subtract(
    image: Dataset2d,
    mask: Optional[NDArray] = ...,
    thresh_bg: Optional[float] = ...,
    order: Tuple[int, int] = ...,
    sigma: Optional[float] = ...,
    num_iter: int = ...,
    plot_result: bool = ...,
    axsize: Tuple[int, int] = ...,
    cmap: str = ...,
    return_background_and_mask: Literal[False] = ...,
    **show_kwargs,
) -> Dataset2d: ...
@overload
def background_subtract(
    image: Dataset2d,
    mask: Optional[NDArray] = ...,
    thresh_bg: Optional[float] = ...,
    order: Tuple[int, int] = ...,
    sigma: Optional[float] = ...,
    num_iter: int = ...,
    plot_result: bool = ...,
    axsize: Tuple[int, int] = ...,
    cmap: str = ...,
    return_background_and_mask: Literal[True] = ...,
    **show_kwargs,
) -> Tuple[Dataset2d, Dataset2d, NDArray]: ...

def background_subtract(
    image: NDArray | Dataset2d,
    mask: Optional[NDArray] = None,        # boolean numpy array or None
    thresh_bg: Optional[float] = None,
    order: Tuple[int, int] = (2, 2),
    sigma: Optional[float] = None,
    num_iter: int = 10,
    plot_result: bool = True,
    axsize: Tuple[int, int] = (3.1, 3),
    cmap: str = "turbo",
    return_background_and_mask: bool = False,
    **show_kwargs,
):
    """
    Background subtraction via bi-variate Bernstein (Bézier) polynomial fitting.

    Parameters
    ----------
    image : numpy.ndarray or Dataset2d
        2D input image. If a Dataset2d is provided, only its array is used for the fit;
        outputs are returned as Dataset2d (same metadata preserved).
    mask : numpy.ndarray of bool, optional
        Boolean mask (same shape as `image`) indicating valid pixels for fitting and for
        range calculations in the diagnostic plot. If None, all pixels are valid.
    thresh_bg : float, optional
        Background threshold to classify pixels as background during robust iterations.
        If None, initialized from the median of `image[mask]`.
    order : tuple[int, int], default (2, 2)
        Polynomial order (u, v) for the Bernstein basis along the two axes.
        Number of basis terms = (order[0] + 1) * (order[1] + 1).
    sigma : float, optional
        Gaussian sigma (pixels) to smooth residuals before thresholding at each iteration.
        If None or 0, no smoothing is applied.
    num_iter : int, default 10
        Number of robust fitting iterations (refit background, update mask).
    plot_result : bool, default True
        If True, displays a 3-panel diagnostic:
          (1) Input (mean-centered by background mean),
          (2) Fitted background shown only where `mask_bg` is True (else black),
          (3) Background-subtracted image with `RdBu_r`, zero-centered, symmetric ±min(|min|,|max|)
              computed from `im_sub[mask]`. Colorbar is shown on panel (3).
    axsize : tuple[int, int], default (3.1, 3)
        Panel size in inches, forwarded to `show_2d`.
    cmap : str, default "turbo"
        Colormap for panels (1) and (2). Panel (3) always uses "RdBu_r".
    return_background_and_mask : bool, default False
        If True, also return the fitted background and the final background mask.

    **show_kwargs
        Additional arguments passed to `quantem.core.visualization.show_2d`.

    Returns
    -------
    im_sub : numpy.ndarray or Dataset2d
        Background-subtracted image (same type as `image`).
    im_bg : numpy.ndarray or Dataset2d, optional
        Fitted background (same type as `image`). Returned if `return_background_and_mask=True`.
    mask_bg : numpy.ndarray of bool, optional
        Final background mask (always NumPy). Returned if `return_background_and_mask=True`.

    Notes
    -----
    - The right-panel plot uses `im_sub[mask]` to compute a symmetric, zero-centered range
      so white corresponds to 0.0 in RdBu_r.
    """
    # --- inputs ---
    im = _as_array(image).astype(float, copy=True)
    if im.ndim != 2:
        raise ValueError("`image` must be a 2D numpy array or Dataset2d")

    if mask is None:
        mask_arr = np.ones_like(im, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != im.shape:
            raise ValueError("`mask` must have the same shape as `image`.")

    # --- basis ---
    order = (int(order[0]), int(order[1]))
    A_full = _build_basis_matrix(im.shape, order)  # (H*W, K)
    H, W = im.shape
    im_flat = im.ravel()

    # --- init background & mask ---
    im_bg = np.zeros_like(im)
    thresh_val = np.median(im[mask_arr]) if thresh_bg is None else float(thresh_bg)

    resid = im - im_bg
    if sigma is not None and sigma > 0:
        resid = gaussian_filter(resid, sigma=sigma, mode="nearest")
    mask_bg = resid < thresh_val
    mask_bg &= mask_arr

    # --- iterations ---
    for _ in range(int(num_iter)):
        idx = mask_bg.ravel()
        if not np.any(idx):
            idx = mask_arr.ravel()  # ensure solvable if mask collapses

        coefs, *_ = np.linalg.lstsq(A_full[idx, :], im_flat[idx], rcond=None)
        im_bg = (A_full @ coefs).reshape(H, W)

        resid = im - im_bg
        if sigma is not None and sigma > 0:
            resid = gaussian_filter(resid, sigma=sigma, mode="nearest")

        thr = thresh_val if thresh_bg is None else float(thresh_bg)
        mask_bg = (resid < thr)
        mask_bg &= mask_arr

    # --- final subtraction ---
    im_sub = im - im_bg

    # --- plotting (right panel: zero-centered RdBu_r; limits from im_sub[mask]) ---
    if plot_result:
        # Range from masked region
        vals = im_sub[mask_arr]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = np.array([0.0])

        vmin_sub = float(np.min(vals))
        vmax_sub = float(np.max(vals))
        vrange   = float(np.maximum(abs(vmin_sub), abs(vmax_sub)))  # ±min(|min|,|max|)
        if vrange <= 0:
            vrange = 1e-12

        # Background panel: show only fitted region; black elsewhere
        bg_disp = (im_bg - np.mean(im_bg)).copy()
        bg_disp[~mask_bg] = np.nan

        # Colormaps
        cmap_base = cm.get_cmap(cmap).with_extremes(bad="black")  # NaNs -> black
        cmap_div  = "RdBu_r"

        # Panels: input, background(masked), background-subtracted
        disp = [
            im - np.mean(im_bg),
            bg_disp,
            im_sub,
        ]

        # Per-panel normalization:
        norm = [
            {"interval_type": "manual",   "stretch_type": "linear", "vmin": vmin_sub, "vmax": vmax_sub},   # input
            {"interval_type": "manual",   "stretch_type": "linear", "vmin": vmin_sub, "vmax": vmax_sub},   # background
            {"interval_type": "centered", "stretch_type": "linear", "vcenter": 0.0,   "half_range": vrange},  # zero-centered
        ]

        show_2d(
            disp,
            cmap=[cmap_base, cmap_base, cmap_div],
            norm=norm,
            cbar=[False, False, True],  # colorbar only on right panel
            title=["Input Image", "Background (fit region)", "Background Subtracted"],
            axsize=axsize,
            **show_kwargs,
        )

    # --- outputs (match image type for images; mask always ndarray) ---
    if isinstance(image, Dataset2d):
        meta = dict(origin=image.origin, sampling=image.sampling, units=image.units)
        name_base = getattr(image, "name", "image")
        im_sub_ds = Dataset2d.from_array(im_sub, name=f"{name_base} (bg-sub)", **meta)
        im_bg_ds  = Dataset2d.from_array(im_bg,  name=f"{name_base} (bg-fit)", **meta)
        if return_background_and_mask:
            return im_sub_ds, im_bg_ds, mask_bg
        else:
            return im_sub_ds
    else:
        if return_background_and_mask:
            return im_sub, im_bg, mask_bg
        else:
            return im_sub
