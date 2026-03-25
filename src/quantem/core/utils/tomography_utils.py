from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm  # TODO: Temporary
from numpy.typing import NDArray
from scipy.ndimage import center_of_mass, gaussian_filter, shift
from scipy.special import comb
from tqdm.auto import tqdm

from quantem.core.utils.imaging_utils import cross_correlation_shift
from quantem.core.utils.utils import to_numpy
from quantem.core.visualization import show_2d

ImageType = NDArray[Any]
BoolArray = NDArray[np.bool_]


def _bernstein_basis_1d(n: int, t: NDArray[Any]) -> NDArray[Any]:
    k = np.arange(n + 1, dtype=int)
    return (
        comb(n, k)[None, :] * (t[:, None] ** k[None, :]) * ((1.0 - t)[:, None] ** (n - k)[None, :])
    )


def _build_basis_matrix(im_shape: Tuple[int, int], order: Tuple[int, int]) -> NDArray[Any]:
    H, W = im_shape
    ou, ov = int(order[0]), int(order[1])
    u = np.linspace(0.0, 1.0, H)
    v = np.linspace(0.0, 1.0, W)
    Bu = _bernstein_basis_1d(ou, u)
    Bv = _bernstein_basis_1d(ov, v)
    basis_cube = np.einsum("ik,jl->ijkl", Bu, Bv)
    return basis_cube.reshape(H * W, (ou + 1) * (ov + 1))


def background_subtract(
    image: ImageType,
    mask: Optional[BoolArray] = None,
    thresh_bg: Optional[float] = None,
    order: Tuple[int, int] = (1, 1),
    sigma: Optional[float] = None,
    num_iter: int = 10,
    plot_result: bool = True,
    axsize: Tuple[int, int] = (3, 3),
    cmap: str = "turbo",
    return_background_and_mask: bool = False,
    **show_kwargs,
) -> ImageType | Tuple[ImageType, NDArray[Any], BoolArray]:
    """
    Background subtraction via bivariate Bernstein polynomial fitting.

    Returns
    -------
    - If `return_background_and_mask=False`: ImageType (same as input)
    - If `True`: (ImageType, numpy.ndarray, numpy.ndarray[bool])
      where background and mask are always NumPy.
    """
    im = to_numpy(image).astype(float, copy=True)
    if im.ndim != 2:
        raise ValueError("`image` must be 2D")

    mask_arr: BoolArray = (
        np.ones_like(im, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    if mask_arr.shape != im.shape:
        raise ValueError("`mask` must match `image` shape")

    order = (int(order[0]), int(order[1]))
    A_full = _build_basis_matrix(im.shape, order)
    H, W = im.shape
    im_flat = im.ravel()

    im_bg = np.zeros_like(im)
    thresh_val = np.median(im[mask_arr]) if thresh_bg is None else float(thresh_bg)

    resid = im - im_bg
    if sigma and sigma > 0:
        resid = gaussian_filter(resid, sigma=sigma, mode="nearest")
    mask_bg: BoolArray = (resid < thresh_val) & mask_arr

    for _ in range(int(num_iter)):
        idx = mask_bg.ravel()
        if not np.any(idx):
            idx = mask_arr.ravel()
        coefs, *_ = np.linalg.lstsq(A_full[idx, :], im_flat[idx], rcond=None)
        im_bg = (A_full @ coefs).reshape(H, W)

        resid = im - im_bg
        if sigma and sigma > 0:
            resid = gaussian_filter(resid, sigma=sigma, mode="nearest")

        thr = thresh_val if thresh_bg is None else float(thresh_bg)
        mask_bg = (resid < thr) & mask_arr

    im_sub_np = im - im_bg

    if plot_result:
        vals = im_sub_np[mask_arr]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = np.array([0.0])
        vmin_sub = float(np.min(vals))
        vmax_sub = float(np.max(vals))
        vrange = float(max(abs(vmin_sub), abs(vmax_sub))) or 1e-12

        bg_disp = (im_bg - np.mean(im_bg)).copy()
        bg_disp[~mask_bg] = np.nan

        cmap_base = cm.get_cmap(cmap).with_extremes(bad="black")
        cmap_div = "RdBu_r"

        disp = [im - np.mean(im_bg), bg_disp, im_sub_np]
        norm = [
            {
                "interval_type": "manual",
                "stretch_type": "linear",
                "vmin": vmin_sub,
                "vmax": vmax_sub,
            },
            {
                "interval_type": "manual",
                "stretch_type": "linear",
                "vmin": vmin_sub,
                "vmax": vmax_sub,
            },
            {
                "interval_type": "centered",
                "stretch_type": "linear",
                "vcenter": 0.0,
                "half_range": vrange,
            },
        ]

        show_2d(
            disp,
            cmap=[cmap_base, cmap_base, cmap_div],
            norm=norm,
            cbar=[False, False, True],
            title=["Input Image", "Background (fit region)", "Background Subtracted"],
            axsize=axsize,
            **show_kwargs,
        )

    # # preserve  if needed
    # if isinstance(
    #     image,
    # ):
    #     meta = dict(origin=image.origin, sampling=image.sampling, units=image.units)
    #     name_base = getattr(image, "name", "image")
    # im_sub: ImageType = .from_array(im_sub_np, name=f"{name_base} (bg-sub)", **meta)  # type: ignore[assignment]
    # else:
    im_sub = im_sub_np  # type: ignore[assignment]

    if return_background_and_mask:
        return im_sub, im_bg, mask_bg
    return im_sub


# --- Tilt Series Processing Utility Functions ---


def fourier_binning(img, crop_size):
    """
    Crop the img in Fourier space to the specified size.
    """
    center = np.array(img.shape) // 2

    fft_img = np.fft.fftshift(np.fft.fft2(img))

    cropped_fft = fft_img[
        center[0] - crop_size[0] // 2 : center[0] + crop_size[0] // 2,
        center[1] - crop_size[1] // 2 : center[1] + crop_size[1] // 2,
    ]
    cropped_img = np.fft.ifft2(np.fft.ifftshift(cropped_fft)).real
    return cropped_img


def cross_correlation_align_stack(ref_img, stack, print_pred=False):
    """
    Aligns a stack of images to a reference image using cross-correlation.

    This function assumes the stack does not contain the reference image itself.

    Stack shape should be (N, H, W) where N is the number of images.
    """

    new_images = []
    pred_shifts = []

    prev_img = ref_img
    for img in tqdm(stack):
        shift_pred = cross_correlation_shift(prev_img, img)
        if print_pred:
            print(f"Shift prediction: {shift_pred}")
        shifted_image = shift(img, shift=shift_pred, mode="constant", cval=0.0)

        pred_shifts.append(shift_pred)
        new_images.append(shifted_image)

        prev_img = shifted_image

    return new_images, pred_shifts


def centering_com_alignment(image_stack):
    """
    Aligns the image stack to the center of mass of the whole image_stack to the
    image center. This is useful for aligning the tilt series to the invariant line.
    """

    aligned_stack = np.zeros_like(image_stack)
    h, w = image_stack.shape[1:]
    image_center = np.array([h // 2, w // 2])

    com_reference = np.array(center_of_mass(image_stack.mean(axis=0)))

    for i, img in enumerate(image_stack):
        com_img = np.array(center_of_mass(img))
        shift_vec = com_reference - com_img
        aligned_stack[i] = shift(img, shift=shift_vec, mode="constant", cval=0.0)

    final_shift = image_center - com_reference
    for i in range(aligned_stack.shape[0]):
        aligned_stack[i] = shift(aligned_stack[i], shift=final_shift, mode="constant", cval=0.0)

    return aligned_stack


def differentiable_shift_2d(image, shift_x, shift_y, sampling_rate):
    """
    Shifts a 2D image using grid_sample in a differentiable manner with zero-pad boundary conditions applied.

    Args:
        image: Tensor of shape [H, W]
        shift_x: Scalar tensor (dx) for shift in x-direction (in physical units)
        shift_y: Scalar tensor (dy) for shift in y-direction (in physical units)
        sampling_rate: Scalar value (physical units per pixel) to correctly normalize shifts

    Returns:
        Shifted image of shape [H, W]
    """
    H, W = image.shape

    # Convert physical shift to pixel shift
    shift_x_pixel = shift_x
    shift_y_pixel = shift_y

    # Normalize shift for grid_sample (assuming align_corners=True)
    normalized_shift_x = shift_x_pixel * 2 / (W - 1)
    normalized_shift_y = shift_y_pixel * 2 / (H - 1)

    # Create normalized grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij",
    )

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Apply shift (ensure it's differentiable)
    grid[:, :, :, 0] -= normalized_shift_x
    grid[:, :, :, 1] -= normalized_shift_y

    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Sample using grid_sample (fully differentiable)
    shifted_image = F.grid_sample(
        image, grid, mode="bicubic", padding_mode="zeros", align_corners=True
    )

    return shifted_image.squeeze(0).squeeze(0)  # Back to [H, W]


# --- TV loss ---


def get_TV_loss(tensor, factor=1e-3):
    tv_d = torch.pow(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :], 2).sum()
    tv_h = torch.pow(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :], 2).sum()
    tv_w = torch.pow(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1], 2).sum()
    tv_loss = tv_d + tv_h + tv_w

    return tv_loss * factor / (torch.prod(torch.tensor(tensor.shape)))


# Circular mask


def torch_phase_cross_correlation(im1, im2):
    f1 = torch.fft.fft2(im1)
    f2 = torch.fft.fft2(im2)
    cc = torch.fft.ifft2(f1 * torch.conj(f2))
    cc_abs = torch.abs(cc)

    max_idx = torch.argmax(cc_abs)
    shifts = torch.tensor(np.unravel_index(max_idx.item(), im1.shape), device=im1.device).float()

    for i, dim in enumerate(im1.shape):
        if shifts[i] > dim // 2:
            shifts[i] -= dim

    # return shifts.flip(0)  # (dx, dy)
    return shifts
