import math
from itertools import product
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.patches import Circle, Ellipse
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from quantem.core import config

if TYPE_CHECKING:
    import cupy as cp  # type: ignore
    import torch  # type: ignore
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


# region --- array module stuff ---


def get_array_module(array: "np.ndarray | cp.ndarray"):
    """Returns np or cp depending on the array type."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    raise ValueError(f"Input is not a numpy array or cupy array: {type(array)}")


def get_tensor_module(tensor: "np.ndarray | cp.ndarray | torch.Tensor"):
    """
    This is like get_array_module but includes torch. It is kept explicitly separate as in most
    cases get_array_module is used, and that fails if given a torch.Tensor.
    """
    if config.get("has_torch"):
        if isinstance(tensor, torch.Tensor):
            return torch
    if config.get("has_cupy"):
        if isinstance(tensor, cp.ndarray):
            return cp
    if isinstance(tensor, np.ndarray):
        return np
    raise ValueError(f"Input is not a numpy array, cupy array, or torch tensor: {type(tensor)}")


def to_numpy(array: "np.ndarray | cp.ndarray | torch.Tensor") -> np.ndarray:
    """Convert a torch.Tensor or cupy.ndarray to a numpy.ndarray. Always returns
    a copy for consistency."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return array.cpu().detach().numpy()
    if isinstance(array, np.ndarray):
        return np.array(array)
    elif isinstance(array, (tuple, list)):
        return np.array(array)
    #     return np.array([to_numpy(i) for i in array])
    # elif isinstance(array, (float, int, bool)):
    # return np.array(array)
    raise TypeError(
        f"Input should be np.ndarray, cp.ndarray, or torch.Tensor, tuple, or list. Got: {type(array)}"
    )


def to_cpu(arrs: Any) -> np.ndarray | Sequence:
    """
    Similar to to_numpy but also allows lists and handles Datasets, only called in show_2d
    so could likely be replaced with a more specific function.
    """
    from quantem.core.datastructures import Dataset2d

    if config.get("has_cupy"):
        if isinstance(arrs, cp.ndarray):
            return cp.asnumpy(arrs)
    if config.get("has_torch"):
        if isinstance(arrs, torch.Tensor):
            return arrs.cpu().detach().numpy()
    if isinstance(arrs, np.ndarray):
        return np.array(arrs)
    elif isinstance(arrs, list):
        return [to_cpu(i) for i in arrs]
    elif isinstance(arrs, tuple):
        return tuple([to_cpu(i) for i in arrs])
    elif isinstance(arrs, Dataset2d):
        return to_cpu(arrs.array)
    else:
        raise NotImplementedError(f"Unkown type: {type(arrs)}")


# endregion


# region --- TEM ---


def electron_wavelength_angstrom(E_eV: float):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = h / math.sqrt(2 * m * e * E_eV) / math.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
    return lam


# endregion


# region --- generally useful bits ---


def tqdmnd(*iterables, **kwargs):
    return tqdm(list(product(*iterables)), **kwargs)


def subdivide_batches(
    num_items: int,
    num_batches: Optional[int] = None,
    max_batch: Optional[int] = None,
) -> List[int]:
    """
    Split `num_items` into a list of batch sizes.

    Parameters
    ----------
    num_items : int
        Total number of items to split into batches.
    num_batches : int, optional
        Number of desired batches. Cannot be used with `max_batch`.
    max_batch : int, optional
        Maximum batch size. Cannot be used with `num_batches`.

    Returns
    -------
    List[int]
        List of batch sizes that sum to `num_items`.
    """
    if num_batches is not None and max_batch is not None:
        raise RuntimeError("Specify only one of `num_batches` or `max_batch`.")

    if num_batches is None:
        if max_batch is None:
            raise RuntimeError("Must provide either `num_batches` or `max_batch`.")
        num_batches = (num_items + max_batch - 1) // max_batch

    if num_items < num_batches:
        raise ValueError("`num_batches` may not exceed `num_items`.")

    base_size = num_items // num_batches
    remainder = num_items % num_batches

    return [base_size + 1] * remainder + [base_size] * (num_batches - remainder)


def generate_batches(
    num_items: int,
    num_batches: Optional[int] = None,
    max_batch: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[Tuple[int, int]]:
    """
    Yield (start, end) index tuples for each batch.

    Parameters
    ----------
    num_items : int
        Total number of items to batch.
    num_batches : int, optional
        Number of batches. Cannot be used with `max_batch`.
    max_batch : int, optional
        Maximum size of each batch. Cannot be used with `num_batches`.
    start_index : int, default = 0
        Optional offset to start indexing from.

    Yields
    ------
    (int, int)
        Tuple of (start, end) indices for each batch.
    """
    batch_sizes = subdivide_batches(num_items, num_batches, max_batch)
    idx = start_index
    for size in batch_sizes:
        yield idx, idx + size
        idx += size


def median_filter(
    array: np.ndarray, size: int = 3, mask: np.ndarray | None = None, axes=(-2, -1)
) -> np.ndarray:
    """
    Apply a median filter to an array.
    """
    if mask is None:
        return ndi.median_filter(array, size=size, axes=axes)
    else:
        mask = mask.astype(bool)
        # make sure that the mask shape is of same dimension as axes specified
        if mask.ndim != len(axes):
            raise ValueError(
                "Mask must have same number of dimensions as axes. "
                + f"Mask has {mask.ndim} dimensions, axes has {len(axes)} dimensions."
            )

        arr = array.copy()
        for inds in np.argwhere(mask):
            med_slices = [slice(None)] * arr.ndim
            ind_slices = [slice(None)] * arr.ndim
            for ax in axes:
                axis_ind = inds[ax]
                ind_slices[ax] = axis_ind
                axis_size = arr.shape[ax]
                min_val = max(0, axis_ind - size // 2)
                max_val = min(axis_size, axis_ind + size // 2 + 1)
                med_slices[ax] = slice(min_val, max_val)
            window = arr[tuple(med_slices)]
            arr[tuple(ind_slices)] = np.median(window, axis=tuple(axes))
        return arr


def filter_hot_pixels(
    array: np.ndarray,
    threshold_std: float = 5,
    kernel_size: int = 9,
    use_channel_mean: bool = True,
) -> np.ndarray:
    """
    Filter hot pixels from an array.
    assumes that the last two axes are the image dimensions
    use_channel_mean: if True, use the mean of the channel to filter the hot pixels, if False will
    use the local region around each pixel in the image dimensions
    """
    if int(kernel_size) % 2 != 1:
        kernel_size = int(kernel_size) + 1

    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 0
    kernel = kernel / np.sum(kernel)

    if use_channel_mean:
        channel_mean = np.mean(array, axis=tuple(range(array.ndim - 2)))
        dimy, dimx = channel_mean.shape
        inds = np.mgrid[0:dimy, 0:dimx].reshape(2, dimy * dimx)
        patches = extract_patches(channel_mean, inds, patch_size=kernel_size)[None]
        local_mean = np.mean(patches, axis=(-2, -1)).reshape(dimy, dimx)
        dif = np.abs(channel_mean - local_mean)
        std = np.std(patches, axis=(-2, -1)).reshape(dimy, dimx)
        mask = dif > threshold_std * std
        # print("bads: ", np.where(mask))
        if np.any(mask):
            return median_filter(array, size=kernel_size, mask=mask)
        else:
            return array
    else:
        # use the local region around each pixel in the image dimensions
        raise NotImplementedError("Not implemented")


def extract_patches(
    array: np.ndarray | np.ma.MaskedArray, indices: np.ndarray | tuple, patch_size: int = 3
) -> np.ndarray | np.ma.MaskedArray:
    """
    Extract patches from an array around the given indices.

    Args:
        array (np.ndarray): The input array.
        indices (np.ndarray): The indices around which to extract patches.
        patch_size (int): The size of the patches to extract. Default is 3.

    Returns:
        np.ndarray: The extracted patches.
    """
    if patch_size % 2 == 0:
        patch_size += 1
    ys, xs = np.array(indices)
    patch2 = patch_size // 2

    y_offsets = np.arange(-patch2, patch2 + 1)
    x_offsets = np.arange(-patch2, patch2 + 1)
    y_grid, x_grid = np.meshgrid(y_offsets, x_offsets, indexing="ij")

    y_indices = ys[:, None, None] + y_grid
    x_indices = xs[:, None, None] + x_grid

    y_indices = np.clip(y_indices, 0, array.shape[0] - 1)
    x_indices = np.clip(x_indices, 0, array.shape[1] - 1)

    patches = array[y_indices, x_indices]

    return patches


def otsu_threshold(img: np.ndarray, bins: int = 256) -> float:
    """
    Calculate Otsu's threshold for image binarization.

    Args:
        img (np.ndarray): Input image array.
        bins (int): Number of histogram bins. Default is 256.

    Returns:
        float: The optimal threshold value.
    """
    hist, bin_edges = np.histogram(img.ravel(), bins=bins)
    total = img.size
    current_max, threshold = 0, 0
    sum_total = np.dot(hist, bin_edges[:-1])
    sum_foreground, weight_background = 0, 0

    for i in range(bins):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_foreground += hist[i] * bin_edges[i]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        between_var = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if between_var > current_max:
            current_max = between_var
            threshold = bin_edges[i]
    return threshold


def fit_probe_circle(
    img: np.ndarray, threshold: Optional[float] = None, show: bool = True
) -> Tuple[float, float, float]:
    """
    Fit a circle to the probe shape in an image.

    Args:
        img (np.ndarray): Input image containing the probe.
        threshold (Optional[float]): Threshold for binarization. If None, Otsu's method is used.
        show (bool): Whether to display the fitted circle. Default is True.

    Returns:
        Tuple[float, float, float]: Center coordinates (xc, yc) and radius R of the fitted circle.
    """
    if threshold is None:
        threshold = otsu_threshold(img)
    binary: np.ndarray = ndi.binary_fill_holes(img > threshold)  # type: ignore

    smoothed = ndi.gaussian_filter(binary.astype(float), sigma=1)
    grad_mag = np.hypot(ndi.sobel(smoothed, axis=1), ndi.sobel(smoothed, axis=0))
    edge_points = np.argwhere(grad_mag > grad_mag.mean())
    y, x = edge_points[:, 0], edge_points[:, 1]

    def calc_R(xc: float, yc: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c: Tuple[float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        Ri = calc_R(*c, x, y)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    result = least_squares(f_2, center_estimate, args=(x, y))
    xc, yc = result.x
    R = calc_R(xc, yc, x, y).mean()

    if show:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.add_patch(Circle((xc, yc), R, color="red", alpha=0.3, linewidth=2))
        plt.show()

    return yc, xc, R


def fit_probe_ellipse(
    img: np.ndarray, threshold: Optional[float] = None, show: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Fit an ellipse to the probe shape in an image using the direct least squares fitting method for
    ellipses, which solves the general conic equation Ax² + Bxy + Cy² + Dx + Ey + F = 0 subject to
    the constraint B² - 4AC < 0 (ellipse condition).

    Args:
        img (np.ndarray): Input image containing the probe.
        threshold (Optional[float]): Threshold for binarization. If None, Otsu's method is used.
        show (bool): Whether to display the fitted ellipse. Default is True.

    Returns:
        Tuple[float, float, float, float, float]: Center coordinates (xc, yc),
        semi-major axis (a_axis), semi-minor axis (b_axis), and rotation angle (theta) in radians.
    """
    # Threshold the image to create a binary mask
    if threshold is None:
        threshold = otsu_threshold(img)
    binary: np.ndarray = ndi.binary_fill_holes(img > threshold)  # type: ignore

    # Find edge points using gradient magnitude
    smoothed = ndi.gaussian_filter(binary.astype(float), sigma=1)
    grad_x = ndi.sobel(smoothed, axis=1)
    grad_y = ndi.sobel(smoothed, axis=0)
    grad_mag = np.hypot(grad_x, grad_y)
    edge_points = np.argwhere(grad_mag > grad_mag.mean())
    y, x = edge_points[:, 0], edge_points[:, 1]

    # Set up the design matrix for the conic equation Ax² + Bxy + Cy² + Dx + Ey + F = 0
    # Each row represents one edge point [x², xy, y², x, y, 1]
    D = np.vstack([x**2, x * y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)  # Scatter matrix

    # Constraint matrix to enforce ellipse condition (B² - 4AC < 0)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2  # Constraint: 4AC
    C[1, 1] = -1  # Constraint: -B²

    # Solve the generalized eigenvalue problem to find ellipse coefficients
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S).dot(C))
    cond = np.logical_and(np.isreal(eigvals), eigvals > 0)
    a = eigvecs[:, cond][:, 0].real  # type: ignore

    # Extract coefficients from the conic equation
    # General form: a0*x² + a1*xy + a2*y² + a3*x + a4*y + a5 = 0
    b, c, d, f, g, a0 = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]

    # Calculate ellipse center coordinates
    num = b * b - a0 * c
    xc = (c * d - b * f) / num
    yc = (a0 * f - b * d) / num

    # Calculate semi-major and semi-minor axes lengths
    up = 2 * (a0 * f * f + c * d * d + g * b * b - 2 * b * d * f - a0 * c * g)
    down1 = (b * b - a0 * c) * ((c - a0) * np.sqrt(1 + 4 * b * b / ((a0 - c) ** 2)) - (c + a0))
    down2 = (b * b - a0 * c) * ((a0 - c) * np.sqrt(1 + 4 * b * b / ((a0 - c) ** 2)) - (c + a0))
    a_axis = np.sqrt(up / down1)  # Semi-major axis
    b_axis = np.sqrt(up / down2)  # Semi-minor axis

    # Calculate rotation angle of the ellipse
    theta = 0.5 * np.arctan(2 * b / (a0 - c))

    if show:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ellipse = Ellipse(
            (xc, yc),
            2 * a_axis,
            2 * b_axis,
            angle=np.degrees(theta),
            color="red",
            alpha=0.3,
            linewidth=2,
        )
        ax.add_patch(ellipse)
        plt.show()

    return yc, xc, a_axis, b_axis, theta


class RNGMixin:
    """
    Mixin class providing consistent RNG functionality with both numpy and torch generators.
    If you do not provide a seed, the RNG will be initialized with a random seed, and subsequently
    resetting the RNG will use a _new_ random seed. Setting with a fixed seed or generator and then
    resetting the RNG will use the same seed.

    Provides:
    - self.rng: np.random.Generator property
    - self._rng_torch: torch.Generator for torch operations
    - self._reset_rng(): reset the RNG to the current seed
    """

    def __init__(self, rng: np.random.Generator | int | None = None, *args, **kwargs):
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            self._rng_seed = None
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            self._rng_seed = rng
            rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self._rng_seed = rng.bit_generator._seed_seq.entropy  # type:ignore ## get the seed
        else:
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")

        self._rng = rng
        self._update_torch_rng()

    def _update_torch_rng(self):
        """Update the torch generator with current seed and device."""
        if self._rng_seed is not None:
            device = getattr(self, "device", "cpu")
            if self._rng_seed is None:
                self._rng_torch = torch.Generator(device=device)
            else:
                self._rng_torch = torch.Generator(device=device).manual_seed(
                    self._rng_seed % 2**32
                )

    def _reset_rng(self):
        """Reset RNG to current seed, useful for reproducible iterations."""
        if self._rng is not None:
            self.rng = self._rng_seed  # sets rng and _rng_torch

    def _rng_to_device(self, device: str | int | torch.device):
        ## Could consider renaming this as just to, allowing super calls
        """Update torch RNG when device changes."""
        if hasattr(self, "_rng_seed") and self._rng_seed is not None:
            dev, _id = config.validate_device(device)
            self._rng_torch = torch.Generator(device=dev).manual_seed(self._rng_seed % 2**32)
