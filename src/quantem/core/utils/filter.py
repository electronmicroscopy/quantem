import numpy as np
import scipy.ndimage as ndi

from quantem.core.utils.utils import extract_patches


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
