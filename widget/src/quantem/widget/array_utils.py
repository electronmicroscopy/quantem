"""Array utilities for widgets. NumPy + PyTorch input."""
import numpy as np


def to_numpy(data, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert NumPy / PyTorch / Dataset to NumPy.

    Upcasts torch dtypes numpy can't represent (bfloat16, float8) to float32 first
    so the user sees their data instead of "Got unsupported ScalarType BFloat16".
    """
    try:
        import torch
        is_tensor = isinstance(data, torch.Tensor)
    except ImportError:
        is_tensor = False
    if is_tensor:
        if data.dtype in (torch.bfloat16,) or str(data.dtype).startswith("torch.float8"):
            data = data.to(torch.float32)
        result = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        result = data
    else:
        # Last-resort fallback covers Dataset.__array__, dlpack-compatible objects, etc.
        try:
            result = np.asarray(data)
        except Exception as e:
            raise TypeError(
                f"to_numpy expected a NumPy array or PyTorch tensor, got {type(data).__name__}."
            ) from e
    if dtype is not None:
        result = np.asarray(result, dtype=dtype)
    return result


def _resize_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-pad image to (target_h, target_w) with zeros. For gallery alignment."""
    h, w = img.shape[-2:]
    if h == target_h and w == target_w:
        return img
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    return np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0)


def bin2d(img: np.ndarray, factor: int, mode: str = "mean") -> np.ndarray:
    """Reduce 2D image by integer binning factor. mean or sum of f×f blocks."""
    if factor <= 1:
        return img
    h, w = img.shape[-2:]
    h2, w2 = h - h % factor, w - w % factor
    img = img[..., :h2, :w2]
    blocks = img.reshape(*img.shape[:-2], h2 // factor, factor, w2 // factor, factor)
    if mode == "sum":
        return blocks.sum(axis=(-3, -1))
    return blocks.mean(axis=(-3, -1))
