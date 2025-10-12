import numpy as np
import torch

from quantem.core.utils.validators import validate_tensor


class TomoDataset(torch.utils.data.Dataset):
    """
    A simple dataset class for fitting a volume to a tilt series of images.
    To be expanded/extended/renamed once we know more about the use cases.
    """

    def __init__(
        self,
        images: np.ndarray | torch.Tensor,
    ):
        self.images = validate_tensor(
            images,
            name="images",
            dtype=torch.float64,
            ndim=3,
            expand_dims=True,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index]


class SimpleImageDataset(torch.utils.data.Dataset):
    """
    A simple dataset class for fitting a single image.
    To be expanded/extended/renamed once we know more about the use cases.
    """

    def __init__(
        self,
        image: np.ndarray | torch.Tensor,
    ):
        self.image = validate_tensor(
            image,
            name="image",
            dtype=torch.float64,
            ndim=2,
            expand_dims=False,
        )

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        return self.image


class SimpleVolumeDataset(torch.utils.data.Dataset):
    """
    Simple dataset class for fitting a single volume. To be used for debugging and testing
    config params, not for real applications.
    """

    def __init__(
        self,
        volume: np.ndarray | torch.Tensor,
    ):
        self.volume = validate_tensor(
            volume,
            name="volume",
            dtype=torch.float64,
            ndim=3,
            expand_dims=False,
        )

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        return self.volume
