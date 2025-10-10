from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from quantem.core import config


def inverse_softplus(y):
    return np.log(np.exp(y) - 1)


def inverse_softplus_torch(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.exp(y) - 1)


# TODO -- probably make this not a dataclass so we can have validators in setters/getters
@dataclass
class Config:
    model_type: Literal["2dgs", "3dgs"]
    # shape of the rasterized images [y,x]
    raster_shape: tuple[int, int]
    # extent of recon volume [z,y,x] in A
    volume_size: tuple[float, float, float]

    # isotropic or non-isotropic splats
    isotropic_splats: bool = True
    # add random initialization for quaternions when using anisotropic splats
    random_quaternion_init: bool = False

    # Random crop size for training (unused maybe unecessary)
    patch_size: int | None = None
    # A global scaler that applies to the scene size related parameters
    # currently not used but long term  might be useful for handling scaling issues
    global_scale: float = 1.0
    # device -- maybe unecessary since we have config
    device: str = config.get_device()

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # Number of training steps # TODO rename to iter to be consistent
    max_steps: int = 30_000
    print_every: int = 0  # <= 0 means no printing

    # Activation/normalization function for enforcing sigma > 0
    activation_sigma: Callable = torch.nn.functional.softplus
    # Activation/normalization function for enforcing intensity > 0
    activation_intensity: Callable = torch.nn.functional.softplus

    # Initialization strategy
    init_type: Literal["grid", "random"] = "grid"
    # Initial number of GSs
    init_num_pts: int = 1_000
    # alternatively, grid spacing (A)
    init_grid_sampling: float = 1.0
    # Initial sigma of GSs in A
    init_sigma: float = 0.2
    # Initial intensity of GS # TODO will need to be fixed with intensity scaling
    init_intensity: float = 1.0
    # Weight for SSIM loss, 0 = pure l2, 1 = pure ssim
    ssim_lambda: float = 0.0

    # Near plane clipping distance # this could be sample top surface
    near_plane: float = 0.0  # A? check TODO remove, redundant with volume_size
    # Far plane clipping distance # this could be sample exit surface
    far_plane: float = 200  # A? check

    # GSs with intensity below this value * init_intensity will be pruned
    prune_intensity_fac: float = 0.5
    # GSs with image plane gradient above this value will be split or duplicated
    split_dup_grad2d: float = 0.005  # 0.0002 # TODO - no 2d/3d
    # GSs with sigma below this value (A) will be duplicated. Above will be split
    grow_sigma3d: float = 0.2  # TODO need to play with these values
    # GSs with sigma above this value will be pruned.
    prune_sigma_big_A: float = 0.4
    # GSs with sigma below this value will be pruned.
    prune_sigma_small_A: float = 0.1
    # Padding (A) around edge for where gaussians can be placed
    prune_pad_A: float = 0.5

    # mode for adding new gaussians, "grid", "random", "density"
    add_gaussians_mode: Literal["grid", "random", "density"] = "grid"
    add_gaussian_sampling: float = 1.0  # A
    add_start_iter: int = -1
    add_stop_iter: int = -1

    # merging of splats that are too close
    xy_merge_A: float = 0.5  # cutoff distance
    # in order to merge, the max sigma * fac > other sigmas
    xy_merge_sigma_fac: float = 0.5

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = int(9e9)  # 15_000
    # Pause refining until this number of steps after reset
    pause_refine_after_reset: int = 0
    # Reset intensities every this steps
    reset_every: int = 3000
    # don't reset after this iteration
    reset_stop_iter: int = 10_000
    # Refine GSs every this steps
    refine_every: int = 500  # 100

    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False

    lr_base: float = 0.1

    @property
    def raster_sampling(self) -> tuple[float, float]:
        """sampling in yx of the rasterized image"""
        return (
            self.volume_size[-2] / self.raster_shape[-2],
            self.volume_size[-1] / self.raster_shape[-1],
        )

    def activation_sigma_inverse(self, val: float) -> float:
        act_name = self.activation_sigma.__name__
        if act_name == "softplus":
            if val < 100:
                return inverse_softplus(val)
            else:
                return val  # avoid overflow error
        else:
            raise NotImplementedError(f"Unknown activation for sigma: {act_name}")

    # TODO combine with above, though might have some issues with in place as well
    def activation_sigma_inverse_torch(self, arr: torch.Tensor) -> torch.Tensor:
        act_name = self.activation_sigma.__name__
        if act_name == "softplus":
            arr2 = torch.where(arr > 100, arr, inverse_softplus_torch(arr))
            return arr2
        else:
            raise NotImplementedError(f"Unknown activation for sigma: {act_name}")

    def activation_intensity_inverse(self, val: float) -> float:
        act_name = self.activation_intensity.__name__
        if act_name == "softplus":
            if val < 100:
                return inverse_softplus(val)
            else:
                return val  # avoid overflow error
        else:
            raise NotImplementedError(f"Unknown activation for intensity: {act_name}")

    # TODO combine activation_sigma, could just have a single inverse activation function
    def activation_intensity_inverse_torch(self, arr: torch.Tensor) -> torch.Tensor:
        act_name = self.activation_intensity.__name__
        if act_name == "softplus":
            arr2 = torch.where(arr > 100, arr, inverse_softplus_torch(arr))
            return arr2
        else:
            raise NotImplementedError(f"Unknown activation for sigma: {act_name}")

    @property
    def init_intensity_scaled(self) -> float:
        """The actual intensity is scaled by the sigma due to projection"""
        return self.init_intensity / np.sqrt(2 * np.pi) / self.init_sigma
