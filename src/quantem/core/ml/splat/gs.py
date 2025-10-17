import math
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Slider
from torch.optim import Adam, Optimizer
from torchmetrics.image import StructuralSimilarityIndexMeasure  # PeakSignalNoiseRatio
from tqdm.auto import tqdm

from quantem.core.visualization import show_2d

from .datasets import SimpleImageDataset, SimpleVolumeDataset, TomoDataset
from .gs_config import Config
from .gs_rendering import (  # , rasterization_2dgs_inria_wrapper
    quaternion_to_2d_angle,
    random_quaternion,
    rasterization_2dgs,
    rasterization_volume,
)

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from .gs_strategy import DefaultStrategy


class GS:
    """Base Gaussian splatting class inherited by 2D and 3D"""

    def __init__(self, cfg: Config, rng: int | np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.rng = rng
        self._losses = []

        # Model
        self.splats, self.optimizers, self.schedulers = self._initialize_splats_and_optimizers()
        print("Model initialized. Number of GS:", len(self.splats["positions"]))
        self.model_type = cfg.model_type

        # Densification Strategy
        self.strategy = DefaultStrategy(
            cfg=cfg,
            verbose=True,
            key_for_gradient="positions",
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # TODO reimplement (note data range)
        # self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # Precompute the 2D grid for the target image
        yy, xx = torch.meshgrid(
            torch.arange(cfg.raster_shape[0], device=self.device) * cfg.raster_sampling[0],
            torch.arange(cfg.raster_shape[1], device=self.device) * cfg.raster_sampling[1],
            indexing="ij",
        )
        self.grid_y = yy.type(torch.float64)
        self.grid_x = xx.type(torch.float64)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")
        self._rng = rng
        seed = rng.bit_generator._seed_seq.entropy  # type:ignore ## get seed from the generator
        self._rng_torch = torch.Generator(device=self.device).manual_seed(seed % 2**32)

    @property
    def losses(self) -> np.ndarray:
        return np.array(self._losses)

    def _initialize_splats_and_optimizers(
        self,
    ) -> tuple[torch.nn.ParameterDict, dict[str, Optimizer], list]:
        # TODO set up optimizers and schedulers passable like in ptycho
        # currently not scaling the lrs for more consistent/faster testing
        if self.cfg.init_type == "random":
            positions = torch.tensor(self.cfg.volume_size) * (
                torch.rand((self.cfg.init_num_pts, 3), generator=self._rng_torch)
            )
        elif self.cfg.init_type == "grid":
            if self.cfg.model_type == "2dgs":
                yy, xx = np.mgrid[
                    0 : self.cfg.volume_size[-2] : self.cfg.init_grid_sampling,
                    0 : self.cfg.volume_size[-1] : self.cfg.init_grid_sampling,
                ]
                yy += yy[1, 1] / 2
                xx += xx[1, 1] / 2
                positions = torch.tensor(
                    np.stack([np.zeros_like(yy.ravel()), yy.ravel(), xx.ravel()]).T,
                    dtype=torch.float64,
                )
            elif self.cfg.model_type == "3dgs":
                z, y, x = np.mgrid[
                    0 : self.cfg.volume_size[0] : self.cfg.init_grid_sampling,
                    0 : self.cfg.volume_size[1] : self.cfg.init_grid_sampling,
                    0 : self.cfg.volume_size[2] : self.cfg.init_grid_sampling,
                ]
                z += z[1, 1, 1] / 2
                y += y[1, 1, 1] / 2
                x += x[1, 1, 1] / 2
                positions = torch.tensor(
                    np.stack([z.ravel(), y.ravel(), x.ravel()]).T, dtype=torch.float64
                )
            else:
                raise ValueError(f"Unknown model type: {self.cfg.model_type}")
        else:
            raise ValueError(f"Unknown init_type: {self.cfg.init_type}")

        N = positions.shape[0]
        sigmas = torch.ones((N, 3), dtype=torch.float64) * self.cfg.activation_sigma_inverse(
            self.cfg.init_sigma
        )

        intensities = torch.ones(N, dtype=torch.float64) * self.cfg.activation_intensity_inverse(
            self.cfg.init_intensity_scaled
        )

        params = [
            ("positions", torch.nn.Parameter(positions), self.cfg.lr_base),
            ("sigmas", torch.nn.Parameter(sigmas), self.cfg.lr_base / 10),
            ("intensities", torch.nn.Parameter(intensities), self.cfg.lr_base / 10),
        ]

        # Add quaternions for rotation when using anisotropic splats
        if not self.cfg.isotropic_splats:
            if self.cfg.random_quaternion_init:
                # Random quaternions blended with identity for stability
                quaternions = F.normalize(
                    0.9 * torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float64)
                    + 0.1
                    * random_quaternion((N,), device=self.device, generator=self._rng_torch).to(
                        torch.float64
                    ),
                    dim=-1,
                )
            else:
                # Identity quaternions [w=1, x=0, y=0, z=0]
                quaternions = torch.zeros((N, 4), dtype=torch.float64, device=self.device)
                quaternions[:, 0] = 1.0

            params.append(("quaternions", torch.nn.Parameter(quaternions), self.cfg.lr_base / 10))

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.cfg.device)
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1
        optimizers = {
            name: Adam(
                [{"params": splats[name], "lr": lr * math.sqrt(self.cfg.batch_size)}],
                eps=1e-15 / math.sqrt(self.cfg.batch_size),
                betas=(1 - self.cfg.batch_size * (1 - 0.9), 1 - self.cfg.batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        }

        schedulers = []
        # schedulers = [
        # torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizers["positions"], gamma=1 ** (1.0 / max_steps)
        # ),
        # torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizers["sigmas"], gamma=1 ** (1.0 / max_steps)
        # ),
        # torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizers["intensities"], gamma=1 ** (1.0 / max_steps)
        # ),
        # ]
        # Add scheduler for quaternions if they exist
        # if "quaternions" in self.optimizers:
        #     schedulers.append(
        #         torch.optim.lr_scheduler.ExponentialLR(
        #             self.optimizers["quaternions"], gamma=0.1 ** (1.0 / max_steps)
        #         )
        #     )

        return splats, optimizers, schedulers  # type:ignore ## an instance of Adam is an Optimizer... idk.

    def rasterize_splats(  # rasterize and project the splats to 2D image(s)
        self,
        rescale_sigmas: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # rasterization does normalization internally
        sigmas = self.cfg.activation_sigma(self.splats["sigmas"])  # [N, 3]
        if rescale_sigmas is not None:  # for visualization
            sigmas *= rescale_sigmas
        intensities = self.cfg.activation_intensity(self.splats["intensities"])  # [N,]
        positions = self.splats["positions"]  # don't normalize here, take care of in strategy

        # Get quaternions if available for non-isotropic splats
        quaternions = None
        if not self.cfg.isotropic_splats and "quaternions" in self.splats:
            quaternions = self.splats["quaternions"]

        if self.model_type == "2dgs":
            rendered_ims = rasterization_2dgs(
                positions=positions,
                sigmas=sigmas,
                intensities=intensities,
                grids=(self.grid_y, self.grid_x),
                isotropic_splats=self.cfg.isotropic_splats,
                quaternions=quaternions,
                **kwargs,
            )
        else:
            raise NotImplementedError

        return rendered_ims

    def rasterize_volume_splats(  # rasterize the splats to a 3D volume
        self,
        volume_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Rasterize the splats to a 3D volume.

        Args:
            volume_shape: (D, H, W) output volume dimensions. If None, inferred from config.
            **kwargs: Additional arguments passed to rasterization_volume.

        Returns:
            torch.Tensor: (1, D, H, W) rasterized volume
        """
        # Apply activations to get actual parameter values
        sigmas = self.cfg.activation_sigma(self.splats["sigmas"])  # [N, 3]
        intensities = self.cfg.activation_intensity(self.splats["intensities"])  # [N,]
        positions = self.splats["positions"]  # don't normalize here, take care of in strategy

        # Get quaternions if available for non-isotropic splats
        quaternions = None
        if not self.cfg.isotropic_splats and "quaternions" in self.splats:
            quaternions = self.splats["quaternions"]

        # Determine volume shape - if not provided, create a cube based on raster_shape
        if volume_shape is None:
            min_sampling = min(self.cfg.raster_sampling)
            z_shape = int(round(self.cfg.volume_size[0] / min_sampling))
            volume_shape = (z_shape, self.cfg.raster_shape[0], self.cfg.raster_shape[1])

        # Use volume size from config
        volume_size = self.cfg.volume_size

        # Call the volume rasterization function
        rendered_volume = rasterization_volume(
            positions=positions,
            sigmas=sigmas,
            intensities=intensities,
            volume_shape=volume_shape,
            volume_size=volume_size,
            isotropic_splats=self.cfg.isotropic_splats,
            quaternions=quaternions,
            **kwargs,
        )
        return rendered_volume

    def compute_loss(self, renders: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        l2loss = F.mse_loss(renders, images)
        if self.cfg.ssim_lambda > 0:
            ssimloss = 1.0 - self.ssim(renders[None,], images[None,])
            loss = l2loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda
        else:
            loss = l2loss
        # add regularization, other loss terms here
        return loss

    def fit_image(  # Single 2D image fitting
        self,
        trainset: SimpleImageDataset,
        max_steps: int | None = None,
        reset: bool = False,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
        cfg = self.cfg
        if reset:
            self._losses = []
            self.splats, self.optimizers, self.schedulers = (
                self._initialize_splats_and_optimizers()
            )
        if max_steps is None:
            max_steps = cfg.max_steps

        init_step = len(self.losses)

        image = trainset[0].to(self.device)
        renders = torch.ones(1)  # for typing
        # Training loop.
        start_time = datetime.now()
        pbar = tqdm(range(init_step, init_step + max_steps))
        for step in pbar:
            # forward
            renders = self.rasterize_splats(
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            if (  # lol at this -- temp of course, but want to see before/after the refinement
                step == 1
                or (
                    ((step + 1) % cfg.refine_every) == 0
                    and cfg.refine_stop_iter > (step + 1) >= cfg.refine_start_iter
                )
                or (
                    (step % cfg.refine_every) == 0
                    and cfg.refine_stop_iter > step >= cfg.refine_start_iter
                )
            ):
                r = torch.squeeze(renders).cpu().detach().numpy()
                show_2d(r, title=f"iter {step} render", force_show=True, cbar=True, norm="minmax")
                # print(f"Intensities iter {step}: \n", self.cfg.activation_intensity(self.splats["intensities"]))
                # print(f"Scaled intensities iter {step}:\n", self.cfg.activation_intensity(self.splats["intensities"]) * np.sqrt(2*np.pi) * self.cfg.activation_sigma(self.splats["sigmas"]))
                # print(f"Sigmas iter {step}: \n", self.cfg.activation_sigma(self.splats["sigmas"]))

            # the way info is done is a bit redundant, can be cleaned up later
            info = {
                "height": self.cfg.raster_shape[0],
                "width": self.cfg.raster_shape[1],
                "positions": self.splats["positions"],
                "sigmas": self.splats["sigmas"],
                "n_images": 1,  # len(trainset),
                "mask": torch.ones_like(self.splats["sigmas"], dtype=torch.float64),
            }

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step + 1,
                info=info,
            )

            loss = self.compute_loss(renders[None,], image[None,])
            loss.backward()
            desc = f"loss={loss.item():.3e}"
            pbar.set_description(desc)

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step + 1,
                info=info,
            )

            # print and/or checkpoint
            if cfg.print_every > 0:
                if (step + 1) % cfg.print_every == 0 or step == 0 or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    d = timedelta(seconds=(datetime.now() - start_time).seconds)
                    print(
                        f"Step: {step} | num_GS: {len(self.splats['positions'])} | mem {mem:.2f} GB | ellapsed time (h:m:s) {d}"
                    )
                # write a checkpointing function

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()

            self._losses.append(loss.item())

        return torch.squeeze(renders).cpu().detach().numpy(), self.losses

    def fit_volume(  # Single 3D volume fitting (rasterized, not projected)
        self,
        trainset: SimpleVolumeDataset,
        max_steps: int | None = None,
        reset: bool = False,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
        cfg = self.cfg
        if cfg.model_type != "3dgs":
            raise ValueError(f"Model type {cfg.model_type} is not supported for volume fitting.")
        if reset:
            self._losses = []
            self.splats, self.optimizers, self.schedulers = (
                self._initialize_splats_and_optimizers()
            )
        if max_steps is None:
            max_steps = cfg.max_steps

        init_step = len(self.losses)

        volume = trainset[0].to(self.device)
        renders = torch.ones(1)  # for typing
        # Training loop.
        start_time = datetime.now()
        pbar = tqdm(range(init_step, init_step + max_steps))
        for step in pbar:
            # forward
            renders = self.rasterize_volume_splats()
            if (
                step == init_step + 1
                or (
                    ((step + 1) % cfg.refine_every) == 0
                    and cfg.refine_stop_iter > (step + 1) >= cfg.refine_start_iter
                )
                or (
                    (step % cfg.refine_every) == 0
                    and cfg.refine_stop_iter > step >= cfg.refine_start_iter
                )
            ):
                r = torch.squeeze(renders).cpu().detach().numpy()
                ## show slices of the 3D volume
                show_2d(
                    [
                        r.sum(0),
                        r.sum(1),
                        r[:, r.shape[1] // 2, :],
                    ],
                    norm="minmax",
                    cbar=True,
                    title=[f"step {step} render (yx) sum", "render (zx) sum", "render (zx) slice"],
                    force_show=True,
                )
                # print(f"Intensities iter {step}: \n", self.cfg.activation_intensity(self.splats["intensities"]))
                # print(f"Scaled intensities iter {step}:\n", self.cfg.activation_intensity(self.splats["intensities"]) * np.sqrt(2*np.pi) * self.cfg.activation_sigma(self.splats["sigmas"]))
                # print(f"Sigmas iter {step}: \n", self.cfg.activation_sigma(self.splats["sigmas"]))

            # the way info is done is a bit redundant, can be cleaned up later
            info = {
                "height": self.cfg.raster_shape[0],
                "width": self.cfg.raster_shape[1],
                "positions": self.splats["positions"],
                "sigmas": self.splats["sigmas"],
                "n_images": 1,  # len(trainset),
                "mask": torch.ones_like(self.splats["sigmas"], dtype=torch.float64),
            }

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step + 1,
                info=info,
            )

            loss = self.compute_loss(renders, volume)
            loss.backward()
            desc = f"loss={loss.item():.3e}"
            pbar.set_description(desc)

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step + 1,
                info=info,
            )

            # print and/or checkpoint
            if cfg.print_every > 0:
                if (step + 1) % cfg.print_every == 0 or step == 0 or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    d = timedelta(seconds=(datetime.now() - start_time).seconds)
                    print(
                        f"Step: {step} | num_GS: {len(self.splats['positions'])} | mem {mem:.2f} GB | ellapsed time (h:m:s) {d}"
                    )
                # write a checkpointing function

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()

            self._losses.append(loss.item())

        return torch.squeeze(renders).cpu().detach().numpy(), self.losses

    def fit_tomo(  # Placeholder for 3D volume fitting
        self, trainset: TomoDataset, max_steps: int | None = None
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
        ### attn: Nalini
        ### conceptually I don't like the way this gsplat did the training loop, it feels silly to not just iterate
        ### through the trainloader like a normal training loop, but I guess it's necessary because
        ### we're counting "steps" (i.e. each time a backwards call is performed) rather than
        ### epochs. What makes the most sense to me would be more like:

        # for epoch in range(num_epochs):
        #     for image_batch, poses in trainloader:
        #        image_batch = image_batch.to(self.device)
        #        renders = self.rasterize_splats(poses, *args)
        #        self.strategy.step_pre_backward(*args, step = epoch) ## necessary??
        #        loss = self.compute_loss_all(*args)
        #        loss.backward()
        #        self.strategy.update_step(*args) ## accumulating gradients
        #        optimizers.step()
        #     self.strategy.step_post_backward(*args, step = epoch) # so splitting etc. only happens once on the appropriate epoch
        #     ## and of course the update_step() stuff would have to be removed from the post_backward call
        #     self.schedulers.step()

        ### this might just be worse than the way it was done, but it just makes more sense to me
        ### it would explicitly separate the batch size from gradient accumlation and the strategy,

        # trainloader = DataLoader(
        #     trainset,
        #     batch_size=cfg.batch_size,
        #     shuffle=True,
        #     num_workers=1,
        #     persistent_workers=True,
        #     pin_memory=True,
        # )
        # trainloader_iter = iter(trainloader)
        raise NotImplementedError

    ##### Visualization functions -- move to a viz module later #####

    def plot_gaussians(
        self,
        rescale_sigma: float = 1,
        cmap: str = "viridis",
        alpha: float = 0.8,
    ):
        """
        Plot 2D Gaussian splats as circles with radius ~ sigma and color ~ intensity.

        Args:
            positions: (N, 3) tensor, only y and x used for 2D plotting.
            sigmas: (N, 3) tensor, only y and x used for 2D plotting.
            intensities: (N,) tensor, mapped to color.
            ax: matplotlib axis to plot on.
            rescale_sigma: float, multiply sigma by this for visualization.
            cmap: matplotlib colormap name.
            alpha: float, transparency for overlap.
        """
        if self.model_type != "2dgs":
            raise NotImplementedError

        fig, ax = plt.subplots(figsize=(8, 8))

        pos = self.splats["positions"].detach().cpu().numpy()
        sig = self.cfg.activation_sigma(self.splats["sigmas"]).detach().cpu().numpy()
        inten = self.cfg.activation_intensity(self.splats["intensities"]).detach().cpu().numpy()

        y = pos[:, 1]
        x = pos[:, 2]

        if self.cfg.isotropic_splats:
            radius = np.mean(sig, axis=1) * rescale_sigma
            patches = [Circle((xi, yi), ri) for xi, yi, ri in zip(x, y, radius)]
        else:
            # Non-isotropic splats: use sigmas for each axis, plot as ellipses
            sig_y = sig[:, 1] * rescale_sigma
            sig_x = sig[:, 2] * rescale_sigma

            # Handle rotation via quaternions if available
            if "quaternions" in self.splats:
                # Extract rotation angles in degrees using PyTorch
                angles_deg = (
                    -1 * quaternion_to_2d_angle(self.splats["quaternions"]) * 180.0 / np.pi
                )
                angles_deg_np = angles_deg.detach().cpu().numpy()

                patches = [
                    Ellipse((xi, yi), width=2 * sx, height=2 * sy, angle=angle_deg)
                    for xi, yi, sy, sx, angle_deg in zip(x, y, sig_y, sig_x, angles_deg_np)
                ]
            else:
                # No rotation, just use standard ellipses
                patches = [
                    Ellipse((xi, yi), width=2 * sx, height=2 * sy)
                    for xi, yi, sy, sx in zip(x, y, sig_y, sig_x)
                ]

        collection = PatchCollection(
            patches,
            cmap=plt.cm.get_cmap(cmap),
            alpha=alpha,
            linewidth=0,
        )
        collection.set_array(inten)
        ax.add_collection(collection)
        ax.set_title(
            f"Mean Intensity: {np.mean(inten):.3f} | Mean Sigma: {np.mean(sig):.3f} | "
            f"Num GS: {len(self.splats['positions'])}"
        )
        ax.set_aspect("equal")
        ax.set_ylim(0, self.cfg.raster_shape[0] * self.cfg.raster_sampling[0])
        ax.set_xlim(0, self.cfg.raster_shape[1] * self.cfg.raster_sampling[1])
        ax.invert_yaxis()
        ax.autoscale_view()
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Intensity")
        plt.show()

    def plot_gaussians_3d(
        self,
        rescale_sigma: float = 1,
        cmap: str = "viridis",
        alpha: float = 0.8,
        projected: bool = False,  # new argument
    ):
        """
        Plot 3D Gaussian splats as ellipses/circles on three orthogonal slices through the center of the volume,
        or as projections along each axis if projected=True.

        Args:
            rescale_sigma: float, multiply sigma by this for visualization.
            cmap: matplotlib colormap name.
            alpha: float, transparency for overlap.
            projected: bool, if True, show projections along each axis with single-color colormap.
        """
        if self.model_type != "3dgs":
            raise NotImplementedError("plot_gaussians_3d is only for 3dgs models.")

        pos = self.splats["positions"].detach().cpu().numpy()
        sig = self.cfg.activation_sigma(self.splats["sigmas"]).detach().cpu().numpy()
        inten = self.cfg.activation_intensity(self.splats["intensities"]).detach().cpu().numpy()

        zc = self.cfg.volume_size[0] / 2
        yc = self.cfg.volume_size[1] / 2
        xc = self.cfg.volume_size[2] / 2

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        slice_names = ["xy (z=center)", "xz (y=center)", "yz (x=center)"]
        proj_names = ["xy (proj z)", "xz (proj y)", "yz (proj x)"]
        slice_centers = [zc, yc, xc]
        slice_axes = [
            (1, 2, 0),  # y, x, z
            (0, 2, 1),  # z, x, y
            (0, 1, 2),  # z, y, x
        ]

        if projected:
            cmap = "Reds"
            for i, (ax, name, axes) in enumerate(zip(axs, proj_names, slice_axes)):
                # Project along the orthogonal axis (i.e., ignore that coordinate)
                y = pos[:, axes[0]]
                x = pos[:, axes[1]]
                if self.cfg.isotropic_splats:
                    radius = np.mean(sig, axis=1) * rescale_sigma
                    patches = [Circle((xi, yi), ri) for xi, yi, ri in zip(x, y, radius)]
                else:
                    sig_a = sig[:, axes[0]] * rescale_sigma
                    sig_b = sig[:, axes[1]] * rescale_sigma
                    if "quaternions" in self.splats:
                        # For projections, ignore rotation or set to zero for simplicity
                        angles_deg = np.zeros(len(x))
                        patches = [
                            Ellipse((xi, yi), width=2 * sb, height=2 * sa, angle=angle_deg)
                            for xi, yi, sa, sb, angle_deg in zip(x, y, sig_a, sig_b, angles_deg)
                        ]
                    else:
                        patches = [
                            Ellipse((xi, yi), width=2 * sb, height=2 * sa)
                            for xi, yi, sa, sb in zip(x, y, sig_a, sig_b)
                        ]
                collection = PatchCollection(
                    patches,
                    cmap=plt.cm.get_cmap(cmap),
                    alpha=alpha,
                    linewidth=0,
                )
                collection.set_array(inten)
                ax.add_collection(collection)
                ax.set_title(
                    f"{name} Num GS: {len(pos)}\n"
                    + f"Intensity μ: {np.mean(inten):.3f}, std: {np.std(inten):.3f} | "
                    + f"Sigma μ: {np.mean(sig):.3f}, std: {np.std(sig):.3f}"
                )
                ax.set_aspect("equal")
                ax.set_xlim(0, self.cfg.volume_size[axes[1]])
                ax.set_ylim(0, self.cfg.volume_size[axes[0]])
                ax.invert_yaxis()
                ax.set_xlabel(["z", "y", "x"][axes[1]])
                ax.set_ylabel(["z", "y", "x"][axes[0]])
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label="Intensity")
            plt.tight_layout()
            plt.show()
            return

        for i, (ax, name, center, axes) in enumerate(
            zip(axs, slice_names, slice_centers, slice_axes)
        ):
            mask = np.abs(pos[:, axes[2]] - center) < 1.25
            pos_slice = pos[mask]
            sig_slice = sig[mask]
            inten_slice = inten[mask]
            if pos_slice.shape[0] == 0:
                ax.set_title(f"No splats in {name} slice")
                continue

            y = pos_slice[:, axes[0]]
            x = pos_slice[:, axes[1]]

            if self.cfg.isotropic_splats:
                radius = np.mean(sig_slice, axis=1) * rescale_sigma
                patches = [Circle((xi, yi), ri) for xi, yi, ri in zip(x, y, radius)]
            else:
                sig_a = sig_slice[:, axes[0]] * rescale_sigma
                sig_b = sig_slice[:, axes[1]] * rescale_sigma

                # Handle rotation via quaternions if available
                if "quaternions" in self.splats:
                    quats = self.splats["quaternions"].detach().cpu().numpy()[mask]
                    # For each quaternion, compute rotation angle in the slice plane
                    # For simplicity, set angle=0 (no rotation) for now
                    # TODO: implement proper 3D quaternion to 2D slice angle if needed
                    angles_deg = np.zeros(len(quats))
                    patches = [
                        Ellipse((xi, yi), width=2 * sb, height=2 * sa, angle=angle_deg)
                        for xi, yi, sa, sb, angle_deg in zip(x, y, sig_a, sig_b, angles_deg)
                    ]
                else:
                    patches = [
                        Ellipse((xi, yi), width=2 * sb, height=2 * sa)
                        for xi, yi, sa, sb in zip(x, y, sig_a, sig_b)
                    ]

            collection = PatchCollection(
                patches,
                cmap=plt.cm.get_cmap(cmap),
                alpha=alpha,
                linewidth=0,
            )
            collection.set_array(inten_slice)
            ax.add_collection(collection)
            ax.set_title(
                f"{name}\nMean Intensity: {np.mean(inten_slice):.3f} | Mean Sigma: {np.mean(sig_slice):.3f} | "
                f"Num GS: {len(pos_slice)}"
            )
            ax.set_aspect("equal")
            # Set axis limits based on volume size
            ax.set_xlim(0, self.cfg.volume_size[axes[1]])
            ax.set_ylim(0, self.cfg.volume_size[axes[0]])
            ax.invert_yaxis()
            ax.set_xlabel(["z", "y", "x"][axes[1]])
            ax.set_ylabel(["z", "y", "x"][axes[0]])
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array(inten_slice)
            plt.colorbar(sm, ax=ax, label="Intensity")

        plt.tight_layout()
        plt.show()

    def show_orthogonal_slices(
        self,
        volume: np.ndarray | None = None,
        volume_size: tuple[float, float, float] | None = None,
        z_pos: float | None = None,
        y_pos: float | None = None,
        x_pos: float | None = None,
        cmap: str = "viridis",
        title: str | None = None,
        figsize: tuple[int, int] = (15, 5),
    ) -> tuple[Any, Any]:
        """
        Show three orthogonal slices (XY, XZ, YZ) in a single figure.
        If volume is None, rasterize the current splats to a volume.
        """
        if volume is None:
            rendered_volume = self.rasterize_volume_splats()
            volume = rendered_volume.squeeze().cpu().detach().numpy()
        if volume_size is None:
            volume_size = self.cfg.volume_size

        z_coords = np.arange(volume.shape[0]) * (volume_size[0] / volume.shape[0])
        y_coords = np.arange(volume.shape[1]) * (volume_size[1] / volume.shape[1])
        x_coords = np.arange(volume.shape[2]) * (volume_size[2] / volume.shape[2])

        if z_pos is None:
            z_pos = volume_size[0] / 2
        if y_pos is None:
            y_pos = volume_size[1] / 2
        if x_pos is None:
            x_pos = volume_size[2] / 2

        z_idx = np.argmin(np.abs(z_coords - z_pos))
        y_idx = np.argmin(np.abs(y_coords - y_pos))
        x_idx = np.argmin(np.abs(x_coords - x_pos))

        xy_slice = volume[z_idx, :, :]
        xz_slice = volume[:, y_idx, :]
        yz_slice = volume[:, :, x_idx]

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        im1 = axs[0].imshow(
            xy_slice, cmap=cmap, extent=[0, volume_size[2], volume_size[1], 0], aspect=1
        )
        axs[0].set_xlabel("X (Å)")
        axs[0].set_ylabel("Y (Å)")
        axs[0].set_title(f"XY slice at Z = {z_coords[z_idx]:.2f} Å")

        im2 = axs[1].imshow(
            xz_slice, cmap=cmap, extent=[0, volume_size[2], volume_size[0], 0], aspect=1
        )
        axs[1].set_xlabel("X (Å)")
        axs[1].set_ylabel("Z (Å)")
        axs[1].set_title(f"XZ slice at Y = {y_coords[y_idx]:.2f} Å")

        im3 = axs[2].imshow(
            yz_slice, cmap=cmap, extent=[0, volume_size[1], volume_size[0], 0], aspect=1
        )
        axs[2].set_xlabel("Y (Å)")
        axs[2].set_ylabel("Z (Å)")
        axs[2].set_title(f"YZ slice at X = {x_coords[x_idx]:.2f} Å")

        for i, im in enumerate([im1, im2, im3]):
            cbar = plt.colorbar(im, ax=axs[i], shrink=0.8)
            if i == 1:
                cbar.set_label("Intensity")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        plt.show()
        return fig, axs

    def show_slices(
        self,
        axis: str = "z",
        sigma_cutoff: float = 3.0,
        rescale_sigma: float = 1.0,
        cmap: str = "viridis",
        alpha: float = 0.7,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 8),
    ) -> tuple[Any, Any]:
        """
        Show interactive slices displaying true 2D Gaussian cross-sections.
        """
        if axis not in ["z", "y", "x"]:
            raise ValueError(f"axis must be 'z', 'y', or 'x', got {axis}")

        pos = self.splats["positions"].detach().cpu().numpy()
        sig = self.cfg.activation_sigma(self.splats["sigmas"]).detach().cpu().numpy()
        inten = self.cfg.activation_intensity(self.splats["intensities"]).detach().cpu().numpy()

        axis_map = {"z": 0, "y": 1, "x": 2}
        axis_idx = axis_map[axis]

        # Define slice coordinate range
        volume_size = self.cfg.volume_size
        if axis == "z":
            slice_range = (0, volume_size[0])
            xlabel, ylabel = "X (Å)", "Y (Å)"
            x_lim, y_lim = volume_size[2], volume_size[1]
            coord_indices = (2, 1)  # (x, y)
            sigma_indices = (2, 1)  # (sigma_x, sigma_y)
        elif axis == "y":
            slice_range = (0, volume_size[1])
            xlabel, ylabel = "X (Å)", "Z (Å)"
            x_lim, y_lim = volume_size[2], volume_size[0]
            coord_indices = (2, 0)  # (x, z)
            sigma_indices = (2, 0)  # (sigma_x, sigma_z)
        else:  # axis == 'x'
            slice_range = (0, volume_size[2])
            xlabel, ylabel = "Y (Å)", "Z (Å)"
            x_lim, y_lim = volume_size[1], volume_size[0]
            coord_indices = (1, 0)  # (y, z)
            sigma_indices = (1, 0)  # (sigma_y, sigma_z)

        splat_positions_axis = pos[:, axis_idx]
        splat_sigmas_axis = sig[:, axis_idx]
        splat_positions_xy = pos[:, coord_indices]
        splat_sigmas_xy = sig[:, sigma_indices]

        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.2)
        initial_pos = slice_range[0] + (slice_range[1] - slice_range[0]) / 2

        def get_true_gaussian_cross_section(slice_pos):
            distances = np.abs(splat_positions_axis - slice_pos)
            mask = distances <= sigma_cutoff * splat_sigmas_axis

            if not np.any(mask):
                return [], []

            pos_visible_xy = splat_positions_xy[mask]
            sig_visible_xy = splat_sigmas_xy[mask]
            inten_visible = inten[mask]  # Keep original intensities constant
            distances_visible = distances[mask]
            slice_sigmas_visible = splat_sigmas_axis[mask]

            x_coords = pos_visible_xy[:, 0]
            y_coords = pos_visible_xy[:, 1]

            sig_x_base = sig_visible_xy[:, 0]
            sig_y_base = sig_visible_xy[:, 1]

            # Calculate the effective radius scaling based on distance from slice
            gaussian_falloff = np.exp(-(distances_visible**2) / (2 * slice_sigmas_visible**2))
            radius_scaling = np.sqrt(gaussian_falloff) * rescale_sigma
            sig_x_effective = sig_x_base * radius_scaling
            sig_y_effective = sig_y_base * radius_scaling

            patches = []
            if self.cfg.isotropic_splats:
                radii_effective = np.mean([sig_x_effective, sig_y_effective], axis=0)
                patches = [
                    Circle((x, y), r) for x, y, r in zip(x_coords, y_coords, radii_effective)
                ]
            else:
                if "quaternions" in self.splats and not self.cfg.isotropic_splats:
                    angles_deg = np.zeros(len(x_coords))  # Placeholder for now
                    patches = [
                        Ellipse((x, y), width=2 * sx, height=2 * sy, angle=angle)
                        for x, y, sx, sy, angle in zip(
                            x_coords, y_coords, sig_x_effective, sig_y_effective, angles_deg
                        )
                    ]
                else:
                    patches = [
                        Ellipse((x, y), width=2 * sx, height=2 * sy)
                        for x, y, sx, sy in zip(
                            x_coords, y_coords, sig_x_effective, sig_y_effective
                        )
                    ]

            return patches, inten_visible

        initial_patches, initial_intensities = get_true_gaussian_cross_section(initial_pos)
        collection = PatchCollection(
            initial_patches,
            cmap=plt.cm.get_cmap(cmap),
            alpha=alpha,
            linewidth=0,
        )
        if len(initial_intensities) > 0:
            collection.set_array(initial_intensities)
        ax.add_collection(collection)
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        num_splats = len(initial_patches)
        if title:
            ax.set_title(f"{title} - {axis.upper()} = {initial_pos:.2f} Å ({num_splats} splats)")
        else:
            ax.set_title(
                f"True Gaussian Cross-Sections - {axis.upper()} = {initial_pos:.2f} Å ({num_splats} splats)"
            )

        if len(initial_intensities) > 0:
            cbar = plt.colorbar(collection, ax=ax, shrink=0.8)
            cbar.set_label("Cross-Section Intensity")

        ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
        slider = Slider(
            ax_slider,
            f"{axis.upper()} position (Å)",
            slice_range[0],
            slice_range[1],
            valinit=initial_pos,
            valfmt="%.2f",
        )
        current_collection = [collection]  # Use a mutable container for closure

        def update_slice(val):
            slice_pos = slider.val
            patches, intensities = get_true_gaussian_cross_section(slice_pos)
            if current_collection[0] is not None:
                try:
                    current_collection[0].remove()
                except ValueError:  # Handle case where collection is already removed
                    pass
            if patches:
                new_collection = PatchCollection(
                    patches,
                    cmap=plt.cm.get_cmap(cmap),
                    alpha=alpha,
                    linewidth=0,
                )
                new_collection.set_array(intensities)
                ax.add_collection(new_collection)
                current_collection[0] = new_collection

            num_splats = len(patches)
            if title:
                ax.set_title(f"{title} - {axis.upper()} = {slice_pos:.2f} Å ({num_splats} splats)")
            else:
                ax.set_title(
                    f"True Gaussian Cross-Sections - {axis.upper()} = {slice_pos:.2f} Å ({num_splats} splats)"
                )
            fig.canvas.draw_idle()

        slider.on_changed(update_slice)
        plt.show()
        return fig, ax
