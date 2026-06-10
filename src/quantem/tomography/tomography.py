import os
from pathlib import Path
from typing import Literal, Self, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.models.kplanes import CPTilted
from quantem.core.utils.filter import gaussian_filter_2d_stack, gaussian_kernel_1d
from quantem.core.utils.tomography_utils import torch_phase_cross_correlation
from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    DatasetConstraintsType,
    DatasetModelType,
    DeviceBatchSampler,
    TomographyINRDataset,
    TomographyPixDataset,
)
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjConstraintsType,
    ObjectINR,
    ObjectPixelated,
    ObjectTensorDecomp,
)
from quantem.tomography.radon.radon import iradon_torch, radon_torch
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_context import ReconstructionContext
from quantem.tomography.tomography_opt import TomographyOpt


class Tomography(TomographyOpt, TomographyBase):
    """
    Class for handling all ML tomography reconstruction methods.
    Automatic handling between AD and INR-based tomography.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectINR | ObjectTensorDecomp,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        batch_size: int = 1024,
        num_workers: int = 32,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        obj_constraints: dict | ObjConstraintsType | None = None,
        dset_constraints: dict | DatasetConstraintsType | None = None,
        num_samples_per_ray: int | list[tuple[int, int]] | None = None,
        profiling_mode: bool = False,
        val_fraction: float = 0.0,
        loss_type: Literal[
            "l2",
            "l1",
            "smooth_l1",
            "charbonnier",
            "llmse",
            "mse_log_mse",
        ] = "l2",
        loss_func_kwargs: dict = {},
        reset_dset: DatasetModelType | None = None,
        show_metrics: bool = False,
    ):
        """
        This function should be able to handle both AD and INR-based tomography reconstruction methods.
        I.e, auto-detection through the obj model type, while both share the same pose optimization.
        """

        # Check device consistency
        self.obj_model.to(self.device)

        # Saving batch size, num workers, and val fraction for reloading
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

        if profiling_mode:
            if self.global_rank == 0:
                print("Profiling mode enabled.")

        if reset:
            raise NotImplementedError("Reset is not implemented yet.")

        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

            self.obj_model.constraints = obj_constraints

        if dset_constraints is not None:
            if isinstance(dset_constraints, dict):
                dset_constraints = DatasetConstraintParams.parse_dict(dset_constraints)

            self.dset.constraints = dset_constraints
        # Setting up DDP
        if not hasattr(self, "dataloader") or reset_dset is not None:
            if reset_dset is not None:
                print("Resetting Dataloader")
                print("Putting in params from previous dataset.")

                self.dset = reset_dset
                self.dset.to(self.device)

                if optimizer_params is not None:
                    self.optimizer_params = optimizer_params
                    self.set_optimizers()
                if scheduler_params is not None:
                    self.scheduler_params = scheduler_params
                    self.set_schedulers(self.scheduler_params, num_iter=num_iter)

            self._setup_recon_dataloaders(batch_size, num_workers, val_fraction)

        # Type check for INR-based reconstruction
        if not isinstance(self.dset, TomographyINRDataset):
            raise NotImplementedError(
                "Only TomographyINRDataset is supported for this reconstruction method."
            )

        N = max(self.obj_model.shape)

        if num_samples_per_ray is None:
            num_samples_per_ray = max(self.obj_model.shape)
        else:
            if isinstance(num_samples_per_ray, int):
                num_samples_per_ray = num_samples_per_ray
            else:
                if len(num_samples_per_ray) != num_iter:
                    raise ValueError(
                        "num_samples_per_ray schedule must have the same length as num_iter"
                    )
                if self.global_rank == 0:
                    print("num_samples_per_ray schedule provided.")

        loss_func = get_loss_module(name=loss_type, dtype=self.obj_model.dtype, **loss_func_kwargs)

        pbar = tqdm(range(num_iter), disable=not self.verbose)
        for a0 in pbar:
            consistency_loss = torch.tensor(0.0, device=self.device)
            total_loss = torch.tensor(0.0, device=self.device)
            epoch_soft_constraint_loss = torch.tensor(0.0, device=self.device)
            if isinstance(self.obj_model, ObjectINR) or isinstance(
                self.obj_model, ObjectTensorDecomp
            ):
                self.obj_model.model.train()
            else:
                raise NotImplementedError(
                    "AD Pixelated reconstruction is not yet implemented. Use ObjectINR instead."
                )
            self.dset.train()
            # self._reset_iter_constraints()

            if self.sampler is not None:
                self.sampler.set_epoch(a0)

            if isinstance(num_samples_per_ray, list):
                curr_num_samples_per_ray = num_samples_per_ray[a0][1]
            else:
                curr_num_samples_per_ray = num_samples_per_ray

            for batch_idx, batch in enumerate(self.dataloader):
                self.zero_grad_all()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=False,
                ):
                    all_coords = self.dset.get_coords(batch, N, curr_num_samples_per_ray)

                    all_densities = self.obj_model.forward(all_coords)

                    integrated_densities = self.dset.integrate_rays(
                        all_densities,
                        curr_num_samples_per_ray,
                        len(batch["target_value"]),
                    )

                pred = integrated_densities.float()

                soft_constraints_loss = self.obj_model.apply_soft_constraints(
                    ctx=ReconstructionContext(
                        coords=all_coords,
                        pred=pred,
                        all_densities=all_densities,
                    )
                )

                target = batch["target_value"].to(self.device, non_blocking=True).float()

                batch_consistency_loss = loss_func(pred, target)

                soft_constraints_loss += self.dset.apply_soft_constraints()

                epoch_soft_constraint_loss += soft_constraints_loss.detach()

                batch_loss = batch_consistency_loss.float() + soft_constraints_loss.float()

                batch_loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.obj_model.model.parameters(), max_norm=1.0)
                self.step_optimizers()
                total_loss += batch_loss.detach()
                consistency_loss += batch_consistency_loss.detach()

            if isinstance(self.obj_model.model, CPTilted):
                if a0 == 0:
                    prev_R = self.obj_model.model.so3.as_matrix().detach().clone()
                elif (a0 + 1) % 20 == 0:
                    R_now = self.obj_model.model.so3.as_matrix().detach()
                    # Cumulative angular change per rotation over the last 20 iters.
                    # trace(R_prev^T R_now) = 1 + 2*cos(theta), so theta = acos((trace - 1) / 2).
                    rel_trace = torch.einsum("tij,tij->t", prev_R, R_now)
                    angle = torch.acos(((rel_trace - 1) / 2).clamp(-1, 1))  # (T,) radians
                    angle_deg = torch.rad2deg(angle)
                    per_tau_str = ", ".join(f"{a:.2f}°" for a in angle_deg.tolist())
                    print(
                        f"iter {a0}: 20-iter τ change "
                        f"max={angle_deg.max().item():.2f}°, "
                        f"mean={angle_deg.mean().item():.2f}°, "
                        f"per-τ=[{per_tau_str}]"
                    )
                    prev_R = R_now.clone()

            # One stacked all_reduce and one host sync instead of three of each.
            losses = torch.stack([total_loss, consistency_loss, epoch_soft_constraint_loss])
            if self.world_size > 1:
                dist.all_reduce(losses, dist.ReduceOp.AVG)
            total_loss, consistency_loss, epoch_soft_constraint_loss = (
                losses / len(self.dataloader)
            ).tolist()

            self.step_schedulers(loss=total_loss)
            # TODO: Maybe reorganize the losses so that the order makes sense lol.

            avg_val_loss = None
            if self.val_dataloader is not None:
                print("Validating...")
                self.obj_model.model.eval()
                self.dset.eval()
                with torch.no_grad():
                    val_loss = torch.tensor(0.0, device=self.device)

                    for batch in self.val_dataloader:
                        # Match the training pass (enabled=False): bf16 autocast
                        # breaks the so3 pose solve (lu_factor has no BFloat16
                        # kernel) and would make the val loss inconsistent with
                        # the fp32 training loss it is compared to.
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=False,
                        ):
                            all_coords = self.dset.get_coords(batch, N, curr_num_samples_per_ray)

                            all_densities = self.obj_model.forward(all_coords)

                            integrated_densities = self.dset.integrate_rays(
                                all_densities,
                                curr_num_samples_per_ray,
                                len(batch["target_value"]),
                            )

                            target = (
                                batch["target_value"].to(self.device, non_blocking=True).float()
                            )

                            batch_val_loss = torch.nn.functional.mse_loss(
                                integrated_densities, target
                            )

                            val_loss += batch_val_loss.detach()

                    avg_val_loss = val_loss.item() / len(self.val_dataloader)

            metrics = torch.tensor(
                [total_loss, consistency_loss, epoch_soft_constraint_loss], device=self.device
            )

            if self.world_size > 1:
                dist.all_reduce(metrics, dist.ReduceOp.AVG)

            total_loss, consistency_loss, epoch_soft_constraint_loss = metrics.tolist()

            pbar.set_description(
                f"Reconstruction | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e}"
            )

            self._epoch_losses.append(total_loss)
            self._consistency_losses.append(consistency_loss)
            self.append_learning_rates(self.get_current_lrs())
            self.obj_model._soft_constraint_losses.append(epoch_soft_constraint_loss)
            if avg_val_loss is not None:
                self._val_losses.append(avg_val_loss)

            if self.logger is not None:
                if (
                    self.logger.log_images_every > 0
                    and self.num_epochs % self.logger.log_images_every == 0
                ):
                    pred_full = self.obj_model.obj_view

                    if self.global_rank == 0:
                        self.logger.log_iter_images(
                            pred_volume=pred_full,
                            dataset_model=self.dset,
                            iter=self.num_epochs,
                        )
                    pbar.set_description(
                        f"Reconstruction | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e} | Images Logged"
                    )

                if self.global_rank == 0:
                    self.logger.log_iter(
                        object_model=self.obj_model,
                        iter=self.num_epochs,
                        consistency_loss=consistency_loss,
                        total_loss=total_loss,
                        learning_rates=self.get_current_lrs(),
                        num_samples_per_ray=curr_num_samples_per_ray,
                        val_loss=avg_val_loss if self.val_dataloader is not None else None,
                    )

                self.logger.flush()
            if not self.verbose:
                if self.global_rank == 0:
                    print(
                        f"Reconstruction Epoch {self.num_epochs} | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e}"
                    )
        if show_metrics and self.world_size == 1:
            self.plot_losses()

    # --- Helper Functions ---

    def save_volume(self, path: str = "recon_volume.npz", overwrite: bool = False):
        """
        Saves volume to a numpy array file. Does not save the full Tomography object.
        """
        if self.global_rank == 0:
            if not overwrite and os.path.exists(path):
                raise FileExistsError(
                    f"File {path} already exists. Use overwrite=True to overwrite."
                )
            print(f"Saving volume to {path}")
            np.savez(path, volume=self.obj_model.obj_view)

        if torch.distributed.is_initialized():
            print("Barrier")
            torch.distributed.barrier()

    # Loading and Saving
    @classmethod
    def _recursive_load_from_path(cls, path: str):
        return autoserialize_load(path)

    @classmethod
    def from_file(
        cls,
        path: str,
        device: str = "cpu",
    ) -> Self:
        tomography = cls._recursive_load_from_path(path)
        tomography.to(device)
        tomography._rebuild_dataloader(
            batch_size=tomography.batch_size,
            num_workers=tomography.num_workers,
            val_fraction=tomography.val_fraction,
        )
        return tomography

    def _rebuild_dataloader(self, batch_size: int, num_workers: int, val_fraction: float):
        """
        Rebuilds the dataloader due to persistent workers error when reloading the object.
        """
        self._setup_recon_dataloaders(batch_size, num_workers, val_fraction)

    def _setup_recon_dataloaders(self, batch_size: int, num_workers: int, val_fraction: float):
        """Build the train/val batch iterators.

        INR datasets use ``DeviceBatchSampler`` — batches are built with
        tensor ops on the compute device from a device-resident tilt stack,
        removing the per-pixel ``__getitem__`` / collate / H2D-copy
        dataloader bottleneck (``num_workers`` is ignored on this path).
        Distributed runs shard the same seeded epoch permutation across
        ranks (DistributedSampler semantics; the loop's ``set_epoch`` drives
        reshuffling). Non-INR datasets keep the DataLoader path.
        """
        if isinstance(self.dset, TomographyINRDataset):
            n = len(self.dset)
            n_val = int(n * val_fraction)
            # Fixed-seed split: identical across DDP ranks (no train/val
            # leakage between ranks) and stable across save/reload, so a
            # resumed run keeps validating on the same held-out pixels.
            split_gen = torch.Generator()
            split_gen.manual_seed(0)
            perm = torch.randperm(n, generator=split_gen)
            ddp = dict(rank=self.global_rank, world_size=self.world_size)
            self.dataloader = DeviceBatchSampler(
                self.dset, batch_size, self.device, indices=perm[n_val:], **ddp
            )
            # The val sampler keeps its own device-resident copy of the tilt
            # stack; acceptable, since val_fraction > 0 is the rare case.
            val = (
                DeviceBatchSampler(
                    self.dset, batch_size, self.device, indices=perm[:n_val], shuffle=False, **ddp
                )
                if n_val > 0
                else None
            )
            # A per-rank val shard smaller than one batch would divide by
            # zero in the val-loss average.
            self.val_dataloader = val if val is not None and len(val) > 0 else None
            # The training loop calls set_epoch on self.sampler.
            self.sampler = self.dataloader
            self.val_sampler = None
            return

        self.dataloader, self.sampler, self.val_dataloader, self.val_sampler = (
            self.setup_dataloader(
                self.dset,
                batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
            )
        )

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = ["dataloader"],
        compression_level: int | None = 4,
    ) -> None:
        super(Tomography, self).save(
            path=path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
        )

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 4), ncols=2)

        ax[0].plot(self._epoch_losses, label="Total Training Loss")
        if len(self._val_losses) > 0:
            ax[0].plot(self._val_losses, label="Validation Loss")
        ax[0].legend()

        for key, value in self._lrs.items():
            ax[1].plot(value, label=key)

        ax[1].legend()
        ax[0].legend()
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].set_xlabel("Epoch")
        ax[1].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Learning Rate")


class TomographyConventional(TomographyBase):
    """
    Class for handling all conventional tomography reconstruction methods.
    Will also handle choosing the appropriate dataset model to use.
    """

    @classmethod
    def from_models(
        cls,
        dset: TomographyPixDataset,
        obj_model: ObjectPixelated,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        obj_constraints: dict | ObjConstraintsType | None = None,
        mode: Literal["sirt", "fbp"] = "sirt",
        relaxation: float = 0.25,
        reset: bool = False,
        inline_alignment: bool = False,
        smoothing_sigma: float | None = None,
        show_metrics: bool = False,
    ):
        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

            self.obj_model.constraints = obj_constraints

        pbar = tqdm(
            range(num_iter),
            desc=f"{mode} Reconstruction | Loss: {0:.4f}",
            disable=not self.verbose,
        )
        if mode == "sirt" or mode == "fbp":
            proj_forward = torch.zeros_like(self.dset.tilt_stack).permute(2, 0, 1)
        else:
            proj_forward = torch.zeros_like(self.dset.tilt_stack)

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        patience = self.dset.tilt_angles.max() // 10
        for iter in pbar:
            proj_forward, loss = self._reconstruction_epoch(
                inline_alignment=inline_alignment,
                mode=mode,
                proj_forward=proj_forward,
                gaussian_kernel=gaussian_kernel,
                relaxation=relaxation,
            )

            pbar.set_description(f"{mode} Reconstruction | Loss: {loss.item():.4f}")

            self._epoch_losses.append(loss.item())

            # Change relaxation parameter if loss greater than last epoch
            if len(self._epoch_losses) > 1 and self._epoch_losses[-1] > self._epoch_losses[-2]:
                if patience == 0:
                    relaxation *= 0.85
                    print(f"Relaxation parameter changed to: {relaxation}")
                    patience = 10
                else:
                    patience -= 1

            if mode == "fbp":
                break

        if show_metrics:
            self.plot_losses()

    # --- Conventional reconstruction method ---
    def _adaptive_relaxation(self, n_power_iter: int = 10) -> float:
        raise NotImplementedError(
            "Adaptive relaxation hasn't been implemented, please input a valid relaxation parameter."
        )

    def _reconstruction_epoch(
        self,
        inline_alignment: bool,
        mode: Literal["sirt", "fbp"],
        proj_forward: torch.Tensor,
        relaxation: float,
        gaussian_kernel: torch.Tensor | None = None,
    ):
        loss = 0

        if relaxation == 0.0:
            relaxation = self._adaptive_relaxation()
            print(f"Adaptive relaxation: {relaxation}")
        if inline_alignment:
            for ind in range(len(self.dset.tilt_angles)):
                im_proj = proj_forward[:, ind, :]
                # proj_forward rows are volume slices (the tilt axis), i.e. the
                # transpose of the stored tilt image -- the same orientation the
                # error term below compares against.
                im_meas = self.dset.forward(ind).target.T  # type: ignore
                shift = torch_phase_cross_correlation(im_proj, im_meas)
                if torch.linalg.norm(shift) <= 32:
                    shifted = torch.fft.ifft2(
                        torch.fft.fft2(im_meas)
                        * torch.exp(
                            -2j
                            * np.pi
                            * (
                                shift[0]
                                * torch.fft.fftfreq(
                                    im_meas.shape[0], device=im_meas.device
                                ).unsqueeze(1)
                                + shift[1]
                                * torch.fft.fftfreq(im_meas.shape[1], device=im_meas.device)
                            )
                        )
                    ).real

                    # Persist the aligned measurement in the tilt stack: the error
                    # term reads the stack, and proj_forward is overwritten by
                    # radon_torch below, so writing the aligned image there
                    # silently discarded the alignment.
                    self.dset.tilt_stack[ind] = shifted.T

        if mode == "sirt" or mode == "fbp":
            proj_forward = radon_torch(
                self.obj_model.obj,
                theta=self.dset.tilt_angles,
                device=self.device,
            )

            error = self.dset.tilt_stack.permute(2, 0, 1) - proj_forward

            correction = iradon_torch(
                error,
                theta=self.dset.tilt_angles,
                device=self.device,
                filter_name="ramp",
                circle=True,
            )

            normalization = iradon_torch(
                torch.ones_like(error),
                theta=self.dset.tilt_angles,
                device=self.device,
                circle=True,
                filter_name=None,
            )

            normalization[normalization == 0] = 1e-6

            correction /= normalization

            self.obj_model.obj += correction * relaxation

            if gaussian_kernel is not None:
                self.obj_model.obj = gaussian_filter_2d_stack(self.obj_model.obj, gaussian_kernel)

        loss = torch.mean(torch.abs(error))

        return proj_forward, loss

    # --- Helper Functions ---

    def plot_losses(self):
        fig, ax = plt.subplots()
        ax.plot(self._epoch_losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Reconstruction Loss")
        ax.set_yscale("log")
        plt.show()
