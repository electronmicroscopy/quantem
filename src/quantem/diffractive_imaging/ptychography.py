from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from tqdm.auto import tqdm

from quantem.core import config
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeModelType
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_opt import PtychographyOpt
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class Ptychography(PtychographyOpt, PtychographyVisualizations, PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        return cls(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            logger=logger,
            device=device,
            verbose=verbose,
            rng=rng,
            _token=cls._token,
        )

    # region --- explicit properties and setters ---

    @property
    def autograd(self) -> bool:
        return self._autograd

    @autograd.setter
    def autograd(self, autograd: bool) -> None:
        self._autograd = bool(autograd)

    # endregion --- explicit properties and setters ---

    # region --- methods ---
    # TODO reset RNG as well
    def reset_recon(self) -> None:
        super().reset_recon()
        self.obj_model.reset_optimizer()
        self.probe_model.reset_optimizer()
        self.dset.reset_optimizer()

    def _record_epoch(self, epoch_loss: float) -> None:
        self._epoch_losses.append(epoch_loss)
        optimizers = self.optimizers
        all_keys = set(self._epoch_lrs.keys()) | set(optimizers.keys())
        for key in all_keys:
            if key in self._epoch_lrs.keys():
                if key in optimizers.keys():
                    self._epoch_lrs[key].append(optimizers[key].param_groups[0]["lr"])
                else:
                    self._epoch_lrs[key].append(0.0)
            else:  # new optimizer
                # For new optimizers, backfill with 0.0 LR for previous epochs
                current_epoch = self.num_epochs - 1  # -1 because loss was just appended
                prev_lrs = [0.0] * current_epoch
                prev_lrs.append(optimizers[key].param_groups[0]["lr"])
                self._epoch_lrs[key] = prev_lrs

    def _reset_epoch_constraints(self) -> None:
        """Reset constraint loss accumulation for all models."""
        self.obj_model.reset_epoch_constraint_losses()
        self.probe_model.reset_epoch_constraint_losses()
        self.dset.reset_epoch_constraint_losses()

    def _soft_constraints(self) -> torch.Tensor:
        """Calculate soft constraints by calling apply_soft_constraints on each model."""
        total_loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)

        obj_loss = self.obj_model.apply_soft_constraints(
            self.obj_model.obj, mask=self.obj_model.mask
        )
        total_loss += obj_loss

        probe_loss = self.probe_model.apply_soft_constraints(self.probe_model.probe)
        total_loss += probe_loss

        dataset_loss = self.dset.apply_soft_constraints(self.dset.descan_shifts)
        total_loss += dataset_loss

        return total_loss

    # endregion --- methods ---

    # region --- reconstruction ---

    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        obj_type: Literal["complex", "pure_phase", "potential"] | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        store_iterations: bool | None = None,
        store_iterations_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        autograd: bool = True,
        loss_type: Literal[
            "l2_amplitude", "l1_amplitude", "l2_intensity", "l1_intensity", "poisson"
        ] = "l2_amplitude",
    ) -> Self:
        """
        reason for having a single reconstruct() is so that updating things like constraints
        or recon_types only happens in one place, reason for having separate reoconstruction_
        methods would be to simplify the flags for this and not have to include all

        """
        # TODO maybe make an "process args" method that handles things like:
        # mode, store_iterations, device,
        self._check_preprocessed()
        self.set_obj_type(obj_type, force=reset)  # TODO update this or remove, DIPs...
        if device is not None:
            self.to(device)
        self.batch_size = batch_size
        self.store_iterations_every = store_iterations_every
        if store_iterations_every is not None and store_iterations is None:
            self.store_iterations = True
        else:
            self.store_iterations = store_iterations

        if reset:
            self.reset_recon()
        self.constraints = constraints

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

        learn_descan = True if "dataset" in self.optimizer_params.keys() else False
        self.dset._set_targets(loss_type, learn_descan)
        batcher = SimpleBatcher(self.dset.num_gpts, self.batch_size, rng=self.rng)
        pbar = tqdm(range(num_iter), disable=not self.verbose)

        for a0 in pbar:
            consistency_loss = 0.0
            total_loss = 0.0
            self._reset_epoch_constraints()

            for batch_indices in batcher:
                self.zero_grad_all()
                patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                    self.dset.forward(batch_indices, self.obj_padding_px, learn_descan)
                )
                shifted_probes = self.probe_model.forward(positions_px_fractional)
                obj_patches = self.obj_model.forward(patch_indices)
                propagated_probes, overlap = self.forward_operator(
                    obj_patches, shifted_probes, descan_shifts
                )
                pred_intensities = self.detector_model.forward(overlap)

                batch_consistency_loss, targets = self.error_estimate(
                    pred_intensities,
                    batch_indices,
                    loss_type=loss_type,
                )

                batch_soft_constraint_loss = self._soft_constraints()
                batch_loss = batch_consistency_loss + batch_soft_constraint_loss

                self.backward(
                    batch_loss,
                    autograd,
                    obj_patches,
                    propagated_probes,
                    overlap,
                    patch_indices,
                    targets,
                )
                self.step_optimizers()
                consistency_loss += batch_consistency_loss.item()
                total_loss += batch_loss.item()

            num_batches = len(batcher)
            total_loss = total_loss / num_batches
            consistency_loss = consistency_loss / num_batches
            self._record_epoch(total_loss)

            # Step schedulers with current loss
            self.step_schedulers(total_loss)

            if self.store_iterations and (a0 % self.store_iterations_every) == 0:
                self.append_recon_iteration()

            if self.logger is not None:
                self.logger.log_epoch(
                    self.obj_model,
                    self.probe_model,
                    self.dset,
                    a0,
                    consistency_loss,
                    num_batches,
                    self._get_current_lrs(),
                )

            pbar.set_description(f"Epoch {a0 + 1}/{num_iter}, Loss: {total_loss:.3e}")

        torch.cuda.empty_cache()
        return self

    def _get_current_lrs(self) -> dict[str, float]:
        return {
            param_name: optimizer.param_groups[0]["lr"]
            for param_name, optimizer in self.optimizers.items()
            if optimizer is not None
        }

    def backward(
        self,
        loss: torch.Tensor,
        autograd: bool,
        obj_patches: torch.Tensor,
        propagated_probes: torch.Tensor,
        overlap: torch.Tensor,
        patch_indices: torch.Tensor,
        amplitudes: torch.Tensor,
    ):
        if autograd:
            loss.backward()
            # scaling pixelated ad gradients to closer match analytic
            if isinstance(self.obj_model, ObjectPixelated):
                obj_grad_scale = self.dset.upsample_factor**2 / 2  # factor of 2 from l2 grad
                self.obj_model._obj.grad.mul_(obj_grad_scale)  # type:ignore

        else:
            gradient = self.gradient_step(amplitudes, overlap)
            prop_gradient = self.obj_model.backward(
                gradient,
                obj_patches,
                propagated_probes,
                self._propagators,
                patch_indices,
            )
            self.probe_model.backward(prop_gradient, obj_patches)

    def gradient_step(self, amplitudes, overlap):
        """Computes analytical gradient using the Fourier projection modified overlap"""
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        ## mod_overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        ## grad shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        return modified_overlap - overlap

    def fourier_projection(self, measured_amplitudes, overlap_array):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        # corner centering measured amplitudes
        measured_amplitudes = torch.fft.fftshift(measured_amplitudes, dim=(-2, -1))
        fourier_overlap = torch.fft.fft2(overlap_array, norm="ortho")
        if self.num_probes == 1:  # faster
            fourier_modified_overlap = measured_amplitudes * torch.exp(
                1.0j * torch.angle(fourier_overlap)
            )
        else:  # necessary for mixed state # TODO check this with normalization
            farfield_amplitudes = self.estimate_amplitudes(overlap_array, corner_centered=True)
            farfield_amplitudes[farfield_amplitudes == 0] = torch.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification[None] * fourier_overlap

        return torch.fft.ifft2(fourier_modified_overlap, norm="ortho")

    # endregion --- reconstruction ---

    def dummy(self):
        print("Hi this is a test1")
