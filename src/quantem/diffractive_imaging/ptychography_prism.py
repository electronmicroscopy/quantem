from typing import Literal, Self

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbePRISM
from quantem.diffractive_imaging.ptychography_base import PtychographyBase


class PtychoPRISM(PtychographyBase):
    """
    PRISM-accelerated ptychographic reconstruction using plane wave decomposition.

    This class implements the PRISM algorithm for efficient 4D-STEM simulation and
    reconstruction. Instead of computing a multislice calculation for each probe
    position, it:
    1. Computes multislice for a set of tilted plane waves
    2. Builds PRISM coefficients in reciprocal space
    3. Reconstructs exit waves by linear combinations of propagated plane waves

    The reconstruction uses automatic differentiation (autograd) for gradient computation.
    """

    _token = object()

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbePRISM,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        """
        Create PtychoPRISM from component models.

        Parameters
        ----------
        dset : DatasetModelType
            Dataset model containing diffraction patterns and scan positions
        obj_model : ObjectModelType
            Object model (ObjectPixelated or ObjectDIP )
        probe_model : ProbePRISM
            PRISM probe model with plane wave decomposition
        detector_model : DetectorModelType
            Detector model for converting exit waves to intensities
        logger : LoggerPtychography | None
            Optional logger for tracking reconstruction progress
        device : str | int
            Device to run on ('cpu', 'gpu', 'cuda:0', etc.)
        verbose : int | bool
            Verbosity level
        rng : np.random.Generator | int | None
            Random number generator

        Returns
        -------
        PtychoPRISM
            Initialized PRISM ptychography reconstruction object
        """
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

    def _compute_propagated_plane_waves(self, max_batch_size):
        """
        Propagate PRISM plane waves through the object.

        This is the core PRISM operation: instead of propagating probes at each
        position, we propagate a compact set of plane waves once and reuse them.
        """
        sampling = self.sampling
        gpts = self.obj_shape_full[-2:]
        extent = gpts * sampling

        # Get plane waves from probe model on object FOV
        plane_waves = self.probe_model._prism_plane_waves(
            self.probe_model.wave_vectors, extent, gpts
        )

        # Get full object transmission functions
        obj_array = self.obj_model.obj
        if self.obj_model.obj_type == "potential":
            transmission = torch.exp(1.0j * obj_array)
        else:
            transmission = obj_array

        all_waves = []
        for start, end in generate_batches(plane_waves.shape[0], max_batch=max_batch_size):
            waves = plane_waves[start:end]

            # Multislice propagation
            for s in range(self.num_slices):
                # transmit
                transmission_slice = transmission[s]
                waves = waves * transmission_slice

                # Propagate
                if s < self.num_slices - 1:
                    waves = self._propagate_array(waves, self._propagators[s])

            all_waves.append(waves)

        return torch.cat(all_waves, dim=0)

    def forward_operator(
        self,
        prism_coefs: torch.Tensor,
        patch_indices: torch.Tensor,
        max_batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        PRISM forward operator: reconstruct exit waves from plane wave basis.

        Parameters
        ----------
        prism_coefs : torch.Tensor
            PRISM coefficients [num_probes, batch_size, num_waves]
        patch_indices : torch.Tensor
            Indices for extracting patches [batch_size, roi_h, roi_w]

        Returns
        -------
        torch.Tensor
            Exit waves [num_probes, batch_size, roi_h, roi_w]
        """

        # Extract patches from propagated plane waves
        propagated_plane_waves = self._compute_propagated_plane_waves(max_batch_size)

        exit_waves = torch.tensordot(
            prism_coefs,  # [num_probes, num_positions, num_waves]
            propagated_plane_waves,  # [num_waves, obj_h, obj_w]
            dims=((2,), (0,)),
        )  # -> [num_probes, num_positions, obj_h, obj_w]

        num_probes, num_positions, obj_h, obj_w = exit_waves.shape
        exit_flat = exit_waves.reshape(num_probes, num_positions, obj_h * obj_w)

        patch_idx_expanded = patch_indices.unsqueeze(0).expand(num_probes, -1, -1, -1)
        batch_idx = torch.arange(num_positions, device=exit_flat.device)
        batch_idx = batch_idx[None, :, None, None].expand_as(patch_idx_expanded)

        patch_idx_flat = patch_idx_expanded.reshape(num_probes, num_positions, -1).to(torch.int64)
        real_patches = torch.gather(exit_flat.real, 2, patch_idx_flat)
        imag_patches = torch.gather(exit_flat.imag, 2, patch_idx_flat)

        roi_h, roi_w = patch_indices.shape[-2:]
        real_patches = real_patches.reshape(num_probes, num_positions, roi_h, roi_w)
        imag_patches = imag_patches.reshape(num_probes, num_positions, roi_h, roi_w)

        exit_patches = torch.complex(real_patches, imag_patches)

        return exit_patches

    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        store_iterations: bool | None = None,
        store_iterations_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        loss_type: Literal[
            "l2_amplitude", "l1_amplitude", "l2_intensity", "l1_intensity", "poisson"
        ] = "l2_amplitude",
    ) -> Self:
        """
        Perform PRISM-accelerated ptychographic reconstruction.

        Parameters
        ----------
        num_iter : int
            Number of iterations to run
        reset : bool
            Whether to reset the reconstruction before starting
        optimizer_params : dict | None
            Optimizer parameters for object model
        scheduler_params : dict | None
            Learning rate scheduler parameters
        constraints : dict
            Constraints to apply during reconstruction
        batch_size : int | None
            Batch size for processing
        store_iterations : bool | None
            Whether to store snapshots of iterations
        store_iterations_every : int | None
            How often to store snapshots
        device : Literal["cpu", "gpu"] | None
            Device to use for computation
        loss_type : str
            Loss function type

        Returns
        -------
        Self
            The reconstruction object
        """
        self._check_preprocessed()

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

        # Setup optimizers (only object model has parameters to optimize)
        if optimizer_params is not None:
            # Only object model needs optimization
            if "object" in optimizer_params:
                self.obj_model.set_optimizer(optimizer_params["object"])
            else:
                self.obj_model.set_optimizer(optimizer_params)

            # Add probe optimizer if parameters are learnable
            if "probe" in optimizer_params and self.probe_model.params is not None:
                self.probe_model.set_optimizer(optimizer_params["probe"])

        if scheduler_params is not None:
            if "object" in scheduler_params:
                self.obj_model.set_scheduler(scheduler_params["object"], num_iter)
            else:
                self.obj_model.set_scheduler(scheduler_params, num_iter)
            if "probe" in scheduler_params:
                self.probe_model.set_scheduler(scheduler_params["probe"], num_iter)
            else:
                self.probe_model.set_scheduler(scheduler_params, num_iter)

        self.dset._set_targets(loss_type)
        pbar = tqdm(range(num_iter), disable=not self.verbose)
        indices = torch.arange(self.dset.num_gpts)

        for epoch in pbar:
            epoch_loss = 0.0
            epoch_consistency_loss = 0.0
            self._reset_epoch_constraints()

            # Zero gradients
            if self.obj_model.optimizer is not None:
                self.obj_model.optimizer.zero_grad()
            if self.probe_model.optimizer is not None:
                self.probe_model.optimizer.zero_grad()

            # Get batch data
            patch_indices = self.dset.patch_indices
            positions_px = self.dset.scan_positions_px
            positions = positions_px * torch.as_tensor(self.sampling, dtype=torch.float32)

            prism_coefs = self.probe_model.forward(positions)
            exit_waves = self.forward_operator(prism_coefs, patch_indices, self.batch_size)
            pred_intensities = self.detector_model.forward(exit_waves) / self.roi_shape.prod()

            # Compute loss
            batch_consistency_loss, _targets = self.error_estimate(
                pred_intensities,
                indices,
                loss_type=loss_type,
            )

            # Apply soft constraints
            batch_soft_constraint_loss = self._soft_constraints()

            batch_loss = batch_consistency_loss + batch_soft_constraint_loss

            # Backward pass (autograd)
            self.backward(batch_loss)

            # Optimizer step
            if self.obj_model.optimizer is not None:
                self.obj_model.optimizer.step()
            if self.probe_model.optimizer is not None:
                self.probe_model.optimizer.step()

            epoch_consistency_loss += batch_consistency_loss.item()
            epoch_loss += batch_loss.item()

            # Record epoch
            self._record_epoch(epoch_loss)

            # Step scheduler
            if self.obj_model.scheduler is not None:
                if isinstance(
                    self.obj_model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.obj_model.scheduler.step(epoch_loss)
                else:
                    self.obj_model.scheduler.step()
            if self.probe_model.scheduler is not None:
                if isinstance(
                    self.probe_model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.probe_model.scheduler.step(epoch_loss)
                else:
                    self.probe_model.scheduler.step()

            # Store iteration
            if self.store_iterations and (epoch % self.store_iterations_every) == 0:
                self.append_recon_iteration()

            # Logging
            if self.logger is not None:
                self.logger.log_epoch(
                    self.obj_model,
                    self.probe_model,
                    self.dset,
                    self.num_epochs - 1,
                    epoch_consistency_loss,
                    1,
                    self._get_current_lrs(),
                )

            pbar.set_description(f"Epoch {epoch + 1}/{num_iter}, Loss: {epoch_loss:.3e}")

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
    ):
        """ """
        loss.backward()
        # scaling pixelated ad gradients to closer match analytic
        if isinstance(self.obj_model, ObjectPixelated):
            obj_grad_scale = self.dset.upsample_factor**2 / 2  # factor of 2 from l2 grad
            self.obj_model._obj.grad.mul_(obj_grad_scale)  # type:ignore

    def _record_epoch(self, epoch_loss: float) -> None:
        """Record epoch metrics."""
        self._epoch_losses.append(epoch_loss)

        # Record learning rates
        if "object" not in self._epoch_lrs:
            self._epoch_lrs["object"] = []

        if self.obj_model.optimizer is not None:
            lr = self.obj_model.optimizer.param_groups[0]["lr"]
        else:
            lr = 0.0

        self._epoch_lrs["object"].append(lr)

    def _reset_epoch_constraints(self) -> None:
        """Reset constraint loss accumulation."""
        self.obj_model.reset_epoch_constraint_losses()

    def _soft_constraints(self) -> torch.Tensor:
        """Calculate soft constraints for object model."""
        return self.obj_model.apply_soft_constraints(self.obj_model.obj, mask=self.obj_model.mask)

    def reset_recon(self) -> None:
        """Reset reconstruction to initial state."""
        super().reset_recon()
        self.obj_model.reset_optimizer()
        self._propagated_plane_waves = None
