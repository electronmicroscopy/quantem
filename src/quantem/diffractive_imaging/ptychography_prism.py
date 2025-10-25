from typing import Literal, Self

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType
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
        )  # [num_waves, obj_h, obj_w]

        # Get full object transmission functions
        obj_array = self.obj_model.obj  # [num_slices, obj_h, obj_w]
        if self.obj_model.obj_type == "potential":
            transmission = torch.exp(1.0j * obj_array)
        else:
            transmission = obj_array

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

            plane_waves[start:end] = waves

        return plane_waves

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

        return exit_waves

        # # MPS-safe indexing
        # real = exit_waves.real[:, patch_indices]
        # imag = exit_waves.imag[:, patch_indices]
        # exit_patches = torch.complex(real, imag)  # [num_probes, num_positions, roi_h, roi_w]

        # return exit_patches

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

        # # Setup optimizers (only object model has parameters to optimize)
        # new_scheduler = reset
        # if optimizer_params is not None:
        #     # Only object model needs optimization
        #     if "object" in optimizer_params:
        #         self.obj_model.set_optimizer(optimizer_params["object"])
        #     else:
        #         self.obj_model.set_optimizer(optimizer_params)
        #     new_scheduler = True

        # if scheduler_params is not None:
        #     if "object" in scheduler_params:
        #         self.obj_model.set_scheduler(scheduler_params["object"], num_iter)
        #     else:
        #         self.obj_model.set_scheduler(scheduler_params, num_iter)
        #     new_scheduler = True

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

            # Get batch data
            patch_indices, _positions_px, positions_px_fractional, _descan = self.dset.forward(
                indices, self.obj_padding_px
            )
            positions = _positions_px * self.dset.scan_sampling

            prism_coefs = self.probe_model.forward(positions)
            exit_waves = self.forward_operator(prism_coefs, patch_indices, self.batch_size)
            pred_intensities = self.detector_model.forward(exit_waves)

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
            batch_loss.backward()

            # Optimizer step
            if self.obj_model.optimizer is not None:
                self.obj_model.optimizer.step()

            epoch_consistency_loss += batch_consistency_loss.item()
            epoch_loss += batch_loss.item()

            # Average losses over batches
            num_batches = len(indices)
            epoch_loss /= num_batches
            epoch_consistency_loss /= num_batches

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

            # Store iteration
            if self.store_iterations and (epoch % self.store_iterations_every) == 0:
                self.append_recon_iteration()

            # Logging
            if self.logger is not None:
                current_lr = (
                    self.obj_model.optimizer.param_groups[0]["lr"]
                    if self.obj_model.optimizer is not None
                    else 0.0
                )
                self.logger.log_epoch(
                    self.obj_model,
                    self.probe_model,
                    self.dset,
                    self.num_epochs - 1,
                    epoch_consistency_loss,
                    num_batches,
                    {"object": current_lr},
                )

            pbar.set_description(f"Epoch {epoch + 1}/{num_iter}, Loss: {epoch_loss:.3e}")

        torch.cuda.empty_cache()
        return self

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
