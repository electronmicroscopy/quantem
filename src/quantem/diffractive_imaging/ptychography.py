import contextlib
import copy
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, Sequence, cast
from warnings import warn

import numpy as np
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import load as autoserialize_load
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeModelType, ProbeParametric
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
    ) -> Self:
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

    @classmethod
    def from_ptychography(
        cls,
        ptycho: Self,
        obj_model: ObjectModelType | None = None,
        probe_model: ProbeModelType | None = None,
        logger: LoggerPtychography | None = None,
    ) -> Self:
        _tmp_logger = ptycho.logger
        ptycho.logger = None
        cloned = ptycho.clone()
        ptycho.logger = _tmp_logger
        if obj_model is not None:
            cloned.obj_model = obj_model
        if probe_model is not None:
            cloned.probe_model = probe_model
        if logger is not None:
            cloned.logger = logger

        cloned.reset_recon()
        return cloned

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

        self.dset._set_targets(loss_type)
        batcher = SimpleBatcher(self.dset.num_gpts, self.batch_size, rng=self.rng)
        pbar = tqdm(range(num_iter), disable=not self.verbose)

        for a0 in pbar:
            consistency_loss = 0.0
            total_loss = 0.0
            self._reset_epoch_constraints()

            for batch_indices in batcher:
                self.zero_grad_all()
                patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                    self.dset.forward(batch_indices, self.obj_padding_px)
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
                    self.num_epochs - 1,
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

            if isinstance(self.probe_model, ProbeParametric):
                probe_grad_scale = np.sqrt(self.probe_model._mean_diffraction_intensity)
                for par in self.probe_model.params:
                    par.grad.mul_(probe_grad_scale)  # type:ignore

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

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = (),
        compression_level: int | None = 4,
        save_raw_data: bool = False,
        verbose: int | bool = True,
    ):
        """
        Save the ptychography object, optionally excluding raw dataset data.

        By default, this method saves the ptychography object without the raw dataset
        to save space and allow for dataset reloading. Use save_raw_data=True if you
        want to include the complete dataset.

        When saving without raw data, the system automatically saves:
        - Dataset file path and file type
        - All preprocessing parameters (CoM fitting, rotation, padding, etc.)
        - Reconstruction state (losses, constraints, etc.)

        On load, if no dataset is provided, the system will automatically:
        - Reload the dataset from the saved file path
        - Reapply all preprocessing with the exact same parameters
        - Restore the reconstruction state

        Parameters
        ----------
        path : str | Path
            Path to save the object
        mode : Literal["w", "o"]
            Write mode ('w' for write, 'o' for overwrite)
        store : Literal["auto", "zip", "dir"]
            Storage format
        skip : str | type | Sequence[str | type]
            Additional items to skip during serialization
        compression_level : int | None
            Compression level for zip storage
        save_raw_data : bool
            Whether to save the raw dataset data (default: False)

        Examples
        --------
        # Save without raw data (default behavior) - includes dataset metadata
        ptycho.save("my_reconstruction.zip")

        # Save with raw data included
        ptycho.save("my_reconstruction_with_data.zip", save_raw_data=True)

        # Load a saved reconstruction - automatically reloads dataset
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip")

        # Load and move to GPU
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip", device="gpu")

        # Load with custom dataset (overrides automatic reloading)
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip", dset=my_dataset)

        """
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip = list(skip)

        # Always skip raw dataset data unless explicitly requested
        if not save_raw_data:
            skip.extend(
                [
                    "_dset",  # Skip the dataset object itself
                    "dset",  # Skip dataset references
                ]
            )

            # Save dataset metadata for automatic reloading
            self._dataset_metadata = {
                "file_path": str(self.dset.dset.file_path) if self.dset.dset.file_path else None,
                "preprocessing_params": self.dset._preprocessing_params,
                "learned_scan_positions_px": self.dset.scan_positions_px.data.cpu(),
                "learned_descan_shifts": self.dset.descan_shifts.data.cpu(),
            }

        # Add other common skips for ptychography objects
        skips = skip

        current_device = self.device
        self.to("cpu")

        if self.verbose and verbose:
            print(f"Saving ptychography object to {path}")

        super().save(
            path,
            mode=mode,
            store=store,
            skip=skips,
            compression_level=compression_level,
        )

        self.to(current_device)  # TODO figure out why this isn't working for DDIP sometimes?

        # Clean up temporary metadata
        if not save_raw_data and hasattr(self, "_dataset_metadata"):
            delattr(self, "_dataset_metadata")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dset: DatasetModelType | None = None,
        device: str | int | None = None,
        verbose: int | bool | None = None,
        auto_reload_dataset: bool = True,
    ):
        """
        Load a ptychography object from a saved file.

        Parameters
        ----------
        path : str | Path
            Path to the saved ptychography object
        dset : DatasetModelType | None
            Dataset to use (if None and auto_reload_dataset=True, will try to reload from saved metadata)
        device : str | int | None
            Device to load the object on
        verbose : int | bool | None
            Verbosity level
        rng : np.random.Generator | int | None
            Random number generator
        auto_reload_dataset : bool
            Whether to automatically reload and preprocess the dataset from saved metadata

        Returns
        -------
        Ptychography
            Loaded ptychography object
        """
        # Load the base object without the dataset
        ptycho = cls._recursive_load_from_path(path)

        if not isinstance(ptycho, Ptychography):
            raise ValueError("Loaded object is not a Ptychography object")

        # If no dataset was provided, try to reload it from saved metadata
        if dset is None and auto_reload_dataset and not hasattr(ptycho, "dset"):
            if hasattr(ptycho, "_dataset_metadata") and ptycho._dataset_metadata:
                metadata = ptycho._dataset_metadata
                file_path = metadata.get("file_path")

                if file_path:
                    # Import here to avoid circular imports
                    from quantem.core.io.file_readers import read_4dstem
                    from quantem.diffractive_imaging.dataset_models import (
                        PtychographyDatasetRaster,
                    )

                    # Reload the dataset
                    print(f"reloading dataset from {file_path}", end="\r")
                    try:
                        raw_dset = read_4dstem(file_path)
                    except (ValueError, ModuleNotFoundError) as _e:
                        try:
                            raw_dset = autoserialize_load(file_path)
                            raw_dset.file_path = file_path  # legacy support
                        except Exception as e:
                            raise ValueError(
                                f"Could not automatically reload dataset from {file_path}: {e}"
                            )

                    dset = PtychographyDatasetRaster.from_dataset4dstem(
                        raw_dset, verbose=verbose or 1
                    )
                    # Apply preprocessing with saved parameters
                    preprocessing_params = metadata.get("preprocessing_params", {})
                    _v = dset.verbose
                    dset.verbose = 0
                    dset.preprocess(**preprocessing_params)
                    dset.verbose = _v

                    print(f"Successfully reloaded dataset from {file_path}")
                else:
                    dset = None
            else:
                print("Warning: No dataset metadata found in saved object.")
                dset = None
        elif dset is not None:
            dset._set_initial_scan_positions_px(ptycho.obj_padding_px)
            dset._set_patch_indices(ptycho.obj_padding_px)
            if hasattr(ptycho, "_dataset_metadata") and ptycho._dataset_metadata:
                metadata = ptycho._dataset_metadata
                # preserve learned scan positions and descan shifts
                if "learned_scan_positions_px" in metadata:
                    dset.scan_positions_px.data = metadata["learned_scan_positions_px"]
                if "learned_descan_shifts" in metadata:
                    dset.descan_shifts.data = metadata["learned_descan_shifts"]

        # check if dset was attached to ptycho object
        if dset is not None:
            ptycho.dset = dset
        elif not (hasattr(ptycho, "_dset") and ptycho._dset is not None):
            warn(
                "No dataset provided and could not automatically reload dataset.\n"
                "Please provide a dataset parameter or ensure the object was saved with dataset metadata.\n"
                "Many functionalities will not work without the dataset attached."
            )
            # raise ValueError(
            #     "No dataset provided and could not automatically reload dataset. "
            #     "Please provide a dataset parameter or ensure the object was saved with dataset metadata."
            # )

        if device is not None:
            ptycho.to(device)

        return ptycho

    @classmethod
    def _recursive_load_from_path(cls, path: str | Path):
        """Helper method to load an object from a path using AutoSerialize."""
        return autoserialize_load(path)

    def clone(self, device: str | int = "cpu") -> Self:
        """
        Create a deep-copy clone of this Ptychography instance.

        The clone is placed on CPU by default (device="cpu"). You can override
        the output device by passing a different device string.

        This method first attempts a Python deepcopy for speed. If that fails
        (e.g., due to non-copyable objects), it falls back to serializing the
        object to a temporary file and reloading it, which is robust and includes
        the dataset by default.
        """
        try:
            cloned: Self = copy.deepcopy(self)
        except Exception:
            # Robust fallback: save then reload including raw dataset data so that
            # the in-memory state is fully preserved without relying on external files.
            tmp_path = (
                Path(tempfile.gettempdir()) / f"ptycho_clone_{self.rng.integers(int(1e7))}.zip"
            )
            try:
                self.save(tmp_path, mode="o", store="zip", save_raw_data=True, verbose=0)
                cloned = cast(
                    Self, Ptychography.from_file(tmp_path, device=None, auto_reload_dataset=False)
                )
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()

        if self.logger is not None:
            cloned.logger = self.logger.clone()

        cloned.to(device)
        return cloned
