from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from tqdm.auto import tqdm

from quantem.core import config
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.object_models import ObjectModelType
from quantem.diffractive_imaging.probe_models import ProbeModelType
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_ml import PtychographyML
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class Ptychography(PtychographyML, PtychographyVisualizations, PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    OPTIMIZABLE_VALS = ["object", "probe", "descan", "scan_positions"]
    DEFAULT_LRS = {
        "object": 5e-3,
        "probe": 1e-3,
        "descan": 1e-3,
        "scan_positions": 1e-3,
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
    }
    DEFAULT_OPTIMIZER_TYPE = "adam"
    # _token = object()

    def __init__(
        self,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
        _token: None | object = None,
    ):
        super().__init__(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            device=device,
            verbose=verbose,
            rng=rng,
            _token=_token,
        )
        self._autograd = True

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        return cls(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
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

    def get_tv_loss(
        self, array: torch.Tensor, weights: None | tuple[float, float] = None
    ) -> torch.Tensor:
        """
        weight is tuple (weight_z, weight_yx) or float -> (weight, weight)
        for 2D array, only weight_yx is used

        """
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
        if weights is None:
            w = (
                self.constraints["object"]["tv_weight_z"],
                self.constraints["object"]["tv_weight_yx"],
            )
        elif isinstance(weights, (float, int)):
            if weights == 0:
                return loss
            w = (weights, weights)
        else:
            if not any(weights):
                return loss
            if len(weights) != 2:
                raise ValueError(f"weights must be a tuple of length 2, got {weights}")
            w = weights

        if array.is_complex():
            ph = array.angle()
            loss += self._calc_tv_loss(ph, w)
            amp = array.abs()
            if torch.max(amp) - torch.min(amp) > 1e-3:  # is complex and not pure_phase
                loss += self._calc_tv_loss(amp, w)
        else:
            loss += self._calc_tv_loss(array, w)

        return loss

    def _calc_tv_loss(self, array: torch.Tensor, weight: tuple[float, float]) -> torch.Tensor:
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
        calc_dim = 0
        for dim in range(array.ndim):
            if dim == 0 and array.ndim == 3:  # there's surely a cleaner way but whatev
                w = weight[0]
            else:
                w = weight[1]
            if w > 0:
                calc_dim += 1
                loss += w * torch.mean(torch.abs(array.diff(dim=dim)))  # careful w/ mean -> NaN
        loss /= calc_dim
        return loss

    def get_surface_zero_loss(
        self, array: torch.Tensor, weight: float | int = 0.0
    ) -> torch.Tensor:
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
        if weight == 0:
            return loss
        if array.shape[0] < 3:
            return loss  # no surfaces to zero
        if array.is_complex():
            amp = array.abs()
            ph = array.angle()
            loss += weight * (torch.mean(amp[0]) + torch.mean(amp[-1]))
            loss += weight * (torch.mean(torch.diff(ph[0])) + torch.mean(torch.diff(ph[-1])))
        else:
            loss += weight * (torch.mean(array[0]) + torch.mean(array[-1]))
        return loss

    def reset_recon(self) -> None:
        super().reset_recon()
        self._optimizers = {}
        self._schedulers = {}

    def _record_lrs(self) -> None:
        optimizers = self.optimizers
        all_keys = set(self._epoch_lrs.keys()) | set(optimizers.keys())
        for key in all_keys:
            if key in self._epoch_lrs.keys():
                if key in self.optimizers.keys():
                    self._epoch_lrs[key].append(self.optimizers[key].param_groups[0]["lr"])
                else:
                    self._epoch_lrs[key].append(0.0)
            else:  # new optimizer
                prev_lrs = [0.0] * self.num_epochs
                # prev_lrs = [0.0] * (self.num_epochs - 1)
                prev_lrs.append(self.optimizers[key].param_groups[0]["lr"])
                self._epoch_lrs[key] = prev_lrs

    def _soft_constraints(self) -> torch.Tensor:
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
        if (
            self.constraints["object"]["tv_weight_z"] > 0
            or self.constraints["object"]["tv_weight_yx"] > 0
        ):
            loss += self.get_tv_loss(
                self.obj_model.obj,
                weights=(
                    self.constraints["object"]["tv_weight_z"],
                    self.constraints["object"]["tv_weight_yx"],
                ),
            )
        if self.constraints["object"]["surface_zero_weight"] > 0:
            loss += self.get_surface_zero_loss(
                self.obj_model.obj,
                weight=self.constraints["object"]["surface_zero_weight"],
            )
        if self.constraints["dataset"]["descan_tv_weight"] > 0:
            loss += self.get_tv_loss(
                self.dset.descan_shifts[:, 0],
                weights=self.constraints["dataset"]["descan_tv_weight"],
            )
            loss += self.get_tv_loss(
                self.dset.descan_shifts[:, 1],
                weights=self.constraints["dataset"]["descan_tv_weight"],
            )

        return loss

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
    ) -> Self:
        """
        reason for having a single reconstruct() is so that updating things like constraints
        or recon_types only happens in one place, reason for having separate reoconstruction_
        methods would be to simplify the flags for this and not have to include all

        """
        # TODO maybe make an "process args" method that handles things like:
        # mode, store_iterations, device,
        self._check_preprocessed()
        self.set_obj_type(obj_type, force=reset)
        if device is not None:
            self.to(device)
        batch_size = self.dset.num_gpts if batch_size is None else batch_size
        self.store_iterations = store_iterations
        self.store_iterations_every = store_iterations_every
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

        if "descan" in self.optimizer_params.keys():
            learn_descan = True  # TODO clean this up... not sure how
        else:
            learn_descan = False

        batcher = SimpleBatcher(self.dset.num_gpts, batch_size, rng=self.rng)

        pbar = tqdm(range(num_iter), disable=not self.verbose)
        for a0 in pbar:
            epoch_loss = 0.0

            for batch_indices in batcher:
                patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                    self.dset.forward(batch_indices, self.obj_padding_px, learn_descan)
                )
                shifted_probes = self.probe_model.forward(positions_px_fractional)
                obj_patches = self.obj_model.forward(patch_indices)
                propagated_probes, overlap = self.forward_operator(
                    obj_patches, shifted_probes, descan_shifts
                )
                pred_intensities = self.detector_model.forward(overlap)

                loss, targets = self.error_estimate(
                    pred_intensities,
                    batch_indices,
                    amplitude_error=True,
                    use_unshifted=learn_descan,
                    loss_type="l2",
                )

                loss += self._soft_constraints() / len(batch_indices)

                self.backward(
                    loss,
                    autograd,
                    obj_patches,
                    propagated_probes,
                    overlap,
                    patch_indices,
                    targets,
                )
                for opt in self.optimizers.values():
                    opt.step()
                    opt.zero_grad()

                epoch_loss += loss.item()

            for sch in self.schedulers.values():
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(epoch_loss)
                elif sch is not None:
                    sch.step()

            self._record_lrs()
            self._epoch_losses.append(epoch_loss)
            self._epoch_recon_types.append(f"{self.obj_model.name}-{self.probe_model.name}")
            if self.store_iterations and ((a0 + 1) % self.store_iterations_every == 0 or a0 == 0):
                self.append_recon_iteration(self.obj, self.probe)

            pbar.set_description(f"Epoch {a0 + 1}/{num_iter}, Loss: {epoch_loss:.4f}, ")

        return self

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
        """ """
        if autograd:
            loss.backward()
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
            # TODO -- connect other optimizable values to this, e.g. descan

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
        fourier_overlap = torch.fft.fft2(overlap_array)
        # from quantem.core.visualization.visualization import show_2d
        # show_2d([fourier_overlap[0,0], torch.abs(fourier_overlap[0,0])])
        if self.num_probes == 1:  # faster
            fourier_modified_overlap = measured_amplitudes * torch.exp(
                1.0j * torch.angle(fourier_overlap)
            )
        else:  # necessary for mixed state
            farfield_amplitudes = self.estimate_amplitudes(overlap_array, corner_centered=True)
            farfield_amplitudes[farfield_amplitudes == 0] = torch.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification * fourier_overlap

        return torch.fft.ifft2(fourier_modified_overlap)

    # endregion --- reconstruction ---
