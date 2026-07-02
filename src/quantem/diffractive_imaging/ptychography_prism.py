from typing import Any, Self

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType
from quantem.diffractive_imaging.probe_models import ProbePRISM
from quantem.diffractive_imaging.ptycho_losses import DataCriterion
from quantem.diffractive_imaging.ptycho_utils import fourier_translation_operator
from quantem.diffractive_imaging.ptychography import Ptychography


class PtychographyPRISM(Ptychography):
    """
    Partitioned-PRISM ptychography engine.

    Instead of shifting a converged probe to each scan position, this engine
    propagates a compact set of tilted parent plane waves through the *full*
    object once per batch (multislice), gathers ROI patches of the propagated
    parent waves at each scan position, and reduces them with the ROI-sized
    coefficient maps produced by ``ProbePRISM``:

        ``exit[p, n] = sum_b S_patch[b, n] * ifft2(CTF_p * w_b * c_pb * pos_phase_n)``

    By linearity this reproduces the conventional exit waves (exactly so for a
    ``dense=True`` probe model), while partial coherence and position correction
    enter naturally through the learnable beam coefficients and the analytic
    position phases.

    Single-slice objects take a collapsed fast path (the parent sum folds into one
    effective CTF before the ifft2), matching the conventional engine's speed; the
    per-parent machinery below only runs for multislice objects.

    Memory knobs (all settable via ``reconstruct``; multislice path only):
    - ``batch_size``: scan positions per batch (inherited).
    - ``parent_batch_size``: parent beams reduced per chunk.
    - ``use_checkpointing``: recompute each parent chunk during backward instead
      of retaining its activations (~halves peak memory at ~2x forward compute).

    For multislice objects the parent waves are back-propagated over the total
    thickness and the beamlet CTFs are defocus-compensated (``C10 += thickness``),
    which keeps the parent-beam expansion accurate for thick specimens; disable
    with ``thickness_compensation=False``. PRISM amortizes the per-batch
    multislice of the parent beams best with large (ideally full) batch sizes.
    """

    _supports_prism_probe = True

    # class-level defaults so objects loaded from file (which skip __init__) behave
    parent_batch_size: int | None = None
    use_checkpointing: bool = False
    thickness_compensation: bool = True

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbePRISM,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        if not isinstance(probe_model, ProbePRISM):
            raise TypeError(
                f"PtychographyPRISM requires a ProbePRISM probe model, got "
                f"{type(probe_model).__name__}. Use Ptychography for other probe models."
            )
        return super().from_models(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            logger=logger,
            device=device,
            verbose=verbose,
            rng=rng,
        )

    def preprocess(self, *args, **kwargs) -> Self:
        super().preprocess(*args, **kwargs)
        if self.num_slices > 1 and np.any(self.obj_shape_full[-2:] % self.roi_shape != 0):
            self.vprint(
                f"Warning: padded object shape {tuple(self.obj_shape_full[-2:])} is not an "
                f"integer multiple of roi_shape {tuple(self.roi_shape)}; the PRISM parent "
                "plane waves are not periodic on the object grid and multislice propagation "
                "acquires wrap-around error. Consider adjusting obj_padding_px."
            )
        return self

    def reconstruct(
        self,
        num_iters: int = 0,
        reset: bool = False,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        batch_size: int | None = None,
        parent_batch_size: int | None = None,
        use_checkpointing: bool | None = None,
        thickness_compensation: bool | None = None,
        store_snapshots: bool | None = None,
        store_snapshots_every: int | None = None,
        device: str | int | list[int] | None = None,
        autograd: bool = True,
        loss_type: "str | DataCriterion" = "l2_amplitude",
        num_workers: int = 0,
    ) -> Self:
        """Run iterative PRISM ptychography reconstruction.

        See ``Ptychography.reconstruct`` for the shared arguments. PRISM-specific:

        parent_batch_size : int | None
            Number of parent beams propagated/reduced per chunk (None = all at once).
        use_checkpointing : bool
            Gradient-checkpoint each parent chunk (recomputed during backward).
        thickness_compensation : bool
            Back-propagate parent waves over the total thickness and compensate the
            beamlet CTFs with a matching defocus offset (multislice only).
        """
        if not autograd:
            raise ValueError("PtychographyPRISM only supports autograd=True.")
        if isinstance(device, list) or getattr(self, "_multi_gpu_devices", None) is not None:
            raise NotImplementedError("Multi-GPU is not supported for PtychographyPRISM.")

        if parent_batch_size is not None:
            self.parent_batch_size = int(parent_batch_size)
        if use_checkpointing is not None:
            self.use_checkpointing = bool(use_checkpointing)
        if thickness_compensation is not None:
            self.thickness_compensation = bool(thickness_compensation)

        return super().reconstruct(
            num_iters=num_iters,
            reset=reset,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            constraints=constraints,
            batch_size=batch_size,
            store_snapshots=store_snapshots,
            store_snapshots_every=store_snapshots_every,
            device=device,
            autograd=autograd,
            loss_type=loss_type,
            num_workers=num_workers,
        )

    @property
    def probe_model(self) -> ProbePRISM:
        return self._probe_model  # type: ignore[return-value]

    @probe_model.setter
    def probe_model(self, model: ProbePRISM):
        if not isinstance(model, ProbePRISM) and "ProbePRISM" not in str(type(model)):
            raise TypeError(
                f"PtychographyPRISM requires a ProbePRISM probe model, got {type(model)}"
            )
        Ptychography.probe_model.fset(self, model)  # type: ignore[attr-defined]

    # region --- forward model ---

    def _forward_batch(
        self, batch_indices: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """PRISM forward for one batch: no analytic-gradient aux tensors."""
        patch_indices, _positions_px, positions_px_fractional, descan_shifts = self.dset.forward(
            batch_indices, self.obj_padding_px
        )
        exit_waves = self.forward_operator(patch_indices, positions_px_fractional, descan_shifts)
        pred_intensities = self.detector_model.forward(exit_waves)
        return pred_intensities, {}

    def forward_operator(  # pyright: ignore[reportIncompatibleMethodOverride]  # PRISM consumes indices/positions, not patches/probes
        self,
        patch_indices: torch.Tensor,
        fract_positions: torch.Tensor,
        descan: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PRISM forward operator: parent-wave multislice + coefficient reduction.

        Parameters
        ----------
        patch_indices : torch.Tensor
            Flattened-index ROI lookups into the padded object, (batch, *roi_shape).
        fract_positions : torch.Tensor
            Fractional (subpixel) scan positions in pixels, (batch, 2).
        descan : torch.Tensor | None
            Per-position descan shifts in pixels, (batch, 2).

        Returns
        -------
        torch.Tensor
            Exit waves (num_probes, batch, *roi_shape).
        """
        obj = self.obj_model.obj
        if not obj.is_complex():  # potential or pure_phase -> transmission function
            transmission = torch.exp(1.0j * obj)
        else:
            transmission = obj

        accumulated_thickness = 0.0
        if self.num_slices > 1 and self.thickness_compensation:
            accumulated_thickness = float(np.sum(self.slice_thicknesses))
        self._compute_object_propagators()

        beamlets_fft, position_coefs = self.probe_model.forward(
            fract_positions, accumulated_thickness=accumulated_thickness
        )

        if self.num_slices == 1:
            # Fast path: without propagation every parent wave reduces to the
            # transmission function, so the parent sum folds into a single
            # effective CTF before the ifft2 — one FFT per position instead of
            # one per (parent, position). Mathematically identical (gradients
            # to aberrations and beam coefficients flow through the sum).
            transmission_flat = transmission.reshape(1, -1)
            obj_patches = torch.complex(
                transmission_flat.real[:, patch_indices],
                transmission_flat.imag[:, patch_indices],
            )  # (1, batch, *roi_shape)
            probes = torch.fft.ifft2(beamlets_fft.sum(dim=1)[:, None] * position_coefs[None])
            exit_waves = obj_patches * probes
        else:
            parent_wave_vectors = self.probe_model.parent_wave_vectors
            num_parent = parent_wave_vectors.shape[0]
            max_batch = self.parent_batch_size or num_parent

            exit_waves: torch.Tensor | None = None
            for start, end in generate_batches(num_parent, max_batch=max_batch):
                args = (
                    transmission,
                    parent_wave_vectors[start:end],
                    beamlets_fft[:, start:end],
                    position_coefs,
                    patch_indices,
                )
                if self.use_checkpointing and torch.is_grad_enabled():
                    chunk = checkpoint(self._reduce_parent_chunk, *args, use_reentrant=False)
                else:
                    chunk = self._reduce_parent_chunk(*args)
                exit_waves = chunk if exit_waves is None else exit_waves + chunk
            assert exit_waves is not None

        if descan is not None:
            shifts = fourier_translation_operator(descan, tuple(self.roi_shape))
            exit_waves = exit_waves * shifts[None]
        return exit_waves

    def _reduce_parent_chunk(
        self,
        transmission: torch.Tensor,
        parent_wave_vectors: torch.Tensor,
        beamlets_fft: torch.Tensor,
        position_coefs: torch.Tensor,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate a chunk of parent waves and reduce them to ROI exit waves.

        The checkpointable unit: gathers ROI patches of the propagated parent
        waves *before* the reduction, so no full-object per-position tensor is
        ever materialized.
        """
        propagated = self._propagate_parent_waves(parent_wave_vectors, transmission)
        propagated_flat = propagated.reshape(propagated.shape[0], -1)
        # MPS does not support complex gather kernels
        patches = torch.complex(
            propagated_flat.real[:, patch_indices],
            propagated_flat.imag[:, patch_indices],
        )  # (chunk, batch, *roi_shape)

        coef_maps = torch.fft.ifft2(
            beamlets_fft[:, :, None] * position_coefs[None, None]
        )  # (num_probes, chunk, batch, *roi_shape)
        return (coef_maps * patches[None]).sum(dim=1)

    def _propagate_parent_waves(
        self, parent_wave_vectors: torch.Tensor, transmission: torch.Tensor
    ) -> torch.Tensor:
        """Multislice-propagate tilted plane waves through the full object.

        Returns the tilt-corrected, refocused parent waves
        ``conj(incident) * P_{-t}[M[incident]]`` of shape (num_waves, H, W).
        """
        gpts = transmission.shape[-2:]
        sampling = self.sampling
        device = transmission.device

        x = torch.arange(gpts[0], dtype=torch.float32, device=device) * float(sampling[0])
        y = torch.arange(gpts[1], dtype=torch.float32, device=device) * float(sampling[1])
        phase = parent_wave_vectors[:, 0, None, None] * x[None, :, None] + (
            parent_wave_vectors[:, 1, None, None] * y[None, None, :]
        )
        incident = torch.exp(2.0j * torch.pi * phase)

        waves = incident
        for s in range(self.num_slices):
            waves = waves * transmission[s]
            if s < self.num_slices - 1:
                waves = self._propagate_array(waves, self._obj_propagators[s])

        if self._total_back_propagator is not None:
            waves = self._propagate_array(waves, self._total_back_propagator)

        return waves * incident.conj()

    def _compute_object_propagators(self) -> None:
        """Fresnel propagators on the (padded) object grid, plus the total-thickness
        back-propagator used to refocus the parent waves to the entrance plane.

        Cheap relative to the multislice itself, so recomputed each forward call;
        this also keeps ``learn_probe_tilt`` gradients fresh.
        """
        if self.num_slices == 1:
            self._obj_propagators = None
            self._total_back_propagator = None
            return

        gpts = tuple(int(n) for n in self.obj_shape_full[-2:])
        self._obj_propagators = self.probe_model._compute_propagator_arrays(
            self.sampling, self.num_slices, self.slice_thicknesses, gpts=gpts
        )
        if self.thickness_compensation:
            total_thickness = float(np.sum(self.slice_thicknesses))
            self._total_back_propagator = self.probe_model._compute_propagator_arrays(
                self.sampling, 2, [-total_thickness], gpts=gpts
            )[0]
        else:
            self._total_back_propagator = None

    # endregion --- forward model ---
