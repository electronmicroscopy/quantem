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

    When PRISM pays off
    -------------------
    The parent multislice propagates all P parent beams through the *full* (padded)
    object, which is much larger than a conventional per-position ROI propagation,
    and it re-runs every batch (the object updates per batch). So the two costs are
    the parent multislice (``num_batches x P x num_slices`` full-object FFTs) and the
    reduction (``P x num_positions`` ROI FFTs). PRISM beats conventional multislice
    when these amortize:
    - **Large batches** — the full-object parent multislice is shared across all
      positions in a batch, so it only amortizes when ``batch_size`` is large
      (roughly above ``P x object_area / roi_area``). Small batches on a large field
      of view are the pathological case (a warning is emitted).
    - **Frozen scan positions** — the reduction kernels are then deduplicated to the
      unique sub-pixel offsets (one per commensurate scan), collapsing the reduction
      FFTs from ``P x num_positions`` to ``P x num_unique_offsets``.
    - **Many slices / mixed states** — the parent multislice is independent of the
      number of probe modes, so it amortizes over modes as well as positions.
    Conversely, a single mode with few slices, small batches, and learnable positions
    is conventional multislice's sweet spot; use it there. P scales as ``1 /
    interpolation_factor^2`` and grows with the ROI, so raise ``interpolation_factor``
    for large ROIs.

    Memory / performance knobs (all settable via ``reconstruct``; multislice path only):
    - ``batch_size``: scan positions per batch (inherited). Large is best for PRISM.
    - ``parent_batch_size``: parent beams propagated/reduced per chunk.
    - ``position_batch_size``: scan positions reduced at a time within a batch; bounds
      the peak (parents x positions x roi) tensor so a large ``batch_size`` fits.
    - ``position_quantization``: with frozen positions, round sub-pixel offsets to
      ``1/Q`` px to bound the number of unique reduction kernels for rotated scans.
    - ``use_checkpointing``: recompute each parent chunk during backward instead
      of retaining its activations (~halves peak memory at ~2x forward compute).

    For multislice objects the propagated parent waves are always back-propagated
    over the total thickness (refocused to the entrance plane). This is a pure
    k-space phase in the far field — intensity-neutral in the dense limit — but it
    removes the thickness-dependent quadratic phase across parents, which is what
    keeps the parent-beam interpolation accurate for thick specimens.
    """

    _supports_prism_probe = True

    # class-level defaults so objects loaded from file (which skip __init__) behave
    parent_batch_size: int | None = None
    position_batch_size: int | None = None
    position_quantization: int | None = None
    use_checkpointing: bool = False

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
        position_batch_size: int | None = None,
        position_quantization: int | None = None,
        use_checkpointing: bool | None = None,
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
        position_batch_size : int | None
            Number of scan positions reduced at a time within a batch (None = all at
            once). Bounds the peak (parents x positions x roi) reduction tensor so a
            large ``batch_size`` can be used to amortize the per-batch parent
            multislice without running out of memory.
        position_quantization : int | None
            When scan positions are frozen, round each fractional (sub-pixel) offset to
            ``1 / position_quantization`` of a pixel before deduplicating the reduction
            kernels, bounding the number of unique offsets to at most its square (max
            error ``1 / (2 * position_quantization)`` px). None keeps exact offsets
            (grid-commensurate scans already collapse to a single kernel).
        use_checkpointing : bool
            Gradient-checkpoint each parent chunk (recomputed during backward).
        """
        if not autograd:
            raise ValueError("PtychographyPRISM only supports autograd=True.")
        if isinstance(device, list) or getattr(self, "_multi_gpu_devices", None) is not None:
            raise NotImplementedError("Multi-GPU is not supported for PtychographyPRISM.")

        if parent_batch_size is not None:
            self.parent_batch_size = int(parent_batch_size)
        if position_batch_size is not None:
            self.position_batch_size = int(position_batch_size)
        if position_quantization is not None:
            self.position_quantization = int(position_quantization)
        if use_checkpointing is not None:
            self.use_checkpointing = bool(use_checkpointing)

        self._warn_if_slow_config(batch_size)

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

    def _warn_if_slow_config(self, batch_size: int | None) -> None:
        """Warn when the PRISM forward will not amortize (large field of view + small
        batch): the full-object parent multislice re-runs every batch, so it only pays
        off above ``P x object_area / roi_area`` positions per batch."""
        if self.num_slices == 1:
            return
        bs = batch_size if batch_size is not None else getattr(self, "batch_size", None)
        if not bs:
            return
        num_parent = self.probe_model.num_parent_beams
        obj_ratio = float(np.prod(self.obj_shape_full[-2:]) / np.prod(self.roi_shape))
        crossover = num_parent * obj_ratio
        if bs < crossover:
            self.vprint(
                f"Warning: PRISM re-propagates {num_parent} parent beams through the full "
                f"object every batch; with batch_size={bs} (< ~{crossover:.0f}) this multislice "
                "is not amortized and will likely be slower than conventional multislice. "
                "Prefer a larger batch_size, a larger interpolation_factor (fewer parents), or "
                "frozen scan positions (enables the deduplicated reduction)."
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

        self._compute_object_propagators()
        # Build one reduction kernel per *unique* sub-pixel offset when the scan is
        # frozen (positions reused across the scan), else one per position.
        unique_fracs, offset_ids = self._reduction_offsets(fract_positions)
        beamlets_fft, position_coefs = self.probe_model.forward(unique_fracs)

        if self.num_slices == 1:
            exit_waves = self._reduce_single_slice(
                transmission, beamlets_fft, position_coefs, offset_ids, patch_indices
            )
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
                    offset_ids,
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

    def _reduction_offsets(
        self, fract_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sub-pixel offsets to build reduction kernels for, plus the per-position index
        into them.

        Frozen scans reuse a small set of unique offsets (``offset_ids`` maps each
        position to its kernel), collapsing the O(P x positions) reduction FFTs to
        O(P x unique); grid-commensurate scans reduce to a single kernel. Learnable
        scans keep per-position offsets (``offset_ids is None``) so the analytic
        position phases stay differentiable.
        """
        if self.dset.learn_scan_positions:
            return fract_positions, None
        fracs = fract_positions.detach()
        if self.position_quantization:
            q = float(self.position_quantization)
            fracs = torch.round(fracs * q) / q
        else:  # round to a tight tolerance so exactly-equal offsets group
            fracs = torch.round(fracs * 1.0e6) / 1.0e6
        unique, inverse = torch.unique(fracs, dim=0, return_inverse=True)
        return unique.to(fract_positions.dtype), inverse

    def _position_chunks(self, batch: int):
        yield from generate_batches(batch, max_batch=self.position_batch_size or batch)

    def _reduce_single_slice(
        self,
        transmission: torch.Tensor,
        beamlets_fft: torch.Tensor,
        position_coefs: torch.Tensor,
        offset_ids: torch.Tensor | None,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Single-slice fast path: the parent sum folds into one effective aperture, so
        the reduction is one FFT per (unique) offset instead of per (parent, position)."""
        eff = beamlets_fft.sum(dim=1)  # (num_probes, *roi_shape)
        transmission_flat = transmission.reshape(1, -1)
        window = self.probe_model.coefficient_window
        batch = patch_indices.shape[0]

        probe_unique = None
        if offset_ids is not None:  # frozen: one probe per unique offset, reused
            probe_unique = torch.fft.ifft2(eff[:, None] * position_coefs[None])
            if window is not None:
                probe_unique = probe_unique * window

        chunks = []
        for p0, p1 in self._position_chunks(batch):
            if probe_unique is not None:
                probes = probe_unique[:, offset_ids[p0:p1]]
            else:
                probes = torch.fft.ifft2(eff[:, None] * position_coefs[p0:p1][None])
                if window is not None:
                    probes = probes * window
            obj_patches = torch.complex(
                transmission_flat.real[:, patch_indices[p0:p1]],
                transmission_flat.imag[:, patch_indices[p0:p1]],
            )  # (1, pos_chunk, *roi_shape)
            chunks.append(obj_patches * probes)
        return torch.cat(chunks, dim=1)

    def _reduce_parent_chunk(
        self,
        transmission: torch.Tensor,
        parent_wave_vectors: torch.Tensor,
        beamlets_fft: torch.Tensor,
        position_coefs: torch.Tensor,
        offset_ids: torch.Tensor | None,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate a chunk of parent waves and reduce them to ROI exit waves.

        The checkpointable unit: propagates the chunk once, then reduces the scan
        positions in sub-chunks (``position_batch_size``), gathering ROI patches of the
        propagated parents before the reduction so no full-object per-position tensor is
        materialized. When the scan is frozen the per-parent coefficient maps are built
        once per unique offset and reused across positions.
        """
        propagated = self._propagate_parent_waves(parent_wave_vectors, transmission)
        propagated_flat = propagated.reshape(propagated.shape[0], -1)
        window = self.probe_model.coefficient_window
        batch = patch_indices.shape[0]

        coef_unique = None
        if offset_ids is not None:  # frozen: coef maps per unique offset, reused
            coef_unique = torch.fft.ifft2(beamlets_fft[:, :, None] * position_coefs[None, None])
            if window is not None:
                coef_unique = coef_unique * window

        chunks = []
        for p0, p1 in self._position_chunks(batch):
            # MPS does not support complex gather kernels
            patches = torch.complex(
                propagated_flat.real[:, patch_indices[p0:p1]],
                propagated_flat.imag[:, patch_indices[p0:p1]],
            )  # (parent_chunk, pos_chunk, *roi_shape)
            if coef_unique is not None:
                coef_maps = coef_unique[:, :, offset_ids[p0:p1]]
            else:
                coef_maps = torch.fft.ifft2(
                    beamlets_fft[:, :, None] * position_coefs[p0:p1][None, None]
                )
                if window is not None:
                    coef_maps = coef_maps * window
            # (num_probes, parent_chunk, pos_chunk, *roi_shape) x patches -> sum parents
            chunks.append((coef_maps * patches[None]).sum(dim=1))
        return torch.cat(chunks, dim=1)

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
        # always refocus the parent waves to the entrance plane (removes the
        # thickness-dependent quadratic phase that would spoil the interpolation)
        total_thickness = float(np.sum(self.slice_thicknesses))
        self._total_back_propagator = self.probe_model._compute_propagator_arrays(
            self.sampling, 2, [-total_thickness], gpts=gpts
        )[0]

    # endregion --- forward model ---
