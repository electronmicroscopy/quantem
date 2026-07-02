import contextlib
import copy
import gc
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Self, Sequence, cast
from warnings import warn

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.dist_utils import (
    init_process_group,
    is_distributed_launch,
    spawn_distributed_workers,
)
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectINR, ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import (
    ProbeModelType,
    ProbeParametric,
    ProbePRISM,
)
from quantem.diffractive_imaging.ptycho_losses import DataCriterion
from quantem.diffractive_imaging.ptycho_utils import compute_train_val_split
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_opt import PtychographyOpt
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations


def _ddp_ptycho_worker(
    rank: int,
    world_size: int,
    ptycho_path: str,
    devices: list[int],
    recon_kwargs: dict[str, Any],
    result_path: str,
) -> None:
    """Module-level worker for mp.start_processes — must live at module scope to be picklable.

    Receives a file path rather than the Ptychography object directly so that no
    large tensors cross the process boundary via pickle (which triggers PyTorch's
    shared-memory tensor mechanism and fails in some Linux environments).
    """
    device_id = devices[rank]
    # Bind the CUDA device BEFORE init_process_group so NCCL allocates its
    # communicator buffers on the correct GPU. Without this, NCCL grabs cuda:0
    # at init time, stranding small per-rank buffers on GPUs the user didn't
    # ask for.
    init_process_group(
        rank,
        world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
        local_device=device_id if torch.cuda.is_available() else None,
    )

    # mmap=True so all workers share one memory-mapped RAM copy of the (potentially large,
    # CPU-resident) state instead of each duplicating it.
    ptycho = torch.load(ptycho_path, map_location="cpu", weights_only=False, mmap=True)
    ptycho.to(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    if dist.is_available() and dist.is_initialized():
        ptycho._broadcast_parameters(src=0)

    ptycho._reconstruct_inner(**recon_kwargs, _dist_rank=rank, _dist_world_size=world_size)

    if rank == 0:
        obj_opt = ptycho.optimizers.get("object")
        probe_opt = ptycho.optimizers.get("probe")
        dset_opt = ptycho.optimizers.get("dataset")
        torch.save(
            {
                "obj_state": {k: v.cpu() for k, v in ptycho.obj_model.state_dict().items()},
                "probe_state": {k: v.cpu() for k, v in ptycho.probe_model.state_dict().items()},
                # Dataset learnable params (scan positions / descan) are optimized and all-reduced
                # in the workers; ship them back so the main process keeps the refinement.
                "dset_scan_positions_px": ptycho.dset._scan_positions_px.detach().cpu(),
                "dset_descan_shifts": ptycho.dset._descan_shifts.detach().cpu(),
                "obj_optimizer_params": ptycho.obj_model._optimizer_params,
                "probe_optimizer_params": ptycho.probe_model._optimizer_params,
                "dset_optimizer_params": ptycho.dset._optimizer_params,
                "obj_optimizer_state": obj_opt.state_dict() if obj_opt is not None else None,
                "probe_optimizer_state": probe_opt.state_dict() if probe_opt is not None else None,
                "dset_optimizer_state": dset_opt.state_dict() if dset_opt is not None else None,
                "iter_losses": ptycho._iter_losses,
                "iter_val_losses": ptycho._iter_val_losses,
                "iter_lrs": ptycho._iter_lrs,
                "iter_recon_types": ptycho._iter_recon_types,
            },
            result_path,
        )

    # Synchronize before teardown so every rank finishes
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()
    dist.destroy_process_group()


class Ptychography(PtychographyOpt, PtychographyVisualizations, PtychographyBase):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    _autograd: bool = True
    _dataset_metadata: "dict[str, Any] | None" = None

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
        if isinstance(probe_model, ProbePRISM) and not getattr(
            cls, "_supports_prism_probe", False
        ):
            raise TypeError(
                "ProbePRISM requires the PRISM engine; use PtychographyPRISM.from_models(...)"
            )
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

    def _record_iter(self, iter_loss: float) -> None:
        self._iter_losses.append(iter_loss)
        optimizers = self.optimizers
        all_keys = set(self._iter_lrs.keys()) | set(optimizers.keys())
        for key in all_keys:
            if key in self._iter_lrs.keys():
                if key in optimizers.keys():
                    self._iter_lrs[key].append(optimizers[key].param_groups[0]["lr"])
                else:
                    self._iter_lrs[key].append(0.0)
            else:  # new optimizer
                # For new optimizers, backfill with 0.0 LR for previous iterations
                current_iter = self.num_iters - 1  # -1 because loss was just appended
                prev_lrs = [0.0] * current_iter
                prev_lrs.append(optimizers[key].param_groups[0]["lr"])
                self._iter_lrs[key] = prev_lrs

    def _reset_iter_constraints(self) -> None:
        """Reset constraint loss accumulation for all models."""
        self.obj_model.reset_iter_constraint_losses()
        self.probe_model.reset_iter_constraint_losses()
        self.dset.reset_iter_constraint_losses()

    def _soft_constraints(self) -> torch.Tensor:
        """Calculate soft constraints by calling apply_soft_constraints on each model."""
        total_loss = torch.tensor(0, device=self._single_device, dtype=self._dtype_real)

        if isinstance(self.obj_model, ObjectINR):
            # Implicit objects evaluate soft constraints at sampled coordinates and don't
            # need the materialized grid (which would force full-grid inference each iter).
            obj_loss = self.obj_model.apply_soft_constraints(mask=self.obj_model.mask)
        else:
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
        num_iters: int = 0,
        reset: bool = False,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        batch_size: int | None = None,
        store_snapshots: bool | None = None,
        store_snapshots_every: int | None = None,
        device: str | int | list[int] | None = None,
        autograd: bool = True,
        loss_type: "str | DataCriterion" = "l2_amplitude",
        num_workers: int = 0,
    ) -> Self:
        """Run iterative ptychography reconstruction.

        ``device`` accepts:
          - ``None``             — keep current device
          - ``"cpu"`` / ``"gpu"`` — existing string form
          - ``int``              — specific GPU index, e.g. ``device=2`` → cuda:2
          - ``list[int]``        — multi-GPU, e.g. ``device=[0,1,2,3]``

        ``constraints`` is a dict keyed by ``"object"``, ``"probe"``, ``"dataset"``
        (any subset). Each leaf may be:

        - a ``Constraints`` dataclass instance (e.g. ``PtychoObjConstraintParams.Raster(...)``),
          which replaces that model's constraint state wholesale, or
        - a plain ``dict`` of field-name -> value, which does a per-key partial update
          on the existing constraint state.

        Multi-GPU (``device`` is a list) launches worker processes via ``mp.spawn`` when called
        from a notebook, or uses the existing distributed process group when launched with
        ``torchrun``. Only autograd mode is supported for multi-GPU in this release.

        ``loss_type`` selects the data-fidelity criterion: a registered name
        (``"l2_amplitude"`` [default], ``"l1_amplitude"``, ``"l2_intensity"``, ``"l1_intensity"``,
        ``"poisson"``, ``"smooth_l1_amplitude"``, ``"s3im_amplitude"``) or a ``DataCriterion``
        instance for custom parameters (e.g. ``AmplitudeS3IM(lambda_s3im=0.5)``). See
        ``ptycho_losses``.

        """
        self._check_preprocessed()

        if constraints:
            self.constraints = constraints

        # Determine effective device list: explicit arg takes priority, else fall back to stored.
        devices_to_use = (
            device if isinstance(device, list) else getattr(self, "_multi_gpu_devices", None)
        )

        # Route to multi-GPU path
        if isinstance(devices_to_use, list) and not is_distributed_launch():
            if not autograd:
                raise ValueError("Multi-GPU reconstruction requires autograd=True.")
            return self._spawn_reconstruct(
                devices=devices_to_use,
                num_iters=num_iters,
                reset=reset,
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
                constraints=constraints,
                batch_size=batch_size,
                store_snapshots=store_snapshots,
                store_snapshots_every=store_snapshots_every,
                autograd=autograd,
                loss_type=loss_type,
                num_workers=num_workers,
            )

        # Handle torchrun distributed launch (RANK env var present)
        if is_distributed_launch():
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if not torch.distributed.is_initialized():
                # Bind the device BEFORE init_process_group so NCCL allocates
                # its communicator buffers on the correct GPU.
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                torch.distributed.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="env://",
                )
            dev = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            self.to(dev)
            self._broadcast_parameters(src=0)
        else:
            rank, world_size = 0, 1
            if device is not None and not isinstance(device, list):
                self.to(device)

        return self._reconstruct_inner(
            num_iters=num_iters,
            reset=reset,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            constraints=constraints,
            batch_size=batch_size,
            store_snapshots=store_snapshots,
            store_snapshots_every=store_snapshots_every,
            autograd=autograd,
            loss_type=loss_type,
            num_workers=num_workers,
            _dist_rank=rank,
            _dist_world_size=world_size,
        )

    def _forward_batch(
        self, batch_indices: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the forward model for one batch of scan positions.

        Returns the predicted intensities and an ``aux`` dict of intermediate
        tensors consumed by the analytic-gradient ``backward`` path. Subclasses
        with a different forward model (e.g. ``PtychographyPRISM``) override this.
        """
        patch_data, _positions_px, positions_px_fractional, descan_shifts = self.dset.forward(
            batch_indices, self.obj_padding_px
        )
        shifted_probes = self.probe_model.forward(positions_px_fractional)
        obj_patches = self.obj_model.forward(patch_data)
        propagated_probes, overlap = self.forward_operator(
            obj_patches, shifted_probes, descan_shifts
        )
        pred_intensities = self.detector_model.forward(overlap)
        aux = {
            "obj_patches": obj_patches,
            "propagated_probes": propagated_probes,
            "overlap": overlap,
            "patch_indices": patch_data,
        }
        return pred_intensities, aux

    def _reconstruct_inner(
        self,
        num_iters: int = 0,
        reset: bool = False,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        batch_size: int | None = None,
        store_snapshots: bool | None = None,
        store_snapshots_every: int | None = None,
        autograd: bool = True,
        loss_type: "str | DataCriterion" = "l2_amplitude",
        num_workers: int = 0,
        _dist_rank: int = 0,
        _dist_world_size: int = 1,
    ) -> Self:
        """Core reconstruction loop. Called by reconstruct() for all launch modes."""
        if batch_size is not None:
            self.batch_size = batch_size
        self.store_snapshot_every = store_snapshots_every
        if store_snapshots_every is not None and store_snapshots is None:
            self.store_snapshots = True
        else:
            self.store_snapshots = store_snapshots

        if reset:
            self.reset_recon()
        if constraints:
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
            self.set_schedulers(self.scheduler_params, num_iter=num_iters)

        self.criterion = loss_type  # resolve name/instance -> DataCriterion
        self.dset._set_targets(self._criterion.target_space)
        self.compute_propagator_arrays()  # required to avoid issue if stopped learning probe tilt

        # Compute the global scan count once — needed to keep loss scale consistent across world.
        global_n = self.dset.num_positions

        train_indices, val_indices = compute_train_val_split(
            self.dset.num_positions,
            self.val_ratio,
            self.val_mode,
            self.rng,
        )
        train_loader, train_sampler, val_loader = self._build_dataloaders(
            train_indices,
            val_indices,
            world_size=_dist_world_size,
            rank=_dist_rank,
            num_workers=num_workers,
        )

        pbar = tqdm(range(num_iters), disable=not self.verbose or _dist_rank != 0)

        for a0 in pbar:
            if _dist_world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(a0)
            consistency_loss = 0.0
            total_loss = 0.0
            self._reset_iter_constraints()

            for batch in train_loader:
                self.zero_grad_all()
                batch_indices = batch["index"].to(self._single_device)
                targets = batch["target"].to(self._single_device, non_blocking=True)
                pred_intensities, aux = self._forward_batch(batch_indices)

                batch_consistency_loss, targets = self.error_estimate(
                    pred_intensities,
                    targets=targets,
                    global_n=global_n,
                )

                batch_soft_constraint_loss = self._soft_constraints()
                batch_loss = batch_consistency_loss + batch_soft_constraint_loss

                self.backward(
                    batch_loss,
                    autograd,
                    aux.get("obj_patches"),
                    aux.get("propagated_probes"),
                    aux.get("overlap"),
                    aux.get("patch_indices"),
                    targets,
                )
                if _dist_world_size > 1:
                    self._all_reduce_gradients()
                self.step_optimizers()
                # Post-step parameter projection (only for positivity_mode="shrink")
                self.obj_model.project_parameters()
                consistency_loss += batch_consistency_loss.item()
                total_loss += batch_loss.item()

            num_batches = len(train_loader)
            total_loss = total_loss / num_batches
            consistency_loss = consistency_loss / num_batches

            # Average loss across ranks so rank-0 reports the global mean
            if _dist_world_size > 1:
                loss_t = torch.tensor(
                    [total_loss, consistency_loss], device=self._single_device, dtype=torch.float64
                )
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                total_loss, consistency_loss = loss_t[0].item(), loss_t[1].item()

            # Validation pass (no gradient, no optimizer steps)
            val_loss = None
            if val_loader is not None:
                val_consistency_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_indices = batch["index"].to(self._single_device)
                        targets = batch["target"].to(self._single_device, non_blocking=True)
                        pred_intensities, _aux = self._forward_batch(batch_indices)
                        batch_val_loss, _ = self.error_estimate(
                            pred_intensities,
                            targets=targets,
                            global_n=global_n,
                        )
                        val_consistency_loss += batch_val_loss.item()
                        val_batches += 1
                if val_batches > 0:
                    val_loss = val_consistency_loss / val_batches
                    # Average the val loss across ranks so rank-0 records the global mean
                    if _dist_world_size > 1:
                        val_t = torch.tensor(
                            val_loss, device=self._single_device, dtype=torch.float64
                        )
                        dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
                        val_loss = val_t.item()
                    if _dist_rank == 0:
                        self._iter_val_losses.append(val_loss)

            if _dist_rank == 0:
                self._record_iter(total_loss)  # TODO record val loss as well

            # Step schedulers with current loss
            self.step_schedulers(total_loss)

            if _dist_rank == 0 and self.store_snapshots and (a0 % self.store_snapshot_every) == 0:
                self._store_current_iter_snapshot()

            if _dist_rank == 0 and self.logger is not None:
                self.logger.log_iter(
                    self.obj_model,
                    self.probe_model,
                    self.dset,
                    self.num_iters - 1,
                    consistency_loss,
                    num_batches,
                    self._get_current_lrs(),
                )

            if _dist_rank == 0:
                if val_loss is not None:
                    pbar.set_description(
                        f"Iter {a0 + 1}/{num_iters}, Loss: {total_loss:.3e}, Val: {val_loss:.3e}"
                    )
                else:
                    pbar.set_description(f"Iter {a0 + 1}/{num_iters}, Loss: {total_loss:.3e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    def _spawn_reconstruct(self, devices: list[int], **recon_kwargs) -> Self:
        """Notebook multi-GPU: spawn one worker process per device via forkserver.

        State is saved to a temp file so that no tensors cross the process boundary
        via pickle.  PyTorch's ForkingPickler automatically moves all CPU tensors to
        shared memory when pickling for multiprocessing, which fails on some Linux
        systems (EINVAL from ftruncate).  Passing only a file path (a plain string)
        avoids that mechanism entirely.
        """
        restore_device = f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"
        # Persist batch_size on the main process so it carries into the saved file and
        # is remembered on future calls that omit batch_size.
        bs = recon_kwargs.get("batch_size")
        if bs is not None:
            self.batch_size = bs
        self.to("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ptycho_path = str(tmpdir_path / "ptycho_state.pt")
            result_path = str(tmpdir_path / "result.pt")

            torch.save(self, ptycho_path, pickle_protocol=4)

            # forkserver: workers fork from a clean pre-started server (no inherited
            # CUDA, no Jupyter FDs).  Only plain Python scalars/strings cross the
            # process boundary, so tensor pickling is never triggered.
            spawn_distributed_workers(
                _ddp_ptycho_worker,
                devices,
                ptycho_path,
                devices,
                recon_kwargs,
                result_path,
            )
            result = torch.load(result_path, map_location="cpu", weights_only=False)

        # --- model weights ---
        self.obj_model.load_state_dict(result["obj_state"])
        self.probe_model.load_state_dict(result["probe_state"])
        self.to(restore_device)

        # --- dataset learnable params (scan positions / descan), refined in the workers ---
        dset_positions = result.get("dset_scan_positions_px")
        if dset_positions is not None:
            self.dset._scan_positions_px.data = dset_positions.to(restore_device)
        dset_descan = result.get("dset_descan_shifts")
        if dset_descan is not None:
            self.dset._descan_shifts.data = dset_descan.to(restore_device)

        # --- restore optimizer params (worker may have set/changed them) so that future
        #     spawns (e.g. reset=True without optimizer_params) can re-init the optimizer ---
        for model, key in (
            (self.obj_model, "obj_optimizer_params"),
            (self.probe_model, "probe_optimizer_params"),
            (self.dset, "dset_optimizer_params"),
        ):
            saved = result.get(key)
            if saved is not None:
                model._optimizer_params = saved

        # Re-create optimizers on the restored device (main process never ran set_optimizers).
        # set_optimizers() skips models whose _optimizer_params is NoneOptimizer.
        self.set_optimizers()

        # --- optimizer states (params and device must be set before loading) ---
        for name, key in (
            ("object", "obj_optimizer_state"),
            ("probe", "probe_optimizer_state"),
            ("dataset", "dset_optimizer_state"),
        ):
            opt_state = result.get(key)
            opt = self.optimizers.get(name)
            if opt_state is not None and opt is not None:
                opt.load_state_dict(opt_state)
                # State tensors were saved on CPU; move them to restore_device
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(restore_device)

        # --- iteration tracking ---
        # When reset=True the worker ran reset_recon() internally, so its lists start from 0.
        # When reset=False the worker inherited existing history, so its lists are [old...new...].
        # n_before lets us take only the genuinely new tail in both cases.
        n_before = len(self._iter_losses)
        is_reset = recon_kwargs.get("reset", False)

        if is_reset:
            self._iter_losses.clear()
            self._iter_val_losses.clear()
            self._iter_lrs.clear()
            self._iter_recon_types.clear()
            self._iter_losses.extend(result["iter_losses"])
            self._iter_val_losses.extend(result["iter_val_losses"])
            for k, v in result.get("iter_lrs", {}).items():
                self._iter_lrs[k] = list(v)
            self._iter_recon_types.extend(result.get("iter_recon_types", []))
        else:
            self._iter_losses.extend(result["iter_losses"][n_before:])
            self._iter_val_losses.extend(result["iter_val_losses"][n_before:])
            for k, v in result.get("iter_lrs", {}).items():
                if k not in self._iter_lrs:
                    self._iter_lrs[k] = []
                self._iter_lrs[k].extend(list(v)[n_before:])
            self._iter_recon_types.extend(result.get("iter_recon_types", [])[n_before:])

        self._multi_gpu_devices = devices
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
                if self.obj_model._obj.grad is not None:
                    self.obj_model._obj.grad.mul_(obj_grad_scale)

            if isinstance(self.probe_model, ProbeParametric):
                probe_grad_scale = np.sqrt(self.probe_model._mean_diffraction_intensity)
                for par in self.probe_model.params:
                    if par.grad is not None:
                        par.grad.mul_(probe_grad_scale)

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
        # The data-fidelity criterion is transient config (re-set each reconstruct); don't
        # serialize it (avoids a dill fallback and keeps the archive model-agnostic).
        skip.append("_criterion")

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

        _dev = self.device
        current_device: str = f"cuda:{_dev[0]}" if isinstance(_dev, list) else _dev
        self.to("cpu")

        if self.verbose and verbose:
            print(f"Saving ptychography object to {Path(path).resolve()}")

        super().save(
            path,
            mode=mode,
            store=store,
            skip=skips,
            compression_level=compression_level,
        )

        self.to(current_device)  # TODO figure out why this isn't working for DDIP sometimes?

        # Clean up temporary metadata
        if not save_raw_data and self._dataset_metadata is not None:
            self._dataset_metadata = None

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
            if ptycho._dataset_metadata is not None:
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
            if ptycho._dataset_metadata is not None:
                metadata = ptycho._dataset_metadata
                # preserve learned scan positions and descan shifts
                if "learned_scan_positions_px" in metadata:
                    dset.scan_positions_px.data = metadata["learned_scan_positions_px"]  # type: ignore[assignment]
                if "learned_descan_shifts" in metadata:
                    dset.descan_shifts.data = metadata["learned_descan_shifts"]  # type: ignore[assignment]

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

    def clone(self, device: str | int = "cpu") -> Self:  # TODO make this faster
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
