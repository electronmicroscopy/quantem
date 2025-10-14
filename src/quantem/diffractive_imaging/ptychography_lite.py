import os
from pathlib import Path
from typing import Any, Literal, Self, Sequence

import numpy as np
import torch

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.ml.cnn2d import CNN2d
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectDIP, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeDIP, ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography


class PtychoLite(Ptychography):
    """
    High-level convenience wrapper around Ptychography.

    Provides a from_dataset() constructor that builds pixelated object and probe
    models from simple flags, then initializes a full Ptychography instance.
    """

    @classmethod
    def from_dataset(
        cls,
        dset: Dataset4dstem | PtychographyDatasetRaster,
        *,
        # object settings
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        # probe settings
        num_probes: int = 1,
        energy: float | None = None,
        defocus: float | None = None,
        semiangle_cutoff: float | None = None,
        polar_parameters: dict | None = None,
        vacuum_probe_intensity: np.ndarray | Dataset4dstem | None = None,
        initial_probe_weights: list[float] | np.ndarray | None = None,
        # preprocessing
        obj_padding_px: tuple[int, int] = (0, 0),
        # logging/device
        log_dir: os.PathLike | str | None = None,
        log_prefix: str = "",
        log_images_every: int = 10,
        log_probe_images: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        """
        Build a PFTM instance from a 4D-STEM dataset with simple flags.

        Parameters
        ----------
        dset : Dataset4dstem | PtychographyDatasetRaster
            Raw 4D-STEM dataset or an already-wrapped raster dataset model.
        num_slices : int
            Number of object slices.
        slice_thicknesses : float | Sequence | None
            Slice thickness(es) in Å. If None and num_slices>1, must be set later.
        obj_type : {"complex","pure_phase","potential"}
            Object parameterization.
        num_probes : int
            Number of probe components (mixed state when >1).
        energy, defocus, semiangle_cutoff, rolloff, polar_parameters
            Probe settings passed to ProbePixelated.
        vacuum_probe_intensity : np.ndarray | Dataset4dstem | None
            Optional corner-centered vacuum probe intensity for scaling/centering.
        initial_probe_weights : list[float] | np.ndarray | None
            Optional initial component weights (length=num_probes).
        log_dir, log_prefix, log_suffix, log_images_every, log_probe_images, device, verbose, rng
            Standard Ptychography configuration.
        """

        # Ensure dataset model
        if isinstance(dset, PtychographyDatasetRaster):
            dset_model = dset
        elif isinstance(dset, Dataset4dstem):
            dset_model = PtychographyDatasetRaster.from_dataset4dstem(dset)
        else:
            raise TypeError(
                f"dset must be Dataset4dstem or PtychographyDatasetRaster, got {type(dset)}"
            )

        if not dset_model.preprocessed:
            dset_model.preprocess(com_fit_function="constant")

        # Object model
        obj_model = ObjectPixelated.from_uniform(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            obj_type=obj_type,
            device="cpu",
            rng=rng,
        )

        # Probe model
        probe_params: dict[str, Any] = {
            "energy": energy,
            "defocus": defocus,
            "semiangle_cutoff": semiangle_cutoff,
        }

        if polar_parameters is not None:
            probe_params["polar_parameters"] = polar_parameters

        probe_model = ProbePixelated.from_params(
            probe_params=probe_params,
            num_probes=num_probes,
            device="cpu",
            rng=rng,
            initial_probe_weights=initial_probe_weights,
            vacuum_probe_intensity=vacuum_probe_intensity,
        )

        detector_model = DetectorPixelated()

        if log_dir is not None:
            logger = LoggerPtychography(
                log_dir=log_dir,
                run_prefix=log_prefix,
                run_suffix="pix",
                log_images_every=log_images_every,
                log_probe_images=log_probe_images,
            )
            if verbose:
                print(f"Logging to {logger.log_dir}")
        else:
            logger = None

        ptycho = cls.from_models(
            dset=dset_model,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            logger=logger,
            device=device,
            verbose=verbose,
            rng=rng,
        )
        ptycho.preprocess(obj_padding_px=obj_padding_px)
        return ptycho

    def reconstruct(  # type:ignore could do overloads but this is simpler...
        self,
        num_iter: int = 0,
        reset: bool = False,
        lr_obj: float = 5e-3,
        learn_probe: bool = True,
        lr_probe: float = 1e-3,
        batch_size: int | None = None,
        scheduler_type: Literal["exp", "cyclic", "plateau", "none"] = "none",
        scheduler_factor: float = 0.5,
        new_optimizers: bool = False,  # not sure what the default should be
        constraints: dict = {},  # TODO add constraints flags
        store_iterations_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        verbose: int | bool = True,
    ) -> Self:
        self.verbose = verbose

        if new_optimizers or reset or self.num_epochs == 0:
            opt_params = {
                "object": {
                    "type": "adam",
                    "lr": lr_obj,
                },
            }
            scheduler_params = {
                "object": {
                    "type": scheduler_type,
                    "factor": scheduler_factor,
                }
            }
            if learn_probe:
                opt_params["probe"] = {
                    "type": "adam",
                    "lr": lr_probe,
                }
                scheduler_params["probe"] = {
                    "type": scheduler_type,
                    "factor": scheduler_factor,
                }
        else:
            opt_params = None
            scheduler_params = None

        constraints = constraints  # placeholder for constraints flags

        return super().reconstruct(
            num_iter=num_iter,
            reset=reset,
            optimizer_params=opt_params,
            scheduler_params=scheduler_params,
            constraints=constraints,
            batch_size=batch_size,
            store_iterations_every=store_iterations_every,
            device=device,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dset: Any | None = None,
        device: str | int | None = None,
        verbose: int | bool | None = None,
        auto_reload_dataset: bool = True,
    ) -> Self:
        """
        Load a saved reconstruction and ensure the returned instance is PtychoLite.

        This is a thin wrapper over the base loader that preserves the full
        saved state while upgrading the loaded instance to this subclass when
        necessary, without mutating the underlying data.
        """
        base = Ptychography.from_file(
            path=path,
            dset=dset,
            device=device,
            verbose=verbose,
            auto_reload_dataset=auto_reload_dataset,
        )

        if not isinstance(base, cls):
            # Safely upgrade the loaded instance to this subclass without altering state
            try:
                base.__class__ = cls  # type: ignore[assignment]
            except Exception:
                # Fallback: rely on the serialized class if available
                upgraded = cls._recursive_load_from_path(path)
                return upgraded  # type: ignore[return-value]

        return base  # type: ignore[return-value]


class PtychoLiteDIP(Ptychography):
    """
    High-level convenience wrapper around Ptychography.

    Provides a from_dataset() constructor that builds pixelated object and probe
    models from simple flags, then initializes a full Ptychography instance.
    """

    @classmethod
    def from_ptycholite(
        cls,
        ptycholite: PtychoLite,
        pretrain_iters: int | None = None,
        pretrain_lr: float = 1e-3,
        # model settings
        cnn_num_layers: int = 3,
        # logging/device
        log_dir: os.PathLike | str | None = None,
        log_prefix: str = "",
        log_images_every: int = 10,
        log_probe_images: bool = False,
        device: Literal["cpu", "gpu", "cuda"] = "cpu",
        verbose: int | bool = True,
    ) -> Self:
        if device == "gpu":
            device = "cuda"
        # Object model
        obj_dip = CNN2d(
            in_channels=ptycholite.obj_model.num_slices,
            out_channels=ptycholite.obj_model.num_slices,
            num_layers=cnn_num_layers,
            dtype=torch.float32 if ptycholite.obj_model.obj_type == "complex" else torch.float32,
        )

        obj_model = ObjectDIP.from_pixelated(
            model=obj_dip,
            pixelated=ptycholite.obj_model,
            device=device,
        )

        # Probe model
        probe_dip = CNN2d(
            in_channels=ptycholite.probe_model.num_probes,
            out_channels=ptycholite.probe_model.num_probes,
            num_layers=cnn_num_layers,
            dtype=torch.complex64,
        )

        probe_model = ProbeDIP.from_pixelated(
            model=probe_dip,
            pixelated=ptycholite.probe_model,
            device=device,
        )

        if pretrain_iters is not None:
            obj_model.pretrain(
                reset=True,
                num_epochs=pretrain_iters,
                optimizer_params={
                    "type": "adam",
                    "lr": pretrain_lr,
                },
                scheduler_params={
                    "type": "plateau",
                    "factor": 0.5,
                },
                apply_constraints=False,
                device=config.get("device"),
            )

            probe_model.pretrain(
                reset=True,
                num_epochs=pretrain_iters,
                optimizer_params={
                    "type": "adam",
                    "lr": 1e-3,
                },
                scheduler_params={
                    "type": "plateau",
                    "factor": 0.5,
                },
                apply_constraints=False,
            )

        if log_dir is not None:
            logger = LoggerPtychography(
                log_dir=log_dir,
                run_prefix=log_prefix,
                run_suffix="pix",
                log_images_every=log_images_every,
                log_probe_images=log_probe_images,
            )
            if verbose:
                print(f"Logging to {logger.log_dir}")
        else:
            logger = None

        # Build a fresh instance of this subclass using the original components
        ptycho = cls.from_models(
            dset=ptycholite.dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=ptycholite.detector_model,
            logger=logger if logger is not None else ptycholite.logger,
            device=device,
            verbose=ptycholite.verbose,
            rng=ptycholite.rng,
        )

        ptycho.preprocess(
            obj_padding_px=(int(ptycholite.obj_padding_px[0]), int(ptycholite.obj_padding_px[1]))
        )
        return ptycho

    def reconstruct(  # type:ignore could do overloads but this is simpler...
        self,
        num_iter: int = 0,
        reset: bool = False,
        lr_obj: float = 5e-4,
        learn_probe: bool = True,
        lr_probe: float = 5e-4,
        batch_size: int | None = None,
        scheduler_type: Literal["exp", "cyclic", "plateau", "none"] = "none",
        scheduler_factor: float = 0.5,
        new_optimizers: bool = False,  # not sure what the default should be
        constraints: dict = {},  # TODO add constraints flags
        store_iterations_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        verbose: int | bool = True,
    ) -> Self:
        self.verbose = verbose

        if new_optimizers or reset or self.num_epochs == 0:
            opt_params = {
                "object": {
                    "type": "adam",
                    "lr": lr_obj,
                },
            }
            scheduler_params = {
                "object": {
                    "type": scheduler_type,
                    "factor": scheduler_factor,
                }
            }
            if learn_probe:
                opt_params["probe"] = {
                    "type": "adam",
                    "lr": lr_probe,
                }
                scheduler_params["probe"] = {
                    "type": scheduler_type,
                    "factor": scheduler_factor,
                }
        else:
            opt_params = None
            scheduler_params = None

        constraints = constraints  # placeholder for constraints flags

        return super().reconstruct(
            num_iter=num_iter,
            reset=reset,
            optimizer_params=opt_params,
            scheduler_params=scheduler_params,
            constraints=constraints,
            batch_size=batch_size,
            store_iterations_every=store_iterations_every,
            device=device,
        )
