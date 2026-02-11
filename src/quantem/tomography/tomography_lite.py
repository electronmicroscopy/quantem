import os
from typing import Literal, Self

import numpy as np

from quantem.core.ml.inr import HSiren
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import ObjectINR, ObjectPixelated
from quantem.tomography.tomography import Tomography, TomographyConventional


class TomographyLiteINR(Tomography):
    """
    A lite version of the Tomography class.
    """

    @classmethod
    def from_dataset(
        cls,
        dset: DatasetModelType,
        device: str = "cuda",
        log_dir: os.PathLike | str | None = None,
        log_images_every: int = 10,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        dset_model = dset

        # Define the object model
        model = HSiren(alpha=1, winner_initialization=True)
        obj_model = ObjectINR.from_model(
            model=model,
            shape=(
                max(dset_model.tilt_stack.shape),
                max(dset_model.tilt_stack.shape),
                max(dset_model.tilt_stack.shape),
            ),
            device=device,
            rng=rng,
        )

        # TODO: Implement pretrain

        if log_dir is not None:
            logger = LoggerTomography(
                log_dir=log_dir,
                run_prefix="tomography_lite_inr",
                run_suffix="",
                log_images_every=log_images_every,
            )
        else:
            logger = None

        tomography = cls.from_models(
            dset=dset_model,
            obj_model=obj_model,
            device=device,
            rng=rng,
            logger=logger,
        )

        return tomography

    def reconstruct(
        self,
        num_iter: int = 10,
        reset: bool = False,
        obj_lr: float = 1e-4,
        pose_lr: float = 1e-2,
        batch_size: int = 1024,
        num_workers: int = 32,
        learn_pose: bool = True,
        warmup_routine: bool = True,
        scheduler_type: Literal[
            "exp", "cyclic", "plateau", "cosine_annealing", "linear", "full_warmup"
        ] = "none",
        scheduler_factor: float = 0.5,
        new_optimizers: bool = False,
        constraints: dict = {},
    ):
        if self.num_epochs == 0:
            opt_params = {
                "object": {
                    "type": "adam",
                    "lr": obj_lr,
                },
            }

            scheduler_params = {
                "object": {
                    "type": scheduler_type,
                    "factor": scheduler_factor,
                },
            }

            if learn_pose:
                opt_params["pose"] = {
                    "type": "adam",
                    "lr": pose_lr,
                }
                scheduler_params["pose"] = {
                    "type": scheduler_type,
                    "factor": scheduler_factor,
                }

        else:
            opt_params = None
            scheduler_params = None

        constraints = constraints
        num_samples_per_ray = int(max(self.dset.tilt_stack.shape))
        return super().reconstruct(
            num_iter=num_iter,
            batch_size=batch_size,
            num_workers=num_workers,
            reset=reset,
            num_samples_per_ray=num_samples_per_ray,
            optimizer_params=opt_params,
            scheduler_params=scheduler_params,
            constraints=constraints,
        )


class TomographyLiteConv(TomographyConventional):
    @classmethod
    def from_dataset(
        cls,
        dset: DatasetModelType,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        dset_model = dset

        obj_model = ObjectPixelated(
            shape=(
                max(dset_model.tilt_stack.shape),
                max(dset_model.tilt_stack.shape),
                max(dset_model.tilt_stack.shape),
            ),
            device=device,
            rng=rng,
        )

        tomography = cls.from_models(
            dset=dset_model,
            obj_model=obj_model,
            device=device,
            rng=rng,
        )

        return tomography
