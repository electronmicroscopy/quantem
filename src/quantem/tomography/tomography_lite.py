import os
from typing import Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.ml.inr import HSiren
from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    SchedulerParams,
)
from quantem.tomography.dataset_models import (
    DatasetConstraintsType,
    TomographyINRDataset,
    TomographyPixDataset,
)
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import ObjConstraintsType, ObjectINR, ObjectPixelated
from quantem.tomography.tomography import Tomography, TomographyConventional


class TomographyLiteINR(Tomography):
    """
    A lite version of the Tomography class.
    """

    @classmethod
    def from_dataset(
        cls,
        tilt_series: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        device: str = "cuda",
        log_dir: os.PathLike | str | None = None,
        log_images_every: int = 10,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        dset_model = TomographyINRDataset.from_data(
            tilt_stack=tilt_series,
            tilt_angles=tilt_angles,
        )

        # Define the object model
        model = HSiren(alpha=1, winner_initialization=72)
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
                log_dir=str(log_dir),
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

    def reconstruct(  # type:ignore[reportIncompatibleMethodOverride] ## easier than overloads
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
            "exp", "cyclic", "plateau", "cosine_annealing", "linear", "full_warmup", "none"
        ] = "none",
        scheduler_params: dict = {},
        new_optimizers: bool = False,
        obj_constraints: ObjConstraintsType | dict | None = None,
        dset_constraints: DatasetConstraintsType | dict | None = None,
        show_metrics: bool = False,
    ):
        if self.num_epochs == 0:
            opt_params = {
                "object": OptimizerParams.Adam(lr=obj_lr),
            }

            all_scheduler_params = {
                "object": SchedulerParams.parse_dict(
                    {
                        "name": scheduler_type,
                        **scheduler_params,
                    }
                ),
            }

            if learn_pose:
                opt_params["pose"] = OptimizerParams.Adam(lr=pose_lr)
                all_scheduler_params["pose"] = SchedulerParams.parse_dict(
                    {
                        "name": scheduler_type,
                        **scheduler_params,
                    }
                )
        else:
            opt_params = None
            all_scheduler_params = None
            obj_constraints = None
            dset_constraints = None

        num_samples_per_ray = int(max(self.dset.tilt_stack.shape))
        return super().reconstruct(
            num_iter=num_iter,
            batch_size=batch_size,
            num_workers=num_workers,
            reset=reset,
            num_samples_per_ray=num_samples_per_ray,
            optimizer_params=opt_params,
            scheduler_params=all_scheduler_params,
            obj_constraints=obj_constraints,
            dset_constraints=dset_constraints,
            show_metrics=show_metrics,
        )


class TomographyLiteConv(TomographyConventional):
    @classmethod
    def from_dataset(
        cls,
        tilt_series: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        dset_model = TomographyPixDataset.from_data(
            tilt_stack=tilt_series,
            tilt_angles=tilt_angles,
        )

        obj_model = ObjectPixelated.from_uniform(
            shape=(
                max(tilt_series.shape),
                max(tilt_series.shape),
                max(tilt_series.shape),
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
