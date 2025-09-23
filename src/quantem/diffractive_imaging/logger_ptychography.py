import os
from warnings import warn

import numpy as np

from quantem.core import config
from quantem.core.ml.logger import LoggerBase
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.object_models import ObjectModelType
from quantem.diffractive_imaging.probe_models import ProbeModelType


class LoggerPtychography(LoggerBase):
    """
    Ptychography-specific logger optimized for minimal performance impact during reconstruction.

    Performance optimizations:
    - Selective logging with early returns to minimize overhead
    - Cached tensor conversions and computations
    - Reduced memory allocations and copies
    - Smart image logging only when needed
    """

    def __init__(
        self,
        log_dir: os.PathLike | str,
        run_prefix: str = "",
        run_suffix: str = "",
        log_images_every: int = 10,
        log_probe_images: bool = True,
    ):
        super().__init__(log_dir, run_prefix, run_suffix, log_images_every)
        self._phase_cmap = config.get("viz.phase_cmap")
        self._log_probe_images = log_probe_images

    def object_image(self, volume_obj: ObjectModelType, epoch: int, logger_cmap: str = "turbo"):
        """Log object images with object type-aware visualization (optimized)."""
        try:
            obj = volume_obj.obj.cpu().detach().numpy()
            obj_type = volume_obj.obj_type

            # log z-sum only for speed
            obj_sum = np.sum(obj, axis=0)

            if obj_type == "potential":
                self.log_image(
                    tag="object/potential_zsum",
                    image=obj_sum,
                    step=epoch,
                    cmap=logger_cmap,
                )
            elif obj_type == "pure_phase":
                self.log_image(
                    tag="object/phase_zsum",
                    image=np.angle(obj_sum),
                    step=epoch,
                    cmap=self._phase_cmap,
                )
            elif obj_type == "complex":
                self.log_image(
                    tag="object/amplitude_zsum",
                    image=np.abs(obj_sum),
                    step=epoch,
                    cmap=logger_cmap,
                )
                self.log_image(
                    tag="object/phase_zsum",
                    image=np.angle(obj_sum),
                    step=epoch,
                    cmap=self._phase_cmap,
                )

        except Exception as e:
            print(f"Warning: Failed to log object images at epoch {epoch}: {e}")

    def probe_image(self, probe_model: ProbeModelType, epoch: int, logger_cmap: str = "turbo"):
        """Log probe images showing both real-space and fourier-space representations (optimized)."""
        try:
            probe = probe_model.probe

            # Single tensor conversion
            if hasattr(probe, "detach"):
                probe = probe.detach().cpu().numpy()

            # Log probe (real space) for each probe state
            for probe_idx in range(probe.shape[0]):
                probe_data = np.fft.fftshift(probe[probe_idx])

                # Complex probe - log amplitude and phase
                self.log_image(
                    tag=f"probe/amplitude/probe_{probe_idx}",
                    image=np.abs(probe_data),
                    step=epoch,
                    cmap=logger_cmap,
                )
                self.log_image(
                    tag=f"probe/phase/probe_{probe_idx}",
                    image=np.angle(probe_data),
                    step=epoch,
                    cmap=self._phase_cmap,
                )

        except Exception as e:
            print(f"Warning: Failed to log probe images at epoch {epoch}: {e}")

    def organize_constraint_losses(
        self,
        object_model: ObjectModelType,
        probe_model: ProbeModelType,
        dataset_model: DatasetModelType,
        num_batches: int = 1,
    ) -> dict[str, dict[str, float]]:
        """Organize constraint losses with minimal overhead."""
        organized_losses = {}

        models = [
            ("object_constraints", object_model),
            ("probe_constraints", probe_model),
            ("dataset_constraints", dataset_model),
        ]

        for model_name, model in models:
            losses = model.get_epoch_constraint_losses()
            if losses:
                # Only create dict if there are non-zero losses
                nonzero_losses = {k: v / num_batches for k, v in losses.items() if v != 0.0}
                if nonzero_losses:
                    organized_losses[model_name] = nonzero_losses

        return organized_losses

    def log_epoch(
        self,
        object_model: ObjectModelType,
        probe_model: ProbeModelType,
        dataset_model: DatasetModelType,
        epoch: int,
        consistency_loss: float,
        num_batches: int,
        # epoch_loss: float,
        # batch_losses: list[dict],
        learning_rates: dict | None = None,
        logger_cmap: str = "turbo",
    ):
        """Condensed epoch logging that handles losses, learning rates, and images."""
        try:
            organized_losses = self.organize_constraint_losses(
                object_model, probe_model, dataset_model, num_batches
            )
            total_constraint_loss = 0.0
            for category, constraint_losses in organized_losses.items():
                for constraint_name, value in constraint_losses.items():
                    self.log_scalar(f"constraints/{category}/{constraint_name}", value, epoch)
                    total_constraint_loss += value

            self.log_scalar("loss/consistency", consistency_loss, epoch)
            self.log_scalar("loss/constraint", total_constraint_loss, epoch)
            self.log_scalar("loss/total", total_constraint_loss + consistency_loss, epoch)

            # Learning rates
            if learning_rates:
                for param_name, lr_value in learning_rates.items():
                    if hasattr(lr_value, "item"):
                        lr_value = lr_value.item()
                    self.log_scalar(f"learning_rate/{param_name}", float(lr_value), epoch)

            # Images (only when needed)
            if epoch % self.log_images_every == 0 and self.log_images_every > 0:
                self.object_image(object_model, epoch, logger_cmap)
                if self._log_probe_images:
                    self.probe_image(probe_model, epoch, logger_cmap)

            # Flush occasionally, default is 120 sec
            if epoch % 50 == 0:
                self.flush()

        except Exception as e:
            warn(f"Warning: Epoch logging failed at epoch {epoch}: {e}")
            # Allow caller to decide whether to continue
            # Avoid unreachable code
