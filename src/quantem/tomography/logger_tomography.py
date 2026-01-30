import matplotlib.pyplot as plt
import torch

from quantem.core.ml.logger import LoggerBase
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.object_models import ObjectModelType


class LoggerTomography(LoggerBase):
    """
    Logger for ML-based tomography reconstructions.
    """

    def __init__(
        self,
        log_dir: str,
        run_prefix: str,
        run_suffix: str = None,
        log_images_every: int = 10,
    ):
        super().__init__(log_dir, run_prefix, run_suffix, log_images_every)

    def log_epoch(self, epoch: int, loss: float, tilt_series_loss: float, soft_loss: float):
        self.log_scalar("loss/total", loss, epoch)
        self.log_scalar("loss/tilt_series", tilt_series_loss, epoch)
        self.log_scalar("loss/soft", soft_loss, epoch)

    def log_iter(
        self,
        object_model: ObjectModelType,
        iter: int,
        consistency_loss: float,
        total_loss: float,
        learning_rates: dict[str, float],
        num_samples_per_ray: int,
        val_loss: float | None = None,
    ):
        self.log_scalar("loss/consistency", consistency_loss, iter)
        self.log_scalar("loss/total", total_loss, iter)
        self.log_scalar("loss/soft", object_model._soft_constraint_losses[-1], iter)
        self.log_scalar("num_samples_per_ray", num_samples_per_ray, iter)
        for param_name, lr_value in learning_rates.items():
            self.log_scalar(f"learning_rate/{param_name}", float(lr_value), iter)
        if val_loss is not None:
            self.log_scalar("loss/val", val_loss, iter)

    def log_iter_images(
        self,
        pred_volume: torch.Tensor,
        dataset_model: DatasetModelType,
        iter: int,
        logger_cmap: str = "turbo",
    ):
        with torch.no_grad():
            z1_vals = dataset_model.z1_params.detach().cpu().numpy()
            z3_vals = dataset_model.z3_params.detach().cpu().numpy()
            shifts_vals = dataset_model.shifts_params.detach().cpu().numpy()

        print("Logging volume...")
        self.log_image("volume/sum_z", pred_volume.sum(axis=0), iter, logger_cmap)
        self.log_image("volume/sum_y", pred_volume.sum(axis=1), iter, logger_cmap)
        self.log_image("volume/sum_x", pred_volume.sum(axis=2), iter, logger_cmap)

        # Plotting z1 and z3 vals
        print("Plotting z1 and z3 angles...")
        fig, ax = plt.subplots()
        ax.plot(z1_vals, label="Z1")
        ax.plot(z3_vals, label="Z3")
        ax.legend()
        ax.set_title("Z1 and Z3 Angles")
        ax.set_xlabel("Tilt Image")
        ax.set_ylabel("Degree")
        self.log_figure("z1_z3_angles", fig, iter)
        plt.close(fig)

        # Plotting shifts
        print("Plotting shifts...")
        fig, ax = plt.subplots()
        ax.plot(shifts_vals[:, 0], label="Shifts X")
        ax.plot(shifts_vals[:, 1], label="Shifts Y")
        ax.legend()
        ax.set_title("Shifts")
        ax.set_xlabel("Tilt Image")
        ax.set_ylabel("Pixel")
        self.log_figure("shifts", fig, iter)
        plt.close(fig)
