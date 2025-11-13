import matplotlib.pyplot as plt

from quantem.core.ml.logger import LoggerBase
from quantem.tomography.object_models import ObjectModelType
from quantem.tomography.tomography_dataset import TomographyDataset


class LoggerTomography(LoggerBase):
    def __init__(self, log_dir: str, run_prefix: str, run_suffix: str = None):
        super().__init__(log_dir, run_prefix, run_suffix)

    # --- Tomography focused logging methods ---

    def tilt_angles_figure(self, dataset: TomographyDataset, step: int):
        figs = []
        for angle_array, title in zip(
            [dataset.z1_angles, dataset.tilt_angles, dataset.z3_angles],
            ["Z1 Angles", "Tilt/ X Angles", "Z3 Angles"],
        ):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(angle_array.detach().cpu().numpy())
            ax.set_title(title)
            ax.set_xlabel("Index")
            ax.set_ylabel("Angle")
            figs.append(fig)
            plt.close(fig)

        self.log_figure(
            tag="tilt_angles/z1",
            fig=figs[0],
            step=step,
        )
        self.log_figure(
            tag="tilt_angles/x",
            fig=figs[1],
            step=step,
        )
        self.log_figure(
            tag="tilt_angles/z3",
            fig=figs[2],
            step=step,
        )

    def projection_images(
        self, volume_obj: ObjectModelType, epoch: int, logger_cmap: str = "turbo"
    ):
        sum_0 = volume_obj.obj.sum(axis=0)
        sum_1 = volume_obj.obj.sum(axis=1)
        sum_2 = volume_obj.obj.sum(axis=2)

        self.log_image(
            tag="projections/Y-X Projection",
            image=sum_0,
            step=epoch,
            cmap=logger_cmap,
        )
        self.log_image(
            tag="projections/Z-X Projection",
            image=sum_1,
            step=epoch,
            cmap=logger_cmap,
        )
        self.log_image(
            tag="projections/Z-Y Projection",
            image=sum_2,
            step=epoch,
            cmap=logger_cmap,
        )
