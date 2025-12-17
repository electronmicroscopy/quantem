from quantem.core.ml.logger import LoggerBase


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
