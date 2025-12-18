import contextlib
import copy
import datetime
import os
import shutil
import tempfile
from pathlib import Path
from typing import Self, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch._tensor import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from quantem.core.io.serialize import AutoSerialize, load

"""
Tensorboard logger class for AD/ML reconstruction methods
"""


class LoggerBase(AutoSerialize):
    def __init__(
        self,
        base_log_dir: os.PathLike | str,
        run_prefix: str,
        run_suffix: str = "",
        log_images_every: int = 10,
    ) -> None:
        self._timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # This should never be reinstantiated.
        self.run_prefix = run_prefix
        self.run_suffix = run_suffix
        self.log_dir = base_log_dir
        self.log_images_every = log_images_every
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def log_image(self, tag: str, image: NDArray | Tensor, step: int, cmap: str = "turbo") -> None:
        cmap_image = self.apply_colormap(image, cmap_name=cmap)
        self.writer.add_image(tag, cmap_image, step)

    def log_figure(self, tag: str, fig: Figure, step: int) -> None:
        self.writer.add_figure(tag, fig, step)

    def log_histogram(self, tag: str, values: NDArray | Tensor, step: int) -> None:
        """Log histogram of values for monitoring distributions."""
        if isinstance(values, Tensor):
            values = values.detach().cpu().numpy()
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text for configuration, hyperparameters, or notes."""
        self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def new_timestamp(self) -> None:
        self.close()
        self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.run_prefix + "_" + self._timestamp
        if self.run_suffix:
            name += f"_{self.run_suffix}"
        new_log_dir = self.log_dir.parent / name
        new_log_dir.mkdir(exist_ok=True)
        self._log_dir = new_log_dir
        self.writer = SummaryWriter(str(self.log_dir))

    def clone(self) -> Self:
        try:
            cloned: Self = copy.deepcopy(self)
        except Exception:
            # using tempfile saving as fallback
            tmp_path = Path(tempfile.gettempdir()) / f"logger_clone_{self.run_prefix}.zip"
            try:
                self.save(
                    tmp_path,
                    mode="o",
                    store="zip",
                )
                cloned = cast(Self, load(tmp_path))
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()
        cloned.new_timestamp()

        # copy old log file to new log dir
        files = list(self.log_dir.glob("events.out.tfevents.*"))
        for file in files:
            shutil.copy(file, cloned.log_dir)
        return cloned

    # --- Properties ---

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @log_dir.setter
    def log_dir(self, dir: str | os.PathLike) -> None:
        if not isinstance(dir, str | os.PathLike):
            raise TypeError("Log directory must be a str or Path.")

        dir = Path(dir)
        name = self.run_prefix + "_" + self._timestamp
        if self.run_suffix:
            name += f"_{self.run_suffix}"

        full_path = dir / name
        full_path.mkdir(parents=True, exist_ok=True)

        self._log_dir = full_path

    @property
    def run_prefix(self) -> str:
        return self._run_prefix

    @run_prefix.setter
    def run_prefix(self, prefix: str) -> None:
        if not isinstance(prefix, str):
            raise TypeError("Prefix must be a string")

        self._run_prefix = prefix

    @property
    def run_suffix(self) -> str:
        return self._run_suffix

    @run_suffix.setter
    def run_suffix(self, suffix: str) -> None:
        if not isinstance(suffix, str):
            raise TypeError("Suffix must be a string")

        self._run_suffix = suffix

    @property
    def log_images_every(self) -> int:
        return self._log_images_every

    @log_images_every.setter
    def log_images_every(self, value: int) -> None:
        self._log_images_every = int(value)

    # --- Helper Functions ---

    @staticmethod
    def apply_colormap(tensor_2d: Tensor | NDArray, cmap_name: str = "turbo") -> NDArray:
        """
        Apply colormap to a 2D tensor and return a [3, H, W] NumPy float32 array in [0, 1].
        """
        if isinstance(tensor_2d, Tensor):
            tensor_2d = tensor_2d.detach().cpu().numpy()

        tensor_2d = (tensor_2d - np.min(tensor_2d)) / (np.ptp(tensor_2d) + 1e-8)
        cmap = plt.get_cmap(cmap_name)
        colored = cmap(tensor_2d)[..., :3].transpose(2, 0, 1)  # type: ignore # [3, H, W]
        return colored.astype(np.float32)
