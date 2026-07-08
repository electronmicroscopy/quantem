from importlib.metadata import PackageNotFoundError, version

from quantem.widget.show2d import Show2D
from quantem.widget.show4dstem import Show4DSTEM

try:
    __version__ = version("quantem.widget")
except PackageNotFoundError:
    # Source-tree imports (e.g. `PYTHONPATH=src pytest`) skip pip install.
    __version__ = "0.0.0+local"

__all__ = ["Show2D", "Show4DSTEM"]
