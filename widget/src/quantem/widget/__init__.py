from importlib.metadata import version
import pathlib
import anywidget
import traitlets

__version__ = version("quantem.widget")

_static = pathlib.Path(__file__).parent / "static"


class CounterWidget(anywidget.AnyWidget):
    _esm = _static / "index.js"

    count = traitlets.Int(0).tag(sync=True)


def show4dstem():
    # TODO: Implement 4D-STEM visualization widget
    print("show4dstem: not yet implemented")


def counter():
    """Create a minimal counter widget for testing."""
    return CounterWidget()
