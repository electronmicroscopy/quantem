from quantem.diffraction.diffraction_tomography import (
    DiffractionTomography as DiffractionTomography,
)
from quantem.diffraction.show_diffraction_tilts import (
    show_diffraction_tilts as show_diffraction_tilts,
)

# deprecated explicit-6D model (likely to be removed)
from quantem.diffraction.full_diffraction_tomography import (
    DiffractionTomography as DiffractionTomography6D,
)
from quantem.diffraction.simulate_diffraction_tomography import (
    SimDiffractionTomography as SimDiffractionTomography,
)
from quantem.diffraction.show_diffraction_patterns import (
    show_diffraction_patterns as show_diffraction_patterns,
)

# the anywidget viewers need the optional "widgets" extra, so they are
# imported on first use rather than with the package
_LAZY = {
    "show_diffraction_tomography": "quantem.diffraction.show_diffraction_tomography",
    "DiffractionTomographyViewer": "quantem.diffraction.show_diffraction_tomography_6d",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
