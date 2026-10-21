"""Deprecated: use :mod:`quantem.core.utils.rotations` and
:mod:`quantem.diffraction.crystal_rotations`.

Kept so existing scripts and notebooks keep working.
"""

import warnings

from quantem.core.utils.rotations import *  # noqa: F401,F403
from quantem.core.utils.rotations import __all__ as _core_all
from quantem.diffraction.crystal_rotations import *  # noqa: F401,F403
from quantem.diffraction.crystal_rotations import __all__ as _crystal_all

__all__ = _core_all + _crystal_all

warnings.warn(
    "quantem.diffraction.rotations is deprecated; import from "
    "quantem.core.utils.rotations (rotation math) or "
    "quantem.diffraction.crystal_rotations (zone axes and symmetry).",
    DeprecationWarning,
    stacklevel=2,
)
