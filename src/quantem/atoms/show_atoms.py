"""Compatibility shim: the interactive viewer lives in ``quantem.widget``.

``ShowAtoms3D`` moved to the ``quantem.widget`` package
(``quantem.widget.show_atoms3d``) so it shares the widget infrastructure with
the other quantem viewers.  This module re-exports it for older imports and
can be removed.
"""

from quantem.widget.show_atoms3d import ShowAtoms3D as ShowAtoms3D
