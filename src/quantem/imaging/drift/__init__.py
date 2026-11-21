"""Drift-correction public package surface.

**Primary user API** - one class, chainable stages::

    from quantem.imaging.drift import DriftCorrection

    dc = DriftCorrection.from_emd(f0, f1)   # angles from EMD metadata
    dc.correct_affine()                    # automatic; no solve knobs required
    dc.plot_combined(stage=("initial", "affine"), interactive=True)
    dc.report()
    dc.show()
    dc.save("drift.zip", mode="o")

Optional residual stages after affine::

    dc.correct_strip(...)       # piecewise-rigid bands
    dc.correct_nonrigid(...)    # per-scanline polish (tiny max_image_shift on lattices)

Manual rigid registration remains available when needed::

    dc.align_translation(max_image_shift=32)

Also: :meth:`~DriftCorrection.from_images`,
:meth:`~DriftCorrection.from_reference`,
:meth:`~DriftCorrection.from_4dstem`, and an explicit
:meth:`~DriftCorrection.preprocess` when you need a fixed canvas.

Advanced troubleshooting stays on the same object through
``diagnose_affine()`` and ``diagnose_nonrigid()``. Numerical code lives under
``drift.core`` and is not a notebook entry point.

Free residual helpers are not re-exported here; use ``dc.correct_strip()``.
"""

from quantem.imaging.drift.correction import DriftCorrection as DriftCorrection
from quantem.imaging.drift.core.strip import StripPass as StripPass
from quantem.imaging.drift.fourdstem import CorrectionResult as CorrectionResult
from quantem.imaging.drift import plot as plot
from quantem.imaging.drift import io as io
from quantem.imaging.drift import fourdstem as fourdstem
from quantem.imaging.drift.io import (
    pair_spectrum_image_references as pair_spectrum_image_references,
    read_emd as read_emd,
    read_emd_eds as read_emd_eds,
    read_emd_metadata as read_emd_metadata,
)

__all__ = [
    "DriftCorrection",
    "CorrectionResult",
    "StripPass",
    "plot",
    "io",
    "fourdstem",
    "pair_spectrum_image_references",
    "read_emd",
    "read_emd_eds",
    "read_emd_metadata",
]
