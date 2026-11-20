"""Diffraction analysis interfaces."""

from quantem.diffraction.polar import PairDistributionFunction as PairDistributionFunction
from quantem.diffraction.bragg_vectors import BraggVectors as BraggVectors
from quantem.diffraction.crystal import Crystal as Crystal
from quantem.diffraction.crystal_map import CrystalMap as CrystalMap
from quantem.diffraction.orientation import OrientationMap as OrientationMap
from quantem.diffraction.phase import PhaseMap as PhaseMap
from quantem.diffraction.reverse_monte_carlo import ReverseMonteCarlo as ReverseMonteCarlo
from quantem.diffraction.strain import StrainMap as StrainMap
from quantem.diffraction.strain_autocorrelation import StrainMapAutocorrelation as StrainMapAutocorrelation
from quantem.diffraction.model_fitting import ModelDiffraction as ModelDiffraction
from quantem.diffraction import bloch as bloch
from quantem.diffraction import calibration as calibration
from quantem.diffraction import crystal_rotations as crystal_rotations
from quantem.diffraction import digital_dark_field as digital_dark_field
from quantem.diffraction import ellipticity as ellipticity
from quantem.diffraction import illumination as illumination

from quantem.diffraction.bragg_peaks import BraggPeaksPolymer
from quantem.diffraction.scan_mask import ScanMaskShape
from quantem.diffraction.polymer_models import (
    PAPER_MODEL_ID,
    PAPER_MODEL_VERSION,
    PolymerModelError,
    PolymerModelResolution,
    resolve_polymer_model,
)
from quantem.diffraction.polymer_ice import (
    IceDetectionResult,
    IceFlaggerDebug,
    IceFlaggerParams,
    apply_ice_mask_to_vector,
    collect_peak_widths,
    compute_global_intensity_threshold,
    detect_ice,
    flag_ice_peaks_in_dataset,
    flag_ice_peaks_in_pattern,
    measure_peak_widths,
    sharpness_mask,
    plot_q_intensity_density,
)
from quantem.diffraction.polymer_normalization import (
    GlobalPercentileNormalization,
    GlobalPercentileStrategy,
    LegacyNormalizationAdapter,
    NormalizationStrategy,
    PerImageMinMaxPercentileNormalization,
    PerImageMinMaxPercentileStrategy,
    resolve_normalization_strategy,
)

__all__ = [
    "BraggVectors",
    "Crystal",
    "CrystalMap",
    "OrientationMap",
    "PairDistributionFunction",
    "PhaseMap",
    "ReverseMonteCarlo",
    "StrainMap",
    "bloch",
    "calibration",
    "digital_dark_field",
    "illumination",
    "rotations",
    "BraggPeaksPolymer",
    "ScanMaskShape",
    "GlobalPercentileNormalization",
    "GlobalPercentileStrategy",
    "IceDetectionResult",
    "IceFlaggerDebug",
    "IceFlaggerParams",
    "measure_peak_widths",
    "sharpness_mask",
    "LegacyNormalizationAdapter",
    "NormalizationStrategy",
    "PAPER_MODEL_ID",
    "PAPER_MODEL_VERSION",
    "PolymerModelError",
    "PolymerModelResolution",
    "PerImageMinMaxPercentileNormalization",
    "PerImageMinMaxPercentileStrategy",
    "apply_ice_mask_to_vector",
    "compute_global_intensity_threshold",
    "collect_peak_widths",
    "detect_ice",
    "flag_ice_peaks_in_dataset",
    "flag_ice_peaks_in_pattern",
    "plot_q_intensity_density",
    "resolve_polymer_model",
    "resolve_normalization_strategy",
]
