"""Diffraction analysis interfaces."""

from quantem.diffraction.bragg_peaks import BraggPeaksPolymer, ScanMaskEditor
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
    compute_global_intensity_threshold,
    detect_ice,
    flag_ice_peaks_in_dataset,
    flag_ice_peaks_in_pattern,
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
    "BraggPeaksPolymer",
    "ScanMaskEditor",
    "GlobalPercentileNormalization",
    "GlobalPercentileStrategy",
    "IceDetectionResult",
    "IceFlaggerDebug",
    "IceFlaggerParams",
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
    "detect_ice",
    "flag_ice_peaks_in_dataset",
    "flag_ice_peaks_in_pattern",
    "plot_q_intensity_density",
    "resolve_polymer_model",
    "resolve_normalization_strategy",
]
