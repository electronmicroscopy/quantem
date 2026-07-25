from __future__ import annotations

import numpy as np
import pytest

from quantem.core.datastructures import Vector
from quantem.diffraction import IceFlaggerParams, detect_ice


def _vectors(shape=(1, 2)):
    polar = Vector.from_shape(
        shape=shape, fields=["r_invA", "theta"], units=["1/A", "rad"]
    )
    intensity = Vector.from_shape(
        shape=shape, fields=["intensities"], units=["normalized"]
    )
    return polar, intensity


def test_detect_ice_and_filter_does_not_mutate_source():
    polar, intensity = _vectors()
    polar[0, 0] = np.column_stack(
        [np.full(3, 1.61), np.deg2rad([5, 65, 140])]
    )
    intensity[0, 0] = np.array([[0.9], [0.8], [0.7]])
    polar[0, 1] = np.empty((0, 2))
    intensity[0, 1] = np.empty((0, 1))
    original = polar[0, 0].array.copy()

    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(
            intensity_cutoff=0.5, min_matches=2, dtheta_deg=6
        ),
        return_debug=True,
    )
    assert result.threshold == 0.5
    assert result.flagged_peaks_count_map.tolist() == [[2, 0]]
    filtered = result.filter(polar)
    np.testing.assert_array_equal(polar[0, 0].array, original)
    assert len(filtered[0, 0].array) == 1
    assert (0, 1) in result.debug_records


def test_masked_cells_are_not_analyzed():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack(
        [np.full(2, 1.61), np.deg2rad([0, 60])]
    )
    intensity[0, 0] = np.ones((2, 1))
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(intensity_cutoff=0.0),
        scan_mask=np.zeros((1, 1), dtype=bool),
    )
    assert result.flagged_peaks_count_map[0, 0] == 0


def test_misaligned_ragged_vectors_fail_clearly():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.zeros((2, 2))
    intensity[0, 0] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="Row count mismatch"):
        detect_ice(
            polar,
            intensity,
            params=IceFlaggerParams(intensity_cutoff=0.0),
        )
