import numpy as np

from quantem.diffraction.grain_clustering import (
    GrainResult,
    SignalTable,
    _rasterize_window,
    consolidate_signals_per_probe,
    grain_rgb_overlay,
)


def _result(labels, confidence=None):
    return GrainResult(
        labels=np.asarray(labels, dtype=np.int64),
        n_grains=1,
        grains=[],
        label_map=np.full((1, 1, 2), -1, dtype=np.int64),
        confidence=None if confidence is None else np.asarray(confidence, dtype=float),
    )


def test_rasterize_window_outlier_does_not_overwrite_assigned_signal():
    signals = SignalTable(
        pos=[[0, 0], [0, 0], [0, 1]],
        theta=[25.0, 80.0, 40.0],
        r=[0.1, 0.1, 0.1],
        intensity=[2.0, 10.0, 3.0],
        window=[0, 0, 0],
        map_shape=(1, 2),
    )
    result = _result([0, -1, -1], confidence=[0.75, 0.1, 0.2])

    labels, theta, confidence, filled = _rasterize_window(signals, result, 0)

    assert labels.tolist() == [[0, -1]]
    assert theta[0, 0] == 25.0
    assert confidence[0, 0] == 0.75
    assert filled.tolist() == [[True, True]]


def test_overlay_colors_only_outlier_probe_without_hiding_assignment():
    signals = SignalTable(
        pos=[[0, 0], [0, 0], [0, 1]],
        theta=[25.0, 80.0, 40.0],
        r=[0.1, 0.1, 0.1],
        intensity=[2.0, 10.0, 3.0],
        window=[0, 0, 0],
        map_shape=(1, 2),
    )
    result = _result([0, -1, -1])
    outlier_color = np.array([0.2, 0.3, 0.4])

    rgb = grain_rgb_overlay(
        signals,
        result,
        mode="orientation",
        boundary=False,
        outlier_color=outlier_color,
    )

    assert not np.allclose(rgb[0, 0], outlier_color)
    np.testing.assert_allclose(rgb[0, 1], outlier_color)


def test_consolidation_merges_only_compatible_same_probe_same_window_signals():
    signals = SignalTable(
        pos=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
        theta=[179.0, 1.0, 45.0, 0.0, 0.0],
        r=[0.100, 0.102, 0.101, 0.101, 0.101],
        intensity=[2.0, 3.0, 4.0, 5.0, 6.0],
        window=[0, 0, 0, 1, 0],
        map_shape=(1, 2),
    )

    consolidated, inverse = consolidate_signals_per_probe(
        signals,
        theta_tol_deg=5.0,
        r_tol_rel=0.05,
        intensity_reducer="sum",
        return_inverse=True,
    )

    assert len(consolidated) == 4
    assert inverse[0] == inverse[1]
    assert inverse[2] != inverse[0]
    assert inverse[3] != inverse[0]
    assert inverse[4] != inverse[0]
    merged = inverse[0]
    assert min(consolidated.theta[merged], 180.0 - consolidated.theta[merged]) < 1.0
    assert consolidated.intensity[merged] == 5.0


def test_consolidation_uses_complete_linkage_to_avoid_chaining():
    signals = SignalTable(
        pos=[[0, 0], [0, 0], [0, 0]],
        theta=[0.0, 4.0, 8.0],
        r=[0.1, 0.1, 0.1],
        intensity=[1.0, 1.0, 1.0],
        window=[0, 0, 0],
        map_shape=(1, 1),
    )

    consolidated = consolidate_signals_per_probe(
        signals, theta_tol_deg=5.0, r_tol_rel=0.05
    )

    assert len(consolidated) == 2
