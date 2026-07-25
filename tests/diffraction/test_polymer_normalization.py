from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import quantem.diffraction.bragg_peaks as bragg_peaks_module
from quantem.diffraction import (
    BraggPeaksPolymer,
    GlobalPercentileNormalization,
    LegacyNormalizationAdapter,
    PerImageMinMaxPercentileNormalization,
    resolve_normalization_strategy,
)


@pytest.mark.parametrize("as_torch", [False, True])
def test_global_percentile_matches_reference(as_torch):
    array = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    batch = torch.from_numpy(array) if as_torch else array
    strategy = GlobalPercentileNormalization(10, 90)
    parameters = strategy.fit(batch)
    lower, upper = np.percentile(array, [10, 90])
    expected = (np.clip(array, lower, upper) - lower) / (upper - lower + 1e-8)
    actual = strategy.transform(batch, parameters)
    if as_torch:
        actual = actual.numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("as_torch", [False, True])
def test_per_image_minmax_percentile_matches_reference(as_torch):
    array = np.array(
        [
            [[1, 2], [3, 5]],
            [[10, 20], [30, 50]],
            [[7, 7], [7, 7]],
        ],
        dtype=np.float32,
    )
    batch = torch.from_numpy(array) if as_torch else array
    per_image = np.stack(
        [
            (image - image.min()) / (image.max() - image.min())
            if image.max() > image.min()
            else np.zeros_like(image)
            for image in array
        ]
    )
    lower, upper = np.percentile(per_image, [5, 95])
    expected = (
        np.clip(per_image, lower, upper) - lower
    ) / (upper - lower + 1e-8)
    strategy = PerImageMinMaxPercentileNormalization(5, 95)
    actual = strategy.transform(batch, strategy.fit(batch))
    if as_torch:
        actual = actual.numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_registered_resolution_and_legacy_adapter():
    strategy = resolve_normalization_strategy(
        {
            "mode": "per_image_minmax_percentile",
            "lower_percentile": 2,
            "upper_percentile": 98,
        }
    )
    assert isinstance(strategy, PerImageMinMaxPercentileNormalization)

    def compute(batch, lower_percentile, upper_percentile):
        return {"low": float(batch.min()), "high": float(batch.max())}

    def transform(batch, parameters):
        return (batch - parameters["low"]) / (
            parameters["high"] - parameters["low"]
        )

    adapter = LegacyNormalizationAdapter(compute, transform, 1, 99)
    batch = np.array([[1.0, 3.0]])
    np.testing.assert_allclose(
        adapter.transform(batch, adapter.fit(batch)), [[0.0, 1.0]]
    )


def test_strategy_and_model_changes_invalidate_inference_caches():
    detector = object.__new__(BraggPeaksPolymer)
    detector._normalization_strategy = GlobalPercentileNormalization()
    detector._normalization_is_explicit = False
    detector._normalization_parameters = object()
    detector._norm_median = 1
    detector._norm_iqr = 2
    detector._bn_adapted = True
    detector._live_chunk_cache = object()

    detector.normalization_strategy = PerImageMinMaxPercentileNormalization()
    assert detector._normalization_parameters is None
    assert detector._bn_adapted is False
    assert detector._live_chunk_cache is None

    detector._normalization_parameters = object()
    detector._bn_adapted = True
    detector.model = object()
    assert detector._normalization_parameters is None
    assert detector._bn_adapted is False


def test_missing_custom_checkpoint_normalization_fails_clearly():
    detector = object.__new__(BraggPeaksPolymer)
    detector._normalization_strategy = None
    with pytest.raises(RuntimeError, match="custom checkpoint"):
        detector._require_normalization_strategy()


def test_named_model_selects_metadata_strategy_and_explicit_override_wins(
    monkeypatch,
):
    class FakeModel:
        def load_state_dict(self, state):
            self.state = state

        def to(self, device):
            return self

    resolution = SimpleNamespace(
        model_id="example",
        weights_path="/unused/weights.pth",
        specification={
            "experimental_normalization": {
                "mode": "per_image_minmax_percentile",
                "lower_percentile": 3,
                "upper_percentile": 97,
            }
        },
    )
    monkeypatch.setattr(bragg_peaks_module, "resolve_polymer_model", lambda **_: resolution)
    monkeypatch.setattr(bragg_peaks_module, "build_polymer_model", lambda _: FakeModel())
    monkeypatch.setattr(torch, "load", lambda *_, **__: {})

    detector = object.__new__(BraggPeaksPolymer)
    detector._device = "cpu"
    detector._model = FakeModel()
    detector._normalization_strategy = None
    detector._normalization_is_explicit = False
    detector._invalidate_inference_caches()
    detector.set_model_weights()
    assert isinstance(
        detector.normalization_strategy, PerImageMinMaxPercentileNormalization
    )
    assert detector.normalization_strategy.lower_percentile == 3

    override = GlobalPercentileNormalization(4, 96)
    detector.normalization_strategy = override
    detector.set_model_weights()
    assert detector.normalization_strategy is override
