import numpy as np
import pytest

from quantem.core.visualization.custom_normalizations import (
    CenteredInterval,
    CustomNormalization,
    HyperbolicSineStretch,
    InverseHyperbolicSineStretch,
    InverseLogarithmicStretch,
    LinearStretch,
    LogarithmicStretch,
    ManualInterval,
    NormalizationConfig,
    PowerLawStretch,
    QuantileInterval,
    _resolve_normalization,
)


@pytest.fixture
def sample_data():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)


class TestIntervals:
    def test_interval_limits(self, sample_data):
        for interval, expected in [
            (ManualInterval(), (1.0, 10.0)),
            (ManualInterval(vmin=2.0, vmax=8.0), (2.0, 8.0)),
            # With default center=0, the range should be [-10, 10] for sample_data [1, 10]
            (CenteredInterval(), (-10.0, 10.0)),
            (CenteredInterval(vcenter=5.0), (0.0, 10.0)),
            (CenteredInterval(vcenter=5.0, half_range=3.0), (2.0, 8.0)),
            (QuantileInterval(), (1.18, 9.82)),  # 2nd and 98th percentiles
            (QuantileInterval(lower_quantile=0.1, upper_quantile=0.9), (1.9, 9.1)),
        ]:
            assert np.allclose(interval.get_limits(sample_data), expected), interval

    def test_manual_interval_call(self, sample_data):
        interval = ManualInterval(vmin=2.0, vmax=8.0)
        result = interval(sample_data)
        assert np.allclose(result, np.clip((sample_data - 2.0) / 6.0, 0.0, 1.0))
        assert result[0] == 0.0  # value below vmin
        assert result[-1] == 1.0  # value above vmax


class TestStretches:
    def test_linear_and_power_stretch(self, sample_data):
        normalized_data = sample_data / 10.0  # Normalize to [0, 1]
        # the identity stretches return their input untouched, even outside [0, 1]
        assert np.allclose(LinearStretch()(sample_data), sample_data)
        assert np.allclose(PowerLawStretch()(sample_data), sample_data)
        assert np.allclose(LinearStretch(slope=2)(normalized_data), 2 * normalized_data)
        assert np.allclose(LinearStretch(intercept=1)(normalized_data), normalized_data + 1)
        assert np.allclose(PowerLawStretch(power=2)(normalized_data), normalized_data**2)

    def test_nonlinear_stretches_map_unit_interval_onto_itself(self):
        x = np.linspace(0.0, 1.0, 11)
        a = 1000.0
        expected = {
            "log": np.log(a * x + 1) / np.log(a + 1),
            "inverse_log": (np.exp(x * np.log(a + 1)) - 1) / a,
        }
        for name, stretch in [
            ("log", LogarithmicStretch(a)),
            ("inverse_log", InverseLogarithmicStretch(a)),
            ("asinh", InverseHyperbolicSineStretch()),
            ("sinh", HyperbolicSineStretch()),
            ("power", PowerLawStretch(power=2)),
        ]:
            y = stretch(x)
            assert y[0] == pytest.approx(0.0, abs=1e-12), name
            assert y[-1] == pytest.approx(1.0), name
            assert np.all(np.diff(y) > 0), name
            assert np.allclose(stretch.inverse(y), x), name
            if name in expected:
                assert np.allclose(y, expected[name]), name
            if name in ("asinh", "sinh"):
                assert y[5] == pytest.approx(0.5), name  # symmetric about the midpoint

    def test_invalid_parameters_raise(self):
        for build in (
            lambda: PowerLawStretch(power=0),
            lambda: LogarithmicStretch(a=0),
            lambda: InverseLogarithmicStretch(a=0),
            lambda: InverseHyperbolicSineStretch(a=0),
            lambda: HyperbolicSineStretch(a=0),
        ):
            with pytest.raises(ValueError):
                build()


class TestCustomNormalization:
    def test_interval_then_stretch(self, sample_data):
        lo, hi = np.quantile(sample_data, (0.02, 0.98))
        quantile = np.clip((sample_data - lo) / (hi - lo), 0.0, 1.0)
        for kwargs, expected in [
            ({}, quantile),
            (
                {"interval_type": "manual", "vmin": 2, "vmax": 8},
                np.clip((sample_data - 2) / 6, 0, 1),
            ),
            ({"interval_type": "centered", "vcenter": 5}, sample_data / 10.0),
            ({"stretch_type": "power", "power": 2}, quantile**2),
            ({"stretch_type": "logarithmic"}, np.log(1000 * quantile + 1) / np.log(1001)),
            (
                {"stretch_type": "asinh"},
                np.arcsinh((2 * quantile - 1) / 0.1) / (2 * np.arcsinh(10.0)) + 0.5,
            ),
        ]:
            assert np.allclose(CustomNormalization(**kwargs)(sample_data), expected), kwargs

    def test_invalid_types_raise(self):
        with pytest.raises(ValueError):
            CustomNormalization(interval_type="invalid")
        with pytest.raises(ValueError):
            CustomNormalization(stretch_type="invalid")


class TestResolveNormalization:
    def test_resolve_normalization(self):
        assert _resolve_normalization(None) == NormalizationConfig()
        assert _resolve_normalization(None, vmin=1, vmax=10) == NormalizationConfig(
            interval_type="manual", vmin=1, vmax=10
        )
        assert _resolve_normalization(None, upper_quantile=0.9) == NormalizationConfig(
            upper_quantile=0.9
        )
        config = _resolve_normalization({"interval_type": "manual", "vmin": 1, "vmax": 10})
        assert (config.interval_type, config.vmin, config.vmax) == ("manual", 1, 10)
        config = _resolve_normalization("log_minmax")
        assert (config.interval_type, config.stretch_type) == ("manual", "logarithmic")
        assert _resolve_normalization("power_sqrt").power == 0.5
        original = NormalizationConfig(power=3.0)
        assert _resolve_normalization(original) is original

    def test_resolve_normalization_invalid(self):
        with pytest.raises(ValueError):
            _resolve_normalization("invalid_preset")
        with pytest.raises(TypeError):
            _resolve_normalization(123)  # type: ignore
