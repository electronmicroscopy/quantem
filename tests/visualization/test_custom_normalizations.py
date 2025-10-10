import numpy as np
import pytest

from quantem.core.visualization.custom_normalizations import (
    NORMALIZATION_PRESETS,
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


@pytest.fixture
def sample_data_with_outliers():
    return np.array(
        [-10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0],
        dtype=np.float64,
    )


class TestManualInterval:
    def test_manual_interval_default(self, sample_data):
        interval = ManualInterval()
        vmin, vmax = interval.get_limits(sample_data)
        assert vmin == 1.0
        assert vmax == 10.0

    def test_manual_interval_with_limits(self, sample_data):
        interval = ManualInterval(vmin=2.0, vmax=8.0)
        vmin, vmax = interval.get_limits(sample_data)
        assert vmin == 2.0
        assert vmax == 8.0

    def test_manual_interval_call(self, sample_data):
        interval = ManualInterval(vmin=2.0, vmax=8.0)
        result = interval(sample_data)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result[0] == 0.0  # value below vmin
        assert result[-1] == 1.0  # value above vmax


class TestCenteredInterval:
    def test_centered_interval_default(self, sample_data):
        interval = CenteredInterval()
        vmin, vmax = interval.get_limits(sample_data)
        # With default center=0, the range should be [-10, 10] for sample_data [1, 10]
        assert vmin == -10.0
        assert vmax == 10.0

    def test_centered_interval_with_center(self, sample_data):
        interval = CenteredInterval(vcenter=5.0)
        vmin, vmax = interval.get_limits(sample_data)
        # With center=5, the range should be [0, 10] for sample_data [1, 10]
        assert vmin == 0.0
        assert vmax == 10.0

    def test_centered_interval_with_half_range(self, sample_data):
        interval = CenteredInterval(vcenter=5.0, half_range=3.0)
        vmin, vmax = interval.get_limits(sample_data)
        assert vmin == 2.0
        assert vmax == 8.0


class TestQuantileInterval:
    def test_quantile_interval_default(self, sample_data):
        interval = QuantileInterval()
        vmin, vmax = interval.get_limits(sample_data)
        assert np.isclose(vmin, 1.18)  # 2nd percentile
        assert np.isclose(vmax, 9.82)  # 98th percentile

    def test_quantile_interval_custom(self, sample_data):
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)
        vmin, vmax = interval.get_limits(sample_data)
        assert np.isclose(vmin, 1.9)  # 10th percentile
        assert np.isclose(vmax, 9.1)  # 90th percentile


class TestLinearStretch:
    def test_linear_stretch_default(self, sample_data):
        stretch = LinearStretch()
        result = stretch(sample_data)
        assert np.allclose(result, sample_data)

    def test_linear_stretch_with_slope(self, sample_data):
        stretch = LinearStretch(slope=2)
        normalized_data = sample_data / 10.0  # Normalize to [0, 1]
        result = stretch(normalized_data)
        assert np.allclose(result, 2 * normalized_data)

    def test_linear_stretch_with_intercept(self, sample_data):
        stretch = LinearStretch(intercept=1)
        normalized_data = sample_data / 10.0  # Normalize to [0, 1]
        result = stretch(normalized_data)
        assert np.allclose(result, normalized_data + 1)


class TestPowerLawStretch:
    def test_power_law_stretch_default(self, sample_data):
        stretch = PowerLawStretch()
        result = stretch(sample_data)
        assert np.allclose(result, sample_data)

    def test_power_law_stretch_square(self, sample_data):
        stretch = PowerLawStretch(power=2)
        normalized_data = sample_data / 10.0  # Normalize to [0, 1]
        result = stretch(normalized_data)
        assert np.allclose(result, normalized_data**2)

    def test_power_law_stretch_invalid_power(self):
        with pytest.raises(ValueError):
            PowerLawStretch(power=0)


class TestLogarithmicStretch:
    def test_logarithmic_stretch_default(self, sample_data):
        stretch = LogarithmicStretch()
        result = stretch(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_logarithmic_stretch_invalid_a(self):
        with pytest.raises(ValueError):
            LogarithmicStretch(a=0)


class TestInverseLogarithmicStretch:
    def test_inverse_logarithmic_stretch_default(self, sample_data):
        stretch = InverseLogarithmicStretch()
        result = stretch(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_inverse_logarithmic_stretch_invalid_a(self):
        with pytest.raises(ValueError):
            InverseLogarithmicStretch(a=0)


class TestInverseHyperbolicSineStretch:
    def test_inverse_hyperbolic_sine_stretch_default(self, sample_data):
        stretch = InverseHyperbolicSineStretch()
        result = stretch(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_inverse_hyperbolic_sine_stretch_invalid_a(self):
        with pytest.raises(ValueError):
            InverseHyperbolicSineStretch(a=0)


class TestHyperbolicSineStretch:
    def test_hyperbolic_sine_stretch_default(self, sample_data):
        stretch = HyperbolicSineStretch()
        result = stretch(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_hyperbolic_sine_stretch_invalid_a(self):
        with pytest.raises(ValueError):
            HyperbolicSineStretch(a=0)


class TestCustomNormalization:
    def test_custom_normalization_default(self, sample_data):
        norm = CustomNormalization()
        result = norm(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_custom_normalization_manual(self, sample_data):
        norm = CustomNormalization(interval_type="manual", vmin=2, vmax=8)
        result = norm(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_custom_normalization_centered(self, sample_data):
        norm = CustomNormalization(interval_type="centered", vcenter=5)
        result = norm(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_custom_normalization_power(self, sample_data):
        norm = CustomNormalization(stretch_type="power", power=2)
        result = norm(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_custom_normalization_logarithmic(self, sample_data):
        norm = CustomNormalization(stretch_type="logarithmic")
        result = norm(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_custom_normalization_asinh(self, sample_data):
        norm = CustomNormalization(stretch_type="asinh")
        result = norm(sample_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_custom_normalization_invalid_interval_type(self, sample_data):
        with pytest.raises(ValueError):
            CustomNormalization(interval_type="invalid")

    def test_custom_normalization_invalid_stretch_type(self, sample_data):
        with pytest.raises(ValueError):
            CustomNormalization(stretch_type="invalid")


class TestNormalizationConfig:
    def test_normalization_config_default(self):
        config = NormalizationConfig()
        assert config.interval_type == "quantile"
        assert config.stretch_type == "linear"
        assert config.lower_quantile == 0.02
        assert config.upper_quantile == 0.98

    def test_normalization_config_custom(self):
        config = NormalizationConfig(
            interval_type="manual",
            stretch_type="power",
            vmin=1,
            vmax=10,
            power=2,
        )
        assert config.interval_type == "manual"
        assert config.stretch_type == "power"
        assert config.vmin == 1
        assert config.vmax == 10
        assert config.power == 2


class TestResolveNormalization:
    def test_resolve_normalization_none(self):
        config = _resolve_normalization(None)
        assert isinstance(config, NormalizationConfig)
        assert config.interval_type == "quantile"

    def test_resolve_normalization_dict(self):
        config = _resolve_normalization({"interval_type": "manual", "vmin": 1, "vmax": 10})
        assert isinstance(config, NormalizationConfig)
        assert config.interval_type == "manual"
        assert config.vmin == 1
        assert config.vmax == 10

    def test_resolve_normalization_preset(self):
        for preset_name in NORMALIZATION_PRESETS:
            config = _resolve_normalization(preset_name)
            assert isinstance(config, NormalizationConfig)

    def test_resolve_normalization_invalid_preset(self):
        with pytest.raises(ValueError):
            _resolve_normalization("invalid_preset")

    def test_resolve_normalization_invalid_type(self):
        with pytest.raises(TypeError):
            _resolve_normalization(123)  # type: ignore
