import numpy as np
import pytest


class TestXEDSBackground:
    def test_calculate_background_polynomial_returns_finite_nonnegative_background(
        self, xeds_factory, line_spectrum_factory
    ):
        dataset = xeds_factory(np.zeros((1, 1, 512), dtype=np.float32))
        true_background = 3.0 + 1.5 * np.linspace(0.0, 1.0, 512, dtype=np.float32)
        spectrum = line_spectrum_factory(
            ["Fe K", "Cu K"],
            [800.0, 400.0],
            n_channels=512,
            energy_min=float(dataset.energy_axis[0]),
            energy_max=float(dataset.energy_axis[-1]),
            background=true_background,
        )

        background = dataset.calculate_background_polynomial(
            spectrum,
            degree=1,
            percentile=10,
            window_size=35,
        )

        assert background.shape == spectrum.shape
        assert np.isfinite(background).all()
        assert np.all(background >= 0.0)
        np.testing.assert_allclose(
            background[[0, -1]], true_background[[0, -1]], rtol=0.3, atol=0.8
        )

    @pytest.mark.parametrize(
        "kwargs,exc_type,match",
        [
            ({"degree": True}, TypeError, "degree"),
            ({"degree": -1}, ValueError, "degree"),
            ({"percentile": 120}, ValueError, "percentile"),
            ({"window_size": 0}, ValueError, "window_size"),
        ],
    )
    def test_calculate_background_polynomial_validates_inputs(
        self, xeds_factory, kwargs, exc_type, match
    ):
        dataset = xeds_factory(np.zeros((1, 1, 64), dtype=np.float32))
        spectrum = np.linspace(1.0, 2.0, 64, dtype=np.float32)

        with pytest.raises(exc_type, match=match):
            dataset.calculate_background_polynomial(spectrum, **kwargs)

    def test_calculate_background_powerlaw_is_alias(self, xeds_factory):
        dataset = xeds_factory(np.zeros((1, 1, 64), dtype=np.float32))
        spectrum = np.linspace(1.0, 2.0, 64, dtype=np.float32)

        poly = dataset.calculate_background_polynomial(spectrum, degree=1, window_size=11)
        alias = dataset.calculate_background_powerlaw(spectrum, degree=1, window_size=11)

        np.testing.assert_allclose(alias, poly)
