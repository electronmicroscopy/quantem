import numpy as np
import pytest


class TestPeakAutoID:
    def test_peak_autoid_returns_detected_elements_and_matches(
        self, xeds_factory, line_spectrum_factory
    ):
        dataset = xeds_factory(np.zeros((3, 3, 512), dtype=np.float32))
        background = 0.2 + 0.1 * np.linspace(0.0, 1.0, 512, dtype=np.float32)
        spectrum = line_spectrum_factory(
            ["Fe K", "Cu K"],
            [1200.0, 900.0],
            n_channels=512,
            energy_min=float(dataset.energy_axis[0]),
            energy_max=float(dataset.energy_axis[-1]),
            background=background,
        )
        cube = np.broadcast_to(spectrum, (3, 3, 512)).copy()
        dataset = xeds_factory(cube)

        details = dataset.peak_autoid(
            elements=["Fe", "Cu"],
            mode="elements_only",
            threshold="mean",
            peaks=6,
            show_text=False,
            return_details=True,
        )

        assert details["mode"] == "elements_only"
        assert details["peak_matches"]
        assert {"Fe", "Cu"} & set(details["detected_elements"])
        assert any(str(match[4]) in {"Fe", "Cu"} for match in details["peak_matches"])

    def test_peak_autoid_invalid_mode_raises(self, xeds_factory):
        dataset = xeds_factory(np.ones((2, 2, 128), dtype=np.float32))

        with pytest.raises(ValueError, match="mode must be one of"):
            dataset.peak_autoid(mode="not-a-mode")

    def test_peak_autoid_elements_preferred_requires_elements(self, xeds_factory):
        dataset = xeds_factory(np.ones((2, 2, 128), dtype=np.float32))

        with pytest.raises(ValueError, match="requires elements"):
            dataset.peak_autoid(mode="elements_preferred")
