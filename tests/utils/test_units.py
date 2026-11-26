import math

import pytest

from quantem.core.utils.units import Unit, parse_units, unit_conversion_factor
from quantem.core.utils.utils import electron_wavelength_angstrom


def test_parse_units_spellings():
    cases = {
        "nm": Unit("nm", "length", 1),
        "Angstroms": Unit("Å", "length", 1),
        "µm": Unit("μm", "length", 1),  # micro sign, stored as Greek mu
        "1/nm": Unit("nm", "length", -1),
        "Å^-1": Unit("Å", "length", -1),
        "A-1": Unit("Å", "length", -1),
        "nm⁻¹": Unit("nm", "length", -1),
        "$\\mathrm{nm}^{-1}$": Unit("nm", "length", -1),
        "m": Unit("m", "length", 1),
        "mrad": Unit("mrad", "angle", 1),
        "MRad": Unit("mrad", "angle", 1),
        "µrad": Unit("μrad", "angle", 1),
        "degrees": Unit("deg", "angle", 1),
    }
    for text, expected in cases.items():
        assert parse_units(text) == expected, text
    for text in ["pixels", "", "1/mrad", None, 3.0]:
        assert parse_units(text) is None, text


def test_unit_conversion_factor_same_kind():
    cases = [
        ("μm", "nm", 1e3),
        ("Å", "pm", 1e2),
        ("1/nm", "1/Å", 0.1),
        ("Å^-1", "nm^-1", 10.0),
        ("mrad", "rad", 1e-3),
        ("deg", "mrad", 1e3 * math.pi / 180),
    ]
    for src, dst, expected in cases:
        assert unit_conversion_factor(src, dst) == pytest.approx(expected, rel=1e-12), (src, dst)


def test_unit_conversion_factor_angle_to_reciprocal_length():
    energy = 300e3
    wavelength = electron_wavelength_angstrom(energy)
    assert wavelength == pytest.approx(0.019687, rel=1e-4)
    forward = unit_conversion_factor("mrad", "1/Å", energy=energy)
    assert forward == pytest.approx(1e-3 / wavelength, rel=1e-12)  # k = theta / lambda
    assert unit_conversion_factor("mrad", "1/nm", energy=energy) == pytest.approx(10 * forward)
    back = unit_conversion_factor("Å^-1", "mrad", energy=energy)
    assert forward * back == pytest.approx(1.0, rel=1e-12)


def test_unit_conversion_factor_errors():
    cases = [
        (("mrad", "1/Å"), {}, "beam energy"),
        (("mrad", "1/Å"), {"energy": 0.0}, "positive"),
        (("nm", "1/nm"), {}, "a length"),
        (("nm", "mrad"), {"energy": 300e3}, "an angle"),
        (("pixels", "nm"), {}, "Unrecognized"),
    ]
    for args, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            unit_conversion_factor(*args, **kwargs)
