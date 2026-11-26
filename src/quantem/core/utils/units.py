"""Parse unit strings and convert between units of length, reciprocal length and angle.

Dataset axes carry their units as free text (``"nm"``, ``"1/nm"``, ``"Å^-1"``,
``"mrad"``, ...), spelled differently by different file formats. :func:`parse_units`
reads such a string into a canonical unit, and :func:`unit_conversion_factor` gives
the factor that converts a value from one unit to another.

Scattering angles and spatial frequencies are related by the electron wavelength,
``k = sin(theta) / lambda ~ theta / lambda`` (no factor of 2 pi), so converting
between an angle and a reciprocal length needs the beam energy. The conversion is
linear in the small-angle approximation, whose relative error is ``theta**2 / 6``
(0.04% at 50 mrad); this is the approximation used throughout quantem to convert a
detector sampling.
"""

import math
from typing import Any, NamedTuple

from quantem.core.utils.utils import electron_wavelength_angstrom

# Length units, as multiples of 1 Å.
LENGTH_UNITS_IN_ANGSTROM = {
    "pm": 1e-2,
    "Å": 1.0,
    "nm": 1e1,
    "μm": 1e4,
    "mm": 1e7,
    "cm": 1e8,
    "m": 1e10,
}

# Angle units, as multiples of 1 radian.
ANGLE_UNITS_IN_RADIANS = {
    "μrad": 1e-6,
    "mrad": 1e-3,
    "rad": 1.0,
    "deg": math.pi / 180.0,
}

# Accepted spellings (compared in lower case) for each unit above.
_LENGTH_UNIT_ALIASES = {
    **dict.fromkeys(["pm", "picometer", "picometre", "picometers", "picometres"], "pm"),
    **dict.fromkeys(["a", "å", "Å", "ang", "angstrom", "angstroms", "ångström", "ångströms"], "Å"),
    **dict.fromkeys(["nm", "nanometer", "nanometre", "nanometers", "nanometres"], "nm"),
    **dict.fromkeys(
        [
            "um",
            "μm",  # Greek mu
            "µm",  # micro sign
            "micrometer",
            "micrometre",
            "micrometers",
            "micrometres",
            "micron",
            "microns",
        ],
        "μm",
    ),
    **dict.fromkeys(["mm", "millimeter", "millimetre", "millimeters", "millimetres"], "mm"),
    **dict.fromkeys(["cm", "centimeter", "centimetre", "centimeters", "centimetres"], "cm"),
    **dict.fromkeys(["m", "meter", "metre", "meters", "metres"], "m"),
}

_ANGLE_UNIT_ALIASES = {
    **dict.fromkeys(["urad", "μrad", "µrad", "microrad", "microradian", "microradians"], "μrad"),
    **dict.fromkeys(["mrad", "millirad", "milliradian", "milliradians"], "mrad"),
    **dict.fromkeys(["rad", "radian", "radians"], "rad"),
    **dict.fromkeys(["deg", "degree", "degrees", "°"], "deg"),
}

_RECIPROCAL_SUFFIXES = ("^-1", "**-1", "⁻¹", "-1")


class Unit(NamedTuple):
    """A parsed unit: canonical ``base`` (e.g. ``"nm"``), ``kind`` and ``power``.

    ``kind`` is ``"length"`` or ``"angle"``; ``power`` is 1, or -1 for a reciprocal
    length such as ``"1/nm"``.
    """

    base: str
    kind: str
    power: int


def parse_units(units: Any) -> Unit | None:
    """Parse a unit string into a :class:`Unit`.

    Accepts lengths (pm, Å, nm, μm, mm, cm, m), reciprocal lengths written as
    ``"1/nm"``, ``"nm^-1"``, ``"nm**-1"``, ``"nm-1"``, ``"nm⁻¹"`` or
    ``"$\\mathrm{nm}^{-1}$"``, and angles (μrad, mrad, rad, deg), in common
    spellings and any letter case.

    Parameters
    ----------
    units : Any
        The unit string, typically a dataset's ``units`` entry.

    Returns
    -------
    Unit or None
        None if ``units`` is not a string or not a recognized unit (e.g.
        ``"pixels"``), including reciprocal angles.

    Examples
    --------
    >>> parse_units("Å^-1")
    Unit(base='Å', kind='length', power=-1)
    >>> parse_units("mrad")
    Unit(base='mrad', kind='angle', power=1)
    """
    if not isinstance(units, str):
        return None
    u = units.strip().replace("$", "").replace("\\mathrm", "").replace(" ", "")
    u = u.replace("{", "").replace("}", "")
    if not u:
        return None
    # Whole-token lookup before any suffix handling, so e.g. "m" and "mrad" stay distinct.
    angle = _ANGLE_UNIT_ALIASES.get(u.lower())
    if angle is not None:
        return Unit(angle, "angle", 1)
    power = 1
    if u.startswith("1/"):
        u, power = u[2:], -1
    else:
        for suffix in _RECIPROCAL_SUFFIXES:
            if u.endswith(suffix):
                u, power = u[: -len(suffix)], -1
                break
    base = _LENGTH_UNIT_ALIASES.get(u.lower())
    return Unit(base, "length", power) if base is not None else None


def unit_conversion_factor(from_units: str, to_units: str, energy: float | None = None) -> float:
    """Factor that converts a value in ``from_units`` to ``to_units``.

    Converts within lengths, within reciprocal lengths and within angles, and
    between angles and reciprocal lengths through the electron wavelength
    ``lambda`` at ``energy``: ``k = theta / lambda`` (small-angle approximation;
    see the module docstring).

    Parameters
    ----------
    from_units, to_units : str
        Unit strings accepted by :func:`parse_units`.
    energy : float, optional
        Beam energy in eV. Required only to convert between an angle and a
        reciprocal length.

    Returns
    -------
    float
        Multiply a value (or a sampling) in ``from_units`` by this to express it
        in ``to_units``.

    Raises
    ------
    ValueError
        If a unit is not recognized, the units are of incompatible kinds (e.g. a
        length and a reciprocal length), or an angle conversion lacks ``energy``.

    Examples
    --------
    >>> unit_conversion_factor("1/nm", "1/Å")
    0.1
    >>> round(unit_conversion_factor("mrad", "1/Å", energy=300e3), 4)
    0.0508
    """
    src, dst = parse_units(from_units), parse_units(to_units)
    for text, parsed in ((from_units, src), (to_units, dst)):
        if parsed is None:
            raise ValueError(
                f"Unrecognized units {text!r}. Use a length ({', '.join(LENGTH_UNITS_IN_ANGSTROM)}), "
                f"a reciprocal length such as '1/nm' or 'Å^-1', or an angle "
                f"({', '.join(ANGLE_UNITS_IN_RADIANS)})."
            )
    assert src is not None and dst is not None

    # Express both in a common reference: Å for lengths, 1/Å for reciprocal lengths,
    # rad for angles.
    def to_reference(unit: Unit) -> float:
        if unit.kind == "angle":
            return ANGLE_UNITS_IN_RADIANS[unit.base]
        scale = LENGTH_UNITS_IN_ANGSTROM[unit.base]
        return scale if unit.power == 1 else 1.0 / scale

    factor = to_reference(src) / to_reference(dst)
    kinds = {src.kind, dst.kind}
    if src.kind == dst.kind and src.power == dst.power:
        return factor
    if kinds == {"angle", "length"} and -1 in (src.power, dst.power):
        if energy is None:
            raise ValueError(
                f"Converting {from_units!r} to {to_units!r} needs the beam energy: pass energy (eV)."
            )
        if energy <= 0:
            raise ValueError(f"energy must be positive (eV), got {energy}.")
        wavelength = electron_wavelength_angstrom(energy)  # Å
        # rad -> 1/Å divides by the wavelength; 1/Å -> rad multiplies by it.
        return factor / wavelength if src.kind == "angle" else factor * wavelength

    def describe(unit: Unit) -> str:
        if unit.kind == "angle":
            return "an angle"
        return "a length" if unit.power == 1 else "a reciprocal length"

    raise ValueError(
        f"Cannot convert {from_units!r} ({describe(src)}) to {to_units!r} ({describe(dst)})."
    )
