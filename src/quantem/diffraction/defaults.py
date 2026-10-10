"""Shared defaults of the matching, phase and refinement chain.

Every function takes these as keyword arguments, so they can be changed per
call; keeping one value for each across the chain makes the matching, the
phase fit, the strain and the dynamical refinement compare the same peaks
in the same way.
"""

# peak pairing distance between simulated and measured peaks, and the width
# of the polar correlation kernel (1/Angstroms)
PAIR_DISTANCE = 0.05

# intensities are compared as I ** POWER_INTENSITY; 0.25 flattens the
# dynamic range so weak reflections count, 0 compares positions only
POWER_INTENSITY = 0.25

# excitation-error envelope of the kinematical patterns (1/Angstroms);
# about twice the physical width, so orientations halfway between library
# zones keep their intensities
SIGMA_EXCITATION = 0.04

# excitation-error cutoff of the Bloch wave beam list (1/Angstroms)
SG_MAX = 0.1

# simulated reflections weaker than this fraction of the strongest one are
# treated as unobservable and do not count against a candidate
MIN_SIM_INTENSITY_REL = 0.02

# detected peaks (including the direct beam) a pattern needs to be matched
# or refined, and paired reflections a least-squares refinement needs
MIN_NUMBER_PEAKS = 5
MIN_PAIRS = 4


def resolve(value, key: str, *sources: dict | None, default=None):
    """First non-None of an explicit value, metadata entries and a default.

    The refinement stages call this so a parameter left as None inherits
    the value the previous stage used.

    Parameters
    ----------
    value : object
        Explicit value; returned whenever it is not None.
    key : str
        Key looked up in each source.
    *sources : dict | None
        Metadata dicts, searched in order; None sources are skipped, as are
        entries that are None.
    default : object, optional
        Returned when nothing else is found.

    Returns
    -------
    object
        The resolved value.
    """
    if value is not None:
        return value
    for src in sources:
        if src is not None and src.get(key) is not None:
            return src[key]
    return default
