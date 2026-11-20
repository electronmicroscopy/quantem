"""SystematicRowMap: phase and in-plane angle from Friedel-paired rows."""

import numpy as np
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import Crystal, SystematicRowMap


def test_row_indexing():
    lif = Crystal.from_ase(bulk("LiF", "rocksalt", a=4.027, cubic=True), name="LiF", verbose=False)
    lif.calculate_structure_factors(k_max=1.0)
    li = Crystal.from_ase(bulk("Li", "bcc", a=3.51, cubic=True), name="Li", verbose=False)
    li.calculate_structure_factors(k_max=1.0)

    def pair(k, deg, inten):
        # deg is measured from the row axis toward the column axis
        g = k * np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
        return [[g[0], g[1], inten], [-g[0], -g[1], inten]]

    k200 = 2 / 4.027
    k110 = np.sqrt(2) / 3.51
    cells = [
        np.array(pair(k200, 30.0, 5.0)),  # LiF (200) at 30 degrees
        np.array(pair(k110, 100.0, 4.0) + pair(k200, 10.0, 1.0)),  # Li wins, two directions
        np.array([[0.3, 0.1, 9.0]]),  # unpaired peak, no row
        np.array(pair(0.3, 45.0, 2.0)),  # paired but on no ring
    ]
    peaks = Vector.from_data(
        [cells[:2], cells[2:]], fields=["qx", "qy", "intensity"], units=["A^-1"] * 2 + ["counts"]
    )
    rm = SystematicRowMap.from_vectors(peaks, [lif, li], k_min=0.2, k_max=0.6).index(k_tol=0.01)

    phase = rm.phase_index()
    assert phase.tolist() == [[0, 1], [-1, -1]], phase
    assert rm.ring_label[rm.ring[0, 0, 0]] == "(200)"
    angle, _ = rm.in_plane_angle("LiF")
    assert np.isclose(angle[0, 0], 30.0) and np.isclose(angle[0, 1], 10.0), angle
    assert rm.num_directions()[0, 1] == 2
    assert rm.intensity[1, 0, 0] == 0, "an unpaired peak must not form a row"
    assert rm.phase[1, 1, 0] == -1 and rm.intensity[1, 1, 0] > 0, "off-ring row kept unassigned"
    single = SystematicRowMap.from_vectors(peaks, [lif, li], k_min=0.2, k_max=0.6)
    single.index(k_tol=0.01, require_pair=False)
    assert single.intensity[1, 0, 0] == 9.0, "require_pair=False keeps single peaks"
