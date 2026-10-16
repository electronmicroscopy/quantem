"""Tests for quantem.diffraction.crystal."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import quat_from_zone_axis


@pytest.fixture
def ti_beta():
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), name="Ti beta")
    xtl.calculate_structure_factors(k_max=1.5)
    return xtl


def test_symmetry_detection(ti_beta):
    assert ti_beta.pointgroup == "m-3m"
    assert ti_beta.laue_group == "m-3m"
    assert ti_beta.sym_quats.shape[0] == 24  # proper rotations of m-3m


def test_hcp_symmetry():
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686))
    assert xtl.pointgroup == "6/mmm"
    assert xtl.sym_quats.shape[0] == 12


def test_bcc_absences(ti_beta):
    # h + k + l odd forbidden in bcc
    parity = ti_beta.hkl.sum(dim=1) % 2
    assert (parity == 0).all()


def test_ring_positions(ti_beta):
    # (110) ring at sqrt(2)/a
    g110 = np.sqrt(2) / 3.31
    assert np.isclose(float(ti_beta.g_len.min()), g110, atol=1e-6)


def test_pseudo_symmetry():
    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    exact = Crystal.from_ase(ortho, pseudo_symmetry_tol=None)
    pseudo = Crystal.from_ase(ortho, pseudo_symmetry_tol=0.01)  # 0.04 A on a 4 A cell
    assert exact.pointgroup_matching == "mmm"
    assert pseudo.pointgroup_matching == "m-3m"
    assert pseudo.sym_quats_matching.shape[0] == 24
    # exact group is retained for reporting/refinement
    assert pseudo.pointgroup == "mmm"


def test_zone_axis_wedge_anchored_001(ti_beta):
    wedge = ti_beta.zone_axis_wedge()
    assert torch.allclose(wedge[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


def test_generate_pattern(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    p = ti_beta.generate_pattern(q, energy_ev=200e3)
    # [001] zone: peaks on a square grid of 110-type spacings
    assert p["qx"].shape[0] > 4
    qr = torch.hypot(p["qx"], p["qy"])
    assert float(qr.min()) > 0.4  # no direct beam
    # pattern symmetric under 90 degree rotation
    rot = torch.stack((-p["qy"], p["qx"]), dim=1)
    orig = torch.stack((p["qx"], p["qy"]), dim=1)
    d = torch.cdist(rot, orig).min(dim=1).values
    assert float(d.max()) < 1e-6


def _images_in_wedge(xtl, n=3000, tol=1e-9):
    """Count, per random direction, its symmetry images inside the wedge."""
    from quantem.diffraction.rotations import quat_to_matrix

    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, 3))
    d = torch.as_tensor(d / np.linalg.norm(d, axis=1, keepdims=True))
    c = xtl.zone_axis_wedge()
    Rs = quat_to_matrix(xtl.sym_quats_matching)
    imgs = torch.einsum("sij,nj->nsi", Rs, d)
    imgs = torch.cat([imgs, -imgs], dim=1).reshape(-1, 3)
    ok = torch.ones(imgs.shape[0], dtype=torch.bool)
    for i in range(3):
        nrm = torch.cross(c[i], c[(i + 1) % 3], dim=0)
        ok &= (imgs @ nrm) * torch.sign(nrm @ c[(i + 2) % 3]) >= -tol
    return ok.reshape(n, -1).sum(dim=1)


@pytest.mark.parametrize(
    "label,spacegroup,symbols,basis,cellpar,laue",
    [
        ("Si", 227, ["Si"], [(0, 0, 0)], [5.43] * 3 + [90] * 3, "m-3m"),
        (
            "pyrite",
            205,
            ["Fe", "S"],
            [(0, 0, 0), (0.385, 0.385, 0.385)],
            [5.42] * 3 + [90] * 3,
            "m-3",
        ),
        ("Ti", 194, ["Ti"], [(1 / 3, 2 / 3, 0.25)], [2.95, 2.95, 4.68, 90, 90, 120], "6/mmm"),
        (
            "CdI2 -3m1",
            164,
            ["Cd", "I"],
            [(0, 0, 0), (1 / 3, 2 / 3, 0.25)],
            [4.24, 4.24, 6.84, 90, 90, 120],
            "-3m",
        ),
        ("Bi R-3m", 166, ["Bi"], [(0, 0, 0.234)], [4.55, 4.55, 11.86, 90, 90, 120], "-3m"),
        (
            "P-31m",
            162,
            ["Cu", "O"],
            [(1 / 3, 2 / 3, 0), (0.4, 0, 0.3)],
            [5.0, 5.0, 7.0, 90, 90, 120],
            "-3m",
        ),
        (
            "ilmenite",
            148,
            ["Fe", "Ti", "O"],
            [(0, 0, 0.355), (0, 0, 0.146), (0.317, 0.023, 0.245)],
            [5.09, 5.09, 14.09, 90, 90, 120],
            "-3",
        ),
        (
            "rutile",
            136,
            ["Ti", "O"],
            [(0, 0, 0), (0.305, 0.305, 0)],
            [4.59, 4.59, 2.96, 90, 90, 90],
            "4/mmm",
        ),
        (
            "Pnma",
            62,
            ["Fe", "C"],
            [(0.18, 0.25, 0.33), (0.04, 0.25, 0.87)],
            [5.0, 6.7, 4.5, 90, 90, 90],
            "mmm",
        ),
    ],
)
def test_wedge_is_fundamental_domain(label, spacegroup, symbols, basis, cellpar, laue):
    from ase.spacegroup import crystal as ase_crystal

    atoms = ase_crystal(symbols, basis=basis, spacegroup=spacegroup, cellpar=cellpar)
    xtl = Crystal.from_ase(atoms, name=label, verbose=False)
    assert xtl.laue_group_matching == laue
    # exactly one symmetry image of every direction lies in the wedge: the
    # wedge covers all of orientation space once (the -3m1 setting used to
    # get a wedge rotated by 30 degrees, covering half the directions twice)
    hits = _images_in_wedge(xtl)
    assert int(hits.min()) == 1 and int(hits.max()) == 1
    labels = xtl.zone_axis_wedge_labels()
    assert len(labels) == 3 and all(len(t) > 2 for t in labels)


def test_wedge_follows_cell_setting():
    # a rotated Cartesian setting moves the symmetry axes; the wedge follows
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    atoms.rotate(37, "z", rotate_cell=True)
    atoms.rotate(20, "x", rotate_cell=True)
    xtl = Crystal.from_ase(atoms, verbose=False)
    hits = _images_in_wedge(xtl)
    assert int(hits.min()) == 1 and int(hits.max()) == 1
    assert xtl.zone_axis_wedge_labels(mathtext=False) == ["[001]", "[011]", "[111]"]


def test_pseudo_symmetry_default_and_warning():
    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    xtl = Crystal.from_ase(ortho, verbose=False)  # default tolerance 1% of the cell
    assert xtl.pointgroup == "mmm" and xtl.pointgroup_matching == "m-3m"
    assert xtl.pseudo_symmetry_report["intensity_mismatch"] < 0.05
    msg = xtl.matching_symmetry_warning()
    assert msg is not None and "pseudo_symmetry_tol=None" in msg
    # the pseudo group's operators are exact rotations (orthonormalized), so
    # its wedge is a fundamental domain up to the cell distortion
    hits = _images_in_wedge(xtl, tol=1e-3)
    assert int(hits.min()) >= 1
    exact = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    assert exact.matching_symmetry_warning() is None


def test_pseudo_symmetry_dimensionless_and_intensity_check():
    # an almost body-centered cell: the center atom 0.002 A off (0.5, 0.5, 0.5)
    # is body centered at the default tolerance, and its 100/010/001
    # patterns are identical within any measurable intensity
    almost_bcc = Atoms(
        "Fe2", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5005]], cell=[4.0, 4.0, 4.0], pbc=True
    )
    xtl = Crystal.from_ase(almost_bcc, verbose=False)
    assert xtl.pointgroup_matching == "m-3m"
    assert xtl.pseudo_symmetry_report["intensity_mismatch"] < 1e-3
    # the same cell at an unmeasurably tight distance tolerance keeps its
    # own (lower) symmetry; the tolerance is a fraction of the lattice
    tight = Crystal.from_ase(almost_bcc, pseudo_symmetry_tol=1e-7, verbose=False)
    assert tight.pointgroup_matching == tight.pointgroup
    # a candidate whose intensities do not match within the intensity
    # tolerance is rejected and the cell keeps its own symmetry
    strict = Crystal.from_ase(almost_bcc, pseudo_symmetry_intensity_tol=1e-9, verbose=False)
    assert strict.pseudo_symmetry_report.get("candidate") == "m-3m"
    assert strict.pseudo_symmetry_report.get("rejected") is True
    assert strict.pointgroup_matching == strict.pointgroup
    assert "rejected" in strict.symmetry_summary()


def _l10(other: str, a: float = 3.58) -> Atoms:
    """Two species ordered in alternating (001) layers of an fcc lattice.

    The lattice stays cubic and every atom sits exactly on its site, so no
    relaxation of the positions recovers the cubic parent: only the
    diffracted intensities can say whether the ordering is visible.
    """
    at = Atoms(
        "Ni4",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        cell=[a, a, a],
        pbc=True,
    )
    at.symbols = ["Ni", "Ni", other, other]
    return at


def test_pseudo_symmetry_from_weak_ordering():
    """Ordering of species that scatter alike is found through the lattice.

    Transition metals next to each other in the periodic table (the Ni, Co,
    Mn of a cathode) give superlattice reflections far too weak to index, so
    the orientation library must fold the variants together. The relaxed
    position search cannot find this: the atoms are already where they
    belong and only the species differ.
    """
    weak = Crystal.from_ase(_l10("Co"), verbose=False)
    assert weak.pointgroup == "4/mmm"
    assert weak.pointgroup_matching == "m-3m"
    assert weak.sym_quats_matching.shape[0] == 3 * weak.sym_quats.shape[0]
    assert weak.pseudo_symmetry_report["route"] == "lattice"
    assert weak.pseudo_symmetry_report["intensity_mismatch"] < 0.01

    # a light partner makes the same ordering plainly visible, and the
    # candidate is rejected
    strong = Crystal.from_ase(_l10("Li"), verbose=False)
    assert strong.pointgroup_matching == strong.pointgroup
    assert strong.pseudo_symmetry_report["rejected"] is True
    assert strong.pseudo_symmetry_report["intensity_mismatch"] > 0.1

    # the intensity tolerance is the decision, and it is the user's
    borderline = Crystal.from_ase(_l10("Al"), verbose=False)
    assert borderline.pointgroup_matching == borderline.pointgroup
    loose = Crystal.from_ase(_l10("Al"), pseudo_symmetry_intensity_tol=0.1, verbose=False)
    assert loose.pointgroup_matching == "m-3m"


def test_true_symmetry_cells_are_unchanged():
    """The lattice route must not disturb cells that are already at their
    lattice's symmetry, nor accept a lattice symmetry the structure breaks."""
    from ase.build import bulk

    for atoms, pg in (
        (bulk("Si", "diamond", a=5.43), "m-3m"),
        (bulk("Ti", "hcp", a=2.95, c=4.686), "6/mmm"),
        (bulk("Ti", "bcc", a=3.26, cubic=True), "m-3m"),
    ):
        xtl = Crystal.from_ase(atoms, verbose=False)
        assert xtl.pointgroup_matching == pg
        assert xtl.sym_quats_matching.shape[0] == xtl.sym_quats.shape[0]

    # corundum sits on a hexagonal lattice but its structure is only -3m;
    # the lattice route proposes 6/mmm and the intensities reject it
    from ase.spacegroup import crystal as ase_crystal

    al2o3 = ase_crystal(
        ("Al", "O"),
        basis=[(0, 0, 0.3522), (0.3064, 0, 0.25)],
        spacegroup=167,
        cellpar=[4.7607, 4.7607, 12.9947, 90, 90, 120],
    )
    xtl = Crystal.from_ase(al2o3, verbose=False)
    assert xtl.pointgroup_matching == "-3m"
    assert xtl.pseudo_symmetry_report["candidate"] == "6/mmm"
    assert xtl.pseudo_symmetry_report["rejected"] is True


def test_projected_rotation_order():
    """Apparent zero-layer symmetry, which limits in-plane indexing."""
    bcc = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), verbose=False
    ).calculate_structure_factors(k_max=1.5)
    hcp = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False
    ).calculate_structure_factors(k_max=1.5)

    def cartesian(xtl, uvw):
        d = torch.as_tensor(np.asarray(uvw, dtype=float), dtype=torch.float64) @ xtl.lat_real
        return (d / torch.linalg.norm(d)).numpy()

    # the zero-layer net of {110} along <111> is hexagonal, so the pattern
    # repeats every 60 degrees while the crystal repeats every 120
    assert bcc.projected_rotation_order(cartesian(bcc, (1, 1, 1))) == 6
    assert bcc.projected_rotation_order(cartesian(bcc, (0, 0, 1))) == 4
    assert bcc.projected_rotation_order(cartesian(bcc, (0, 1, 1))) == 2
    # a general zone axis keeps the two-fold that Friedel's law provides
    assert bcc.projected_rotation_order(cartesian(bcc, (1, 2, 3))) == 2
    assert hcp.projected_rotation_order(cartesian(hcp, (0, 0, 1))) == 6
    assert hcp.projected_rotation_order(cartesian(hcp, (1, 0, 0))) == 2

    # vectorized over a stack
    axes = np.stack([cartesian(bcc, u) for u in ((1, 1, 1), (0, 0, 1), (0, 1, 1))])
    assert list(bcc.projected_rotation_order(axes)) == [6, 4, 2]


def test_projected_order_matches_pattern_degeneracy():
    """The reported order is the rotation that leaves the pattern unchanged."""
    from quantem.diffraction.rotations import (
        qmult,
        qnormalize,
        quat_from_axis_angle,
        quat_from_zone_axis,
    )

    bcc = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), verbose=False
    ).calculate_structure_factors(k_max=1.5)
    d = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) @ bcc.lat_real
    axis = d / torch.linalg.norm(d)
    n = bcc.projected_rotation_order(axis.numpy())
    q = quat_from_zone_axis(axis)
    beam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    spun = qnormalize(qmult(quat_from_axis_angle(beam, torch.tensor(2 * np.pi / n)), q))

    a = bcc.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    b = bcc.generate_pattern(spun, energy_ev=200e3, sigma_excitation=0.02)
    pa = torch.stack([a["qx"], a["qy"]], dim=1)
    pb = torch.stack([b["qx"], b["qy"]], dim=1)
    assert pa.shape == pb.shape
    # every peak of one pattern sits on a peak of the other, same intensity
    dist = torch.cdist(pa, pb)
    dmin, j = dist.min(dim=1)
    assert float(dmin.max()) < 1e-6
    rel = (a["intensity"] - b["intensity"][j]).abs().max() / a["intensity"].max()
    assert float(rel) < 1e-6


_LFSO_PRISTINE_CIF = """data_
_cell_length_a 5.1749
_cell_length_b 8.9426
_cell_length_c 5.1721
_cell_angle_alpha 90
_cell_angle_beta 109.697
_cell_angle_gamma 90
_symmetry_space_group_name_H-M C2/m
loop_
_symmetry_equiv_pos_as_xyz
 'x, y, z'
 '-x, y, -z'
 'x, -y, z'
 '-x, -y, -z'
 'x+1/2, y+1/2, z'
 '-x+1/2, y+1/2, -z'
 'x+1/2, -y+1/2, z'
 '-x+1/2, -y+1/2, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0 0.3380 0.5 0.389
Li1 Li 0 0.3380 0.5 0.611
Sb1 Sb 0 0 0.5 0.360
Fe2 Fe 0 0 0.5 0.640
Li2 Li 0 0.5 0 1
Li3 Li 0 0.1662 0 1
O1 O 0.7359 0.5 0.2706 1
O2 O 0.7665 0.8415 0.2722 1
"""


def test_from_cif_keeps_partial_occupancy(tmp_path):
    # ASE reads a shared site as its majority species alone: this structure
    # would load with no Sb at all, and Sb is its strongest scatterer
    path = tmp_path / "lfso_pristine.cif"
    path.write_text(_LFSO_PRISTINE_CIF)
    xtl = Crystal.from_cif(path, verbose=False)
    content: dict[str, float] = {}
    for s, f in zip(xtl.atoms.get_chemical_symbols(), xtl.occupancy.numpy()):
        content[s] = content.get(s, 0.0) + float(f)
    assert content["Sb"] == pytest.approx(0.72, abs=1e-6)
    assert content["Fe"] == pytest.approx(2.836, abs=1e-6)
    assert content["Li"] == pytest.approx(8.444, abs=1e-6)
    assert content["O"] == pytest.approx(12.0, abs=1e-6)
    assert xtl.spacegroup.startswith("C2/m")


def test_pseudo_symmetry_names_the_breaking_reflection(tmp_path):
    # the three 120 degree twin variants of the honeycomb-ordered cell differ
    # only by the (020) superstructure reflection at ~0.07 of the strongest;
    # at the default tolerance that rejects the layered parent, and the
    # report must name the reflection responsible
    path = tmp_path / "lfso_pristine.cif"
    path.write_text(_LFSO_PRISTINE_CIF)
    strict = Crystal.from_cif(path, verbose=False)
    rep = strict.pseudo_symmetry_report
    assert rep["rejected"] and rep["candidate"] == "-3m"
    assert 0.05 < rep["intensity_mismatch"] < 0.1
    (h0, i0, h1, i1) = rep["broken_by"]
    assert tuple(abs(v) for v in h0) == (0, 2, 0)
    assert "broken by" in strict.symmetry_summary()


def test_pseudo_symmetry_from_parent_lattice(tmp_path):
    # the honeycomb superstructure sits on a layered R-3m parent, itself on a
    # rocksalt parent; neither parent's rotations map the monoclinic cell onto
    # itself, so only the parent-lattice route can find them
    path = tmp_path / "lfso_pristine.cif"
    path.write_text(_LFSO_PRISTINE_CIF)
    layered = Crystal.from_cif(path, pseudo_symmetry_intensity_tol=0.1, verbose=False)
    assert layered.pointgroup_matching == "-3m"
    assert layered.sym_quats_matching.shape[0] == 6
    assert "parent lattice" in layered.pseudo_symmetry_report["route"]
    # the layer normal is ~[103] in the monoclinic cell
    assert any("103" in lab for lab in layered.zone_axis_wedge_labels(mathtext=False))

    rocksalt = Crystal.from_cif(path, pseudo_symmetry_intensity_tol=0.4, verbose=False)
    assert rocksalt.pointgroup_matching == "m-3m"
    assert rocksalt.sym_quats_matching.shape[0] == 24
    # the rocksalt parent is broken only by the (001) layer-ordering reflection
    h0, i0, h1, i1 = rocksalt.pseudo_symmetry_report["broken_by"]
    assert tuple(abs(v) for v in h0) == (0, 0, 1)
    assert 0.3 < rocksalt.pseudo_symmetry_report["intensity_mismatch"] < 0.35


_LFSO_CHARGED_CIF = """data_
_cell_length_a 5.04848
_cell_length_b 5.04848
_cell_length_c 9.4279
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M P-31c
loop_
_symmetry_equiv_pos_as_xyz
 'x, y, z'
 '-x, -y, -z'
 '-x+y, -x, z'
 '-x+y, y, -z+1/2'
 '-y, -x, -z+1/2'
 '-y, x-y, z'
 'y, -x+y, -z'
 'y, x, z+1/2'
 'x-y, -y, z+1/2'
 'x-y, x, -z'
 '-x, -x+y, z+1/2'
 'x, x-y, -z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Me1 Li 0.3333333 0.6666667 0.75 0.673
Me11 Sb 0.3333333 0.6666667 0.75 0.327
Me2 Li 0.3333333 0.6666667 0.25 0.327
Me22 Sb 0.3333333 0.6666667 0.25 0.673
Me3 Fe 0 0 0.75 1
O1 O 0.7409 0.7329 0.8660 1
"""


def test_hexagonal_pseudo_symmetry_can_be_adopted(tmp_path):
    # a hexagonal cell given to five decimals reproduces rotation products to
    # only ~1e-6; a closure test that strict rejected every hexagonal parent
    # group whatever the intensity tolerance
    path = tmp_path / "lfso_charged.cif"
    path.write_text(_LFSO_CHARGED_CIF)
    strict = Crystal.from_cif(path, pseudo_symmetry_intensity_tol=0.4, verbose=False)
    assert strict.pointgroup_matching == "-3m"
    assert strict.pseudo_symmetry_report["candidate"] == "6/mmm"
    loose = Crystal.from_cif(path, pseudo_symmetry_intensity_tol=1.0, verbose=False)
    assert loose.pointgroup_matching == "6/mmm"
    assert loose.sym_quats_matching.shape[0] == 12


def test_direction_vector():
    cubic = Crystal.from_ase(bulk("Au", "fcc", a=4.08, cubic=True), verbose=False)
    v = cubic.direction_vector([1, 1, 0])
    assert torch.allclose(v, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64) / np.sqrt(2))
    hcp = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    a1 = hcp.lat_real[0] / torch.linalg.norm(hcp.lat_real[0])
    c = hcp.lat_real[2] / torch.linalg.norm(hcp.lat_real[2])
    assert torch.allclose(hcp.direction_vector([2, -1, -1, 0]), a1)
    assert torch.allclose(hcp.direction_vector([0, 0, 0, 1]), c)
    # 3- and 4-index forms of the same direction agree
    assert torch.allclose(hcp.direction_vector([1, 0, 0]), hcp.direction_vector([2, -1, -1, 0]))
    with pytest.raises(ValueError):
        hcp.direction_vector([1, 0])


def test_miller_bravais_round_trip():
    from quantem.diffraction.crystal import miller_bravais_to_miller, miller_to_miller_bravais

    assert miller_to_miller_bravais([1, 0, 0]).tolist() == [2, -1, -1, 0]
    assert miller_to_miller_bravais([1, 1, 0]).tolist() == [1, 1, -2, 0]
    assert miller_to_miller_bravais([0, 0, 1]).tolist() == [0, 0, 0, 1]
    assert miller_bravais_to_miller([2, -1, -1, 0]).tolist() == [1, 0, 0]
    rng = np.random.default_rng(0)
    uvw = rng.integers(-4, 5, size=(200, 3))
    uvw = uvw[np.abs(uvw).sum(axis=1) > 0]
    uvtw = miller_to_miller_bravais(uvw)
    assert np.all(uvtw[:, 2] == -(uvtw[:, 0] + uvtw[:, 1]))
    back = miller_bravais_to_miller(uvtw)
    reduced = uvw // np.gcd.reduce(np.abs(uvw), axis=1)[:, None]
    assert np.array_equal(back, reduced)


def test_format_direction():
    from quantem.diffraction.crystal import format_direction

    bar = "̅"
    assert format_direction(None) == ""
    assert format_direction([1, -1, 0], mathtext=False) == "[11" + bar + "0]"
    assert format_direction([1, -1, 0]) == "[1$\\bar{1}$0]"
    assert format_direction([1, 0, 0], hexagonal=True, mathtext=False) == (
        "[21" + bar + "1" + bar + "0]"
    )


def test_spglib_no_deprecation_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    assert xtl.pointgroup == "6/mmm"


def test_generate_pattern_validates_excitation_model(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="excitation_model"):
        ti_beta.generate_pattern(q, excitation_model="slabb")
    with pytest.raises(ValueError, match="thickness_A"):
        ti_beta.generate_pattern(q, excitation_model="slab")


def test_generate_pattern_foil_normal():
    from quantem.diffraction.illumination import excitation_coefficients
    from quantem.diffraction.rotations import qrotate

    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    c_axis = xtl.direction_vector([0, 0, 0, 1])

    # foil normal along the beam: identical to the default geometry
    q0 = quat_from_zone_axis(c_axis, in_plane_deg=10.0)
    p0 = xtl.generate_pattern(q0, energy_ev=200e3)
    p1 = xtl.generate_pattern(q0, energy_ev=200e3, foil_normal=(0, 0, 0, 1))
    for key in ("qx", "qy", "intensity", "s_g"):
        assert torch.allclose(p0[key], p1[key], atol=1e-12)

    # a tilted flake: each spot sits where its rod meets the Ewald sphere
    tilt = xtl.direction_vector([0, 1, -1, 6])
    q = quat_from_zone_axis(tilt, in_plane_deg=10.0)
    p = xtl.generate_pattern(q, energy_ev=200e3, foil_normal=(0, 0, 0, 1))
    assert p["qx"].shape[0] > 5
    n_lab = qrotate(q, c_axis[None])[0]
    assert float(n_lab[2]) < 0.999  # really tilted
    g_lab = qrotate(q, p["hkl"].to(torch.float64) @ xtl.lat_recip)
    spot = g_lab - p["s_g"][:, None] * n_lab[None]
    assert torch.allclose(spot[:, :2], torch.stack([p["qx"], p["qy"]], dim=1), atol=1e-12)
    s_spot, _, _ = excitation_coefficients(spot, 200e3)
    # first order in s_g: the residual is far below the excitation error
    s_g = p["s_g"].numpy()
    big = np.abs(s_g) > 1e-3
    assert np.all(np.abs(s_spot[big]) < 0.05 * np.abs(s_g[big]))
    # and moves off the projection of g, which is where it sits by default
    assert float((spot[:, :2] - g_lab[:, :2]).abs().max()) > 1e-4


def test_wk_factor_without_thermal_motion():
    from quantem.diffraction.wk_scattering_factors import compute_WK_factor

    g = np.linspace(0.0, 3.0, 61)
    f0 = compute_WK_factor(g, 29, 200e3, thermal_sigma=None)
    assert f0.dtype == np.complex128 and f0.shape == g.shape
    assert np.all(np.isfinite(f0))
    # the phonon absorption vanishes continuously as the displacement -> 0
    f_small = compute_WK_factor(g, 29, 200e3, thermal_sigma=1e-3)
    assert np.allclose(f_small, f0, rtol=1e-3, atol=1e-7)
    # elastic part is monotonic in g, imaginary part positive
    assert np.all(np.diff(f0.real) < 0)
    assert np.all(compute_WK_factor(g, 29, 200e3, thermal_sigma=0.08).imag > 0)
