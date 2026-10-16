import numpy as np
import pytest
import torch

from quantem.diffraction import Crystal, ReverseMonteCarlo
from quantem.diffraction.reverse_monte_carlo import cubic_rotations

CIF = """data_VNb
_cell_length_a 3.2
_cell_length_b 3.2
_cell_length_c 3.2
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'I m -3 m'
_symmetry_Int_Tables_number 229
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Nb1 Nb 0 0 0 0.4
V1 V 0 0 0 0.3
Zr1 Zr 0 0 0 0.3
"""


@pytest.fixture
def rmc(tmp_path):
    path = tmp_path / "vnb.cif"
    path.write_text(CIF)
    rng = np.random.default_rng(1)
    images = [rng.random((64, 64)) + 1.0 for _ in range(2)]
    out = ReverseMonteCarlo.from_images(
        images, zone_axes=[(0, 0, 1), (0, 1, 1)], sampling=0.02, bin_factor=2
    )
    out.set_crystal(Crystal.from_cif(path, verbose=False))
    out.geometry = dict(
        centers=[np.array([32.0, 32.0])] * 2,
        matrices=[np.eye(2) / 0.02, np.array([[0.0, -1.0], [1.0, 0.0]]) / 0.02],
        tilts=[np.zeros(2), np.array([0.01, 0.0])],
    )
    out.set_mask(bragg_radius=0.06, q_max=0.6, center_radius=0.1, edge_px=2)
    out.build_supercell(cells=4, seed=0, device="cpu")
    out.fit_background()
    return out


def test_cubic_rotations():
    ops = cubic_rotations()
    assert ops.shape == (24, 3, 3)
    assert np.allclose([np.linalg.det(o) for o in ops], 1)
    assert len({o.tobytes() for o in ops}) == 24


def test_ternary_site_and_composition(rmc):
    assert rmc.species == ["Nb", "V", "Zr"]
    assert np.allclose(rmc.concentrations, [0.4, 0.3, 0.3])
    assert len(rmc.site_x) == 2 * 4**3
    counts = np.bincount(rmc.species_index, minlength=3)
    assert counts.sum() == 128
    assert np.all(np.abs(counts - 128 * rmc.concentrations) <= 1)


def _score(rmc, d_fr, d_fi):
    keep = rmc._keep / len(rmc.site_x)
    d_int = (2 * (rmc._Fr * d_fr + rmc._Fi * d_fi) + d_fr**2 + d_fi**2) * keep
    dm = rmc._scale * rmc._u * rmc._read(d_int[:, rmc._sym_index].mean(dim=1))
    return float((rmc._w * (dm**2 - 2 * rmc._r * dm)).sum())


def test_swap_score_matches_recompute(rmc):
    """The incremental loss change of one swap equals the loss after recomputing F from scratch."""
    loss0 = rmc._update_residual()
    spec = rmc.species_index
    j1 = int(np.nonzero(spec == 0)[0][0])
    j2 = int(np.nonzero(spec == 1)[0][0])
    c1, s1 = rmc._phases(rmc._positions(np.array([j1])))
    c2, s2 = rmc._phases(rmc._positions(np.array([j2])))
    df = rmc._fs[1] - rmc._fs[0]
    dL = _score(rmc, df * (c1 - c2), df * (s2 - s1))
    spec[j1], spec[j2] = 1, 0
    rmc._recompute_F()
    assert rmc._update_residual() - loss0 == pytest.approx(dL, rel=1e-3, abs=1e-4 * loss0)


def test_displacement_score_matches_recompute(rmc):
    loss0 = rmc._update_residual()
    j = np.array([5])
    new = np.array([[1, -1, 1]])
    co, so = rmc._phases(rmc._positions(j))
    cn, sn = rmc._phases(rmc._positions(j, new))
    f = rmc._fs[int(rmc.species_index[5])]
    dL = _score(rmc, f * (cn - co), f * (so - sn))
    rmc.displacement[5] = new[0]
    rmc._recompute_F()
    assert rmc._update_residual() - loss0 == pytest.approx(dL, rel=1e-3, abs=1e-4 * loss0)


def test_run_lowers_loss_and_keeps_composition(rmc):
    counts = np.bincount(rmc.species_index)
    rmc.run(n_sweeps=3, batch=8, progress=False)
    assert np.array_equal(np.bincount(rmc.species_index), counts)
    assert rmc.loss_history[-1] <= rmc.loss_history[0] + 1e-6


def test_warren_cowley_random_is_near_zero(rmc):
    sro = rmc.warren_cowley(n_shells=2)
    assert np.allclose(sro["radius"], [3.2 * np.sqrt(3) / 2, 3.2], atol=1e-3)
    assert sro["alpha"].shape == (2, 3, 3)
    assert np.all(np.abs(sro["alpha"]) < 0.35)


def test_coarse_diffuse_grid_matches_sections(rmc):
    fine = rmc.diffuse_grid()
    coarse = rmc.diffuse_grid(max_size=rmc.grid_size // 2)
    assert coarse.shape[0] == fine.shape[0] // 2
    a, _, d = rmc.diffuse_section((0, 0, 1), extent=1.0, grid=fine)
    b, _, _ = rmc.diffuse_section((0, 0, 1), extent=1.0, grid=coarse)
    assert np.allclose(a[d > 0.2], b[d > 0.2], rtol=1e-4)  # no displacements: identical


def test_mask_is_zero_on_bragg_peaks(rmc):
    w = rmc.mask["w"][0]
    b = rmc.bin_factor
    for r, c in rmc.bragg_positions(0):
        r, c = int(r // b), int(c // b)
        if 0 <= r < w.shape[0] and 0 <= c < w.shape[1]:
            assert w[r, c] < 0.3


def test_sro_section_random_is_near_laue(rmc):
    img, _, node_dist = rmc.diffuse_section((0, 0, 1), extent=1.0, smooth=True)
    between = img[node_dist > 0.2]
    assert 0.5 < between.mean() < 1.5


def test_omega_embryo_is_a_collapsed_row_and_scores_exactly(rmc):
    sites, new = rmc._omega_proposals(4)
    assert sites.shape[1] == 3 and len(set(sites.ravel())) == sites.size
    xc = rmc.site_x[sites] // rmc.refine
    step = np.mod(xc[:, 1] - xc[:, 0], rmc.cells * 2)
    step = np.where(step > rmc.cells, step - 2 * rmc.cells, step)
    assert np.all(np.abs(step) == 1)  # nearest neighbours along <111>
    vec = new
    assert np.all(vec[:, 0] == 0) and np.all(vec[:, 1] == -vec[:, 2])

    loss0 = rmc._update_residual()
    d_fr = torch.zeros(len(rmc._needed))
    d_fi = torch.zeros(len(rmc._needed))
    for k in range(3):
        j = sites[:1, k]
        co, so = rmc._phases(rmc._positions(j))
        cn, sn = rmc._phases(rmc._positions(j, new[:1, k]))
        f = rmc._fs[int(rmc.species_index[j[0]])]
        d_fr += (f * (cn - co))[0]
        d_fi += (f * (so - sn))[0]
    dL = _score(rmc, d_fr[None], d_fi[None])
    rmc.displacement[sites[0]] = new[0]
    rmc._recompute_F()
    assert rmc._update_residual() - loss0 == pytest.approx(dL, rel=1e-3, abs=1e-4 * loss0)


def test_fitted_envelope_does_not_raise_loss(rmc):
    loss_measured = rmc._update_residual()
    rmc.envelope = "fitted"
    rmc._setup_forward()
    rmc._solve_linear(rmc._model_diffuse())
    assert rmc._update_residual() <= loss_measured * (1 + 1e-6)
    assert all(np.all(p >= 0) for p in rmc._env_p)


def test_static_b_cap_holds(rmc):
    rmc.run(n_sweeps=3, batch=8, omega_fraction=1.0, max_static_b=0.2, progress=False)
    assert rmc.static_b() <= 0.2 + 1e-9
    assert len(rmc._omega_vectors) == 16  # two amplitudes x eight <111> senses


def test_size_effect_chi_is_odd_about_bragg_nodes(rmc):
    a = rmc.lattice_parameter
    g = np.array([1.0, 1.0, 0.0]) / a  # allowed BCC reflection
    kappa = np.array([[0.02, 0.005, -0.01], [0.0, 0.03, 0.01]])
    plus = rmc._size_chi(g + kappa)
    minus = rmc._size_chi(g - kappa)
    assert np.all(np.abs(plus) > 0)
    assert np.allclose(plus, -minus, rtol=0.2)
    far = rmc._size_chi(g + 4 * kappa)
    assert np.all(np.abs(far) < np.abs(plus))  # grows toward the node


def test_size_effect_radii_orders_species(rmc):
    rmc.set_size_effect("radii")
    eta = dict(zip(rmc.species, rmc.size_eta))
    assert eta["V"] < eta["Nb"] < eta["Zr"]
    assert abs((rmc.concentrations * rmc.size_eta).sum()) < 1e-12


def test_autoserialize_round_trip_rebuilds_model(rmc, tmp_path):
    from quantem.core.io import load

    rmc.envelope = "fitted"
    rmc._setup_forward()
    rmc._solve_linear(rmc._model_diffuse())
    loss = rmc._update_residual()
    model = rmc.model_images()
    path = tmp_path / "rmc.zip"
    rmc.save(path, mode="o", skip=rmc.DERIVED_ATTRIBUTES)
    back = load(path)
    assert back._update_residual() == pytest.approx(loss, rel=1e-5)
    for a, b in zip(model, back.model_images()):
        assert np.allclose(a, b, rtol=1e-4, atol=1e-6)
    assert np.array_equal(back.species_index, rmc.species_index)


def test_displacement_correlations_see_omega(rmc):
    sites, new = rmc._omega_proposals(8)
    rmc.displacement[sites.reshape(-1)] = new.reshape(-1, 3)
    c = rmc.displacement_correlations(n_shells=2)
    assert c["longitudinal"][0] < 0  # collapsing nearest-neighbour pairs move toward each other
    dist = rmc.displacement_distributions()
    assert set(dist) == {"<100>", "<110>", "<111>"}


def test_random_displacement_scores_exactly_and_stays_bounded(rmc):
    loss0 = rmc._update_residual()
    sites, new = rmc._random_proposals(16)
    assert np.all(np.abs(new) <= rmc._max_steps)
    assert np.all(np.any(new[:, 0] != rmc.displacement[sites[:, 0]], axis=1))
    j = sites[:1, 0]
    co, so = rmc._phases(rmc._positions(j))
    cn, sn = rmc._phases(rmc._positions(j, new[:1, 0]))
    f = rmc._fs[int(rmc.species_index[j[0]])]
    dL = _score(rmc, f * (cn - co), f * (so - sn))
    rmc.displacement[j[0]] = new[0, 0]
    rmc._recompute_F()
    assert rmc._update_residual() - loss0 == pytest.approx(dL, rel=1e-3, abs=1e-4 * loss0)


def test_shell_labels(rmc):
    c = rmc.displacement_correlations(n_shells=4)
    assert c["shell"] == ["1/2<111>", "<100>", "<110>", "1/2<311>"]


def test_set_mask_edge_px_zero_keeps_detector(rmc):
    rmc.set_mask(bragg_radius=0.06, q_max=0.6, center_radius=0.1, edge_px=0)
    assert all(np.any(w > 0.5) for w in rmc.mask["w"])


def test_bloch_envelope_needs_fit_thickness(rmc):
    with pytest.raises(RuntimeError, match="fit_thickness"):
        rmc.set_envelope("bloch")
    assert rmc.envelope == "measured"
    with pytest.raises(ValueError, match="unknown envelope"):
        rmc.set_envelope("nope")
    assert rmc.envelope == "measured"


def test_fit_size_effect_runs_on_model_device(rmc):
    out = rmc.fit_size_effect(verbose=False)
    assert set(out["eta"]) == set(rmc.species)
    assert out["loss"][1] <= out["loss"][0] * (1 + 1e-6)
    assert abs((rmc.concentrations * rmc.size_eta).sum()) < 1e-12


def _synthetic_pattern(rmc, zone, center, theta, shape=(160, 160)):
    """Gaussian spots at the kinematic intensities on a broad halo about the direct beam."""
    from quantem.diffraction.reverse_monte_carlo import _rot2

    _, g2, inten = rmc._zone_reflections(zone, 1.6)
    A = _rot2(theta) / rmc.sampling
    pos = np.vstack([center, np.asarray(center) + g2 @ A.T])
    inten = np.concatenate([[2 * inten.max()], inten])  # direct beam first
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    im = 50.0 * np.exp(-((rr - center[0]) ** 2 + (cc - center[1]) ** 2) / (2 * 40.0**2))
    for (pr, pc), i in zip(pos, inten / inten.max()):
        im += 1e3 * i * np.exp(-((rr - pr) ** 2 + (cc - pc) ** 2) / (2 * 1.2**2))
    return im + 1.0


def test_fit_geometry_on_synthetic_lattice(tmp_path):
    path = tmp_path / "vnb.cif"
    path.write_text(CIF)
    zones = [(0, 0, 1), (0, 1, 1)]
    centers = [np.array([78.3, 81.6]), np.array([80.5, 79.2])]
    out = ReverseMonteCarlo.from_images(
        [np.zeros((160, 160))] * 2, zone_axes=zones, sampling=0.03, bin_factor=4
    )
    out.set_crystal(Crystal.from_cif(path, verbose=False))
    out.images = [
        _synthetic_pattern(out, z, c, th).astype(np.float32)
        for z, c, th in zip(zones, centers, (0.3, -0.7))
    ]
    out.fit_geometry(scale_range=(0.97, 1.03), verbose=False)
    geo = out.geometry
    for c, c_fit in zip(centers, geo["centers"]):
        assert np.allclose(c_fit, c, atol=0.3)
    # the halo biases the peak centroids slightly on so small a detector
    assert out.lattice_parameter == pytest.approx(3.2, rel=1e-2)
    assert all(np.rad2deg(np.linalg.norm(t)) < 0.5 for t in geo["tilts"])
    assert all(rms < 0.3 for rms in geo["rms_px"])
    assert len(geo["_inten_kin"]) == len(geo["excitation_width"]) == 2

    out.fit_thickness(
        thickness=(100.0, 200.0), step=50.0, tilt_range_deg=0.0, k_max=1.0, depth_samples=4,
        verbose=False,
    )  # fmt: skip
    assert out.geometry["thickness_k_max"] == 1.0
    assert all(g is not None for g in out.geometry["bloch_g"])

    out.set_mask(bragg_radius=0.08, q_max=1.0, center_radius=0.15, edge_px=4)
    out.build_supercell(cells=3, seed=0, device="cpu")
    out.fit_background()
    for envelope in ("kinematic", "bloch"):
        out.set_envelope(envelope)
        assert out.envelope == envelope
        assert np.isfinite(out._update_residual())


def test_shell_correlations_match_warren_cowley(rmc):
    sc = rmc.shell_correlations(n_shells=4)
    sro = rmc.warren_cowley(n_shells=4)
    ratio = sc["ratio"]
    np.testing.assert_allclose(sc["radius"], sro["radius"])
    # pair counts are symmetric, and an unlike pair has ratio = 1 - alpha
    np.testing.assert_allclose(ratio, np.swapaxes(ratio, 1, 2), atol=1e-9)
    np.testing.assert_allclose(ratio[:, 0, 1], 1 - sro["alpha"][:, 0, 1], atol=0.02)
    # a random arrangement sits near 1 in every shell
    assert np.abs(ratio - 1).max() < 0.2
    assert sc["shell"][:3] == ["1/2<111>", "<100>", "<110>"]
    # the default radius reaches five lattice parameters, or half the supercell
    far = rmc.shell_correlations()
    assert far["radius"][-1] <= min(5, rmc.cells / 2) * rmc.lattice_parameter + 1e-6
    assert len(far["shell"]) == len(far["radius"])


def test_plot_shell_correlations(rmc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = rmc.plot_shell_correlations()
    K = len(rmc.species)
    assert len(axs) == K * (K + 1) // 2
    plt.close(fig)


def test_to_atoms_and_cif_round_trip(rmc, tmp_path):
    from ase.io import read

    atoms = rmc.to_atoms()
    n = len(rmc.site_x)
    assert len(atoms) == n
    assert np.allclose(atoms.cell.lengths(), rmc._a_crystal * rmc.cells)
    counts = {s: int((np.array(atoms.get_chemical_symbols()) == s).sum()) for s in rmc.species}
    expected = np.bincount(rmc.species_index, minlength=len(rmc.species))
    assert [counts[s] for s in rmc.species] == expected.tolist()
    # displacements move the atoms off the ideal sites by the fitted amount
    first = int(
        np.argsort(np.array([rmc.species[k] for k in rmc.species_index]), kind="stable")[0]
    )
    rmc.displacement[first] = [1, 0, 0]
    step = rmc._a_crystal * rmc.cells / rmc.grid_size
    shifted = rmc.to_atoms().get_positions()[0] - atoms.get_positions()[0]
    assert np.allclose(shifted, [step, 0, 0])
    rmc.displacement[first] = 0
    path = rmc.to_cif(tmp_path / "rmc.cif")
    back = read(path)
    assert len(back) == n
    assert np.allclose(back.cell.lengths(), atoms.cell.lengths())
    assert np.allclose(back.get_scaled_positions(), atoms.get_scaled_positions(), atol=1e-5)
