"""CrystalMap and PhaseMap behaviour on small synthetic alpha/beta titanium scans."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.crystal_map import CrystalMap
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.rotations import misorientation_angle_deg, quat_from_axis_angle

Q_ALPHA = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
Q_BETA = quat_from_axis_angle(
    torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    torch.tensor(np.deg2rad(20.0), dtype=torch.float64),
)


def _crystals():
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    ti_b = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), name="Ti beta", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    return ti_a, ti_b


def _pattern(xtl, q, rng):
    p = xtl.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    arr = np.column_stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()])
    arr[:, :2] += rng.normal(0, 0.003, (arr.shape[0], 2))
    arr[:, 2] *= rng.lognormal(0, 0.3, arr.shape[0])
    return arr


def _vacuum(rng):
    """Direct beam and a few detections around it: enough peaks to be
    matched, but nothing diffracted."""
    arr = np.zeros((5, 3))
    arr[0, 2] = 100.0
    arr[1:, :2] = rng.uniform(-0.03, 0.03, (4, 2))
    arr[1:, 2] = 1.0
    return arr


def _peaks(cells):
    return Vector.from_data(cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t")


@pytest.fixture(scope="module")
def two_phase():
    """Alpha on the left, beta on the right, one column of vacuum."""
    rng = np.random.default_rng(11)
    ti_a, ti_b = _crystals()
    R, C = 3, 7
    cells = []
    for _ in range(R):
        row = []
        for c in range(C):
            if c == C - 1:
                row.append(_vacuum(rng))
            elif c < 3:
                row.append(_pattern(ti_a, Q_ALPHA, rng))
            else:
                row.append(_pattern(ti_b, Q_BETA, rng))
        cells.append(row)
    cm = CrystalMap.from_vectors(_peaks(cells), [ti_a, ti_b], energy_ev=200e3)
    cm.build_plan(angle_step_zone_axis_deg=3.0, verbose=False, progress_bar=False)
    cm.match_orientations(progress_bar=False)
    cm.fit(progress_bar=False)
    return cm


def test_null_hypothesis_leaves_vacuum_unindexed(two_phase):
    cm = two_phase
    ph = cm.phase_index
    # the vacuum column was matched (enough peaks) but diffracts nothing
    assert bool(cm[0].computed[:, -1].all())
    assert (ph[:, -1] == -1).all()
    assert (ph[:, :3] == 0).all() and (ph[:, 3:-1] == 1).all()
    # unfit and unindexed positions carry NaN reliability, not zero
    assert np.isnan(cm.phases.reliability.numpy()[:, -1]).all()
    # per-crystal weights are normalized where indexed, zero elsewhere
    w = cm.phases.crystal_weights.numpy()
    assert np.allclose(w[ph >= 0].sum(-1), 1.0)
    assert np.allclose(w[ph < 0], 0.0)


def test_crystal_map_mask_and_phase_fractions(two_phase):
    cm = two_phase
    ph = cm.phase_index
    m_a = cm.mask("Ti alpha")
    assert np.array_equal(m_a, cm.mask(0))
    assert (m_a[ph != 0] == 0).all() and (m_a[ph == 0] > 0).all()
    m_all = cm.mask()
    assert (m_all[ph == -1] == 0).all() and (m_all[ph >= 0] > 0).all()
    with pytest.raises(KeyError):
        cm.mask("Ti gamma")
    with pytest.raises(KeyError):
        cm.mask(5)

    frac = cm.phase_fractions()
    assert set(frac) == {"unindexed", "Ti alpha", "Ti beta"}
    assert np.isclose(sum(frac.values()), 1.0)
    assert np.isclose(frac["unindexed"], (ph == -1).mean())
    assert np.isclose(frac["Ti beta"], (ph == 1).mean())


def test_plot_phase_majority_filter_blacks_out_removed_positions(two_phase):
    cm = two_phase
    pm = cm.phases
    saved = pm.phase_index.clone()
    try:
        # one isolated indexed position in an unindexed field: the filter
        # removes it, and it must then be black, not the last phase color
        lone = torch.full_like(saved, -1)
        lone[1, 1] = 0
        pm.phase_index = lone
        for radius, lit in ((0, True), (1, False)):
            fig, ax = pm.plot_phase(majority_filter=radius, shade_by="none", scalebar=None)
            rgb = np.asarray(ax.images[0].get_array())
            assert (rgb[1, 1].sum() > 0) == lit
            plt.close(fig)
        # and on the real decision the filter runs and draws vacuum black
        pm.phase_index = saved
        fig, ax = cm.plot_phase(majority_filter=1, scalebar=None)
        rgb = np.asarray(ax.images[0].get_array())
        assert np.allclose(rgb[:, -1], 0.0)
        plt.close(fig)
    finally:
        pm.phase_index = saved


def test_plot_phase_cycles_colors_past_the_palette(two_phase):
    from quantem.diffraction.orientation_visualization import (
        DEFAULT_PHASE_COLORS,
        phase_color_cycle,
    )

    n = len(DEFAULT_PHASE_COLORS) + 2
    colors = phase_color_cycle(n)
    assert colors.shape == (n, 3)
    assert np.allclose(colors[len(DEFAULT_PHASE_COLORS)], DEFAULT_PHASE_COLORS[0])
    # a short palette of names cycles as well
    assert np.allclose(phase_color_cycle(3, ["red", "blue"])[2], (1.0, 0.0, 0.0))
    fig, ax = two_phase.plot_phase(phase_colors=np.array([[1.0, 0.0, 0.0]]), shade_by="none")
    plt.close(fig)


def test_crystal_map_plots_smoke(two_phase):
    cm = two_phase
    figs = cm.plot_orientation()
    assert isinstance(figs, list) and len(figs) == 4
    fig, ax = cm.plot_orientation(phase="Ti beta", direction="z")
    assert len(cm.plot_orientation(phase=0)) == 2
    poles = cm.plot_pole_figure()
    assert isinstance(poles, list) and len(poles) == 2
    fig, ax = cm.plot_pole_figure(pole=(0, 0, 0, 1), phase="Ti alpha", color_by="ipf")
    fig, axs = cm.plot_correlation()
    assert np.asarray(axs).shape == (2, 2)
    fig, axs = cm.plot_correlation(mask=True, shared_scale=False)
    # a single match per crystal: the default matches=(0, 1) skips the
    # second, so there is one panel per crystal
    fig, axs = cm.plot_matches([(0, 0), (0, 4)])
    assert axs.shape == (2, 2)
    fig, axs = cm.plot_matches([(0, 0)], phase="Ti alpha")
    assert axs.shape == (1, 1)
    plt.close("all")


def test_refine_overrides_name_known_crystals(two_phase):
    with pytest.raises(KeyError, match="unknown crystals"):
        two_phase.refine_orientations(overrides={"Ti gamma": {}}, progress_bar=False)


def test_fit_without_any_paired_peak_is_unindexed():
    # every measured peak sits between the crystal's rings, so the best model
    # has zero weight; argmax of zero weights must not name the first phase
    ti_a, _ = _crystals()
    rng = np.random.default_rng(3)
    ring = np.zeros((7, 3))
    ring[0, 2] = 100.0
    phi = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring[1:, 0], ring[1:, 1], ring[1:, 2] = 0.2 * np.cos(phi), 0.2 * np.sin(phi), 5.0
    cells = [[_pattern(ti_a, Q_ALPHA, rng), ring]]
    om = OrientationMap.from_vectors(_peaks(cells), ti_a, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=4.0, verbose=False, progress_bar=False)
    om.match_orientations(progress_bar=False)
    om.corr[0, 1, 0] = 0.5  # force the candidate into the fit
    cm = CrystalMap.from_orientation_maps([om])
    cm.fit(progress_bar=False, k_max=1.0)
    assert cm.phase_index[0, 0] == 0
    assert cm.phase_index[0, 1] == -1


def test_single_crystal_map_end_to_end(tmp_path):
    rng = np.random.default_rng(5)
    ti_a, _ = _crystals()
    cells = [[_pattern(ti_a, Q_ALPHA, rng) for _ in range(4)] + [_vacuum(rng)] for _ in range(2)]
    cm = CrystalMap.from_vectors(_peaks(cells), ti_a, energy_ev=200e3)
    assert len(cm) == 1 and cm.names == ["Ti alpha"]
    cm.build_plan(angle_step_zone_axis_deg=3.0, verbose=False, progress_bar=False)
    cm.match_orientations(progress_bar=False)
    cm.refine_orientations(progress_bar=False)
    cm.fit(progress_bar=False)
    assert "refined" in repr(cm) and "phase fit" in repr(cm)
    ph = cm.phase_index
    assert (ph[:, :4] == 0).all() and (ph[:, 4] == -1).all()
    err = misorientation_angle_deg(Q_ALPHA, cm[0].quats[:, :4, 0].reshape(-1, 4), ti_a.sym_quats)
    assert float(err.max()) < 1.5
    # no runner-up crystal: reliability is NaN, and examples fall back to
    # the correlation
    assert np.isnan(cm.phases.reliability.numpy()).all()
    picks = cm.example_positions(num=2, min_distance=1)
    assert len(picks) == 2 and all(ph[p] == 0 for p in picks)
    assert np.isclose(cm.phase_fractions()["Ti alpha"], 0.8)
    fig, ax = cm.plot_phase()
    fig, ax = cm.plot_orientation(phase=0, direction="z")
    fig, ax = cm.plot_pole_figure(phase="Ti alpha")
    plt.close("all")
    cm.save(tmp_path / "cm.zip", mode="o")
