"""Tests for quantem.atoms: templates, RDF, template matching and the model pipeline."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quantem.atoms import AtomicModel, get_template, match_template
from quantem.atoms.measurements import (
    convex_hull_distance,
    kmeans_1d,
    misorientation,
    sample_volume,
    segment_grains,
    strain_from_deformation,
)
from quantem.atoms.pdf import (
    find_neighbors,
    fit_first_peak,
    nn_distance_from_lattice,
    radial_distribution,
)
from quantem.atoms.templates import _crystal_definition


def make_lattice(name: str, n: int = 8, noise: float = 0.03, seed: int = 0):
    """Spherical particle of structure ``name`` with NN distance 1, randomly rotated."""
    cell, basis, _, _ = _crystal_definition(name)
    rng = np.arange(-n, n + 1)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    xyz = ((ijk[:, None, :] + basis[None, :, :]).reshape(-1, 3)) @ cell
    radius = 0.8 * n
    xyz = xyz[np.linalg.norm(xyz, axis=1) < radius]
    interior = np.linalg.norm(xyz, axis=1) < radius - 2.5
    xyz = xyz @ Rotation.random(random_state=seed).as_matrix().T
    xyz = xyz + np.random.default_rng(seed).normal(0, noise, xyz.shape)
    return xyz, interior


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, num_neighbors, shells, num_symmetry",
    [
        ("fcc", 12, (1.0,), 24),
        ("hcp", 12, (1.0,), 6),
        ("bcc", 14, (1.0, 2 / np.sqrt(3)), 24),
        ("sc", 6, (1.0,), 24),
        ("diamond", 16, (1.0, np.sqrt(8 / 3)), 12),
        ("wurtzite", 16, (1.0, np.sqrt(8 / 3)), 3),
        ("ico", 12, (1.0,), 60),
    ],
)
def test_template_shells_and_symmetry(name, num_neighbors, shells, num_symmetry):
    t = get_template(name)
    assert t.num_neighbors == num_neighbors
    assert np.allclose(t.shells, shells, atol=1e-6)
    assert t.num_symmetry == num_symmetry
    assert np.allclose(np.linalg.norm(t.vectors[: t.shell_counts[0]], axis=1), 1.0)


def test_template_extra_shells():
    assert get_template("fcc", num_shells=2).num_neighbors == 18
    assert get_template("zincblende").name == "diamond"


# --------------------------------------------------------------------------- #
# RDF / neighbors / calibration
# --------------------------------------------------------------------------- #
def test_rdf_first_peak():
    xyz, _ = make_lattice("fcc", n=8, noise=0.02)
    rdf = radial_distribution(xyz * 7.0, r_max=25.0, dr=0.05)
    fit = fit_first_peak(rdf["r"], rdf["g_smooth"])
    assert abs(fit["r_nn"] - 7.0) < 0.1
    assert fit["cutoff"][0] < 7.0 < fit["cutoff"][1]


def test_find_neighbors_padding():
    xyz = np.random.default_rng(0).random((5, 3))
    dist, idx = find_neighbors(xyz, 8)
    assert dist.shape == (5, 8) and idx.shape == (5, 8)
    assert np.all(idx[:, 4:] == -1) and np.all(np.isinf(dist[:, 4:]))
    assert not np.any(idx[:, :4] == np.arange(5)[:, None])


def test_nn_distance_from_lattice():
    assert np.isclose(nn_distance_from_lattice("fcc", 3.89), 3.89 / np.sqrt(2))
    assert np.isclose(nn_distance_from_lattice("bcc", 2.0), np.sqrt(3))


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["fcc", "hcp", "bcc", "diamond", "wurtzite"])
def test_match_classifies_synthetic_lattice(name):
    xyz, interior = make_lattice(name, n=7, noise=0.03)
    dist, idx = find_neighbors(xyz, 24)
    dxyz = xyz[idx] - xyz[:, None, :]
    scores = {}
    for tn in ["fcc", "hcp", "bcc", "diamond", "wurtzite"]:
        t = get_template(tn)
        valid = dist <= 1.15 * t.max_radius
        scores[tn] = match_template(dxyz, valid, t, device="cpu", progress=False)["score"]
    names = list(scores)
    best = np.array(names)[np.stack([scores[n] for n in names], 1).argmax(1)]
    assert (best[interior] == name).mean() > 0.97
    assert scores[name][interior].mean() > 0.8


def test_match_recovers_rotation():
    cell, basis, _, _ = _crystal_definition("fcc")
    rng = np.arange(-4, 5)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    xyz = ((ijk[:, None, :] + basis[None, :, :]).reshape(-1, 3)) @ cell
    r0 = Rotation.random(random_state=3).as_matrix()
    xyz = xyz @ r0.T
    center = int(np.argmin(np.linalg.norm(xyz, axis=1)))
    dist, idx = find_neighbors(xyz, 14)
    dxyz = xyz[idx] - xyz[:, None, :]
    t = get_template("fcc")
    res = match_template(
        dxyz[center : center + 1],
        dist[center : center + 1] < 1.15,
        t,
        device="cpu",
        progress=False,
    )
    assert res["score"][0] > 0.999
    assert res["num_matched"][0] == 12
    ang = misorientation(res["rotation"], np.array([[0]]), t.symmetry)
    # rotation equals r0 up to a cubic symmetry op
    delta = res["rotation"][0].T @ r0
    traces = np.einsum("ij,sji->s", delta, t.symmetry)
    assert np.degrees(np.arccos(np.clip((traces.max() - 1) / 2, -1, 1))) < 0.5
    assert ang.shape == (1, 1)


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def test_misorientation_symmetry_invariance():
    t = get_template("fcc")
    r = Rotation.random(4, random_state=1).as_matrix()
    r_sym = np.einsum("nij,jk->nik", r, t.symmetry[7])
    rot = np.concatenate([r, r_sym])
    idx = np.array([[4], [5], [6], [7], [0], [1], [2], [3]])
    assert np.allclose(misorientation(rot, idx, t.symmetry), 0.0, atol=1e-4)


def test_segment_grains_and_strain():
    labels = segment_grains(
        np.array([[1], [0], [3], [2], [-1]]), np.array([[1], [1], [1], [1], [0]], bool), min_size=2
    )
    assert labels.tolist() == [0, 0, 1, 1, -1]
    f = np.array([[[1.04, 0, 0], [0, 1.0, 0], [0, 0, 0.98]]])
    s = strain_from_deformation(f, np.eye(3)[None])
    assert np.isclose(s["e_xx"][0], 0.04) and np.isclose(s["e_zz"][0], -0.02)
    assert np.isclose(s["dilation"][0], 0.02 / 3)


def test_hull_sample_kmeans():
    xyz = np.random.default_rng(0).random((300, 3))
    d = convex_hull_distance(xyz)
    assert d.min() >= -1e-9 and d.max() < 0.5
    vol = np.zeros((8, 8, 8))
    vol[4, 4, 4] = 27.0
    assert sample_volume(vol, np.array([[4, 4, 4]]), radius=1.0)[0] == pytest.approx(27.0 / 7)
    labels, centers = kmeans_1d(np.r_[np.zeros(20), np.ones(20) * 5])
    assert labels[:20].sum() == 0 and labels[20:].sum() == 20 and np.allclose(centers, [0, 5])


# --------------------------------------------------------------------------- #
# AtomicModel pipeline
# --------------------------------------------------------------------------- #
def make_twinned_fcc(n: int = 9, noise: float = 0.03):
    """FCC sphere with a (111) twin: sites above the plane are mirrored."""
    cell, basis, _, _ = _crystal_definition("fcc")
    rng = np.arange(-n, n + 1)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    xyz = ((ijk[:, None, :] + basis[None, :, :]).reshape(-1, 3)) @ cell
    xyz = xyz[np.linalg.norm(xyz, axis=1) < 0.8 * n]
    normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    h = xyz @ normal
    plane = h[np.argmin(np.abs(h - 0.3))]  # a lattice plane slightly off-center
    below = h < plane - 1e-6
    # the twin is the lower crystal reflected through the plane
    mirrored = xyz[below] - 2.0 * (h[below] - plane)[:, None] * normal[None, :]
    xyz = np.concatenate([xyz[~(h > plane + 1e-6)], mirrored])
    above = xyz @ normal > plane + 1e-6
    return xyz + np.random.default_rng(1).normal(0, noise, xyz.shape), above


def test_atomic_model_pipeline():
    xyz, above = make_twinned_fcc()
    model = AtomicModel.from_array(xyz * 7.2, units="voxels", name="twin")
    assert model.num_sites == xyz.shape[0]
    pdf = model.compute_pdf()
    assert abs(pdf["r_nn"] - 7.2) < 0.15
    scale = model.calibrate("fcc", lattice_constant=3.89)
    assert np.isclose(scale * 7.2, 3.89 / np.sqrt(2), rtol=0.03)
    assert model.units == "A"
    model.find_neighbors(20)
    assert np.median(model["num_neighbors"]) == 12
    model.match_templates(["fcc", "hcp"], progress=False, device="cpu")
    assert set(model.channels) >= {"score_fcc", "score_hcp", "structure", "score_diff"}
    structure = model["structure"]
    hcp = structure == 1
    # twin-plane sites have an hcp environment; there is exactly one such plane
    hcp_heights = (model.positions_native / 7.2) @ (np.ones(3) / np.sqrt(3))
    assert 0.05 < hcp.mean() < 0.25
    full = model["num_neighbors"] >= 12
    assert (np.abs(hcp_heights[hcp & full]) < 0.4).mean() > 0.9
    grains = model.segment_grains("fcc", angle_threshold=5.0, min_size=20)
    assert grains.max() + 1 == 2
    # the two grains are the two sides of the twin plane
    for g in (0, 1):
        side = above[grains == g]
        assert side.mean() > 0.95 or side.mean() < 0.05
    strain = model.compute_strain()
    assert abs(np.nanmedian(strain["dilation"])) < 0.02
    model.surface_distance()
    assert model["surface_distance"].min() >= -1e-6
    # selection keeps channels and categories
    sub = model.select(structure == 0)
    assert sub.num_sites == int((structure == 0).sum()) and "structure" in sub.categories


def test_atomic_model_channels_and_shapes():
    model = AtomicModel.from_array(np.random.default_rng(0).random((3, 50)) * 10)
    assert model.positions.shape == (50, 3)
    model.set_channel("foo", np.arange(50))
    assert model["foo"][-1] == 49
    model.set_channel("foo", np.zeros(50))
    assert model["foo"].sum() == 0
    with pytest.raises(ValueError):
        model.set_channel("bar", np.zeros(3))
    with pytest.raises(KeyError):
        model.get_channel("missing")


def test_merge_close_sites():
    xyz, _ = make_lattice("fcc", n=5, noise=0.01)
    dup = np.vstack([xyz, xyz[:3] + 0.05, xyz[3:4] + np.array([0.02, 0.0, 0.0])])
    model = AtomicModel.from_array(dup)
    model.set_channel("intensity", np.r_[np.ones(xyz.shape[0]), 3.0, 3.0, 3.0, 3.0])
    removed = model.merge_close_sites(min_distance=0.3)
    assert removed == 4 and model.num_sites == xyz.shape[0]
    # merged positions are the intensity-weighted mean and the channel is averaged
    model2 = AtomicModel.from_array(dup)
    model2.set_channel("intensity", np.r_[np.ones(xyz.shape[0]), 3.0, 3.0, 3.0, 3.0])
    model2.merge_close_sites(min_distance=0.3, weight_channel="intensity")
    assert model2["intensity"].max() == pytest.approx(2.5)
    model3 = AtomicModel.from_array(dup)
    assert model3.merge_close_sites(min_distance=0.3, mode="remove") == 4
    assert model3.merge_close_sites(min_distance=0.3) == 0


def test_ideal_structures():
    from quantem.atoms import structures as st

    assert st.icosahedron(3).num_sites == 147
    assert st.cuboctahedron(3).num_sites == 147
    assert st.double_icosahedron(1).num_sites == 19
    d = st.double_icosahedron(4, bond_length=2.7, separation=5)
    xyz = d.positions_native
    from scipy.spatial import cKDTree

    assert cKDTree(xyz).query(xyz, k=2)[0][:, 1].min() > 0.99 * 2.7
    # mirror symmetry through z = 0
    mirrored = xyz * np.array([1, 1, -1])
    assert cKDTree(xyz).query(mirrored)[0].max() < 1e-6
    deca = st.decahedron(3)
    assert deca.num_sites > 100 and len(np.unique(deca["sector"])) == 5


def test_fit_icosahedral_centers_and_layers():
    from quantem.atoms import structures as st

    model = st.double_icosahedron(7, bond_length=2.75, separation=5)
    rng = np.random.default_rng(0)
    model.positions_native = model.positions_native + rng.normal(0, 0.05, (model.num_sites, 3))
    model.compute_pdf()
    model.find_neighbors(20)
    model.match_templates(["fcc", "hcp"], progress=False, device="cpu")
    fit = model.fit_icosahedral_centers()
    assert abs(fit["separation"] / model.bond_length - 5.0) < 0.3
    assert abs(abs(fit["axis"][2]) - 1.0) < 0.02
    centers = fit["centers"] + model.center
    assert np.allclose(np.sort(centers[:, 2]), [-2.5 * 2.75, 2.5 * 2.75], atol=0.6)
    layers = model.layer_positions("z")
    assert layers.size > 10
    assert np.all(np.diff(layers) > 0.4 * model.bond_length)


def test_explode_grains():
    xyz, above = make_twinned_fcc()
    model = AtomicModel.from_array(xyz * 7.2, units="voxels")
    model.compute_pdf()
    model.find_neighbors(20)
    model.match_templates(["fcc", "hcp"], progress=False, device="cpu")
    model.segment_grains("fcc", angle_threshold=5.0, min_size=20)
    ex = model.explode_grains(distance=10.0)
    labels = ex["grain"].astype(int)
    assert set(np.unique(labels)) <= {0, 1, 2}
    assert ex.categories["grain"][2] == "shared"
    # each grain moved rigidly by 10 units
    for g in (0, 1):
        src = ex["source_index"][labels == g].astype(int)
        d = ex.positions[labels == g] - model.positions[src]
        assert np.allclose(np.linalg.norm(d, axis=1), 10.0, atol=1e-6)
    assert (labels == 2).sum() > 0
    assert ex.num_sites > (model["grain"] >= 0).sum()


def test_attached_icosahedra_and_growth_steps():
    from quantem.atoms import structures as st

    a = st.attached_icosahedra(3, bond_length=2.0)
    assert a.num_sites == 2 * 147 - 1
    steps = st.growth_steps(a)
    assert steps.min() == 0 and steps.max() == 9
    d = st.double_icosahedron(4, bond_length=2.0, separation=5)
    steps = st.growth_steps(d)
    # the upper half only appears once the front crosses the mid-plane at 2.5 bonds
    assert steps[d.positions[:, 2] > 0.1].min() >= 3


def test_species_mixture_keeps_unassigned():
    rng = np.random.default_rng(0)
    xyz = rng.random((4001, 3)) * 30
    model = AtomicModel.from_array(xyz)
    # two equal components and one site exactly midway, which is ambiguous by symmetry
    intensity = np.r_[rng.normal(1.0, 0.1, 2000), rng.normal(2.0, 0.1, 2000), 1.5]
    model.set_channel("intensity", intensity)
    labels = model.classify_species(names=["Ni", "Pd"], min_posterior=0.9)
    info = model.metadata["species_model"]
    assert abs(info["means"][0] - 1.0) < 0.05 and abs(info["means"][1] - 2.0) < 0.05
    assert 0.4 < info["weights"][0] < 0.6
    assert (labels[:2000] == 0).mean() > 0.95 and (labels[2000:4000] == 1).mean() > 0.95
    assert model.categories["species"] == ["Ni", "Pd"]
    assert labels[-1] == -1 and model["species_posterior"][-1] < 0.9
    labels = model.classify_species(method="kmeans")
    assert labels.min() >= 0
    mask = np.arange(model.num_sites) < 3000
    labels = model.classify_species(names=["Ni", "Pd"], mask=mask)
    assert np.all(labels[~mask] == -1) and labels[:2000].max() == 0
