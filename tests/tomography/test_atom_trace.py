"""Tests for quantem.tomography.Atoms: splatting and tracing a synthetic lattice."""

import numpy as np
import torch
from scipy.spatial import cKDTree

from quantem.core.datastructures import Dataset3d
from quantem.tomography import Atoms
from quantem.tomography.atom_trace import render_isotropic


def test_render_isotropic_per_site_radius():
    pos = torch.tensor([[5.3, 6.0, 7.7], [10.0, 9.4, 4.1]])
    inten = torch.tensor([1.0, 2.0])
    sigma = torch.tensor([1.0, 2.0])
    vol = render_isotropic(pos, inten, sigma, (16, 16, 16), torch.tensor([4, 8])).numpy()
    grid = np.stack(np.meshgrid(*[np.arange(16)] * 3, indexing="ij"), -1)
    ref = sum(
        a * np.exp(-0.5 * ((grid - p) ** 2).sum(-1) / s**2)
        for p, a, s in zip(pos.numpy(), inten.numpy(), sigma.numpy())
    )
    assert np.abs(vol - ref).max() < 5e-3 * ref.max()


def test_trace_synthetic_lattice(monkeypatch):
    g = np.arange(4) * 6.0 + 7.0
    true = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    true += np.random.default_rng(0).uniform(-0.3, 0.3, true.shape)
    t = torch.tensor(true, dtype=torch.float32)
    vol = render_isotropic(t, torch.ones(len(t)), 1.2, (32, 32, 32), 5).numpy()
    atoms = Atoms.from_dataset(
        Dataset3d.from_array(vol), sigma_init=1.2, background=False, device="cpu"
    )

    def no_spacing(self, **kwargs):
        raise RuntimeError("no autocorrelation peak")

    # refine must fall back to the site spacing for the isolation radius
    monkeypatch.setattr(Atoms, "estimate_spacing", no_spacing)
    atoms.find_initial(progress=False)
    atoms.refine(num_iterations=30, update_every=10, progress=False)
    found = atoms.sites.numpy()[:, :3]
    assert atoms.num_sites == len(true)
    assert cKDTree(found).query(true)[0].max() < 0.1
