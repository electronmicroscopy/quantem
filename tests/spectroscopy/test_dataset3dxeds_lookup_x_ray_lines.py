import numpy as np
import pytest


class TestXRayLookup:
    def test_x_ray_lookup_returns_sorted_unique_lines(self, xeds_factory):
        dataset = xeds_factory(np.zeros((1, 1, 256), dtype=np.float32))

        energies, weights, labels = dataset.x_ray_lookup(["Fe K", "Fe Ka1"])

        assert energies.ndim == 1
        assert weights.shape == energies.shape
        assert len(labels) == len(energies)
        assert np.all(np.diff(energies) >= 0)
        assert len(labels) == len(set(labels))

    def test_x_ray_lookup_supports_element_and_shell_selectors(self, xeds_factory):
        dataset = xeds_factory(np.zeros((1, 1, 256), dtype=np.float32))

        energies, _weights, labels = dataset.x_ray_lookup("Cu K")

        assert len(energies) >= 1
        assert all(label.startswith("Cu") for label in labels)

    def test_x_ray_lookup_unknown_specifier_raises(self, xeds_factory):
        dataset = xeds_factory(np.zeros((1, 1, 256), dtype=np.float32))

        with pytest.raises(ValueError, match="Could not resolve element|No X-ray lines matched"):
            dataset.x_ray_lookup("NotAnElement")
