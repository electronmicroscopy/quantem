import numpy as np
import pytest


class TestCliffLorimerQuantification:
    def test_quantify_composition_from_cached_integration_maps(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 256), dtype=np.float32))
        _fe_energy, _fe_weight, fe_labels = dataset.x_ray_lookup("Fe Ka1")
        _cu_energy, _cu_weight, cu_labels = dataset.x_ray_lookup("Cu Ka1")
        fe_label = fe_labels[0]
        cu_label = cu_labels[0]

        fe_map = np.full((2, 2), 10.0, dtype=np.float32)
        cu_map = np.full((2, 2), 5.0, dtype=np.float32)
        dataset._spectrum_images = {
            fe_label: dataset._map_to_dataset2d(fe_map, name=fe_label),
            cu_label: dataset._map_to_dataset2d(cu_map, name=cu_label),
        }

        atomic_percent, weight_percent = dataset.quantify_composition_cliff_lorimer(
            {fe_label: 1.0, cu_label: 2.0},
            method="integration",
            verbose=False,
        )

        expected_fe_atomic = 100.0 * fe_map.sum() / (fe_map.sum() + 2.0 * cu_map.sum())
        assert atomic_percent["Fe"] == pytest.approx(expected_fe_atomic)
        assert atomic_percent["Fe"] + atomic_percent["Cu"] == pytest.approx(100.0)
        assert weight_percent["Fe"] + weight_percent["Cu"] == pytest.approx(100.0)

    def test_quantify_composition_returns_maps(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 256), dtype=np.float32))
        _fe_energy, _fe_weight, fe_labels = dataset.x_ray_lookup("Fe Ka1")
        _cu_energy, _cu_weight, cu_labels = dataset.x_ray_lookup("Cu Ka1")
        fe_label = fe_labels[0]
        cu_label = cu_labels[0]

        dataset._spectrum_images = {
            fe_label: dataset._map_to_dataset2d(np.full((2, 2), 4.0), name=fe_label),
            cu_label: dataset._map_to_dataset2d(np.full((2, 2), 1.0), name=cu_label),
        }

        (_atomic, _weight), (atomic_maps, weight_maps) = (
            dataset.quantify_composition_cliff_lorimer(
                {fe_label: 1.0, cu_label: 1.0},
                method="integration",
                return_maps=True,
                verbose=False,
            )
        )

        assert set(atomic_maps) == {"Fe", "Cu"}
        assert set(weight_maps) == {"Fe", "Cu"}
        assert atomic_maps["Fe"].signal_units == "%"
        assert weight_maps["Cu"].signal_units == "%"

    def test_quantify_composition_requires_cached_images(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 256), dtype=np.float32))

        with pytest.raises(ValueError, match="No spectrum images available"):
            dataset.quantify_composition_cliff_lorimer({"Fe": 1.0, "Cu": 1.0}, verbose=False)
