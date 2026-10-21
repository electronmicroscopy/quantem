import numpy as np


def _make_two_element_cube(dataset, line_spectrum_factory):
    n_channels = dataset.shape[2]
    fe_map = np.array([[8.0, 2.0], [6.0, 1.0]], dtype=np.float32)
    cu_map = np.array([[1.0, 6.0], [2.0, 8.0]], dtype=np.float32)
    background = 0.5 + 0.2 * np.linspace(0.0, 1.0, n_channels, dtype=np.float32)

    cube = np.empty((2, 2, n_channels), dtype=np.float32)
    for y in range(2):
        for x in range(2):
            cube[y, x] = line_spectrum_factory(
                ["Fe K", "Cu K"],
                [fe_map[y, x] * 100.0, cu_map[y, x] * 100.0],
                n_channels=n_channels,
                energy_min=float(dataset.energy_axis[0]),
                energy_max=float(dataset.energy_axis[-1]),
                background=background,
            )
    return cube, fe_map, cu_map


class TestSpectrumMaps:
    def test_integrate_returns_dataset2d_for_single_selector(
        self, xeds_factory, line_spectrum_factory
    ):
        base = xeds_factory(np.zeros((2, 2, 384), dtype=np.float32))
        cube, fe_map, _cu_map = _make_two_element_cube(base, line_spectrum_factory)
        dataset = xeds_factory(cube)

        result = dataset.integrate("Fe K", width=0.12, show=False)

        assert result.array.shape == (2, 2)
        assert result.array[0, 0] > result.array[0, 1]
        assert result.array[0, 0] > 0.0
        assert np.corrcoef(result.array.ravel(), fe_map.ravel())[0, 1] > 0.9

    def test_generate_spectrum_images_caches_and_returns_maps(
        self, xeds_factory, line_spectrum_factory
    ):
        base = xeds_factory(np.zeros((2, 2, 384), dtype=np.float32))
        cube, _fe_map, _cu_map = _make_two_element_cube(base, line_spectrum_factory)
        dataset = xeds_factory(cube)

        images, titles = dataset.generate_spectrum_images(
            elements=["Fe K", "Cu K"],
            width=0.12,
            return_maps=True,
        )

        assert len(images) == len(titles) >= 2
        assert dataset._spectrum_images
        assert any(str(title).startswith("Fe") for title in titles)
        assert all(image.array.shape == (2, 2) for image in images)

    def test_build_pytorch_spectrum_images_scales_by_line_weights(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 384), dtype=np.float32))
        abundance_maps = np.full((1, 2, 2), 5.0, dtype=np.float32)

        line_maps = dataset._build_pytorch_spectrum_images(abundance_maps, ["Fe"])
        _energies, line_weights, line_labels = dataset.x_ray_lookup("Fe")

        assert line_maps
        first_label = line_labels[0]
        np.testing.assert_allclose(line_maps[first_label].array, 5.0 * line_weights[0])
