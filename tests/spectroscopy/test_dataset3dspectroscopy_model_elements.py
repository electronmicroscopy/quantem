import numpy as np


class TestModelElements:
    def test_add_elements_to_model_tracks_requested_lines(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 256), dtype=np.float32))

        dataset.add_elements_to_model(["Fe K", "Cu"])

        assert "Fe" in dataset.model_elements
        assert "Cu" in dataset.model_elements
        assert dataset.model_elements["Fe"]
        assert all(str(line).lower().startswith("k") for line in dataset.model_elements["Fe"])

    def test_remove_elements_from_model_drops_only_requested_selector(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 256), dtype=np.float32))
        dataset.add_elements_to_model("Fe")

        before = set(dataset.model_elements["Fe"])
        assert any(not str(line).lower().startswith("k") for line in before)

        dataset.remove_elements_from_model("Fe K")

        after = set(dataset.model_elements["Fe"])
        assert after
        assert all(not str(line).lower().startswith("k") for line in after)

    def test_clear_model_elements_resets_container(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 256), dtype=np.float32))
        dataset.add_elements_to_model(["Fe", "Cu"])

        dataset.clear_model_elements()

        assert not dataset.model_elements
        assert "None" in repr(dataset.model_elements)
