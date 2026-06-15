"""Tests for ``quantem.core.ml.models.kplanes`` construction guards."""

import pytest

from quantem.core.ml.models.kplanes import KPlanes, KPlanesTILTED


class TestResolutionValidation:
    def test_isotropic_builds(self):
        model = KPlanes(M_features=2, resolution=(16, 16, 16))
        assert model.grids[0].shape == (3, 2, 16, 16)

    def test_anisotropic_raises(self):
        """Regression: the plane grids are allocated as (res[1], res[0]) for all three
        axis pairs, ignoring res[2] -- anisotropic resolutions silently gave the
        XZ/YZ planes the wrong grid along z instead of erroring."""
        with pytest.raises(ValueError, match="isotropic"):
            KPlanes(M_features=2, resolution=(16, 16, 8))
        with pytest.raises(ValueError, match="isotropic"):
            KPlanesTILTED(M_features=2, T=2, resolution=(16, 8, 16))
