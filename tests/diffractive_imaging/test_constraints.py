"""Tests for the ptychography constraint dataclass API."""

import warnings

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    PtychoDatasetConstraintParams,
    Ptychography,
    PtychographyDatasetRaster,
    PtychoObjConstraintParams,
    PtychoProbeConstraintParams,
)

N_SCAN = 8
N_DET = 16
PROBE_ENERGY = 80e3
PROBE_SEMIANGLE = 20
PROBE_DEFOCUS = 100


@pytest.fixture
def ptycho():
    rng = np.random.default_rng(42)
    array = rng.random((N_SCAN, N_SCAN, N_DET, N_DET)).astype(np.float32)
    dset = Dataset4dstem.from_array(
        array,
        name="test",
        sampling=[1.0, 1.0, 0.05, 0.05],
        units=["A", "A", "A^-1", "A^-1"],
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False)
    obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
    probe = ProbePixelated.from_params(
        probe_params={
            "energy": PROBE_ENERGY,
            "defocus": PROBE_DEFOCUS,
            "semiangle_cutoff": PROBE_SEMIANGLE,
        }
    )
    p = Ptychography.from_models(
        dset=pdset,
        obj_model=obj,
        probe_model=probe,
        detector_model=DetectorPixelated(),
        verbose=False,
        rng=42,
    )
    p.preprocess(obj_padding_px=(4, 4))
    return p


# --- parse_dict tests ---------------------------------------------------------


class TestParseDict:
    def test_object_raster_by_name(self):
        c = PtychoObjConstraintParams.parse_dict({"name": "raster", "tv_weight_z": 5.0})
        assert isinstance(c, PtychoObjConstraintParams.Raster)
        assert c.tv_weight_z == 5.0
        assert c.positivity is True  # default preserved

    def test_object_inr_by_type(self):
        c = PtychoObjConstraintParams.parse_dict({"type": "inr"})
        assert isinstance(c, PtychoObjConstraintParams.INR)

    def test_object_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown object constraint type"):
            PtychoObjConstraintParams.parse_dict({"name": "nope"})

    def test_object_missing_name_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'name' or 'type'"):
            PtychoObjConstraintParams.parse_dict({"tv_weight_z": 5.0})

    def test_probe_raster_with_fields(self):
        c = PtychoProbeConstraintParams.parse_dict(
            {"name": "raster", "center_probe": True, "tv_weight": 0.1}
        )
        assert isinstance(c, PtychoProbeConstraintParams.Raster)
        assert c.center_probe is True
        assert c.tv_weight == 0.1

    def test_dataset_raster_default(self):
        c = PtychoDatasetConstraintParams.parse_dict({"name": "raster"})
        assert isinstance(c, PtychoDatasetConstraintParams.Raster)
        assert c.clip_scan_positions is True  # default preserved


# --- Constraint typo catching -------------------------------------------------


class TestTypoCatching:
    def test_setting_unknown_field_via_dict_raises(self, ptycho):
        with pytest.raises(KeyError, match="Invalid constraint key"):
            ptycho.obj_model.constraints = {"not_a_real_field": True}

    def test_add_constraint_unknown_key_raises(self, ptycho):
        with pytest.raises(KeyError, match="Invalid constraint key"):
            ptycho.obj_model.add_constraint("not_a_real_field", True)


# --- Round-trip: pass dataclass via reconstruct(), read back through getter ---


class TestRoundtrip:
    def test_obj_constraints_dataclass(self, ptycho):
        obj_c = PtychoObjConstraintParams.Raster(tv_weight_z=2.5, identical_slices=True)
        ptycho.constraints = {"object": obj_c}
        assert ptycho.obj_model.constraints is obj_c
        assert ptycho.obj_model.constraints.tv_weight_z == 2.5
        assert ptycho.obj_model.constraints.identical_slices is True

    def test_probe_constraints_dataclass(self, ptycho):
        probe_c = PtychoProbeConstraintParams.Raster(center_probe=True, tv_weight=0.05)
        ptycho.constraints = {"probe": probe_c}
        assert ptycho.probe_model.constraints is probe_c

    def test_dataset_constraints_dataclass(self, ptycho):
        dset_c = PtychoDatasetConstraintParams.Raster(descan_tv_weight=0.01)
        ptycho.constraints = {"dataset": dset_c}
        assert ptycho.dset.constraints is dset_c

    def test_dict_form_still_works(self, ptycho):
        """Backward compatibility: nested-dict form sets individual fields."""
        ptycho.constraints = {
            "object": {"tv_weight_z": 3.0, "positivity": False},
            "probe": {"tv_weight": 0.02},
        }
        assert ptycho.obj_model.constraints.tv_weight_z == 3.0
        assert ptycho.obj_model.constraints.positivity is False
        assert ptycho.probe_model.constraints.tv_weight == 0.02


# --- Reconstruct() with constraints= ------------------------------------------


class TestReconstructKwargs:
    def test_dataclass_leaf_applied(self, ptycho):
        from quantem.core.ml import OptimizerParams

        obj_c = PtychoObjConstraintParams.Raster(tv_weight_z=1.5)
        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            constraints={"object": obj_c},
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.tv_weight_z == 1.5

    def test_dict_leaf_partial_update(self, ptycho):
        from quantem.core.ml import OptimizerParams

        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            constraints={"object": {"surface_zero_weight": 0.7}},
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.surface_zero_weight == 0.7
        # other fields keep their defaults
        assert ptycho.obj_model.constraints.positivity is True

    def test_mixed_dataclass_and_dict_leaves(self, ptycho):
        from quantem.core.ml import OptimizerParams

        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            constraints={
                "object": PtychoObjConstraintParams.Raster(tv_weight_xy=0.4),
                "probe": {"center_probe": True},
            },
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.tv_weight_xy == 0.4
        assert ptycho.probe_model.constraints.center_probe is True


# --- Real-valued pure_phase representation -----------------------------------


class TestPurePhaseRealValued:
    def test_pure_phase_pixelated_obj_is_real(self):
        obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        assert not obj._obj.is_complex(), f"pure_phase _obj should be real, got {obj._obj.dtype}"

    def test_complex_pixelated_obj_is_complex(self):
        obj = ObjectPixelated.from_uniform(obj_type="complex", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        assert obj._obj.is_complex()

    def test_potential_pixelated_obj_is_real(self):
        obj = ObjectPixelated.from_uniform(obj_type="potential", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        assert not obj._obj.is_complex()

    def test_pure_phase_tv_emits_no_phase_warning(self):
        obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        obj.constraints.tv_weight_xy = 0.1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj.get_tv_loss(obj._obj)
        phase_warnings = [w for w in caught if "phase wrapping" in str(w.message)]
        assert not phase_warnings, (
            f"pure_phase should not emit phase-wrap warning, got {phase_warnings}"
        )

    def test_complex_tv_still_emits_phase_warning(self):
        obj = ObjectPixelated.from_uniform(obj_type="complex", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        obj.constraints.tv_weight_xy = 0.1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj.get_tv_loss(obj._obj)
        assert any("phase wrapping" in str(w.message) for w in caught), (
            "complex obj_type should still emit phase-wrap warning"
        )

    def test_pure_phase_apply_hard_constraints_stays_real(self):
        obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        out = obj.apply_hard_constraints(obj._obj)
        assert not out.is_complex()


# --- FOV-mask single application ---------------------------------------------


class TestFovMaskSingleApplication:
    def _make_obj(self, obj_type) -> ObjectPixelated:
        obj = ObjectPixelated.from_uniform(obj_type=obj_type, num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        obj.constraints.apply_fov_mask = True
        # Force a non-trivial _obj so masking is observable
        if obj_type == "complex":
            obj._obj = torch.nn.Parameter(
                torch.ones(1, 16, 16, dtype=torch.complex64) * (0.5 + 0.3j)
            )
        else:
            obj._obj = torch.nn.Parameter(torch.full((1, 16, 16), 0.7))
        return obj

    @pytest.mark.parametrize("obj_type", ["pure_phase", "complex", "potential"])
    def test_mask_applied_once(self, obj_type):
        obj = self._make_obj(obj_type)
        # Half-mask: ones on the left, zeros on the right; if mask is applied
        # twice the masked region squares the multiplication (no observable
        # difference for 0/1 masks), so use a non-binary mask.
        mask = torch.full((1, 16, 16), 0.5)
        obj._mask = mask
        out = obj.apply_hard_constraints(obj._obj, mask=mask)
        # Verify nothing crashed and shape is preserved.
        assert out.shape == obj._obj.shape
        # If mask had been applied twice, |out| would scale by 0.5**2 = 0.25
        # of the unmasked value; once it scales by 0.5. We compare to the
        # per-obj-type expected post-constraint value.
        if obj_type == "pure_phase":
            # phase recentered to zero mean, then *= 0.5 mask
            expected_mag = 0.0  # phase=constant -> recenter to 0 -> *0.5 = 0
        elif obj_type == "potential":
            # positivity clamp keeps 0.7, * 0.5 -> 0.35 (one application)
            expected_mag = 0.35
        else:  # complex
            # amp clamp keeps 0.5+0.3j, * 0.5 -> magnitude 0.5 * |0.5+0.3j|
            expected_mag = 0.5 * abs(0.5 + 0.3j)
        # Sample the magnitude in the masked region
        if out.is_complex():
            sampled = out.abs().mean().item()
        else:
            sampled = out.abs().mean().item()
        assert abs(sampled - expected_mag) < 1e-4, (
            f"{obj_type}: expected mag ~{expected_mag}, got {sampled} "
            f"(would be {expected_mag * 0.5} if mask were applied twice)"
        )
