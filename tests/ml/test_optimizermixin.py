"""Tests for OptimizerParams and SchedulerParams dataclasses."""

import pytest

# Now import the module under test — adjust the path if needed
from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    SchedulerParams,
)

# ─── OptimizerParams defaults ───────────────────────────────────────────────


class TestAdamDefaults:
    def test_defaults(self):
        adam = OptimizerParams.Adam()
        assert adam.lr == 1e-3
        assert adam.betas == (0.9, 0.999)
        assert adam.eps == 1e-8
        assert adam.weight_decay == 0
        assert adam._name == "adam"

    def test_params_dict(self):
        adam = OptimizerParams.Adam(lr=0.01, weight_decay=1e-4)
        p = adam.params()
        assert p == {
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-4,
        }

    def test_custom_betas(self):
        adam = OptimizerParams.Adam(betas=(0.8, 0.99))
        assert adam.params()["betas"] == (0.8, 0.99)


class TestAdamWDefaults:
    def test_defaults(self):
        adamw = OptimizerParams.AdamW()
        assert adamw.lr == 1e-3
        assert adamw._name == "adamw"

    def test_params_dict(self):
        adamw = OptimizerParams.AdamW(lr=5e-4, eps=1e-7)
        p = adamw.params()
        assert p["lr"] == 5e-4
        assert p["eps"] == 1e-7


class TestSGDDefaults:
    def test_defaults(self):
        sgd = OptimizerParams.SGD()
        assert sgd.lr == 1e-3
        assert sgd.momentum == 0
        assert sgd.dampening == 0
        assert sgd.nesterov is False
        assert sgd._name == "sgd"

    def test_params_dict(self):
        sgd = OptimizerParams.SGD(lr=0.1, momentum=0.9, nesterov=True)
        p = sgd.params()
        assert p == {
            "lr": 0.1,
            "momentum": 0.9,
            "dampening": 0,
            "weight_decay": 0,
            "nesterov": True,
        }


class TestNoneOptimizer:
    def test_defaults(self):
        none_opt = OptimizerParams.NoneOptimizer()
        assert none_opt._name == "none"
        assert none_opt.params() == {}


# ─── OptimizerParams.parse_dict ─────────────────────────────────────────────


class TestOptimizerParseDict:
    def test_parse_adam(self):
        result = OptimizerParams.parse_dict({"name": "adam", "lr": 0.01})
        assert isinstance(result, OptimizerParams.Adam)
        assert result.lr == 0.01

    def test_parse_adamw(self):
        result = OptimizerParams.parse_dict({"name": "adamw", "weight_decay": 0.1})
        assert isinstance(result, OptimizerParams.AdamW)
        assert result.weight_decay == 0.1

    def test_parse_sgd(self):
        result = OptimizerParams.parse_dict({"name": "sgd", "momentum": 0.9})
        assert isinstance(result, OptimizerParams.SGD)
        assert result.momentum == 0.9

    def test_parse_case_insensitive(self):
        result = OptimizerParams.parse_dict({"name": "Adam"})
        assert isinstance(result, OptimizerParams.Adam)

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            OptimizerParams.parse_dict({"name": "rmsprop"})

    def test_parse_does_not_mutate_input(self):
        d = {"name": "adam", "lr": 0.01}
        original = dict(d)
        OptimizerParams.parse_dict(d)
        assert d == original

    def test_parse_invalid_name_type_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            OptimizerParams.parse_dict({"name": 42})


# ─── parse_dict "name" vs "type" key handling ───────────────────────────────


class TestOptimizerParseDictKeyHandling:
    def test_parse_with_type_key(self):
        result = OptimizerParams.parse_dict({"type": "adam", "lr": 0.01})
        assert isinstance(result, OptimizerParams.Adam)
        assert result.lr == 0.01

    def test_name_takes_precedence_over_type(self):
        result = OptimizerParams.parse_dict({"name": "adam", "type": "sgd"})
        assert isinstance(result, OptimizerParams.Adam)

    def test_neither_name_nor_type_raises(self):
        with pytest.raises(ValueError, match="Must provide either"):
            OptimizerParams.parse_dict({"lr": 0.01})

    def test_type_key_not_leaked_into_constructor(self):
        """'type' should be popped from d so it doesn't become an unexpected kwarg."""
        result = OptimizerParams.parse_dict({"type": "sgd", "momentum": 0.9})
        assert isinstance(result, OptimizerParams.SGD)
        assert result.momentum == 0.9

    def test_both_keys_popped_when_name_used(self):
        """Even when 'name' is used, 'type' should be popped so it doesn't leak."""
        result = OptimizerParams.parse_dict({"name": "adam", "type": "ignored", "lr": 0.05})
        assert isinstance(result, OptimizerParams.Adam)
        assert result.lr == 0.05


class TestSchedulerParseDictKeyHandling:
    def test_parse_with_type_key(self):
        result = SchedulerParams.parse_dict({"type": "plateau", "patience": 20})
        assert isinstance(result, SchedulerParams.Plateau)
        assert result.patience == 20

    def test_name_takes_precedence_over_type(self):
        result = SchedulerParams.parse_dict({"name": "plateau", "type": "linear"})
        assert isinstance(result, SchedulerParams.Plateau)

    def test_neither_name_nor_type_defaults_to_none(self):
        result = SchedulerParams.parse_dict({"patience": 20})
        assert isinstance(result, SchedulerParams.NoneScheduler)

    def test_type_key_not_leaked_into_constructor(self):
        result = SchedulerParams.parse_dict({"type": "cyclic", "step_size_up": 50})
        assert isinstance(result, SchedulerParams.Cyclic)
        assert result.step_size_up == 50

    def test_both_keys_popped_when_name_used(self):
        result = SchedulerParams.parse_dict({"name": "plateau", "type": "ignored", "patience": 5})
        assert isinstance(result, SchedulerParams.Plateau)
        assert result.patience == 5


# ─── SchedulerParams defaults ───────────────────────────────────────────────


class TestPlateauDefaults:
    def test_defaults(self):
        p = SchedulerParams.Plateau()
        assert p.mode == "min"
        assert p.factor == 0.5
        assert p.patience == 10
        assert p.cooldown == 50
        assert p.min_lr is None
        assert p._name == "plateau"

    def test_params_computes_min_lr(self):
        p = SchedulerParams.Plateau()
        result = p.params(base_LR=0.01)
        assert result["min_lr"] == pytest.approx(0.01 / 20)

    def test_params_explicit_min_lr(self):
        p = SchedulerParams.Plateau(min_lr=1e-6)
        result = p.params(base_LR=0.01)
        assert result["min_lr"] == 1e-6


class TestExponentialDefaults:
    def test_defaults(self):
        e = SchedulerParams.Exponential()
        assert e.gamma == 0.9
        assert e._name == "exponential"

    def test_params_with_num_iter(self):
        e = SchedulerParams.Exponential(factor=None)
        result = e.params(base_LR=0.01, num_iter=100)
        assert result == {"gamma": 0.9}

    def test_params_factor_overrides_gamma(self):
        e = SchedulerParams.Exponential(factor=0.01)
        result = e.params(base_LR=0.01, num_iter=100)
        expected_gamma = 0.01 ** (1.0 / 100)
        assert result["gamma"] == pytest.approx(expected_gamma)

    def test_params_no_num_iter_raises(self):
        e = SchedulerParams.Exponential()
        with pytest.raises(ValueError, match="num_iter must be set"):
            e.params(base_LR=0.01, num_iter=None)

    def test_params_uses_own_num_iter(self):
        e = SchedulerParams.Exponential(num_iter=50, factor=None)
        result = e.params(base_LR=0.01, num_iter=None)
        assert result == {"gamma": 0.9}


class TestCyclicDefaults:
    def test_defaults(self):
        c = SchedulerParams.Cyclic()
        assert c.mode == "triangular2"
        assert c.cycle_momentum is False
        assert c._name == "cyclic"

    def test_params_computes_lr_bounds(self):
        c = SchedulerParams.Cyclic()
        result = c.params(base_LR=0.01)
        assert result["base_lr"] == pytest.approx(0.01 / 4)
        assert result["max_lr"] == pytest.approx(0.01 * 4)

    def test_params_explicit_lr_bounds(self):
        c = SchedulerParams.Cyclic(base_lr=0.001, max_lr=0.1)
        result = c.params(base_LR=999.0)  # should be ignored
        assert result["base_lr"] == 0.001
        assert result["max_lr"] == 0.1


class TestLinearDefaults:
    def test_defaults(self):
        test = SchedulerParams.Linear()
        assert test.start_factor == 0.1
        assert test.end_factor == 1.0
        assert test._name == "linear"

    def test_params_uses_num_iter(self):
        test = SchedulerParams.Linear()
        result = test.params(base_LR=0.01, num_iter=200)
        assert result["total_iters"] == 200

    def test_params_explicit_total_iters(self):
        test = SchedulerParams.Linear(total_iters=50)
        result = test.params(base_LR=0.01, num_iter=200)
        assert result["total_iters"] == 50

    def test_params_no_iters_raises(self):
        test = SchedulerParams.Linear()
        with pytest.raises(ValueError, match="total_iters must be set"):
            test.params(base_LR=0.01, num_iter=None)


class TestCosineAnnealingDefaults:
    def test_defaults(self):
        ca = SchedulerParams.CosineAnnealing()
        assert ca.eta_min == 1e-7
        assert ca.T_max is None
        assert ca._name == "cosine_annealing"

    def test_params_uses_num_iter(self):
        ca = SchedulerParams.CosineAnnealing()
        result = ca.params(base_LR=0.01, num_iter=300)
        assert result["T_max"] == 300

    def test_params_explicit_T_max(self):
        ca = SchedulerParams.CosineAnnealing(T_max=150)
        result = ca.params(base_LR=0.01, num_iter=300)
        assert result["T_max"] == 150

    def test_params_no_T_max_raises(self):
        ca = SchedulerParams.CosineAnnealing()
        with pytest.raises(ValueError, match="T_max must be set"):
            ca.params(base_LR=0.01, num_iter=None)


class TestNoneScheduler:
    def test_defaults(self):
        ns = SchedulerParams.NoneScheduler()
        assert ns._name == "none"
        assert ns.params(base_LR=0.01) == {}


# ─── SchedulerParams.parse_dict ─────────────────────────────────────────────


class TestSchedulerParseDict:
    def test_parse_plateau(self):
        result = SchedulerParams.parse_dict({"name": "plateau", "patience": 20})
        assert isinstance(result, SchedulerParams.Plateau)
        assert result.patience == 20

    def test_parse_exponential(self):
        result = SchedulerParams.parse_dict({"name": "exponential", "gamma": 0.95})
        assert isinstance(result, SchedulerParams.Exponential)
        assert result.gamma == 0.95

    def test_parse_cyclic(self):
        result = SchedulerParams.parse_dict({"name": "cyclic", "step_size_up": 50})
        assert isinstance(result, SchedulerParams.Cyclic)
        assert result.step_size_up == 50

    def test_parse_linear(self):
        result = SchedulerParams.parse_dict({"name": "linear", "start_factor": 0.5})
        assert isinstance(result, SchedulerParams.Linear)
        assert result.start_factor == 0.5

    def test_parse_cosine_annealing(self):
        result = SchedulerParams.parse_dict({"name": "cosine_annealing", "T_max": 100})
        assert isinstance(result, SchedulerParams.CosineAnnealing)
        assert result.T_max == 100

    def test_parse_none(self):
        result = SchedulerParams.parse_dict({"name": "none"})
        assert isinstance(result, SchedulerParams.NoneScheduler)

    def test_parse_case_insensitive(self):
        result = SchedulerParams.parse_dict({"name": "Plateau"})
        assert isinstance(result, SchedulerParams.Plateau)

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            SchedulerParams.parse_dict({"name": "warmup"})

    def test_parse_does_not_mutate_input(self):
        d = {"name": "plateau", "patience": 5}
        original = dict(d)
        SchedulerParams.parse_dict(d)
        assert d == original

    def test_parse_invalid_name_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            SchedulerParams.parse_dict({"name": 3.14})

    def test_parse_default_name_is_none(self):
        result = SchedulerParams.parse_dict({})
        assert isinstance(result, SchedulerParams.NoneScheduler)


# ─── OptimizerMixin.set_optimizer / PPLR behavior ───────────────────────────

from quantem.core import config  # noqa: E402
from quantem.core.ml.optimizer_mixin import OptimizerMixin  # noqa: E402

torch = pytest.importorskip("torch") if config.get("has_torch") else None
requires_torch = pytest.mark.skipif(not config.get("has_torch"), reason="requires torch")


def _param(value=1.0):
    return torch.nn.Parameter(torch.tensor([value]))


class _FakeModel(OptimizerMixin):
    """Minimal concrete OptimizerMixin for exercising set_optimizer/reset_optimizer.

    ``groups`` is the dict[str, list[Parameter]] returned by get_optimization_parameters.
    Pass ``raise_on_params=True`` to prove the disable path short-circuits before the call.
    """

    def __init__(self, groups, raise_on_params=False):
        super().__init__()
        self._groups = groups
        self._raise_on_params = raise_on_params

    def get_optimization_parameters(self):
        if self._raise_on_params:
            raise AssertionError("get_optimization_parameters should not have been called")
        return self._groups


@requires_torch
class TestSetOptimizer:
    def test_single_optimizer_build(self):
        model = _FakeModel({"default": [_param()]})
        model.set_optimizer(OptimizerParams.Adam(lr=1e-3))
        assert model.has_optimizer()
        assert isinstance(model.optimizer, torch.optim.Adam)
        assert len(model.optimizer.param_groups) == 1
        assert model.optimizer.param_groups[0]["lr"] == 1e-3

    def test_single_optimizer_dict_shorthand(self):
        model = _FakeModel({"default": [_param()]})
        model.set_optimizer({"name": "sgd", "lr": 5e-2, "momentum": 0.9})
        assert isinstance(model.optimizer, torch.optim.SGD)
        assert model.optimizer.param_groups[0]["lr"] == 5e-2
        assert model.optimizer.param_groups[0]["momentum"] == 0.9

    def test_none_optimizer_removes_without_touching_params(self):
        # raise_on_params proves the disable path short-circuits before get_optimization_parameters
        model = _FakeModel({"default": [_param()]}, raise_on_params=True)
        model.set_optimizer(OptimizerParams.NoneOptimizer())
        assert model.optimizer is None
        assert not model.has_optimizer()
        assert model.optimizer_params == {
            OptimizerMixin.DEFAULT_OPTIMIZER_KEY: OptimizerParams.NoneOptimizer()
        }

    def test_multi_group_pplr_applies_per_group_lr(self):
        model = _FakeModel({"descan": [_param()], "scan_positions": [_param(2.0)]})
        model.set_optimizer(
            {
                "descan": OptimizerParams.SGD(lr=1e-2),
                "scan_positions": OptimizerParams.SGD(lr=1e-3),
            }
        )
        groups = model.optimizer.param_groups
        assert len(groups) == 2
        # key order is preserved (descan, then scan_positions)
        assert groups[0]["lr"] == 1e-2
        assert groups[1]["lr"] == 1e-3

    def test_key_mismatch_raises(self):
        model = _FakeModel({"default": [_param()]})
        with pytest.raises(ValueError, match="do not match"):
            model.set_optimizer(
                {"descan": OptimizerParams.SGD(lr=1e-2), "scan_positions": OptimizerParams.SGD()}
            )

    def test_mixed_optimizer_classes_raises(self):
        model = _FakeModel({"a": [_param()], "b": [_param()]})
        with pytest.raises(ValueError, match="same optimizer type"):
            model.set_optimizer(
                {"a": OptimizerParams.Adam(lr=1e-3), "b": OptimizerParams.SGD(lr=1e-3)}
            )

    def test_reset_optimizer_on_unconfigured_model_is_noop(self):
        model = _FakeModel({"default": [_param()]})
        # fresh model defaults to {"default": NoneOptimizer()}
        model.reset_optimizer()
        assert model.optimizer is None
        assert not model.has_optimizer()

    def test_set_scheduler_base_lr_uses_max_group_lr(self):
        model = _FakeModel({"descan": [_param()], "scan_positions": [_param(2.0)]})
        model.set_optimizer(
            {
                "descan": OptimizerParams.SGD(lr=1e-2),
                "scan_positions": OptimizerParams.SGD(lr=1e-3),
            }
        )
        model.set_scheduler(SchedulerParams.Plateau(), num_iter=10)
        assert model.scheduler is not None
        # Plateau min_lr defaults to base_LR / 20, with base_LR = max group lr (1e-2)
        assert model.scheduler.min_lrs[0] == pytest.approx(1e-2 / 20)


class TestSchedulerParamsArePure:
    """Regression: params() mutated the dataclass, so a shared instance baked in the
    first call's derived values (min_lr, gamma, bounds, horizons) for every later
    optimizer or reconstruct() call."""

    def test_plateau_min_lr_follows_base_lr(self):
        p = SchedulerParams.Plateau()
        assert p.params(base_LR=1.0)["min_lr"] == pytest.approx(1 / 20)
        assert p.params(base_LR=10.0)["min_lr"] == pytest.approx(10 / 20)
        assert p.min_lr is None  # instance untouched

    def test_exponential_gamma_follows_num_iter(self):
        p = SchedulerParams.Exponential(factor=0.5)
        g10 = p.params(base_LR=1.0, num_iter=10)["gamma"]
        g100 = p.params(base_LR=1.0, num_iter=100)["gamma"]
        assert g10 == pytest.approx(0.5 ** (1 / 10))
        assert g100 == pytest.approx(0.5 ** (1 / 100))
        assert p.num_iter is None and p.gamma == 0.9

    def test_cyclic_bounds_follow_base_lr(self):
        p = SchedulerParams.Cyclic()
        first = p.params(base_LR=1.0)
        second = p.params(base_LR=2.0)
        assert second["base_lr"] == pytest.approx(2 * first["base_lr"])
        assert second["max_lr"] == pytest.approx(2 * first["max_lr"])
        assert p.base_lr is None and p.max_lr is None

    def test_linear_and_cosine_follow_num_iter(self):
        lin = SchedulerParams.Linear()
        assert lin.params(base_LR=1.0, num_iter=5)["total_iters"] == 5
        assert lin.params(base_LR=1.0, num_iter=50)["total_iters"] == 50
        assert lin.total_iters is None
        cos = SchedulerParams.CosineAnnealing()
        assert cos.params(base_LR=1.0, num_iter=5)["T_max"] == 5
        assert cos.params(base_LR=1.0, num_iter=50)["T_max"] == 50
        assert cos.T_max is None

    def test_explicit_values_still_win(self):
        p = SchedulerParams.Plateau(min_lr=1e-6)
        assert p.params(base_LR=1.0)["min_lr"] == 1e-6
        e = SchedulerParams.Exponential(gamma=0.8)
        assert e.params(base_LR=1.0, num_iter=10)["gamma"] == 0.8
