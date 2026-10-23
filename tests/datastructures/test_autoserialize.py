import numpy as np
import torch
import torch.nn as nn

from quantem.core.io.serialize import AutoSerialize, load

# ─────────────────────────────────────────────── #
# 1) Simple “Dummy” with arr/count/child/_private #
# ─────────────────────────────────────────────── #


class Child(AutoSerialize):
    def __init__(self):
        self.values = np.arange(4).reshape(2, 2)
        self.flag = True


class Dummy(AutoSerialize):
    def __init__(self):
        self.arr = np.linspace(0, 1, 5)
        self.count = 123
        self.child = Child()
        self._private = "secret"


def test_dummy_save_load_and_skip(tmp_path):
    """
    For both the zip and the directory store:
    1.  Save/load Dummy with no skips: all fields rehydrate.
    2.  Then save but skip 'count' and skip 'child', verifying each is dropped.
    3.  Then save but skip '_private', verifying only _private is dropped.
    """
    for store in ("zip", "dir"):
        suffix = ".zip" if store == "zip" else "_dir"
        d = Dummy()

        # ——— (a) no skips ———
        tgt = tmp_path / f"dummy_full{suffix}"
        d.save(str(tgt), mode="w", store=store)
        loaded_full = load(str(tgt))

        # types
        assert isinstance(loaded_full, Dummy), store
        assert isinstance(loaded_full.child, Child), store

        # values round-trip
        np.testing.assert_allclose(loaded_full.arr, d.arr, err_msg=store)
        assert loaded_full.count == d.count, store
        np.testing.assert_allclose(loaded_full.child.values, d.child.values, err_msg=store)
        assert loaded_full.child.flag is True, store
        assert loaded_full._private == "secret", store

        # ——— (b) skip 'count' and skip 'child' ———
        tgt2 = tmp_path / f"dummy_skip_cd{suffix}"
        d.save(str(tgt2), mode="w", store=store, skip=["count", "child"])
        loaded_skip_cd = load(str(tgt2))

        # arr still there
        np.testing.assert_allclose(loaded_skip_cd.arr, d.arr, err_msg=store)

        # count & child should be gone
        assert not hasattr(loaded_skip_cd, "count"), store
        assert not hasattr(loaded_skip_cd, "child"), store

        # _private still exists
        assert loaded_skip_cd._private == "secret", store

        # ——— (c) skip "_private" only ———
        tgt3 = tmp_path / f"dummy_skip_priv{suffix}"
        d.save(str(tgt3), mode="w", store=store, skip=["_private"])
        loaded_skip_priv = load(str(tgt3))

        np.testing.assert_allclose(loaded_skip_priv.arr, d.arr, err_msg=store)
        assert loaded_skip_priv.count == d.count, store
        assert isinstance(loaded_skip_priv.child, Child), store
        assert not hasattr(loaded_skip_priv, "_private"), store


# ──────────────────────────── #
# 2) Nested lists/tuples/dicts #
# ──────────────────────────── #


class subClass(AutoSerialize):
    def __init__(self):
        self.info = {"key": "value"}


class TestingClass(AutoSerialize):
    __test__ = False  # tell pytest this is not a test class despite name starting with Test

    def __init__(self):
        self.data = np.random.rand(104, 100)

        # Top‐level dict and top‐level child:
        self.info = {"key2": "value2"}
        self.child = subClass()

        # A simple list and tuple containing a NumPy array:
        self.list_a = ["a", 2, np.ones((2, 3, 4))]
        self.tuple_a = ("a", 2, np.ones((2, 3, 4)))

        # Nested list/tuple of primitives:
        self.list_nested = ["a", ["b", ["c", ["d"]]]]
        self.tuple_nested = ("a", ("b", ("c", ("d",))))

        # Now, a list and a tuple that each contain a subClass instance:
        self.list_with_child = ["start", subClass()]
        self.tuple_with_child = ("start", subClass())

    def method(self):
        return self.data.sum()


def test_nested_save_load_and_skip(tmp_path):
    """
    1.  Save/load TestingClass with no skips:
        - `data` (ndarray)
        - top‐level dict `info`
        - nested list + nested tuple (with ndarrays)
        - top‐level child.info
        - nested subClass inside list_with_child and tuple_with_child
    2.  Save with skip=["info","child"]: `info` and top‐level `child` are removed entirely,
        but the *nested* subClass instances inside list_with_child/tuple_with_child remain.
    """
    obj = TestingClass()
    for skip in ([], ["info", "child"]):
        tgt = tmp_path / f"nested_{len(skip)}.zip"
        obj.save(str(tgt), mode="w", skip=skip)
        loaded = load(str(tgt))

        # — Top‐level class —
        assert isinstance(loaded, TestingClass), skip

        # — data array —
        np.testing.assert_allclose(loaded.data, obj.data, err_msg=str(skip))

        # — top‐level dicts and child: kept, or gone when skipped —
        if skip:
            assert not hasattr(loaded, "info")
            assert not hasattr(loaded, "child")
        else:
            assert loaded.info == obj.info
            assert isinstance(loaded.child, subClass)
            assert loaded.child.info == obj.child.info

        # — list_a / tuple_a: first two entries + ndarray —
        assert loaded.list_a[:2] == obj.list_a[:2], skip
        np.testing.assert_allclose(loaded.list_a[2], obj.list_a[2], err_msg=str(skip))
        assert loaded.tuple_a[:2] == obj.tuple_a[:2], skip
        np.testing.assert_allclose(loaded.tuple_a[2], obj.tuple_a[2], err_msg=str(skip))

        # — nested list & nested tuple of primitives —
        assert loaded.list_nested == obj.list_nested, skip
        assert loaded.tuple_nested == obj.tuple_nested, skip

        # — list_with_child / tuple_with_child: nested subClass survives —
        for name in ("list_with_child", "tuple_with_child"):
            assert getattr(loaded, name)[0] == "start", (name, skip)
            assert isinstance(getattr(loaded, name)[1], subClass), (name, skip)
            if not skip:
                assert getattr(loaded, name)[1].info == getattr(obj, name)[1].info, name


# ──────────────────────────── #
# 3) PyTorch‐specific behavior #
# ──────────────────────────── #


class TorchWithOpt(AutoSerialize):
    def __init__(self):
        self.data = torch.rand(100, 100)
        self.param = self.data.clone().detach().requires_grad_(True)
        self.opt = torch.optim.Adam([self.data.detach().clone()], lr=0.01)

    def method(self):
        return self.data.sum()


def test_torch_tensor_and_optimizer(tmp_path):
    obj = TorchWithOpt()
    tgt = tmp_path / "opt.zip"
    obj.save(str(tgt), mode="w")
    loaded = load(str(tgt))

    assert torch.allclose(obj.data, loaded.data)
    assert isinstance(loaded.opt, torch.optim.Adam)
    assert loaded.opt.param_groups[0]["lr"] == 0.01


class TestModule(torch.nn.Module, AutoSerialize):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.param = torch.nn.Parameter(torch.randn(3, requires_grad=True))

    def forward(self, x):
        return self.linear(x)


def test_torch_nn_module_roundtrip(tmp_path):
    mod = TestModule()
    tgt = tmp_path / "mod.zip"
    mod.save(str(tgt), mode="w")
    mod_reload = load(str(tgt))

    assert isinstance(mod_reload, TestModule)
    out = mod_reload.forward(torch.randn(1, 10))
    assert out.shape[-1] == 5

    for key in mod.state_dict():
        assert torch.allclose(mod.state_dict()[key], mod_reload.state_dict()[key])


class TestHybrid(AutoSerialize, nn.Module):
    __test__ = False

    def __init__(self):
        nn.Module.__init__(self)
        super().__init__()
        self._p1 = nn.Parameter(torch.randn(3, 3))

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, value):
        with torch.no_grad():
            self._p1.copy_(value)


def test_hybrid_module_roundtrip(tmp_path):
    mod = TestHybrid()
    tgt = tmp_path / "hybrid.zip"
    mod.save(str(tgt), mode="w")
    mod_reload = load(str(tgt))

    for key in mod.state_dict():
        assert torch.allclose(mod.state_dict()[key], mod_reload.state_dict()[key])


class WithSets(AutoSerialize):
    def __init__(self):
        self.names = {"C10", "C12", "rotation_angle"}
        self.numbers = {1, 2, 3}
        self.empty = set()


def test_set_roundtrip(tmp_path):
    """Sets load back as sets, whether their items are strings or numbers."""
    target = tmp_path / "sets.zip"
    WithSets().save(str(target), mode="w")
    loaded = load(str(target))

    assert loaded.names == {"C10", "C12", "rotation_angle"}
    assert loaded.numbers == {1, 2, 3}
    assert loaded.empty == set()
