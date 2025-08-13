import numpy as np
import pytest
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


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_dummy_save_load_and_skip(tmp_path, store):
    """
    1.  Save/load Dummy with no skips: all fields rehydrate.
    2.  Then save but skip 'count' and skip 'child', verifying each is dropped.
    3.  Then save but skip '_private', verifying only _private is dropped.
    """
    # ——— (a) no skips ———
    d = Dummy()
    tgt = tmp_path / ("dummy_full.zip" if store == "zip" else "dummy_full_dir")
    d.save(str(tgt), mode="w", store=store)
    loaded_full = load(str(tgt))

    # types
    assert isinstance(loaded_full, Dummy)
    assert isinstance(loaded_full.child, Child)

    # values round-trip
    np.testing.assert_allclose(loaded_full.arr, d.arr)
    assert loaded_full.count == d.count
    np.testing.assert_allclose(loaded_full.child.values, d.child.values)
    assert loaded_full.child.flag is True
    assert loaded_full._private == "secret"

    # ——— (b) skip 'count' and skip 'child' ———
    tgt2 = tmp_path / ("dummy_skip_cd.zip" if store == "zip" else "dummy_skip_cd_dir")
    d.save(str(tgt2), mode="w", store=store, skip=["count", "child"])
    loaded_skip_cd = load(str(tgt2))

    # arr still there
    np.testing.assert_allclose(loaded_skip_cd.arr, d.arr)

    # count & child should be gone
    assert not hasattr(loaded_skip_cd, "count")
    assert not hasattr(loaded_skip_cd, "child")

    # _private still exists
    assert loaded_skip_cd._private == "secret"

    # ——— (c) skip "_private" only ———
    tgt3 = tmp_path / ("dummy_skip_priv.zip" if store == "zip" else "dummy_skip_priv_dir")
    d.save(str(tgt3), mode="w", store=store, skip=["_private"])
    loaded_skip_priv = load(str(tgt3))

    np.testing.assert_allclose(loaded_skip_priv.arr, d.arr)
    assert loaded_skip_priv.count == d.count
    assert isinstance(loaded_skip_priv.child, Child)
    assert not hasattr(loaded_skip_priv, "_private")


# ──────────────────────────── #
# 2) Nested lists/tuples/dicts #
# ──────────────────────────── #


class subClass(AutoSerialize):
    def __init__(self):
        self.info = {"key": "value"}


class TestingClass(AutoSerialize):
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


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_nested_save_load_roundtrip(tmp_path, store):
    """
    1.  Save/load TestingClass with no skips:
        - `data` (ndarray)
        - top‐level dict `info`
        - nested list + nested tuple (with ndarrays)
        - top‐level child.info
        - nested subClass inside list_with_child and tuple_with_child
    2.  Verify each nested structure (including the nested subClass objects) round‐trips.
    """
    obj = TestingClass()
    tgt = tmp_path / ("nested.zip" if store == "zip" else "nested_dir")
    obj.save(str(tgt), mode="w", store=store)
    loaded = load(str(tgt))

    # — Top‐level class —
    assert isinstance(loaded, TestingClass)

    # — data array —
    np.testing.assert_allclose(loaded.data, obj.data)

    # — top‐level dicts and child —
    assert loaded.info == obj.info
    assert isinstance(loaded.child, subClass)
    assert loaded.child.info == obj.child.info

    # — list_a: first two entries + ndarray —
    assert loaded.list_a[:2] == obj.list_a[:2]
    np.testing.assert_allclose(loaded.list_a[2], obj.list_a[2])

    # — tuple_a: first two entries + ndarray —
    assert loaded.tuple_a[:2] == obj.tuple_a[:2]
    np.testing.assert_allclose(loaded.tuple_a[2], obj.tuple_a[2])

    # — nested list & nested tuple of primitives —
    assert loaded.list_nested == obj.list_nested
    assert loaded.tuple_nested == obj.tuple_nested

    # — list_with_child: check nested subClass —
    assert loaded.list_with_child[0] == "start"
    assert isinstance(loaded.list_with_child[1], subClass)
    assert loaded.list_with_child[1].info == obj.list_with_child[1].info

    # — tuple_with_child: check nested subClass —
    assert loaded.tuple_with_child[0] == "start"
    assert isinstance(loaded.tuple_with_child[1], subClass)
    assert loaded.tuple_with_child[1].info == obj.tuple_with_child[1].info


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_nested_skip_fields(tmp_path, store):
    """
    1.  Save with skip=["info","child"], load back.
    2.  Only `data`, `list_a`, `list_nested`, `tuple_a`, `tuple_nested`,
        `list_with_child` and `tuple_with_child` survive.
    3.  `info` and top‐level `child` are removed entirely, but the *nested*
        subClass instances inside list_with_child/tuple_with_child should remain.
    """
    obj = TestingClass()
    tgt = tmp_path / ("skip_nested.zip" if store == "zip" else "skip_nested_dir")
    obj.save(str(tgt), mode="w", store=store, skip=["info", "child"])
    loaded = load(str(tgt))

    # — data must still be present —
    np.testing.assert_allclose(loaded.data, obj.data)

    # — top‐level info & child must be gone —
    assert not hasattr(loaded, "info")
    assert not hasattr(loaded, "child")

    # — list_a and its ndarray still rehydrate —
    assert loaded.list_a[:2] == obj.list_a[:2]
    np.testing.assert_allclose(loaded.list_a[2], obj.list_a[2])

    # — tuple_a and its ndarray still rehydrate —
    assert loaded.tuple_a[:2] == obj.tuple_a[:2]
    np.testing.assert_allclose(loaded.tuple_a[2], obj.tuple_a[2])

    # — nested list & nested tuple still rehydrate —
    assert loaded.list_nested == obj.list_nested
    assert loaded.tuple_nested == obj.tuple_nested

    # — list_with_child: nested subClass should survive —
    assert loaded.list_with_child[0] == "start"
    assert isinstance(loaded.list_with_child[1], subClass)

    # — tuple_with_child: nested subClass should survive —
    assert loaded.tuple_with_child[0] == "start"
    assert isinstance(loaded.tuple_with_child[1], subClass)


# ──────────────────────────── #
# 3) PyTorch‐specific behavior #
# ──────────────────────────── #


class TorchOnly(AutoSerialize):
    def __init__(self):
        self.data = torch.rand(100, 100)

    def method(self):
        return self.data.sum()


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_torch_tensor_roundtrip(tmp_path, store):
    obj = TorchOnly()
    tgt = tmp_path / ("tensor.zip" if store == "zip" else "tensor_dir")
    obj.save(str(tgt), mode="w", store=store)
    loaded = load(str(tgt))
    assert torch.allclose(obj.data, loaded.data)


class TorchWithOpt(AutoSerialize):
    def __init__(self):
        self.data = torch.rand(100, 100)
        self.param = self.data.clone().detach().requires_grad_(True)
        self.opt = torch.optim.Adam([self.data.detach().clone()], lr=0.01)

    def method(self):
        return self.data.sum()


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_torch_tensor_and_optimizer(tmp_path, store):
    obj = TorchWithOpt()
    tgt = tmp_path / ("opt.zip" if store == "zip" else "opt_dir")
    obj.save(str(tgt), mode="w", store=store)
    loaded = load(str(tgt))

    assert torch.allclose(obj.data, loaded.data)
    assert isinstance(loaded.opt, torch.optim.Adam)
    assert set(obj.opt.state_dict().keys()) == set(loaded.opt.state_dict().keys())


class TestModule(torch.nn.Module, AutoSerialize):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.param = torch.nn.Parameter(torch.randn(3, requires_grad=True))

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_torch_nn_module_roundtrip(tmp_path, store):
    mod = TestModule()
    tgt = tmp_path / ("mod.zip" if store == "zip" else "mod_dir")
    mod.save(str(tgt), mode="w", store=store)
    mod_reload = load(str(tgt))

    assert isinstance(mod_reload, TestModule)
    out = mod_reload.forward(torch.randn(1, 10))
    assert out.shape[-1] == 5

    for key in mod.state_dict():
        assert torch.allclose(mod.state_dict()[key], mod_reload.state_dict()[key])


class TestHybrid(AutoSerialize, nn.Module):
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


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_hybrid_module_roundtrip(tmp_path, store):
    mod = TestHybrid()
    tgt = tmp_path / ("hybrid.zip" if store == "zip" else "hybrid_dir")
    mod.save(str(tgt), mode="w", store=store)
    mod_reload = load(str(tgt))

    for key in mod.state_dict():
        assert torch.allclose(mod.state_dict()[key], mod_reload.state_dict()[key])
