import gzip
import json

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import zarr

import quantem
from quantem.core.io.serialize import AutoSerialize, load, print_file
from quantem.core.ml.activation_functions import get_activation_function

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
        self.scalar_array = np.array(2.5)
        self.count = 123
        self.child = Child()
        self._private = "secret"


def test_dummy_save_load_and_skip(tmp_path, capsys):
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
        np.testing.assert_array_equal(loaded_full.scalar_array, 2.5, err_msg=store)
        assert loaded_full.count == d.count, store
        np.testing.assert_allclose(loaded_full.child.values, d.child.values, err_msg=store)
        assert loaded_full.child.flag is True, store
        assert loaded_full._private == "secret", store
        assert not hasattr(loaded_full, "_autoserialize_skip_names"), store

        # the file records the quantem version that wrote it
        print_file(tgt)
        assert f"(quantem {quantem.__version__})" in capsys.readouterr().out, store

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


_unpickled = []


def _record(tag):
    _unpickled.append(tag)
    return tag


class _Payload:
    """Unpickles to a call of _record, which shows whether loading ran code from the file."""

    def __init__(self, tag):
        self.tag = tag

    def __reduce__(self):
        return (_record, (self.tag,))


class _Net(nn.Module):
    """A module from outside quantem and torch, whose methods have not been checked."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((2,), 3.0))


def _local_module() -> nn.Module:
    class Local(nn.Module):  # cannot be imported by name, so it is pickled
        pass

    return Local()


class Untrusted(AutoSerialize):
    def __init__(self):
        self.net = _Net()
        self.payloads = [_Payload("new")]
        self.encode = json.dumps
        self.local = _local_module()


def test_values_that_can_run_code_load_only_with_allow_pickle(tmp_path):
    """Dill fallbacks, in a list or in the older bare-array layout, and modules and
    functions from outside quantem and torch load only with allow_pickle=True;
    without it, nothing stored in them runs."""
    _unpickled.clear()
    tgt = tmp_path / "untrusted"
    Untrusted().save(tgt)
    # older files stored dill fallbacks as gzip-compressed bytes in a bare uint8 array
    legacy = np.frombuffer(gzip.compress(dill.dumps(_Payload("legacy"))), dtype=np.uint8)
    root = zarr.open_group(tgt, mode="r+")
    root.create_array("legacy", shape=legacy.shape, dtype=legacy.dtype)[...] = legacy

    with pytest.raises(ValueError, match="allow_pickle=True") as error:
        load(tgt)
    for entry in (
        f"net ({_Net.__module__}._Net, from outside quantem and torch)",
        "payloads/0 (dill pickle)",
        "legacy (dill pickle)",
        "encode (json.dumps, from outside quantem and torch)",
        "local (dill pickle)",
    ):
        assert entry in str(error.value), entry
    assert _unpickled == []

    loaded = load(tgt, allow_pickle=True)
    assert torch.equal(loaded.net.scale, torch.full((2,), 3.0))
    assert loaded.payloads == ["new"]
    assert loaded.legacy == "legacy"
    assert loaded.encode is json.dumps
    assert type(loaded.local).__name__ == "Local"
    assert sorted(_unpickled) == ["legacy", "new"]

    # the file can only name classes that are stored by their attributes
    meta = dict(root.attrs["_autoserialize"])
    root.attrs["_autoserialize"] = {**meta, "class_module": "subprocess", "class_name": "Popen"}
    with pytest.raises(ValueError, match="subprocess.Popen is not an AutoSerialize class"):
        load(tgt, allow_pickle=True)


class TestModule(torch.nn.Module, AutoSerialize):
    __test__ = False

    def __init__(self):
        super().__init__()
        # more than ten layers, so loading them out of order changes the output
        self.layers = nn.Sequential(*(nn.Linear(3, 3) for _ in range(11)))
        # a module that holds a function, as CNN2d holds torch.cat
        self.activation = get_activation_function(torch.tanh, torch.float32)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.5)
        self.rng = np.random.default_rng(0)

    def forward(self, x):
        return self.activation(self.layers(x))

    def train_step(self):
        self.opt.zero_grad()
        self(torch.from_numpy(self.rng.random((1, 3), dtype=np.float32))).sum().backward()
        self.opt.step()
        self.sched.step()


def test_torch_nn_module_roundtrip(tmp_path):
    """A model with its optimizer, scheduler and random generator loads without
    pickle and trains on as the original does, so the optimizer holds the loaded
    model's parameters."""
    mod = TestModule()
    mod.train_step()
    tgt = tmp_path / "mod.zip"
    mod.save(str(tgt), mode="w")
    loaded = load(str(tgt))

    x = torch.randn(2, 3)
    assert torch.equal(loaded(x), mod(x))
    assert loaded.sched.optimizer is loaded.opt

    mod.train_step()
    loaded.train_step()
    for (name, before), after in zip(mod.named_parameters(), loaded.parameters()):
        assert torch.equal(before, after), name
    assert loaded.sched.get_last_lr() == mod.sched.get_last_lr()


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
