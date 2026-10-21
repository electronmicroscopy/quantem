import numpy as np
import pytest
import torch
from ase import Atoms

from quantem.core.datastructures import Dataset2d
from quantem.core.io import load
from quantem.core.io.serialize import AutoSerialize, Bundle


class _Holder(AutoSerialize):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _partial_atoms() -> Atoms:
    atoms = Atoms(
        "NbVZr",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 0]],
        cell=[3.2, 3.2, 3.2],
        pbc=[True, True, False],
    )
    atoms.set_array("occupancy", np.array([0.4, 1.0, 0.6]))
    atoms.set_tags([1, 2, 1])
    atoms.set_masses([92.9, 50.9, 91.2])
    atoms.info["spacegroup_number"] = 229
    atoms.info["not_json"] = object()
    return atoms


def test_ase_atoms_round_trip_keeps_occupancy(tmp_path):
    atoms = _partial_atoms()
    path = tmp_path / "atoms.zip"
    _Holder(atoms=atoms).save(path, mode="o")
    back = load(path).atoms

    assert isinstance(back, Atoms)
    assert back.get_chemical_symbols() == ["Nb", "V", "Zr"]
    assert np.allclose(back.get_positions(), atoms.get_positions())
    assert np.allclose(back.get_cell(), atoms.get_cell())
    assert np.array_equal(back.get_pbc(), atoms.get_pbc())
    assert np.allclose(back.arrays["occupancy"], [0.4, 1.0, 0.6])
    assert np.array_equal(back.get_tags(), [1, 2, 1])
    assert np.allclose(back.get_masses(), [92.9, 50.9, 91.2])
    assert back.info == {"spacegroup_number": 229}


def test_torch_device_round_trip(tmp_path):
    path = tmp_path / "device.zip"
    _Holder(device=torch.device("cpu"), name="x").save(path, mode="o")
    back = load(path)
    assert isinstance(back.device, torch.device)
    assert back.device == torch.device("cpu")
    assert back.name == "x"


def test_bundle_round_trip(tmp_path):
    image = Dataset2d.from_array(np.arange(12.0).reshape(3, 4), name="adf")
    bundle = Bundle(adf=image, atoms=_partial_atoms(), note="IM689")
    assert "adf: Dataset2d" in repr(bundle)
    path = tmp_path / "bundle.zip"
    bundle.save(path, mode="o")
    back = load(path)
    assert isinstance(back, Bundle)
    assert np.array_equal(back.adf.array, image.array)
    assert back.note == "IM689"
    assert np.allclose(back.atoms.arrays["occupancy"], [0.4, 1.0, 0.6])


@pytest.mark.parametrize("name", ["save", "print_tree", "_recursive_save"])
def test_bundle_rejects_reserved_names(name):
    with pytest.raises(ValueError, match="shadow"):
        Bundle(**{name: 1})
