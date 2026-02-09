import pytest
import torch

from quantem.core import config


def test_config_set_device():
    config.set_device("cpu")
    assert config.get_device() == "cpu"
    if torch.cuda.is_available():
        config.set_device("gpu")
        assert config.get_device() == "cuda:0"
        config.set_device("GPU")
        assert config.get_device() == "cuda:0"
        config.set_device(0)
        assert config.get_device() == "cuda:0"
        NUM_DEVICES = torch.cuda.device_count()
        if NUM_DEVICES > 0:
            config.set_device(NUM_DEVICES - 1)
            assert config.get_device() == f"cuda:{NUM_DEVICES - 1}"
        config.refresh()
        assert config.get_device() == "cpu"
    else:
        with pytest.raises(RuntimeError):
            config.set_device("cuda:0")

    if torch.mps.is_available():
        config.set_device("mps")
        assert config.get_device() == "mps"
        config.set_device("gpu")
        assert config.get_device() == "mps"
        config.set_device("GPU")
        assert config.get_device() == "mps"
    else:
        with pytest.raises(RuntimeError):
            config.set_device("mps")


def test_config_update_defaults():
    start_dtype = config.get("dtype_real")
    config.set({"dtype_real": "int32"})
    assert config.get("dtype_real") == "int32"
    config.refresh()
    assert config.get("dtype_real") == start_dtype
    config.update_defaults({"dtype_real": "int32"})
    assert config.get("dtype_real") == "int32"
    config.refresh()
    assert config.get("dtype_real") == "int32"
    config.update_defaults({"dtype_real": start_dtype})
