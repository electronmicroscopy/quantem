import warnings

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


def test_collect_env_flat_and_nested():
    collected = config.collect_env(
        env={
            "QUANTEM_VERBOSE": "3",
            "QUANTEM_VIZ__CMAP": "magma",
            "PATH": "/usr/bin",
        }
    )
    assert collected == {"verbose": 3, "viz": {"cmap": "magma"}}


def test_collect_env_interprets_values():
    collected = config.collect_env(
        env={
            "QUANTEM_VERBOSE": "3",
            "QUANTEM_HAS_CUPY": "False",
            "QUANTEM_VIZ__DEFAULT_COLORS": "['#000000', '#ffffff']",
        }
    )
    assert collected["verbose"] == 3
    assert collected["has_cupy"] is False
    assert collected["viz"]["default_colors"] == ["#000000", "#ffffff"]


def test_collect_env_ignores_quantem_config():
    """QUANTEM_CONFIG names the config directory, it is not a config key."""
    assert config.collect_env(env={"QUANTEM_CONFIG": "/some/dir"}) == {}


def test_collect_env_overrides_yaml(tmp_path):
    (tmp_path / "quantem.yaml").write_text("verbose: 5\nviz:\n  cmap: viridis\n")

    from_yaml = config.collect(path=tmp_path, env={})
    assert from_yaml == {"verbose": 5, "viz": {"cmap": "viridis"}}

    merged = config.collect(path=tmp_path, env={"QUANTEM_VERBOSE": "9"})
    assert merged == {"verbose": 9, "viz": {"cmap": "viridis"}}


def test_check_key_val_known_key_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.check_key_val("verbose", 1)
    assert caught == []


def test_check_key_val_nested_key_silent():
    """Only the top-level key is checked, a key under it is not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.check_key_val("viz.not_a_real_key", 1)
    assert caught == []


def test_check_key_val_unknown_key_warns_and_still_sets():
    with pytest.warns(UserWarning, match='Unknown configuration key "not_a_real_key"'):
        with config.set({"not_a_real_key": 7}):
            assert config.get("not_a_real_key") == 7
    assert config.get("not_a_real_key", None) is None


def test_check_key_val_deprecated_key_warns_as_deprecated():
    config.deprecations["old_key"] = "verbose"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.check_key_val("old_key", 1)
        assert len(caught) == 1
        assert "has been deprecated" in str(caught[0].message)
    finally:
        del config.deprecations["old_key"]


def test_set_context_manager_rolls_back():
    d = {"verbose": 1}
    with config.set({"verbose": 2, "precision": "float64"}, config=d):
        assert d == {"verbose": 2, "precision": "float64"}
    assert d == {"verbose": 1}


def test_set_context_manager_rolls_back_nested():
    d = {"verbose": 1}
    with config.set({"viz.cmap": "magma"}, config=d):
        assert d == {"verbose": 1, "viz": {"cmap": "magma"}}
    assert d == {"verbose": 1}

    d = {"viz": {"cmap": "gray"}}
    with config.set({"viz.cmap": "magma"}, config=d):
        assert d == {"viz": {"cmap": "magma"}}
    assert d == {"viz": {"cmap": "gray"}}


def test_set_context_manager_rolls_back_global_config():
    start = config.get("verbose")
    with config.set({"verbose": 42}):
        assert config.get("verbose") == 42
    assert config.get("verbose") == start
