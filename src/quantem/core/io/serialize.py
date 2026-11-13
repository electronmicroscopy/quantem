"""Save quantem objects as Zarr trees, in a directory or a .zip file, and load them back.

Arrays, tensors, scalars, strings, paths, containers and ase.Atoms are stored as
Zarr arrays and JSON attributes. AutoSerialize objects, dataclasses, torch
modules and schedulers are stored by their attributes, and torch optimizers by
their state_dict, and are rebuilt without unpickling. A value held in several
places, such as a parameter of a model and its optimizer, is stored once and
stays shared.

Values without such a form are pickled. Unpickling can run code stored in the
file, so `load` reads them only with ``allow_pickle=True``. The same holds for
modules, optimizers, schedulers and dataclasses defined outside quantem and
torch, whose code was not written to take attributes from a file.
"""

import dataclasses
import gzip
import importlib
import inspect
import io
import json
import logging
import os
import shutil
import tempfile
import zlib
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal, Sequence, Union, cast
from zipfile import ZipFile

import dill
import numpy as np
import torch
import zarr
from zarr.storage import LocalStore

SkipSpec = Union[str, type, Sequence[Union[str, type]]]

_FORMAT_VERSION = 1
# Group attributes that describe the stored values rather than being one
_METADATA_KEYS = frozenset(
    {
        "_autoserialize",
        "_autoserialize_skip_names",
        "_autoserialize_skip_types",
        "_container_type",
        "_dict_keys",
        "_sequence_encoding",
    }
)
# Attributes "<name><suffix>" that say how to rebuild the attribute "<name>"
_FLAG_SUFFIXES = (".is_path", ".is_ref", ".is_torch_device", ".is_torch_dtype", ".torch_save")
# Torch classes stored as groups of their state, like AutoSerialize objects
_TORCH_STATEFUL = (
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
)
# Packages whose torch classes and dataclasses load without allow_pickle
_TRUSTED_PACKAGES = ("quantem", "torch")
# Group marker -> (array holding the pickle, description for error messages);
# files written before modules, optimizers and schedulers were stored by
# their attributes hold them pickled
_PICKLED = {
    "_torch_whole_module": ("module", "torch module"),
    "_torch_optimizer": ("optimizer", "torch optimizer"),
    "_torch_scheduler": ("scheduler", "torch scheduler"),
    "_dill": ("payload", "dill pickle"),
}
_BIT_GENERATORS = {
    name: getattr(np.random, name) for name in ("PCG64", "MT19937", "Philox", "SFC64")
}
_SKIP = object()  # returned by _Reader for a value that is not restored


class AutoSerialize:
    """Mixin that saves every instance attribute of an object and loads it back.

    Arrays and tensors are stored as Zarr arrays, scalars and strings as JSON
    attributes, and lists, tuples, sets, dicts, dataclasses, torch modules,
    optimizers and schedulers and nested AutoSerialize objects as Zarr groups.
    Values of any other type are pickled with dill; see `load`. Torch
    generators keep their state; loggers and NumPy generators are recreated
    without it.

    Loading does not call ``__init__``. A class whose stored state differs from
    its working state defines ``_post_load(self)`` to convert it after loading.
    """

    __autoserialize_marker__ = ("AutoSerialize", __module__)

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: SkipSpec = (),
        compression_level: int | None = 4,
    ) -> None:
        """
        Save the object to a Zarr directory or a .zip file.

        Parameters
        ----------
        path : str or Path
            Target path. A '.zip' extension writes one zip file, anything else a directory.
        mode : {'w', 'o'}
            'w' refuses to replace an existing path, 'o' overwrites it.
        store : {'auto', 'zip', 'dir'}
            Storage format. 'auto' infers it from the extension.
        skip : str, type, or list of (str or type)
            Attribute names, at any depth, and types not to save.
        compression_level : int or None
            Zstandard (Blosc) level 0-9 for arrays; 0 stores them uncompressed.

        Notes
        -----
        The file records the quantem version that wrote it and the skipped names
        and types; `print_file` shows them.
        """
        compressors = _compressors(compression_level)
        path = str(path)
        if store == "auto":
            store = "zip" if path.endswith(".zip") else "dir"
        if store == "zip" and not path.endswith(".zip"):
            print(f"Warning: appending .zip to path '{path}'")
            path += ".zip"
        elif store == "dir" and os.path.splitext(path)[1]:
            raise ValueError(
                f"Expected a directory path for store='dir', but got file-like path '{path}'"
            )
        elif store not in ("zip", "dir"):
            raise ValueError(f"Unknown store type: {store}")

        if os.path.exists(path):
            if mode != "o":
                raise FileExistsError(f"File '{path}' already exists. Use mode='o' to overwrite.")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        writer = _Writer(*_split_skip(skip), compressors)
        if store == "dir":
            os.makedirs(path)
            writer.root(self, zarr.group(store=LocalStore(path), overwrite=True))
        else:
            # Every attribute update rewrites zarr.json, which a zip cannot do in
            # place, so write a directory and zip it.
            with tempfile.TemporaryDirectory() as tmpdir:
                writer.root(self, zarr.group(store=LocalStore(tmpdir), overwrite=True))
                with ZipFile(path, mode="w") as zf:
                    for dirpath, _, filenames in os.walk(tmpdir):
                        for filename in filenames:
                            full_path = os.path.join(dirpath, filename)
                            zf.write(full_path, arcname=os.path.relpath(full_path, tmpdir))

    def print_tree(
        self,
        name: str | None = None,
        depth: int | None = None,
        show_values: bool = True,
        show_autoserialize_types: bool = False,
        show_class_origin: bool = False,
    ) -> None:
        """
        Print a visual tree representation of this object's structure.

        Parameters
        ----------
        name : str or None
            Label for the root node; defaults to class name.
        depth : int or None
            Maximum tree depth to print. None = unlimited.
        show_values : bool
            Show primitive scalar values (int, float, str, etc) in output.
        show_autoserialize_types : bool
            Include AutoSerialize container/meta keys and container types.
        show_class_origin : bool
            Show full module path for class names.
        """
        # Determine the root label and class string
        mod_cls = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
            if show_class_origin
            else self.__class__.__name__
        )
        label = name or self.__class__.__name__
        print(f"{label}: class {mod_cls}")

        def _recurse(val, prefix: str, current_depth: int, is_last: bool = True):
            def make_branch(idx, total):
                # Choose tree branch chars based on position
                last = idx == total - 1
                return ("└── ", "    ") if last else ("├── ", "│   ")

            # Handle objects using AutoSerialize
            if _is_autoserialize(val):
                # Filter out metadata keys unless user wants to see them
                keys = [
                    k
                    for k in sorted(val.__dict__.keys())
                    if show_autoserialize_types
                    or k not in {"_container_type", "_autoserialize", "_class_def"}
                ]
                for idx, key in enumerate(keys):
                    subval = val.__dict__[key]
                    branch, new_indent = make_branch(idx, len(keys))
                    suffix = ""
                    # Optionally show container type annotation
                    if show_autoserialize_types and hasattr(subval, "_container_type"):
                        suffix = f" (_container_type = '{getattr(subval, '_container_type', '')}')"
                    # Branch: nested class, tensor, ndarray, container, or primitive
                    if _is_autoserialize(subval):
                        # Optionally show full class path
                        s = (
                            f"{key}: class {subval.__class__.__name__}{suffix}"
                            if not show_class_origin
                            else f"{key}: class {subval.__class__.__module__}.{subval.__class__.__name__}{suffix}"
                        )
                        print(prefix + branch + s)
                    elif isinstance(subval, torch.Tensor):
                        print(prefix + branch + f"{key}: torch.Tensor shape={tuple(subval.shape)}")
                    elif isinstance(subval, np.ndarray):
                        print(prefix + branch + f"{key}: ndarray shape={tuple(subval.shape)}")
                    elif isinstance(subval, (list, tuple)) and show_autoserialize_types:
                        print(prefix + branch + f"{key}: {type(subval).__name__}{suffix}")
                    else:
                        val_str = (
                            f" = {repr(subval)}"
                            if show_values
                            and isinstance(subval, (int, float, str, bool, type(None)))
                            else ""
                        )
                        print(prefix + branch + f"{key}: {type(subval).__name__}{val_str}")
                    # Recursively print children if within depth
                    if depth is None or current_depth < depth - 1:
                        _recurse(
                            subval,
                            prefix + new_indent,
                            current_depth + 1,
                            idx == len(keys) - 1,
                        )

            # Handle containers: list or tuple
            elif isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    branch, new_indent = make_branch(idx, len(val))
                    if isinstance(item, torch.Tensor):
                        print(prefix + branch + f"[{idx}]: torch.Tensor shape={tuple(item.shape)}")
                    elif isinstance(item, np.ndarray):
                        print(prefix + branch + f"[{idx}]: ndarray shape={tuple(item.shape)}")
                    else:
                        val_str = (
                            f" = {repr(item)}"
                            if show_values
                            and isinstance(item, (int, float, str, bool, type(None)))
                            else ""
                        )
                        print(prefix + branch + f"[{idx}]: {type(item).__name__}{val_str}")
                    # Recurse for nested containers/objects
                    if depth is None or current_depth < depth - 1:
                        _recurse(
                            item,
                            prefix + new_indent,
                            current_depth + 1,
                            idx == len(val) - 1,
                        )

            # Handle containers: dict
            elif isinstance(val, dict):
                keys = sorted(val.keys())
                for idx, key in enumerate(keys):
                    item = val[key]
                    branch, new_indent = make_branch(idx, len(keys))
                    if isinstance(item, torch.Tensor):
                        print(
                            prefix
                            + branch
                            + f"{repr(key)}: torch.Tensor shape={tuple(item.shape)}"
                        )
                    elif isinstance(item, np.ndarray):
                        print(prefix + branch + f"{repr(key)}: ndarray shape={tuple(item.shape)}")
                    else:
                        val_str = (
                            f" = {repr(item)}"
                            if show_values
                            and isinstance(item, (int, float, str, bool, type(None)))
                            else ""
                        )
                        print(prefix + branch + f"{repr(key)}: {type(item).__name__}{val_str}")
                    # Recurse for nested dict/containers/objects
                    if depth is None or current_depth < depth - 1:
                        _recurse(
                            item,
                            prefix + new_indent,
                            current_depth + 1,
                            idx == len(keys) - 1,
                        )

        _recurse(self, prefix="", current_depth=0)


class _Writer:
    """Write an object's attributes into a Zarr group, recursing into containers."""

    def __init__(self, skip_names: set[str], skip_types: tuple[type, ...], compressors):
        self.skip_names = skip_names
        self.skip_types = skip_types
        self.compressors = compressors
        self.paths: dict[int, str] = {}  # id of each tensor and object written -> its path
        self.written: list[Any] = []  # keeps those ids from being reused during the save

    def root(self, obj: Any, group: zarr.Group) -> None:
        self.paths[id(obj)] = group.path
        self.object(obj, group, quantem_version=package_version("quantem"))
        group.attrs["_autoserialize_skip_names"] = sorted(self.skip_names)
        group.attrs["_autoserialize_skip_types"] = [
            f"{t.__module__}.{t.__qualname__}" for t in self.skip_types
        ]

    def object(self, obj: Any, group: zarr.Group, **metadata) -> None:
        group.attrs["_autoserialize"] = {
            "version": _FORMAT_VERSION,
            "class_module": type(obj).__module__,
            "class_name": type(obj).__qualname__,
            **metadata,
        }
        # optimizers after models and schedulers after optimizers, so that the
        # values they share are stored with the model and referenced from them
        for name, value in sorted(_get_state(obj).items(), key=lambda kv: _write_order(kv[1])):
            if name in self.skip_names or isinstance(value, self.skip_types):
                continue
            self.value(value, group, name)

    def value(self, value: Any, group: zarr.Group, name: str) -> None:
        """Write `value` under `name` in `group`, in the form its type calls for."""
        if isinstance(value, np.ndarray):
            if name not in group:
                _write_array(group, name, value, self.compressors)
        elif isinstance(value, (int, float, str, bool, type(None))):
            group.attrs[name] = value
        elif isinstance(value, np.generic):
            group.attrs[name] = value.item()
        elif isinstance(value, os.PathLike):
            group.attrs[name] = os.fspath(value)
            group.attrs[f"{name}.is_path"] = True
        elif isinstance(value, torch.device):
            # A device belongs to the machine, not to the data: store the string
            # so the object reloads on a host that does not have that device.
            group.attrs[name] = str(value)
            group.attrs[f"{name}.is_torch_device"] = True
        elif isinstance(value, torch.dtype):
            group.attrs[name] = str(value)  # "torch.float32"
            group.attrs[f"{name}.is_torch_dtype"] = True
        elif isinstance(value, torch.Tensor):
            if self._is_ref(value, group, name):
                return
            # torch.save keeps the dtype, requires_grad and Parameter type
            subgroup = group.require_group(name)
            subgroup.attrs.update(
                {
                    "_torch_tensor": True,
                    "_tensor_shape": list(value.shape),
                    "_tensor_dtype": str(value.dtype),
                    "_tensor_device": str(value.device),
                    "_tensor_requires_grad": bool(value.requires_grad),
                }
            )
            _write_bytes(subgroup, "tensor", _torch_bytes(value))
        elif isinstance(value, torch.Generator):
            # Restored on the CPU, like tensors; a CPU generator keeps its state
            subgroup = group.require_group(name)
            subgroup.attrs.update(
                {
                    "_torch_generator": True,
                    "device": str(value.device),
                    "initial_seed": value.initial_seed(),
                }
            )
            _write_bytes(subgroup, "state", value.get_state().numpy().tobytes())
        elif _is_object_class(type(value)) and hasattr(value, "__dict__"):
            if not self._is_ref(value, group, name):
                self.object(value, group.require_group(name))
        elif isinstance(value, (list, tuple, set, dict)):
            self.container(value, group.require_group(name))
        elif isinstance(value, np.random.Generator):
            subgroup = group.require_group(name)
            subgroup.attrs["_numpy_rng"] = True
            subgroup.attrs["_bit_generator_type"] = type(value.bit_generator).__name__
        elif type(value).__module__.startswith("ase.") and type(value).__name__ == "Atoms":
            _write_atoms(group.require_group(name), name, value, self.compressors)
        elif hasattr(value, "add_scalar") and hasattr(value, "add_image"):
            _write_summary_writer(group.require_group(name), value)
        elif isinstance(value, logging.Logger):
            group.require_group(name).attrs.update(
                {
                    "_python_logger": True,
                    "class_name": type(value).__name__,
                    "logger_name": value.name,
                    "logger_level": value.level,
                }
            )
        elif "torch" in str(getattr(value, "__module__", "")):
            # Other torch objects
            self._pickle(group, name, "_torch_whole_module", _torch_bytes(value))
        else:
            print(
                f"Pickling '{name}' ({type(value).__name__}) with dill; "
                "loading it will need allow_pickle=True"
            )
            self._pickle(group, name, "_dill", dill.dumps(value))

    def container(self, value: list | tuple | set | dict, group: zarr.Group) -> None:
        if isinstance(value, dict):
            group.attrs["_container_type"] = "dict"
            # the order, and str and int keys as they are; other keys become strings
            group.attrs["_dict_keys"] = [k if type(k) in (str, int) else str(k) for k in value]
            for key, item in value.items():
                self.value(item, group, str(key))
            return
        kind = "set" if isinstance(value, set) else "tuple" if isinstance(value, tuple) else "list"
        group.attrs["_container_type"] = kind
        items = list(value)
        if (
            items
            and isinstance(items[0], (int, float, np.number, np.bool_))
            and len({type(v) for v in items}) == 1  # mixed types would come back as one
        ):
            group.attrs["_sequence_encoding"] = "ndarray"
            _write_array(group, "values", np.asarray(items), self.compressors)
        else:
            for index, item in enumerate(items):
                self.value(item, group, str(index))

    def _is_ref(self, value: Any, group: zarr.Group, name: str) -> bool:
        """Store a reference to `value` if it was written before, else note its path."""
        path = self.paths.get(id(value))
        if path is not None:
            group.attrs[name] = path
            group.attrs[f"{name}.is_ref"] = True
            return True
        self.paths[id(value)] = _join(group.path, name)
        self.written.append(value)
        return False

    def _pickle(self, group: zarr.Group, name: str, marker: str, payload: bytes) -> None:
        subgroup = group.require_group(name)
        subgroup.attrs[marker] = True
        _write_bytes(subgroup, _PICKLED[marker][0], payload)


class _Reader:
    """Rebuild the values in a Zarr tree written by `_Writer`.

    Without `allow_pickle`, pickled values and classes from outside quantem and
    torch are not loaded: their paths are collected in `refused` so the caller
    can report them all at once.
    """

    def __init__(
        self,
        root: zarr.Group,
        skip_names: set[str],
        skip_types: tuple[type, ...],
        allow_pickle: bool,
    ):
        self.root = root
        self.skip_names = skip_names
        self.skip_types = skip_types
        self.allow_pickle = allow_pickle
        self.refused: list[str] = []
        self.loaded: dict[str, Any] = {}  # path -> value read from it, for references

    def object(self, group: zarr.Group) -> Any:
        cls = self._class(group)
        if cls is _SKIP:
            return _SKIP
        obj = cls.__new__(cls)
        self.loaded[group.path] = obj  # before its values, which may refer back to it
        state = {}
        for key, kind in _entries(group):
            if key in self.skip_names or (
                kind == "group"
                and self.skip_types
                and issubclass(_stored_class(cast(zarr.Group, group[key])), self.skip_types)
            ):
                continue
            value = self._entry(group, key, kind)
            if value is _SKIP or isinstance(value, self.skip_types):
                continue
            state[key] = value
        if self.refused:  # the load fails, so leave the object as read
            obj.__dict__.update(state)
            return obj
        _set_state(obj, state)
        post_load = getattr(obj, "_post_load", None)
        if callable(post_load):
            post_load()
        return obj

    def container(self, group: zarr.Group) -> Any:
        kind = group.attrs["_container_type"]
        if kind == "dict":
            kinds = dict(_entries(group))
            # older files did not store the keys, which come back as strings
            keys = cast(list, group.attrs.get("_dict_keys", list(kinds)))
            values = {key: self._entry(group, str(key), kinds[str(key)]) for key in keys}
            return {key: value for key, value in values.items() if value is not _SKIP}
        if kind not in ("list", "tuple", "set"):
            raise ValueError(f"Unknown container type {kind!r} at {group.path}")
        if group.attrs.get("_sequence_encoding") == "ndarray":
            items = _read_array(group, "values").tolist()
        else:
            kinds = dict(_entries(group))
            items = [self._entry(group, str(i), kinds[str(i)]) for i in range(len(kinds))]
            items = [item for item in items if item is not _SKIP]
        return {"list": list, "tuple": tuple, "set": set}[kind](items)

    def _entry(self, group: zarr.Group, key: str, kind: str) -> Any:
        if kind == "attr":
            value = group.attrs[key]
            if group.attrs.get(f"{key}.is_ref"):
                path = cast(str, value)
                if path in self.loaded:
                    return self.loaded[path]
                return self._group(cast(zarr.Group, self.root[path]))
            if group.attrs.get(f"{key}.is_path"):
                return Path(cast(str, value))
            if group.attrs.get(f"{key}.is_torch_device"):
                return torch.device(cast(str, value))
            if group.attrs.get(f"{key}.is_torch_dtype"):
                dtype = getattr(torch, cast(str, value).removeprefix("torch."), None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError(f"{value!r} at {_join(group.path, key)} is not a torch dtype")
                return dtype
            return value
        if kind == "array":
            array = _read_array(group, key)
            pickled = _legacy_dill_payload(array)
            if pickled is not None:
                if not self.allow_pickle:
                    return self._refuse(_join(group.path, key), "dill pickle")
                return dill.loads(pickled)
            if group.attrs.get(f"{key}.torch_save"):
                return torch.from_numpy(array)
            return array
        return self._group(cast(zarr.Group, group[key]))

    def _group(self, group: zarr.Group) -> Any:
        # a referenced group may have been read before the traversal reaches it
        if group.path not in self.loaded:
            self.loaded[group.path] = self._read_group(group)
        return self.loaded[group.path]

    def _read_group(self, group: zarr.Group) -> Any:
        attrs = group.attrs
        if "_autoserialize" in attrs:
            return self.object(group)
        if "_container_type" in attrs:
            return self.container(group)
        if attrs.get("_torch_tensor"):
            data = io.BytesIO(_read_array(group, "tensor").tobytes())
            return torch.load(data, map_location="cpu", weights_only=not self.allow_pickle)
        for marker, (key, description) in _PICKLED.items():
            if attrs.get(marker):
                if not self.allow_pickle:
                    return self._refuse(group.path, description)
                payload = _read_array(group, key).tobytes()
                if marker == "_dill":
                    return dill.loads(payload)
                return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if attrs.get("_ase_atoms"):
            return _read_atoms(group)
        if attrs.get("_numpy_rng"):
            bit_generator = _BIT_GENERATORS.get(cast(str, attrs.get("_bit_generator_type")))
            return np.random.Generator((bit_generator or np.random.PCG64)())
        if attrs.get("_torch_generator"):
            generator = torch.Generator()
            if attrs.get("device") == "cpu":
                generator.set_state(torch.from_numpy(_read_array(group, "state")))
            else:  # another device's state does not fit a CPU generator
                generator.manual_seed(cast(int, attrs["initial_seed"]))
            return generator
        if attrs.get("_torch_rng_skipped"):  # older files did not store the state
            return torch.Generator()
        if attrs.get("_torch_logger"):
            return _read_summary_writer(group)
        if attrs.get("_python_logger"):
            if attrs.get("class_name", "Logger") != "Logger":
                print(f"Warning: not restoring logger of type {attrs.get('class_name')}")
                return _SKIP
            logger = logging.getLogger(cast(str, attrs.get("logger_name", "quantem")))
            logger.setLevel(cast(int, attrs.get("logger_level", logging.INFO)))
            return logger
        raise ValueError(f"Unknown subgroup structure: {group.path}")

    def _class(self, group: zarr.Group) -> Any:
        """The class of the object stored in `group`, or _SKIP if it is refused."""
        cls = _import_class(group.attrs["_autoserialize"])
        if not _is_object_class(cls):
            raise ValueError(
                f"{cls.__module__}.{cls.__qualname__} is not an AutoSerialize class, "
                "dataclass, or torch module, optimizer or scheduler"
            )
        if (
            self.allow_pickle
            or _is_autoserialize_class(cls)  # written to be saved and loaded
            or cls.__module__.partition(".")[0] in _TRUSTED_PACKAGES
        ):
            return cls
        return self._refuse(
            group.path, f"{cls.__module__}.{cls.__qualname__}, from outside quantem and torch"
        )

    def _refuse(self, path: str, description: str) -> Any:
        self.refused.append(f"{path} ({description})")
        return _SKIP


def _get_state(obj: Any) -> dict[str, Any]:
    """The values stored for `obj`, from which `_set_state` rebuilds it."""
    if isinstance(obj, torch.optim.Optimizer):
        # state_dict() names the parameters by position; `params` holds them, so
        # they are written once and shared with the model that holds them
        return {
            "defaults": obj.defaults,
            "state_dict": obj.state_dict(),
            "params": [p for group in obj.param_groups for p in group["params"]],
        }
    if isinstance(obj, torch.optim.lr_scheduler.LRScheduler):
        return {**obj.state_dict(), "optimizer": obj.optimizer}
    if isinstance(obj, torch.nn.Module):
        # internals left at their defaults, such as empty hook dicts, are not stored
        default = vars(torch.nn.Module())
        return {
            key: value
            for key, value in obj.__getstate__().items()
            if not (key in default and type(value) is type(default[key]) and value == default[key])
        }
    return vars(obj)


def _set_state(obj: Any, state: dict[str, Any]) -> None:
    """Restore the attributes from `_get_state` onto `obj`, made with ``__new__``."""
    if isinstance(obj, torch.optim.Optimizer):
        # as torch documents: construct over the parameters, then load the state.
        # The defaults can hold settings the constructor fixes itself (AdamW's
        # decoupled_weight_decay); load_state_dict restores each group's settings.
        params, state_dict = state["params"], state["state_dict"]
        groups = [
            {**group, "params": [params[i] for i in group["params"]]}
            for group in state_dict["param_groups"]
        ]
        accepted = inspect.signature(type(obj).__init__).parameters
        obj.__init__(groups, **{k: v for k, v in state["defaults"].items() if k in accepted})
        obj.load_state_dict(state_dict)
    elif isinstance(obj, torch.optim.lr_scheduler.LRScheduler):
        obj.optimizer = state.pop("optimizer")
        # load_state_dict of some schedulers reads attributes that __init__ sets
        obj.__dict__.update(state)
        obj.load_state_dict(dict(state))
    elif isinstance(obj, torch.nn.Module):
        obj.__setstate__({**vars(torch.nn.Module()), **state})
        # files written before sets were stored as sets hold this as a list
        obj._non_persistent_buffers_set = set(obj._non_persistent_buffers_set)
    else:
        obj.__dict__.update(state)


def _write_order(value: Any) -> int:
    if isinstance(value, torch.optim.lr_scheduler.LRScheduler):
        return 2
    return int(isinstance(value, torch.optim.Optimizer))


def _entries(group: zarr.Group) -> list[tuple[str, str]]:
    """(key, kind) of each value stored in `group`: JSON attributes, arrays, then groups."""
    attrs = [
        (key, "attr")
        for key in group.attrs
        if key not in _METADATA_KEYS and not key.endswith(_FLAG_SUFFIXES)
    ]
    arrays = [(key, "array") for key in group.array_keys() if key not in _METADATA_KEYS]
    groups = [(key, "group") for key in group.group_keys() if key not in _METADATA_KEYS]
    return attrs + arrays + groups


def _import_class(metadata: Any) -> type:
    """Import the class named in a group's ``_autoserialize`` metadata."""
    if int(metadata.get("version", 1)) != _FORMAT_VERSION:
        raise ValueError(f"Unsupported AutoSerialize version: {metadata.get('version')}")
    module_name, class_name = metadata["class_module"], metadata["class_name"]
    cls: Any = importlib.import_module(module_name)
    for part in class_name.split("."):  # a qualified name, for nested classes
        cls = getattr(cls, part, None)
    if not isinstance(cls, type):
        raise ValueError(f"{module_name}.{class_name} is not an importable class")
    return cls


def _stored_class(group: zarr.Group) -> type:
    """Class of the object stored in `group`, else `object`."""
    metadata = group.attrs.get("_autoserialize")
    return _import_class(metadata) if metadata is not None else object


def _is_object_class(cls: type) -> bool:
    """True for the classes stored as a group of their state, rebuilt by `_set_state`."""
    return (
        _is_autoserialize_class(cls)
        or issubclass(cls, _TORCH_STATEFUL)
        or dataclasses.is_dataclass(cls)
    )


def _is_autoserialize_class(cls: type) -> bool:
    # Compares the marker rather than using issubclass, so classes stay
    # recognized after an autoreload creates a new AutoSerialize class object.
    return getattr(cls, "__autoserialize_marker__", None) == AutoSerialize.__autoserialize_marker__


def _is_autoserialize(value: Any) -> bool:
    return _is_autoserialize_class(type(value))


def _split_skip(skip: SkipSpec) -> tuple[set[str], tuple[type, ...]]:
    items = [skip] if isinstance(skip, (str, type)) else list(skip)
    return (
        {s for s in items if isinstance(s, str)},
        tuple(s for s in items if isinstance(s, type)),
    )


def _join(path: str, key: str) -> str:
    return f"{path}/{key}" if path else key


def _compressors(level: int | None) -> list[dict] | None:
    if level is None:
        return None
    if not 0 <= level <= 9:
        raise ValueError(f"compression_level must be between 0 and 9, got {level}")
    return [
        {
            "name": "blosc",
            "configuration": {"cname": "zstd", "clevel": int(level), "shuffle": "bitshuffle"},
        }
    ]


def _write_array(group: zarr.Group, name: str, array: np.ndarray, compressors) -> None:
    array = np.asarray(array)
    ds = group.create_array(
        name=name, shape=array.shape, dtype=array.dtype, compressors=compressors
    )
    ds[...] = array


def _write_bytes(group: zarr.Group, name: str, data: bytes) -> None:
    _write_array(group, name, np.frombuffer(data, dtype=np.uint8), compressors=None)


def _read_array(group: zarr.Group, key: str) -> np.ndarray:
    array = cast(zarr.Array, group[key])
    shape = array.attrs.get("_original_shape")
    if shape is not None:  # older files stored zero-size arrays as a placeholder scalar
        return np.empty(tuple(cast(list[int], shape)), dtype=array.dtype)
    return np.asarray(array[...])


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _legacy_dill_payload(array: np.ndarray) -> bytes | None:
    """Pickle stored by older versions as gzip-compressed dill in a plain uint8 array."""
    if array.dtype != np.uint8 or array.ndim != 1 or array[:2].tobytes() != b"\x1f\x8b":
        return None
    try:
        payload = gzip.decompress(array.tobytes())
    except (OSError, EOFError, zlib.error):
        return None
    return payload if payload[:1] == b"\x80" else None


def _write_atoms(group: zarr.Group, name: str, atoms: Any, compressors) -> None:
    # Stored as plain arrays so the file stays readable without pickling an ase
    # version in: cell, pbc, every per-atom array in atoms.arrays (numbers,
    # positions, occupancy, tags, masses, ...) and the JSON-serializable entries
    # of atoms.info. Constraints and attached calculators are not saved.
    group.attrs["_ase_atoms"] = True
    _write_array(group, "cell", np.asarray(atoms.get_cell()), compressors)
    _write_array(group, "pbc", np.asarray(atoms.get_pbc()), compressors)
    arrays_group = group.require_group("arrays")
    text_arrays = {}
    for key, arr in atoms.arrays.items():
        arr = np.asarray(arr)
        if arr.dtype.kind in "biufc":
            _write_array(arrays_group, key, arr, compressors)
        else:
            text_arrays[key] = arr.tolist()
    group.attrs["text_arrays"] = _json_entries(text_arrays, f"{name}.arrays")
    group.attrs["info"] = _json_entries(dict(atoms.info), f"{name}.info")


def _read_atoms(group: zarr.Group) -> Any:
    from ase import Atoms

    if "arrays" in group.group_keys():
        arrays_group = cast(zarr.Group, group["arrays"])
        arrays = {k: _read_array(arrays_group, k) for k in arrays_group.array_keys()}
        text_arrays = cast(dict[str, Any], group.attrs.get("text_arrays", {}))
        arrays.update({k: np.asarray(v) for k, v in text_arrays.items()})
    else:  # files written before per-atom arrays were stored
        arrays = {k: _read_array(group, k) for k in ("numbers", "positions")}
    atoms = Atoms(
        numbers=arrays.pop("numbers"),
        positions=arrays.pop("positions"),
        cell=_read_array(group, "cell"),
        pbc=_read_array(group, "pbc"),
    )
    for key, arr in arrays.items():
        atoms.set_array(key, arr)
    atoms.info.update(cast(dict[str, Any], group.attrs.get("info", {})))
    return atoms


def _write_summary_writer(group: zarr.Group, writer: Any) -> None:
    group.attrs["_torch_logger"] = True
    group.attrs["class_name"] = type(writer).__name__
    for key in ("log_dir", "comment", "filename_suffix"):
        if hasattr(writer, key):
            group.attrs[key] = str(getattr(writer, key) or "")
    for key in ("max_queue", "flush_secs"):
        if hasattr(writer, key):
            group.attrs[key] = int(getattr(writer, key))


def _read_summary_writer(group: zarr.Group) -> Any:
    attrs = group.attrs
    if attrs.get("class_name", "SummaryWriter") != "SummaryWriter":
        print(f"Warning: not restoring logger of type {attrs.get('class_name')}")
        return _SKIP
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(
        log_dir=cast(str | None, attrs.get("log_dir")),
        comment=str(attrs.get("comment", "")),
        max_queue=int(cast(int, attrs.get("max_queue", 10))),
        flush_secs=int(cast(int, attrs.get("flush_secs", 120))),
        filename_suffix=str(attrs.get("filename_suffix", "")),
    )


def _is_saved_object(directory: str) -> bool:
    """True if `directory` is the root of a saved AutoSerialize object."""
    try:
        with open(os.path.join(directory, "zarr.json")) as f:
            return "_autoserialize" in json.load(f).get("attributes", {})
    except (OSError, ValueError):
        return False


def _open_saved(path: str | Path) -> tuple[zarr.Group, Any]:
    """Open a saved object read-only, from a directory or a .zip archive.

    Copies of saved files often gain one extra folder level: a folder dragged
    into a folder of the same name, or a zip made by the operating system,
    which wraps its contents in a folder. If `path` holds no saved object but
    exactly one subfolder that is one, that subfolder is opened.

    Returns
    -------
    root : zarr.Group
        The object's root group.
    tempdir : tempfile.TemporaryDirectory | None
        Holds the extracted zip; keep it alive while reading.
    """
    path = os.fspath(path)
    tempdir = None
    if os.path.isdir(path):
        base = path
    elif os.path.isfile(path):
        tempdir = tempfile.TemporaryDirectory()
        with ZipFile(path, "r") as zf:
            zf.extractall(tempdir.name)
        base = tempdir.name
    else:
        raise FileNotFoundError(f"No such file or directory: {path!r}")

    if not _is_saved_object(base):
        found = [
            os.path.join(base, d)
            for d in sorted(os.listdir(base))
            if not d.startswith((".", "__MACOSX")) and _is_saved_object(os.path.join(base, d))
        ]
        if len(found) != 1:
            hint = (
                f"; it contains {len(found)} saved objects: "
                + ", ".join(os.path.basename(f) for f in found)
                if found
                else ""
            )
            raise FileNotFoundError(f"No saved quantem object at {path!r}{hint}.")
        base = found[0]
    # read-only: loading never writes into the user's folders
    root = zarr.open_group(store=LocalStore(base, read_only=True), mode="r")
    return root, tempdir


def load(path: str | Path, skip: SkipSpec = (), *, allow_pickle: bool = False) -> Any:
    """
    Load an object saved with `AutoSerialize.save`.

    Parameters
    ----------
    path : str or Path
        Directory or .zip file holding the object, or a folder one level above it.
    skip : str, type, or list of (str or type)
        Attribute names, at any depth, and types not to load.
    allow_pickle : bool
        Also load pickled values (dill fallbacks, and torch modules, optimizers
        and schedulers in files written before quantem stored them without
        pickle), and torch modules, optimizers, schedulers and dataclasses
        defined outside quantem and torch. These can run code stored in the
        file, so only allow them for files from a source you trust.

    Returns
    -------
    obj : AutoSerialize
        The reconstructed object.

    Raises
    ------
    ValueError
        If the file holds such values and `allow_pickle` is False. The message
        lists them; nothing from them has been run.
    """
    root, tempdir = _open_saved(path)
    reader = _Reader(root, *_split_skip(skip), allow_pickle=allow_pickle)
    try:
        obj = reader.object(root)
    finally:
        if tempdir is not None:
            tempdir.cleanup()
    if reader.refused:
        listing = "\n".join(f"  {entry}" for entry in reader.refused)
        raise ValueError(
            f"{os.fspath(path)!r} holds values that can run code when loaded:\n"
            f"{listing}\n"
            "Load it with allow_pickle=True if you trust where the file came from."
        )
    return obj


def print_file(
    path: str | Path,
    depth: int | None = None,
    show_values: bool = True,
    show_autoserialize_types: bool = False,
    show_class_origin: bool = False,
) -> None:
    """
    Print a tree view of the saved structure of an AutoSerialize file (directory or zip archive).

    Parameters
    ----------
    path : str or Path
        Path to the directory or .zip archive containing a serialized object.
    depth : int or None, optional
        Maximum tree depth to print (None = no limit).
    show_values : bool, optional
        Print scalar values for simple fields.
    show_autoserialize_types : bool, optional
        Display internal serialization/meta fields and container types.
    show_class_origin : bool, optional
        Show full class import path (module + class name) in output.
    """
    # Open the Zarr group from dir/zip
    root, tempdir = _open_saved(path)

    def _recurse(obj: Any, prefix: str = "", current_depth: int = 0, is_last: bool = True) -> None:
        if isinstance(obj, zarr.Group):
            # Collect all keys: attrs, arrays, and subgroups, for sorting/printing
            keys = sorted(set(obj.attrs.keys()) | set(obj.array_keys()) | set(obj.group_keys()))

            # Print the root label (with class info and quantem version) at the top level
            if current_depth == 0:
                class_info = cast(dict[str, Any] | None, obj.attrs.get("_autoserialize"))
                label = Path(path).name
                if class_info is not None and "class_name" in class_info:
                    mod_cls = (
                        f"{cast(str, class_info['class_module'])}.{cast(str, class_info['class_name'])}"
                        if show_class_origin
                        else cast(str, class_info["class_name"])
                    )
                    label += f": class {mod_cls}"
                    if "quantem_version" in class_info:
                        label += f" (quantem {class_info['quantem_version']})"
                print(label)

            # Optionally filter out internal autoserialize fields for cleaner output
            printable_keys = []
            for key in keys:
                if not show_autoserialize_types and (
                    key in _METADATA_KEYS or key == "_class_def" or key.endswith(_FLAG_SUFFIXES)
                ):
                    continue
                printable_keys.append(key)

            # Print all attributes/arrays/groups in a tree view, using unicode branches
            for idx, key in enumerate(printable_keys):
                last = idx == len(printable_keys) - 1
                branch = "└── " if last else "├── "
                new_prefix = prefix + ("    " if last else "│   ")

                if key in obj.group_keys():
                    # Print nested groups (submodules, containers, etc)
                    child_group = cast(zarr.Group, obj[key])
                    group_type = child_group.attrs.get("_container_type")
                    suffix = (
                        f" (_container_type = '{group_type}')"
                        if group_type and show_autoserialize_types
                        else ""
                    )
                    print(prefix + branch + f"{key}{suffix}")
                    if depth is None or current_depth < depth - 1:
                        _recurse(child_group, new_prefix, current_depth + 1, last)

                elif key in obj.array_keys():
                    # Print info about arrays/tensors
                    arr = cast(zarr.Array, obj[key])
                    is_torch = obj.attrs.get(f"{key}.torch_save", False)
                    tensor_type = "ndarray"  # Default to ndarray for array_keys
                    print(
                        prefix
                        + branch
                        + f"{key}: {tensor_type} shape={arr.shape} is_torch={is_torch}"
                    )

                else:
                    # Print scalar/group attribute values
                    val = obj.attrs[key]
                    type_str = type(val).__name__
                    display_val = (
                        f" = {repr(val)}"
                        if show_values and isinstance(val, (int, float, str, bool, type(None)))
                        else ""
                    )
                    print(prefix + branch + f"{key}: {type_str}{display_val}")

            # Check for tensors in new format that might not be in the main keys
            for key in obj.group_keys():
                if key not in printable_keys:  # Skip if already printed
                    child_group = cast(zarr.Group, obj[key])
                    if child_group.attrs.get("_torch_tensor"):
                        # This is a tensor in the new format
                        tensor_shape = child_group.attrs.get("_tensor_shape", "unknown")
                        requires_grad = child_group.attrs.get("_tensor_requires_grad", False)
                        grad_info = " (requires_grad=True)" if requires_grad else ""
                        print(
                            prefix
                            + "└── "
                            + f"{key}: torch.Tensor shape={tensor_shape}{grad_info}"
                        )

    try:
        _recurse(root)
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def _json_entries(entries: dict, label: str) -> dict:
    """Return the JSON-serializable entries of a dict, warning about the rest."""
    kept, dropped = {}, []
    for key, val in entries.items():
        try:
            json.dumps({str(key): val})
        except (TypeError, ValueError):
            dropped.append(str(key))
        else:
            kept[str(key)] = val
    if dropped:
        print(f"Not saving non-JSON-serializable entries of {label}: {dropped}")
    return kept


class Bundle(AutoSerialize):
    """A named collection of serializable objects, saved as one file.

    Groups any AutoSerialize objects (Datasets, Vectors, ...) plus plain
    metadata values under attribute names::

        bundle = Bundle(adf=dataset2d, peaks=vector, note="IM689")
        bundle.save("data.zip")
        b = load("data.zip"); b.adf, b.peaks

    Parameters
    ----------
    **objects
        Objects to store, keyed by attribute name. Names must not shadow an
        existing attribute or method of the class (e.g. ``save``).

    Raises
    ------
    ValueError
        If a name shadows a class attribute or method.
    """

    def __init__(self, **objects):
        reserved = sorted(name for name in objects if hasattr(type(self), name))
        if reserved:
            raise ValueError(
                f"Bundle names {reserved} shadow Bundle/AutoSerialize attributes; "
                "choose different names."
            )
        for name, obj in objects.items():
            setattr(self, name, obj)

    def __repr__(self) -> str:
        items = ", ".join(
            f"{k}: {type(v).__name__}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"Bundle({items})"
