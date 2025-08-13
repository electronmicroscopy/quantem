import gzip
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence, Union
from zipfile import ZipFile

import dill
import numpy as np
import torch
import zarr
from zarr.storage import LocalStore


# Base class for automatic serialization of classes
class AutoSerialize:
    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: Union[str, type, Sequence[Union[str, type]]] = (),
        compression_level: int | None = 4,
    ) -> None:
        """
        Save the current object to disk using Zarr serialization.

        Parameters
        ----------
        path : str or Path
            Target file path. Use '.zip' extension for zip format, otherwise a directory.
        mode : {'w', 'o'}
            'w' = write only if file doesn't exist, 'o' = overwrite if it does.
        store : {'auto', 'zip', 'dir'}
            Storage format. 'auto' infers from file extension.
        skip : str, type, or list of (str or type)
            Attribute names/types to skip (by name or type) during serialization.
        compression_level : int or None
            If set (0–9), applies Zstandard compression with Blosc backend at that level.
            Level 0 disables compression. Raises ValueError if > 9.

        Notes
        -----
        Skipped attribute names and types are also stored in the file metadata for correct
        round-trip skipping during load().
        """
        # Validate compression level
        if compression_level is not None:
            if not (0 <= compression_level <= 9):
                raise ValueError(
                    f"compression_level must be between 0 and 9, got {compression_level}"
                )
            compressors = [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": int(compression_level),
                        "shuffle": "bitshuffle",
                    },
                }
            ]
        else:
            compressors = None

        path = str(path)
        # Auto-infer storage format if needed
        if store == "auto":
            store = "zip" if path.endswith(".zip") else "dir"

        # Ensure .zip extension if requested
        if store == "zip" and not path.endswith(".zip"):
            print(f"Warning: appending .zip to path '{path}'")
            path += ".zip"

        # Handle overwrite vs. write protection
        if os.path.exists(path):
            if mode == "o":
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            else:
                raise FileExistsError(f"File '{path}' already exists. Use mode='o' to overwrite.")

        # Normalize skip argument (split to names and types)
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip_names = {s for s in skip if isinstance(s, str)}
        skip_types = tuple(s for s in skip if isinstance(s, type))

        def write_skip_metadata(root):
            # Store skip info as attributes for correct deserialization
            root.attrs["_autoserialize_skip_names"] = list(skip_names)
            root.attrs["_autoserialize_skip_types"] = [
                f"{t.__module__}.{t.__qualname__}" for t in skip_types
            ]

        # Main branch: choose between zip and directory storage
        if store == "zip":
            # Always use tempdir for safe atomic write
            with tempfile.TemporaryDirectory() as tmpdir:
                store_obj = LocalStore(tmpdir)
                root = zarr.group(store=store_obj, overwrite=True)
                self._recursive_save(self, root, skip_names, skip_types, compressors)
                write_skip_metadata(root)
                # Zip up all files in tempdir
                with ZipFile(path, mode="w") as zf:
                    for dirpath, _, filenames in os.walk(tmpdir):
                        for filename in filenames:
                            full_path = os.path.join(dirpath, filename)
                            rel_path = os.path.relpath(full_path, tmpdir)
                            zf.write(full_path, arcname=rel_path)
        elif store == "dir":
            # Directory mode requires no extension
            if os.path.splitext(path)[1]:
                raise ValueError(
                    f"Expected a directory path for store='dir', but got file-like path '{path}'"
                )
            os.makedirs(path, exist_ok=True)
            store_obj = LocalStore(path)
            root = zarr.group(store=store_obj, overwrite=True)
            self._recursive_save(self, root, skip_names, skip_types, compressors)
            write_skip_metadata(root)
        else:
            raise ValueError(f"Unknown store type: {store}")

    def _recursive_save(
        self,
        obj,
        group: zarr.Group,
        skip_names: set[str] = set(),
        skip_types: tuple[type, ...] = (),
        compressors=None,
    ) -> None:
        # Store class identity and version metadata at group root if not already set
        if "_autoserialize" not in group.attrs:
            group.attrs["_autoserialize"] = {
                "version": 1,
                "class_module": obj.__class__.__module__,
                "class_name": obj.__class__.__qualname__,
            }

        # Support both attrs and plain Python classes
        attrs_fields = getattr(obj.__class__, "__attrs_attrs__", None)
        if attrs_fields is not None:
            items = [(field.name, getattr(obj, field.name)) for field in attrs_fields]
        else:
            items = obj.__dict__.items()

        for attr_name, attr_value in items:
            # Skip any attributes matching names/types in skip lists
            if attr_name in skip_names or isinstance(attr_value, skip_types):
                continue

            # --- Serialization handlers by type ---
            if isinstance(attr_value, torch.Tensor):
                # Save as ndarray, with flag for torch reloading
                tensor_np = (
                    attr_value.detach().cpu().numpy()
                    if attr_value.requires_grad
                    else attr_value.cpu().numpy()
                )
                if attr_name not in group:
                    arr = group.create_dataset(
                        name=attr_name,
                        shape=tensor_np.shape,
                        dtype=tensor_np.dtype,
                        compressors=compressors,
                    )
                    arr[:] = tensor_np
                    group.attrs[f"{attr_name}.torch_save"] = True

            elif isinstance(attr_value, torch.optim.Optimizer):
                # Save torch optimizer state_dict as compressed byte array
                opt_group = group.require_group(attr_name)
                opt_group.attrs["_torch_optimizer"] = True
                opt_group.attrs["class_name"] = attr_value.__class__.__name__

                buffer = io.BytesIO()
                torch.save(attr_value.state_dict(), buffer)
                buffer.seek(0)
                byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
                opt_group.create_dataset(
                    "state_dict", data=byte_arr, shape=byte_arr.shape, dtype="uint8"
                )

            elif isinstance(attr_value, torch.nn.Module):
                # Save entire torch module as compressed byte array
                subgroup = group.require_group(attr_name)
                subgroup.attrs["_torch_whole_module"] = True
                buffer = io.BytesIO()
                torch.save(attr_value, buffer)
                buffer.seek(0)
                byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
                subgroup.create_dataset(
                    "module", data=byte_arr, shape=byte_arr.shape, dtype="uint8"
                )

            elif isinstance(attr_value, np.ndarray):
                # Save as native array
                if attr_name not in group:
                    arr = group.create_dataset(
                        name=attr_name,
                        shape=attr_value.shape,
                        dtype=attr_value.dtype,
                        compressors=compressors,
                    )
                    arr[:] = attr_value

            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                # Scalars saved as attributes
                group.attrs[attr_name] = attr_value

            elif isinstance(attr_value, AutoSerialize):
                # Nested AutoSerialize subtree
                subgroup = group.require_group(attr_name)
                self._recursive_save(attr_value, subgroup, skip_names, skip_types, compressors)

            elif isinstance(attr_value, (list, tuple, dict)):
                # Save containers recursively (with nested AutoSerialize support)
                subgroup = group.require_group(attr_name)
                self._serialize_container(
                    attr_value, subgroup, skip_names, skip_types, compressors
                )

            else:
                # Fallback: dill-serialize + gzip-compress
                serialized = dill.dumps(attr_value)
                compressed = gzip.compress(serialized)
                ds = group.create_dataset(
                    name=attr_name,
                    shape=(len(compressed),),
                    dtype="uint8",
                    compressors=compressors,
                )
                ds[:] = np.frombuffer(compressed, dtype="uint8")

    @classmethod
    def _recursive_load(
        cls,
        group: zarr.Group,
        skip_names: set[str] = frozenset(),
        skip_types: tuple[type, ...] = (),
    ) -> object:
        """
        Recursively reconstruct an AutoSerialize object from a Zarr group,
        honoring attribute/type skipping for selective deserialization.
        """
        # --- Load class identity and ensure version is compatible ---
        meta = group.attrs["_autoserialize"]
        version = meta.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported AutoSerialize version: {version}")
        module = __import__(meta["class_module"], fromlist=[meta["class_name"]])
        cls_obj = getattr(module, meta["class_name"])
        obj = cls_obj.__new__(cls_obj)  # Avoid __init__ side effects

        # If attrs package is used, only allow whitelisted attribute names
        attrs_fields = getattr(cls_obj, "__attrs_attrs__", None)
        if attrs_fields is not None:
            attrs_item_names = [f.name for f in attrs_fields]
        else:
            attrs_item_names = []

        set_attrs = set()

        # --- Restore simple attributes ---
        for name, val in group.attrs.items():
            if name == "_autoserialize" or name.endswith(".torch_save"):
                continue  # Skip metadata/flags
            if name in skip_names:
                continue
            if attrs_item_names and name not in attrs_item_names:
                continue
            setattr(obj, name, val)
            set_attrs.add(name)

        # --- Restore datasets (arrays/tensors/serialized objects) ---
        for ds in group.array_keys():
            if ds in skip_names:
                continue
            arr = group[ds][:]
            try:
                payload = gzip.decompress(arr.tobytes())
                v = dill.loads(payload)
            except Exception:
                v = arr
                if group.attrs.get(f"{ds}.torch_save", False):
                    v = torch.from_numpy(v)
            if type(v) in skip_types:
                continue
            setattr(obj, ds, v)
            set_attrs.add(ds)

        # --- Restore subgroups (optimizers, modules, nested objects, containers) ---
        for name in group.group_keys():
            if name in skip_names:
                continue
            subgrp = group[name]

            # torch optimizer group
            if subgrp.attrs.get("_torch_optimizer"):
                clsname = subgrp.attrs["class_name"]
                data = bytes(subgrp["state_dict"][:])
                buf = io.BytesIO(data)
                state = torch.load(buf, map_location="cpu")
                opt_cls = getattr(torch.optim, clsname)
                dummy = torch.zeros(1, requires_grad=True)
                opt = opt_cls([dummy])
                opt.load_state_dict(state)
                if type(opt) in skip_types:
                    continue
                setattr(obj, name, opt)
                set_attrs.add(name)

            # torch module group
            elif subgrp.attrs.get("_torch_whole_module"):
                data = bytes(subgrp["module"][:])
                buf = io.BytesIO(data)
                mod = torch.load(buf, map_location="cpu", weights_only=False)
                if type(mod) in skip_types:
                    continue
                setattr(obj, name, mod)
                set_attrs.add(name)

            # nested AutoSerialize group
            elif "_autoserialize" in subgrp.attrs:
                m = subgrp.attrs["_autoserialize"]
                submod = __import__(m["class_module"], fromlist=[m["class_name"]])
                subcls = getattr(submod, m["class_name"])
                if subcls in skip_types:
                    continue
                val = subcls._recursive_load(subgrp, skip_names, skip_types)
                if type(val) in skip_types:
                    continue
                setattr(obj, name, val)
                set_attrs.add(name)

            # containers (list, tuple, dict)
            elif subgrp.attrs.get("_container_type", None) is not None:
                val = cls._deserialize_container(subgrp)
                if type(val) in skip_types:
                    continue
                setattr(obj, name, val)
                set_attrs.add(name)

            else:
                print(f"Unhandled group: {name} with attrs: {dict(subgrp.attrs)}")
                raise ValueError(f"Unknown subgroup structure: {subgrp.path}")

        # Remove attributes in skip_names that may have been set by __init__ (when using __new__)
        for name in skip_names:
            if hasattr(obj, name):
                delattr(obj, name)

        # attrs pattern: call post-init if defined
        if hasattr(obj, "__attrs_post_init__"):
            obj.__attrs_post_init__()

        return obj

    def _serialize_container(
        self,
        value,
        group: zarr.Group,
        skip_names: set[str],
        skip_types: tuple[type, ...],
        compressors=None,
    ):
        """
        Serialize a container (list, tuple, or dict) into a Zarr group.

        Handles nested containers, AutoSerialize instances, PyTorch objects, and primitives,
        with recursive support for arbitrary depth and skipping.
        """

        # Special handling for torch.nn containers: flatten to list and record type
        if isinstance(value, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ParameterList)):
            group.attrs["_torch_iterable_module_type"] = type(value).__name__
            value = list(value)

        # Handle list/tuple containers
        if isinstance(value, (list, tuple)):
            group.attrs["_container_type"] = type(value).__name__
            for i, v in enumerate(value):
                key = str(i)

                # --- If entry is an AutoSerialize object, use full recursive_save (preserves type!) ---
                if isinstance(v, AutoSerialize):
                    subgroup = group.require_group(key)
                    self._recursive_save(v, subgroup, skip_names, skip_types, compressors)

                # --- Recursively handle nested containers ---
                elif isinstance(v, (list, tuple, dict)):
                    subgroup = group.require_group(key)
                    self._serialize_container(v, subgroup, skip_names, skip_types, compressors)

                # --- Torch nn.Module: save as a whole-module byte array with a marker ---
                elif isinstance(v, torch.nn.Module):
                    subgroup = group.require_group(key)
                    subgroup.attrs["_torch_whole_module"] = True
                    buffer = io.BytesIO()
                    torch.save(v, buffer)
                    buffer.seek(0)
                    byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
                    subgroup.create_dataset(
                        "module", data=byte_arr, shape=byte_arr.shape, dtype="uint8"
                    )

                # --- Torch tensor: save as ndarray, with .torch_save flag ---
                elif isinstance(v, torch.Tensor):
                    arr_np = v.detach().cpu().numpy() if v.requires_grad else v.cpu().numpy()
                    ds = group.create_dataset(
                        name=key,
                        data=arr_np,
                        shape=arr_np.shape,
                        dtype=arr_np.dtype,
                        compressors=compressors,
                    )
                    group.attrs[f"{key}.torch_save"] = True

                # --- Standard numpy arrays: save directly as dataset ---
                elif isinstance(v, np.ndarray):
                    group.create_dataset(
                        name=key,
                        data=v,
                        shape=v.shape,
                        dtype=v.dtype,
                        compressors=compressors,
                    )

                # --- Primitive types: store as attribute ---
                elif isinstance(v, (int, float, str, bool, type(None))):
                    group.attrs[key] = v

                # --- Fallback: dill/gzip serialize and store as byte array ---
                else:
                    payload = dill.dumps(v)
                    comp = gzip.compress(payload)
                    ds = group.create_dataset(
                        name=key,
                        shape=(len(comp),),
                        dtype="uint8",
                        compressors=compressors,
                    )
                    ds[:] = np.frombuffer(comp, dtype="uint8")

        # Handle dict containers (very similar logic, using keys)
        elif isinstance(value, dict):
            group.attrs["_container_type"] = "dict"
            for k, v in value.items():
                key = str(k)

                # --- AutoSerialize instance as value ---
                if isinstance(v, AutoSerialize):
                    subgroup = group.require_group(key)
                    self._recursive_save(v, subgroup, skip_names, skip_types, compressors)

                # --- Nested container (list/tuple/dict) as value ---
                elif isinstance(v, (list, tuple, dict)):
                    subgroup = group.require_group(key)
                    self._serialize_container(v, subgroup, skip_names, skip_types, compressors)

                # --- Torch nn.Module as value ---
                elif isinstance(v, torch.nn.Module):
                    subgroup = group.require_group(key)
                    subgroup.attrs["_torch_whole_module"] = True
                    buffer = io.BytesIO()
                    torch.save(v, buffer)
                    buffer.seek(0)
                    byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
                    subgroup.create_dataset(
                        "module", data=byte_arr, shape=byte_arr.shape, dtype="uint8"
                    )

                # --- Torch tensor as value ---
                elif isinstance(v, torch.Tensor):
                    arr_np = v.detach().cpu().numpy() if v.requires_grad else v.cpu().numpy()
                    ds = group.create_dataset(
                        name=key,
                        data=arr_np,
                        shape=arr_np.shape,
                        dtype=arr_np.dtype,
                        compressors=compressors,
                    )
                    group.attrs[f"{key}.torch_save"] = True

                # --- Standard numpy array as value ---
                elif isinstance(v, np.ndarray):
                    group.create_dataset(
                        name=key,
                        data=v,
                        shape=v.shape,
                        dtype=v.dtype,
                        compressors=compressors,
                    )

                # --- Primitive type as value ---
                elif isinstance(v, (int, float, str, bool, type(None))):
                    group.attrs[key] = v

                # --- Fallback: dill/gzip ---
                else:
                    payload = dill.dumps(v)
                    comp = gzip.compress(payload)
                    ds = group.create_dataset(
                        name=key,
                        shape=(len(comp),),
                        dtype="uint8",
                        compressors=compressors,
                    )
                    ds[:] = np.frombuffer(comp, dtype="uint8")

    @classmethod
    def _deserialize_container(cls, group: zarr.Group):
        """
        Reconstructs a list, tuple, or dict container from a Zarr group.

        Supports nested containers, torch module containers, and automatic conversion
        of torch tensors and special objects. Container structure and type info are
        encoded in Zarr group attributes.
        """
        ctype = group.attrs.get("_container_type")
        if ctype is None:
            raise ValueError(f"Missing _container_type in group: {group.path}")

        torch_iterable_type = group.attrs.get("_torch_iterable_module_type")

        # Helper to handle optional torch tensor restoration
        def maybe_tensor(group, key):
            arr = group[key][:]
            return torch.from_numpy(arr) if group.attrs.get(f"{key}.torch_save") else arr

        if ctype in ("list", "tuple"):
            # Determine maximum index to reconstruct order and size
            length = (
                max(
                    (
                        int(k)
                        for k in list(group.attrs)
                        + list(group.array_keys())
                        + list(group.group_keys())
                        if k.isdigit()
                    ),
                    default=-1,
                )
                + 1
            )
            items = []
            for i in range(length):
                key = str(i)
                if key in group.attrs:
                    items.append(group.attrs[key])
                elif key in group.array_keys():
                    items.append(maybe_tensor(group, key))
                elif key in group.group_keys():
                    subgroup = group[key]
                    # Handle recursive containers
                    if "_container_type" in subgroup.attrs:
                        items.append(cls._deserialize_container(subgroup))
                    # Restore nested AutoSerialize objects
                    elif "_autoserialize" in subgroup.attrs:
                        meta = subgroup.attrs["_autoserialize"]
                        submod = __import__(meta["class_module"], fromlist=[meta["class_name"]])
                        subcls = getattr(submod, meta["class_name"])
                        items.append(subcls._recursive_load(subgroup))
                    # Restore nested torch modules
                    elif subgroup.attrs.get("_torch_whole_module"):
                        data = bytes(subgroup["module"][:])
                        buf = io.BytesIO(data)
                        mod = torch.load(buf, map_location="cpu", weights_only=False)
                        items.append(mod)
                    else:
                        raise ValueError(f"Unknown group structure at key '{key}' in {group.path}")
                else:
                    raise KeyError(f"Missing expected key '{key}' in container")
            # Restore container type and special torch containers
            result = items if ctype == "list" else tuple(items)
            if torch_iterable_type == "Sequential":
                return torch.nn.Sequential(*result)
            elif torch_iterable_type == "ModuleList":
                return torch.nn.ModuleList(result)
            elif torch_iterable_type == "ParameterList":
                return torch.nn.ParameterList(result)
            else:
                return result

        elif ctype == "dict":
            result = {}
            # Restore scalars and simple objects stored as attributes
            for key in group.attrs:
                if key == "_container_type" or key.endswith(".torch_save"):
                    continue
                result[key] = group.attrs[key]
            # Restore arrays (including torch tensors)
            for key in group.array_keys():
                result[key] = maybe_tensor(group, key)
            # Restore subgroups
            for key in group.group_keys():
                subgroup = group[key]
                if "_container_type" in subgroup.attrs:
                    result[key] = cls._deserialize_container(subgroup)
                elif "_autoserialize" in subgroup.attrs:
                    meta = subgroup.attrs["_autoserialize"]
                    submod = __import__(meta["class_module"], fromlist=[meta["class_name"]])
                    subcls = getattr(submod, meta["class_name"])
                    result[key] = subcls._recursive_load(subgroup)
                elif subgroup.attrs.get("_torch_whole_module"):
                    data = bytes(subgroup["module"][:])
                    buf = io.BytesIO(data)
                    mod = torch.load(buf, map_location="cpu", weights_only=False)
                    result[key] = mod
                else:
                    raise ValueError(f"Unknown group structure at key '{key}' in {group.path}")
            return result

        else:
            raise ValueError(f"Unknown container type: {ctype}")

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
            if isinstance(val, AutoSerialize):
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
                    if isinstance(subval, AutoSerialize):
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


def load(
    path: str | Path,
    skip: Union[str, type, Sequence[Union[str, type]]] = (),
) -> Any:
    """
    Load an AutoSerialize object from disk.

    Parameters
    ----------
    path : str or Path
        Directory or .zip file containing a serialized object.
    skip : str, type, or list of (str or type)
        Names/types of attributes to skip when loading.
        Combined with skip info stored in the file, if present.

    Returns
    -------
    obj : Any
        Reconstructed AutoSerialize instance.
    """
    # Normalize skip argument to sets/tuples for merging
    if isinstance(skip, (str, type)):
        skip = [skip]
    user_skip_names = {s for s in skip if isinstance(s, str)}
    user_skip_types = tuple(s for s in skip if isinstance(s, type))

    # Load Zarr store from directory or extracted zip
    if os.path.isdir(path):
        store = LocalStore(path)
    else:
        tempdir = tempfile.TemporaryDirectory()
        with ZipFile(path, "r") as zf:
            zf.extractall(tempdir.name)
        store = LocalStore(tempdir.name)

    root = zarr.group(store=store)
    if "_autoserialize" not in root.attrs:
        raise KeyError("Missing '_autoserialize' metadata in Zarr root attrs.")
    meta = root.attrs["_autoserialize"]
    version = meta.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported AutoSerialize version: {version}")

    # Read skip metadata (names/types) stored with the file, if present
    file_skip_names = set(root.attrs.get("_autoserialize_skip_names", []))
    file_skip_types_raw = root.attrs.get("_autoserialize_skip_types", [])
    file_skip_types = (
        tuple(
            # Import each type by fully-qualified name from string
            __import__(t.rpartition(".")[0], fromlist=[t.rpartition(".")[2]]).__dict__[
                t.rpartition(".")[2]
            ]
            for t in file_skip_types_raw
        )
        if file_skip_types_raw
        else tuple()
    )

    # Merge user-specified and file-stored skip lists/types (avoid duplicates)
    skip_names = user_skip_names | file_skip_names
    skip_types = user_skip_types + tuple(t for t in file_skip_types if t not in user_skip_types)

    # Dynamically import target class, then reconstruct from Zarr
    mod = __import__(meta["class_module"], fromlist=[meta["class_name"]])
    cls = getattr(mod, meta["class_name"])
    return cls._recursive_load(root, skip_names=skip_names, skip_types=skip_types)


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
    if os.path.isdir(path):
        store = LocalStore(path)
    else:
        tempdir = tempfile.TemporaryDirectory()
        with ZipFile(path, "r") as zf:
            zf.extractall(tempdir.name)
        store = LocalStore(tempdir.name)
    root = zarr.group(store=store)

    def _recurse(obj: Any, prefix: str = "", current_depth: int = 0, is_last: bool = True) -> None:
        if isinstance(obj, zarr.Group):
            # Collect all keys: attrs, arrays, and subgroups, for sorting/printing
            keys = sorted(set(obj.attrs.keys()) | set(obj.array_keys()) | set(obj.group_keys()))

            # Print the root label (with class info) at the top level
            if current_depth == 0:
                class_info = obj.attrs.get("_autoserialize")
                label = Path(path).name
                if class_info and "class_name" in class_info:
                    mod_cls = (
                        f"{class_info['class_module']}.{class_info['class_name']}"
                        if show_class_origin
                        else class_info["class_name"]
                    )
                    label += f": class {mod_cls}"
                print(label)

            # Optionally filter out internal autoserialize fields for cleaner output
            printable_keys = []
            for key in keys:
                if not show_autoserialize_types and key in {
                    "_container_type",
                    "_autoserialize",
                    "_class_def",
                }:
                    continue
                if not show_autoserialize_types and key.endswith(".torch_save"):
                    continue
                printable_keys.append(key)

            # Print all attributes/arrays/groups in a tree view, using unicode branches
            for idx, key in enumerate(printable_keys):
                last = idx == len(printable_keys) - 1
                branch = "└── " if last else "├── "
                new_prefix = prefix + ("    " if last else "│   ")

                if key in obj.group_keys():
                    # Print nested groups (submodules, containers, etc)
                    child_group = obj[key]
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
                    arr = obj[key]
                    is_torch = obj.attrs.get(f"{key}.torch_save", False)
                    tensor_type = "torch.Tensor" if is_torch else "ndarray"
                    print(prefix + branch + f"{key}: {tensor_type} shape={arr.shape}")

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

    _recurse(root)
