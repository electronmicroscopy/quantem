import gzip
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import AbstractSet, Any, Literal, Sequence, Union, cast
from zipfile import ZipFile

import dill
import numpy as np
import torch
import zarr
from zarr.storage import LocalStore


# Base class for automatic serialization of classes
class AutoSerialize:
    """
    Base class for automatic serialization of Python objects using Zarr storage.

    This class provides comprehensive serialization support for:
    - PyTorch objects (tensors, modules, optimizers, schedulers)
    - PyTorch loggers (SummaryWriter, etc.) - metadata only
    - Python loggers (logging.Logger) - metadata only
    - NumPy arrays and scalars
    - Python primitives (int, float, str, bool, None)
    - pathlib.Path objects (converted to strings for storage)
    - AutoSerialize instances (nested serialization)
    - Python containers (list, tuple, dict, set)
    - NumPy random generators
    - Fallback to dill for other objects

    Path objects are automatically converted to strings during serialization
    and restored as Path objects during deserialization.

    Logger objects (PyTorch and Python) are serialized as metadata only
    and recreated as new instances during deserialization.
    """

    # ---- Helpers to reduce casting noise ----
    @staticmethod
    def _get_group(parent: zarr.Group, key: str) -> zarr.Group:
        return cast(zarr.Group, parent[key])

    @staticmethod
    def _get_array(parent: zarr.Group, key: str) -> zarr.Array:
        return cast(zarr.Array, parent[key])

    @staticmethod
    def _is_numeric_scalar(value: Any) -> bool:
        """Return True if value is a plain numeric scalar (int/float/bool or numpy scalar)."""
        # Avoid treating numpy arrays/tensors/containers as scalars
        if isinstance(value, (np.ndarray, torch.Tensor, list, tuple, dict, set)):
            return False
        # Python numeric types and numpy scalar types
        return isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_))

    @staticmethod
    def _fix_torch_module_sets(mod):
        """Fix PyTorch module set attributes that might have been corrupted during serialization."""
        if isinstance(mod, torch.nn.Module):
            # Fix _non_persistent_buffers_set if it's not a set
            if hasattr(mod, "_non_persistent_buffers_set") and not isinstance(
                mod._non_persistent_buffers_set, set
            ):
                mod._non_persistent_buffers_set = set(mod._non_persistent_buffers_set)
        return mod

    @staticmethod
    def _load_torch_module_with_fix(data_bytes):
        """Load a torch module and fix any corrupted set attributes."""
        buf = io.BytesIO(data_bytes)
        mod = torch.load(buf, map_location="cpu", weights_only=False)
        AutoSerialize._fix_torch_module_sets(mod)
        return mod

    @staticmethod
    def _array_to_np(arr: zarr.Array) -> np.ndarray:
        # Handle empty arrays (any dimension of size 0) and 0-dimensional arrays
        if arr.ndim == 0 or any(s == 0 for s in arr.shape):
            # Check if this was originally an empty array with a specific shape
            if "_original_shape" in arr.attrs:
                original_shape = arr.attrs["_original_shape"]
                # Convert to tuple of ints to ensure proper typing
                if isinstance(original_shape, (list, tuple)):
                    original_shape = tuple(int(cast(Any, x)) for x in original_shape)
                return np.empty(cast(Any, original_shape), dtype=arr.dtype)
            else:
                # For empty or 0-dimensional arrays, return an empty numpy array with the same shape
                return np.empty(arr.shape, dtype=arr.dtype)
        else:
            return cast(np.ndarray, arr[:])

    @staticmethod
    def _read_array_np(parent: zarr.Group, key: str) -> np.ndarray:
        return AutoSerialize._array_to_np(AutoSerialize._get_array(parent, key))

    @staticmethod
    def _write_ndarray(
        group: zarr.Group,
        name: str,
        array: np.ndarray,
        compressors=None,
    ) -> None:
        # Ensure array is a numpy array
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        # Handle scalar arrays (0-dimensional) properly
        if array.ndim == 0:
            ds = group.create_array(
                name=name, shape=(), dtype=array.dtype, compressors=compressors
            )
            ds[()] = array.item()  # Use () for scalar indexing
        else:
            # Handle empty arrays (any dimension of size 0)
            if any(s == 0 for s in array.shape):
                # For empty arrays, create a 0-dimensional array instead of (0,)
                # This avoids indexing issues during loading
                ds = group.create_array(
                    name=name, shape=(), dtype=array.dtype, compressors=compressors
                )
                # Store the original shape as an attribute for reconstruction
                ds.attrs["_original_shape"] = array.shape
                # No need to assign data since it's empty
                return
            # Ensure the shape is valid (no negative dimensions)
            if any(s < 0 for s in array.shape):
                raise ValueError(f"Invalid array shape {array.shape} for array '{name}'")
            ds = group.create_array(
                name=name, shape=array.shape, dtype=array.dtype, compressors=compressors
            )
            ds[:] = array

    @staticmethod
    def _write_bytes(
        group: zarr.Group,
        name: str,
        data: bytes,
        compressors=None,
    ) -> None:
        # Handle empty bytes
        if not data:
            ds = group.create_array(name=name, shape=(0,), dtype="uint8", compressors=compressors)
            return

        buf_arr = np.frombuffer(data, dtype="uint8")
        # Handle scalar arrays (0-dimensional) properly
        if buf_arr.ndim == 0:
            ds = group.create_array(name=name, shape=(), dtype="uint8", compressors=compressors)
            ds[()] = buf_arr.item()
        else:
            ds = group.create_array(
                name=name, shape=buf_arr.shape, dtype="uint8", compressors=compressors
            )
            ds[:] = buf_arr

    @staticmethod
    def _convert_string_to_path_if_needed(val: Any, group: zarr.Group, key: str) -> Any:
        """Convert string back to pathlib.Path if it was originally a Path object."""
        if isinstance(val, str) and group.attrs.get(f"{key}.is_path", False):
            try:
                from pathlib import Path

                return Path(val)
            except (ValueError, OSError):
                # If Path creation fails, keep as string
                return val
        return val

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

    def _serialize_value(
        self,
        value: Any,
        group: zarr.Group,
        name: str,
        skip_names: set[str] = set(),
        skip_types: tuple[type, ...] = (),
        compressors=None,
    ) -> None:
        """
        Unified method to serialize any value type to a Zarr group.
        This eliminates duplication between _recursive_save and _serialize_container.
        """
        # --- Serialization handlers by type ---
        if isinstance(value, torch.Tensor):
            # Save entire tensor with torch.save to preserve requires_grad, grad_fn, etc.
            # This is more robust than converting to numpy which loses gradient information
            subgroup = group.require_group(name)
            subgroup.attrs["_torch_tensor"] = True
            subgroup.attrs["_tensor_shape"] = list(value.shape)
            subgroup.attrs["_tensor_dtype"] = str(value.dtype)
            subgroup.attrs["_tensor_device"] = str(value.device)
            subgroup.attrs["_tensor_requires_grad"] = bool(value.requires_grad)

            buffer = io.BytesIO()
            torch.save(value, buffer)
            buffer.seek(0)
            byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
            self._write_bytes(subgroup, "tensor", byte_arr.tobytes(), compressors=None)

        elif isinstance(value, torch.optim.Optimizer):
            # Save entire optimizer with torch.save for robustness
            subgroup = group.require_group(name)
            subgroup.attrs["_torch_optimizer"] = True
            subgroup.attrs["class_name"] = value.__class__.__name__

            buffer = io.BytesIO()
            torch.save(value, buffer)
            buffer.seek(0)
            byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
            self._write_bytes(subgroup, "optimizer", byte_arr.tobytes(), compressors=None)

        elif hasattr(value, "step") and hasattr(value, "get_last_lr"):
            # Handle LR schedulers with torch.save for robustness
            subgroup = group.require_group(name)
            subgroup.attrs["_torch_scheduler"] = True
            subgroup.attrs["class_name"] = value.__class__.__name__

            buffer = io.BytesIO()
            torch.save(value, buffer)
            buffer.seek(0)
            byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
            self._write_bytes(subgroup, "scheduler", byte_arr.tobytes(), compressors=None)

        elif hasattr(value, "add_scalar") and hasattr(value, "add_image"):
            # Handle PyTorch loggers (SummaryWriter, etc.) - save basic info only
            subgroup = group.require_group(name)
            subgroup.attrs["_torch_logger"] = True
            subgroup.attrs["class_name"] = value.__class__.__name__

            # Store basic logger information that can be reconstructed
            if hasattr(value, "log_dir"):
                subgroup.attrs["log_dir"] = str(value.log_dir)
            if hasattr(value, "comment"):
                subgroup.attrs["comment"] = str(value.comment) if value.comment else ""
            if hasattr(value, "max_queue"):
                subgroup.attrs["max_queue"] = int(value.max_queue)
            if hasattr(value, "flush_secs"):
                subgroup.attrs["flush_secs"] = int(value.flush_secs)
            if hasattr(value, "filename_suffix"):
                subgroup.attrs["filename_suffix"] = (
                    str(value.filename_suffix) if value.filename_suffix else ""
                )
        elif hasattr(value, "log") and hasattr(value, "info"):
            # Handle other logging objects (like Python's logging.Logger)
            subgroup = group.require_group(name)
            subgroup.attrs["_python_logger"] = True
            subgroup.attrs["class_name"] = value.__class__.__name__

            # Store logger name and level if available
            if hasattr(value, "name"):
                subgroup.attrs["logger_name"] = str(value.name)
            if hasattr(value, "level"):
                subgroup.attrs["logger_level"] = int(value.level)

        elif isinstance(value, torch.nn.Module):
            # Save entire torch module with torch.save for robustness
            subgroup = group.require_group(name)
            subgroup.attrs["_torch_whole_module"] = True
            buffer = io.BytesIO()
            torch.save(value, buffer)
            buffer.seek(0)
            byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
            self._write_bytes(subgroup, "module", byte_arr.tobytes(), compressors=None)

        elif isinstance(value, np.ndarray):
            # Save as native array
            if name not in group:
                self._write_ndarray(group, name, value, compressors)

        elif isinstance(value, (int, float, str, bool, type(None))):
            # Scalars saved as attributes
            group.attrs[name] = value
        elif hasattr(value, "dtype") and hasattr(value, "item"):
            # Handle numpy scalar types (np.float32, np.int64, etc.)
            group.attrs[name] = value.item()
        elif hasattr(value, "__fspath__") or str(type(value)).startswith("<class 'pathlib."):
            # Handle pathlib.Path objects and other path-like objects
            group.attrs[name] = str(value)
            group.attrs[f"{name}.is_path"] = True

        elif isinstance(value, AutoSerialize):
            # Nested AutoSerialize subtree
            subgroup = group.require_group(name)
            self._recursive_save(value, subgroup, skip_names, skip_types, compressors)

        elif isinstance(value, (list, tuple, dict)):
            # Save containers recursively (with nested AutoSerialize support)
            subgroup = group.require_group(name)
            self._serialize_container(value, subgroup, skip_names, skip_types, compressors)

        elif isinstance(value, set):
            # Convert set to list for serialization, store type info
            subgroup = group.require_group(name)
            subgroup.attrs["_container_type"] = "set"
            # Convert set items to list and serialize
            list_value = list(value)
            self._serialize_container(list_value, subgroup, skip_names, skip_types, compressors)

        elif hasattr(value, "bit_generator"):
            # NumPy random generator - save state through bit_generator
            subgroup = group.require_group(name)
            subgroup.attrs["_numpy_rng"] = True
            # Get state from the bit_generator
            rng_state = value.bit_generator.state
            if hasattr(rng_state, "tolist"):
                subgroup.attrs["_rng_state"] = rng_state.tolist()
            else:
                subgroup.attrs["_rng_state"] = rng_state
            subgroup.attrs["_rng_type"] = value.__class__.__name__
            subgroup.attrs["_bit_generator_type"] = value.bit_generator.__class__.__name__

        elif hasattr(value, "get_state") and hasattr(value, "set_state"):
            # PyTorch generator - skip for now as state structure is complex
            # Just store a marker that this was a generator
            subgroup = group.require_group(name)
            subgroup.attrs["_torch_rng_skipped"] = True
            subgroup.attrs["_rng_type"] = "torch.Generator"
            # Don't try to save the state - it's not essential for core functionality

        else:
            # Fallback: dill-serialize + gzip-compress
            print(f"falling back in serialize for {name} of type {type(value)}")
            serialized = dill.dumps(value)
            compressed = gzip.compress(serialized)
            self._write_bytes(group, name, compressed, compressors)

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

            # Use unified serialization method
            self._serialize_value(
                attr_value, group, attr_name, skip_names, skip_types, compressors
            )

    @classmethod
    def _recursive_load(
        cls,
        group: zarr.Group,
        skip_names: AbstractSet[str] = frozenset(),
        skip_types: tuple[type, ...] = (),
    ) -> object:
        """
        Recursively reconstruct an AutoSerialize object from a Zarr group,
        honoring attribute/type skipping for selective deserialization.
        """
        # --- Load class identity and ensure version is compatible ---
        meta = cast(dict[str, Any], group.attrs["_autoserialize"])
        version = int(meta.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported AutoSerialize version: {version}")
        module_name = cast(str, meta["class_module"])
        class_name = cast(str, meta["class_name"])
        module = __import__(module_name, fromlist=[class_name])
        cls_obj = getattr(module, class_name)
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
            if (
                name == "_autoserialize"
                or name.endswith(".torch_save")
                or name.endswith(".is_path")
            ):
                continue  # Skip metadata/flags
            if name in skip_names:
                continue
            if attrs_item_names and name not in attrs_item_names:
                continue

            # Convert string paths back to pathlib.Path objects if needed
            val = cls._convert_string_to_path_if_needed(val, group, name)

            setattr(obj, name, val)
            set_attrs.add(name)

        # --- Restore datasets (arrays/tensors/serialized objects) ---
        for ds in group.array_keys():
            if ds in skip_names:
                continue
            arr_np = AutoSerialize._read_array_np(group, ds)
            try:
                payload = gzip.decompress(arr_np.tobytes())
                v = dill.loads(payload)
            except Exception:
                v = arr_np
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
            subgrp = AutoSerialize._get_group(group, name)

            # torch tensor group
            if subgrp.attrs.get("_torch_tensor"):
                data = AutoSerialize._read_array_np(subgrp, "tensor").tobytes()
                buf = io.BytesIO(data)
                tensor = torch.load(buf, map_location="cpu", weights_only=False)
                if type(tensor) in skip_types:
                    continue
                setattr(obj, name, tensor)
                set_attrs.add(name)

            # torch optimizer group
            elif subgrp.attrs.get("_torch_optimizer"):
                data = AutoSerialize._read_array_np(subgrp, "optimizer").tobytes()
                buf = io.BytesIO(data)
                opt = torch.load(buf, map_location="cpu", weights_only=False)
                if type(opt) in skip_types:
                    continue

                setattr(obj, name, opt)
                set_attrs.add(name)

            # torch scheduler group
            elif subgrp.attrs.get("_torch_scheduler"):
                data = AutoSerialize._read_array_np(subgrp, "scheduler").tobytes()
                buf = io.BytesIO(data)
                scheduler = torch.load(buf, map_location="cpu", weights_only=False)
                if type(scheduler) in skip_types:
                    continue
                setattr(obj, name, scheduler)
                set_attrs.add(name)

            # torch logger group
            elif subgrp.attrs.get("_torch_logger"):
                # Recreate logger from saved metadata
                logger_class_name = subgrp.attrs.get("class_name", "SummaryWriter")

                if logger_class_name == "SummaryWriter":
                    from torch.utils.tensorboard import SummaryWriter

                    # Extract logger parameters with explicit type casting
                    log_dir = subgrp.attrs.get("log_dir", None)

                    comment = str(cast(Any, subgrp.attrs.get("comment", "")))
                    max_queue = int(cast(Any, subgrp.attrs.get("max_queue", 10)))
                    flush_secs = int(cast(Any, subgrp.attrs.get("flush_secs", 120)))
                    filename_suffix = str(cast(Any, subgrp.attrs.get("filename_suffix", "")))

                    # Create new logger instance
                    logger = SummaryWriter(
                        log_dir=log_dir,
                        comment=comment,
                        max_queue=max_queue,
                        flush_secs=flush_secs,
                        filename_suffix=filename_suffix,
                    )
                else:
                    # For other logger types, create a basic instance or skip
                    print(
                        f"Warning: Unknown logger type '{logger_class_name}', skipping logger restoration"
                    )
                    continue

                if type(logger) in skip_types:
                    continue
                setattr(obj, name, logger)
                set_attrs.add(name)

            # python logger group
            elif subgrp.attrs.get("_python_logger"):
                # Recreate Python logger from saved metadata
                logger_class_name = subgrp.attrs.get("class_name", "Logger")

                if logger_class_name == "Logger":
                    import logging

                    # Extract logger parameters
                    logger_name = cast(str, subgrp.attrs.get("logger_name", "quantem"))
                    logger_level = int(cast(Any, subgrp.attrs.get("logger_level", logging.INFO)))

                    # Create new logger instance
                    logger = logging.getLogger(logger_name)
                    logger.setLevel(logger_level)
                else:
                    # For other logger types, create a basic instance or skip
                    print(
                        f"Warning: Unknown Python logger type '{logger_class_name}', skipping logger restoration"
                    )
                    continue

                if type(logger) in skip_types:
                    continue
                setattr(obj, name, logger)
                set_attrs.add(name)

            # torch module group
            elif subgrp.attrs.get("_torch_whole_module"):
                data = AutoSerialize._read_array_np(subgrp, "module").tobytes()
                buf = io.BytesIO(data)
                mod = torch.load(buf, map_location="cpu", weights_only=False)
                if type(mod) in skip_types:
                    continue

                # Fix PyTorch module set attributes that might be corrupted
                if isinstance(mod, torch.nn.Module):
                    cls._fix_torch_module_sets(mod)

                setattr(obj, name, mod)
                set_attrs.add(name)

            # nested AutoSerialize group
            elif "_autoserialize" in subgrp.attrs:
                m = cast(dict[str, Any], subgrp.attrs["_autoserialize"])
                submod_name = cast(str, m["class_module"])
                subcls_name = cast(str, m["class_name"])
                submod = __import__(submod_name, fromlist=[subcls_name])
                subcls = getattr(submod, subcls_name)
                if subcls in skip_types:
                    continue
                val = subcls._recursive_load(subgrp, skip_names, skip_types)
                if type(val) in skip_types:
                    continue

                setattr(obj, name, val)
                set_attrs.add(name)

            # containers (list, tuple, dict)
            elif subgrp.attrs.get("_container_type", None) is not None:
                val = cls._deserialize_container(cast(zarr.Group, subgrp))
                if type(val) in skip_types:
                    continue
                setattr(obj, name, val)
                set_attrs.add(name)

            # NumPy random generator
            elif subgrp.attrs.get("_numpy_rng"):
                import numpy.random as npr

                # rng_type = subgrp.attrs.get("_rng_type", "Generator")
                bit_generator_type = subgrp.attrs.get("_bit_generator_type", "PCG64")
                # rng_state = subgrp.attrs["_rng_state"]

                # Create the appropriate bit generator
                if bit_generator_type == "PCG64":
                    bit_gen = npr.PCG64()
                elif bit_generator_type == "MT19937":
                    bit_gen = npr.MT19937()
                elif bit_generator_type == "Philox":
                    bit_gen = npr.Philox()
                elif bit_generator_type == "SFC64":
                    bit_gen = npr.SFC64()
                else:
                    # Fallback to default
                    bit_gen = npr.PCG64()

                # Create generator with fresh state
                rng = npr.Generator(bit_gen)
                # Note: We don't restore the exact state due to type compatibility issues
                # The generator will work fine with fresh state and can be re-seeded if needed

                setattr(obj, name, rng)
                set_attrs.add(name)

            # PyTorch generator (skipped during save)
            elif subgrp.attrs.get("_torch_rng_skipped"):
                # Create a new generator since we didn't save the state
                rng = torch.Generator()
                setattr(obj, name, rng)
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

        # Fix PyTorch module set attributes after all loading is complete
        if isinstance(obj, torch.nn.Module):
            cls._fix_torch_module_sets(obj)

        # Also fix any nested PyTorch modules in the object's attributes
        # Use a more defensive approach to avoid triggering property accessors
        for attr_name in dir(obj):
            if not attr_name.startswith("_"):  # Skip private attributes
                try:
                    # Check if it's a property first to avoid triggering accessors
                    if hasattr(type(obj), attr_name):
                        attr_descriptor = getattr(type(obj), attr_name)
                        if hasattr(attr_descriptor, "__get__") and not hasattr(
                            attr_descriptor, "__set__"
                        ):
                            # This is a read-only property, skip it to avoid triggering computation
                            continue

                    attr_value = getattr(obj, attr_name)
                    if isinstance(attr_value, torch.nn.Module):
                        cls._fix_torch_module_sets(attr_value)
                except (AttributeError, RuntimeError, ValueError, KeyError):
                    # Skip attributes that can't be accessed or cause other errors
                    pass

        return obj

    def _serialize_container(
        self,
        value: Union[list, tuple, dict],
        group: zarr.Group,
        skip_names: set[str] = set(),
        skip_types: tuple[type, ...] = (),
        compressors=None,
    ) -> None:
        """
        Serialize Python containers (list, tuple, dict) to Zarr groups.

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
            # Fast-path: homogeneous numeric scalars → single ndarray
            try:
                is_all_numeric = len(value) > 0 and all(
                    AutoSerialize._is_numeric_scalar(v) for v in value
                )
            except TypeError:
                # If value isn't sized/iterable like expected, fall back
                is_all_numeric = False

            if is_all_numeric:
                group.attrs["_sequence_encoding"] = "ndarray"
                arr = np.asarray(value)
                # Store in a single dataset named 'values'
                self._write_ndarray(group, "values", arr, compressors)
            else:
                for i, v in enumerate(value):
                    key = str(i)
                    # Use unified serialization method
                    self._serialize_value(v, group, key, skip_names, skip_types, compressors)

        # Handle dict containers
        elif isinstance(value, dict):
            group.attrs["_container_type"] = "dict"
            for k, v in value.items():
                key = str(k)
                # Use unified serialization method
                self._serialize_value(v, group, key, skip_names, skip_types, compressors)

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
            arr = AutoSerialize._read_array_np(group, key)
            return torch.from_numpy(arr) if group.attrs.get(f"{key}.torch_save") else arr

        if ctype in ("list", "tuple"):
            # Determine maximum index to reconstruct order and size
            # Fast-path: ndarray-encoded homogeneous sequence
            if (
                group.attrs.get("_sequence_encoding") == "ndarray"
                and "values" in group.array_keys()
            ):
                arr = AutoSerialize._read_array_np(group, "values")
                seq = arr.tolist()
                items = seq
            else:
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
                        val = group.attrs[key]
                        # Convert string paths back to Path objects if needed
                        val = cls._convert_string_to_path_if_needed(val, group, key)
                        items.append(val)
                    elif key in group.array_keys():
                        items.append(maybe_tensor(group, key))
                    elif key in group.group_keys():
                        subgroup = cast(zarr.Group, group[key])
                        # Handle recursive containers
                        if "_container_type" in subgroup.attrs:
                            items.append(cls._deserialize_container(subgroup))
                        # Restore nested AutoSerialize objects
                        elif "_autoserialize" in subgroup.attrs:
                            meta = cast(dict[str, Any], subgroup.attrs["_autoserialize"])
                            submod = __import__(
                                cast(str, meta["class_module"]),
                                fromlist=[cast(str, meta["class_name"])],
                            )
                            subcls = getattr(submod, cast(str, meta["class_name"]))
                            items.append(subcls._recursive_load(subgroup))
                        # Restore nested torch modules
                        elif subgroup.attrs.get("_torch_whole_module"):
                            module_arr = cast(zarr.Array, subgroup["module"])
                            data = cast(np.ndarray, module_arr[:]).tobytes()
                            buf = io.BytesIO(data)
                            # For containers, load to CPU - they'll be moved to the right device when attached to the main object
                            mod = torch.load(buf, map_location="cpu", weights_only=False)
                            items.append(mod)
                        elif subgroup.attrs.get("_torch_tensor"):
                            # Handle new tensor format in containers
                            data = AutoSerialize._read_array_np(subgroup, "tensor").tobytes()
                            buf = io.BytesIO(data)
                            tensor = torch.load(buf, map_location="cpu", weights_only=False)
                            items.append(tensor)
                        elif subgroup.attrs.get("_torch_logger"):
                            # Handle torch logger in containers
                            logger_class_name = subgroup.attrs.get("class_name", "SummaryWriter")

                            if logger_class_name == "SummaryWriter":
                                from torch.utils.tensorboard import SummaryWriter

                                log_dir = subgroup.attrs.get("log_dir", None)
                                comment = str(cast(Any, subgroup.attrs.get("comment", "")))
                                max_queue = int(cast(Any, subgroup.attrs.get("max_queue", 10)))
                                flush_secs = int(cast(Any, subgroup.attrs.get("flush_secs", 120)))
                                filename_suffix = str(
                                    cast(Any, subgroup.attrs.get("filename_suffix", ""))
                                )

                                logger = SummaryWriter(
                                    log_dir=log_dir,
                                    comment=comment,
                                    max_queue=max_queue,
                                    flush_secs=flush_secs,
                                    filename_suffix=filename_suffix,
                                )
                                items.append(logger)
                            else:
                                # Skip unknown logger types in containers
                                continue
                        elif subgroup.attrs.get("_python_logger"):
                            # Handle Python logger in containers
                            logger_class_name = subgroup.attrs.get("class_name", "Logger")

                            if logger_class_name == "Logger":
                                import logging

                                logger_name = cast(
                                    str, subgroup.attrs.get("logger_name", "quantem")
                                )
                                logger_level = int(
                                    cast(Any, subgroup.attrs.get("logger_level", logging.INFO))
                                )

                                logger = logging.getLogger(logger_name)
                                logger.setLevel(logger_level)
                                items.append(logger)
                            else:
                                # Skip unknown logger types in containers
                                continue
                        else:
                            raise ValueError(
                                f"Unknown group structure at key '{key}' in {group.path}"
                            )
                    else:
                        raise KeyError(f"Missing expected key '{key}' in container")
            # Restore container type and special torch containers
            seq_result = items if ctype == "list" else tuple(items)
            if torch_iterable_type == "Sequential":
                return torch.nn.Sequential(*seq_result)
            elif torch_iterable_type == "ModuleList":
                return torch.nn.ModuleList(cast(Sequence[torch.nn.Module], list(seq_result)))
            elif torch_iterable_type == "ParameterList":
                return torch.nn.ParameterList(cast(Sequence[torch.nn.Parameter], list(seq_result)))
            else:
                return seq_result

        elif ctype == "set":
            # Convert back from list to set
            items = []
            for i in range(
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
            ):
                key = str(i)
                if key in group.attrs:
                    val = group.attrs[key]
                    # Convert string paths back to Path objects if needed
                    val = cls._convert_string_to_path_if_needed(val, group, key)
                    items.append(val)
                elif key in group.array_keys():
                    items.append(maybe_tensor(group, key))
                elif key in group.group_keys():
                    subgroup = cast(zarr.Group, group[key])
                    # Handle recursive containers
                    if "_container_type" in subgroup.attrs:
                        items.append(cls._deserialize_container(subgroup))
                    # Restore nested AutoSerialize objects
                    elif "_autoserialize" in subgroup.attrs:
                        meta = cast(dict[str, Any], subgroup.attrs["_autoserialize"])
                        submod = __import__(
                            cast(str, meta["class_module"]),
                            fromlist=[cast(str, meta["class_name"])],
                        )
                        subcls = getattr(submod, cast(str, meta["class_name"]))
                        items.append(subcls._recursive_load(subgroup))
                    # Restore nested torch modules
                    elif subgroup.attrs.get("_torch_whole_module"):
                        module_arr = cast(zarr.Array, subgroup["module"])
                        data = cast(np.ndarray, module_arr[:]).tobytes()
                        buf = io.BytesIO(data)
                        # For containers, load to CPU - they'll be moved to the right device when attached to the main object
                        mod = torch.load(buf, map_location="cpu", weights_only=False)
                        items.append(mod)
                    elif subgroup.attrs.get("_torch_tensor"):
                        # Handle new tensor format in containers
                        data = AutoSerialize._read_array_np(subgroup, "tensor").tobytes()
                        buf = io.BytesIO(data)
                        tensor = torch.load(buf, map_location="cpu", weights_only=False)
                        items.append(tensor)
                    elif subgroup.attrs.get("_torch_logger"):
                        # Handle torch logger in containers
                        logger_class_name = subgroup.attrs.get("class_name", "SummaryWriter")

                        if logger_class_name == "SummaryWriter":
                            from torch.utils.tensorboard import SummaryWriter

                            log_dir = subgroup.attrs.get("log_dir", None)
                            comment = str(cast(Any, subgroup.attrs.get("comment", "")))
                            max_queue = int(cast(Any, subgroup.attrs.get("max_queue", 10)))
                            flush_secs = int(cast(Any, subgroup.attrs.get("flush_secs", 120)))
                            filename_suffix = str(
                                cast(Any, subgroup.attrs.get("filename_suffix", ""))
                            )

                            logger = SummaryWriter(
                                log_dir=log_dir,
                                comment=comment,
                                max_queue=max_queue,
                                flush_secs=flush_secs,
                                filename_suffix=filename_suffix,
                            )
                            items.append(logger)
                        else:
                            # Skip unknown logger types in containers
                            continue
                    elif subgroup.attrs.get("_python_logger"):
                        # Handle Python logger in containers
                        logger_class_name = subgroup.attrs.get("class_name", "Logger")

                        if logger_class_name == "Logger":
                            import logging

                            logger_name = cast(str, subgroup.attrs.get("logger_name", "quantem"))
                            logger_level = int(
                                cast(Any, subgroup.attrs.get("logger_level", logging.INFO))
                            )

                            logger = logging.getLogger(logger_name)
                            logger.setLevel(logger_level)
                            items.append(logger)
                        else:
                            # Skip unknown logger types in containers
                            continue
                    else:
                        raise ValueError(f"Unknown group structure at key '{key}' in {group.path}")
                else:
                    raise KeyError(f"Missing expected key '{key}' in container")
            return set(items)

        elif ctype == "dict":
            result: dict[str, Any] = {}
            # Restore scalars and simple objects stored as attributes
            for key in group.attrs:
                if (
                    key == "_container_type"
                    or key.endswith(".torch_save")
                    or key.endswith(".is_path")
                ):
                    continue
                val = group.attrs[key]
                # Convert string paths back to Path objects if needed
                val = cls._convert_string_to_path_if_needed(val, group, key)
                result[key] = val
            # Restore arrays (including torch tensors)
            for key in group.array_keys():
                result[key] = maybe_tensor(group, key)
            # Restore subgroups
            for key in group.group_keys():
                subgroup = cast(zarr.Group, group[key])
                if "_container_type" in subgroup.attrs:
                    result[key] = cls._deserialize_container(subgroup)
                elif "_autoserialize" in subgroup.attrs:
                    meta = cast(dict[str, Any], subgroup.attrs["_autoserialize"])
                    submod = __import__(
                        cast(str, meta["class_module"]), fromlist=[cast(str, meta["class_name"])]
                    )
                    subcls = getattr(submod, cast(str, meta["class_name"]))
                    result[key] = subcls._recursive_load(subgroup)
                elif subgroup.attrs.get("_torch_whole_module"):
                    module_arr = cast(zarr.Array, subgroup["module"])
                    data = cast(np.ndarray, module_arr[:]).tobytes()
                    buf = io.BytesIO(data)
                    # For containers, load to CPU - they'll be moved to the right device when attached to the main object
                    mod = torch.load(buf, map_location="cpu", weights_only=False)
                    result[key] = mod
                elif subgroup.attrs.get("_torch_tensor"):
                    # Handle new tensor format in containers
                    data = AutoSerialize._read_array_np(subgroup, "tensor").tobytes()
                    buf = io.BytesIO(data)
                    tensor = torch.load(buf, map_location="cpu", weights_only=False)
                    result[key] = tensor
                elif subgroup.attrs.get("_torch_logger"):
                    # Handle torch logger in containers
                    logger_class_name = subgroup.attrs.get("class_name", "SummaryWriter")

                    if logger_class_name == "SummaryWriter":
                        from torch.utils.tensorboard import SummaryWriter

                        log_dir = subgroup.attrs.get("log_dir", None)
                        comment = str(cast(Any, subgroup.attrs.get("comment", "")))
                        max_queue = int(cast(Any, subgroup.attrs.get("max_queue", 10)))
                        flush_secs = int(cast(Any, subgroup.attrs.get("flush_secs", 120)))
                        filename_suffix = str(cast(Any, subgroup.attrs.get("filename_suffix", "")))

                        logger = SummaryWriter(
                            log_dir=log_dir,
                            comment=comment,
                            max_queue=max_queue,
                            flush_secs=flush_secs,
                            filename_suffix=filename_suffix,
                        )
                        result[key] = logger
                    else:
                        # Skip unknown logger types in containers
                        continue
                elif subgroup.attrs.get("_python_logger"):
                    # Handle Python logger in containers
                    logger_class_name = subgroup.attrs.get("class_name", "Logger")

                    if logger_class_name == "Logger":
                        import logging

                        logger_name = cast(str, subgroup.attrs.get("logger_name", "quantem"))
                        logger_level = int(
                            cast(Any, subgroup.attrs.get("logger_level", logging.INFO))
                        )

                        logger = logging.getLogger(logger_name)
                        logger.setLevel(logger_level)
                        result[key] = logger
                    else:
                        # Skip unknown logger types in containers
                        continue
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
    meta = cast(dict[str, Any], root.attrs["_autoserialize"])
    version = int(meta.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported AutoSerialize version: {version}")

    # Read skip metadata (names/types) stored with the file, if present
    file_skip_names = set(cast(Sequence[str], root.attrs.get("_autoserialize_skip_names", [])))
    file_skip_types_raw = cast(
        Sequence[str] | None, root.attrs.get("_autoserialize_skip_types", [])
    )
    file_skip_types = (
        tuple(
            # Import each type by fully-qualified name from string
            __import__(t.rpartition(".")[0], fromlist=[t.rpartition(".")[2]]).__dict__[  # type: ignore[index]
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
    mod = __import__(cast(str, meta["class_module"]), fromlist=[cast(str, meta["class_name"])])
    cls = getattr(mod, cast(str, meta["class_name"]))
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
                class_info = cast(dict[str, Any] | None, obj.attrs.get("_autoserialize"))
                label = Path(path).name
                if class_info is not None and "class_name" in class_info:
                    mod_cls = (
                        f"{cast(str, class_info['class_module'])}.{cast(str, class_info['class_name'])}"
                        if show_class_origin
                        else cast(str, class_info["class_name"])
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

    _recurse(root)
