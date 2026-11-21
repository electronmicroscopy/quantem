"""MAPED operates on supplied datasets without importing accelerator packages."""

import ast
from pathlib import Path

import pytest

import quantem.diffraction.maped as maped


@pytest.mark.parametrize(
    "filename", ["workflow.py", "core/align.py", "core/merge.py", "io.py", "plot.py", "report.py"]
)
def test_maped_has_no_native_accelerator_dependencies(filename: str) -> None:
    """Both public orchestration and bounded merging retain Torch ownership."""
    tree = ast.parse((Path(maped.__file__).parent / filename).read_text())
    imports = set()
    native_kernels = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
            imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Attribute) and node.attr in {"RawKernel", "RawModule"}:
            native_kernels.add(node.attr)

    native_roots = {"cupy", "pycuda", "mlx", "pynvml", "Metal", "objc", "ctypes"}
    assert not {name for name in imports if name.split(".")[0] in native_roots}
    assert not native_kernels
    assert not {name for name in imports if name.startswith(("quantem.gpu", "quantem.widget"))}
