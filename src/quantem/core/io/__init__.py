from quantem.core.io.serialize import AutoSerialize as AutoSerialize
from quantem.core.io.serialize import load as load
from quantem.core.io.serialize import print_file as print_file

_LAZY = {"read_2d", "read_4dstem", "read_emdfile_to_4dstem"}


def __getattr__(name: str):
    if name in _LAZY:
        from quantem.core.io import file_readers

        return getattr(file_readers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
