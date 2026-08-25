from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from quantem.core import config
from math import floor

from quantem.core.ml.activation_functions import get_activation_function
from quantem.core.ml.blocks import Conv2dBlock, Upsample2dBlock, complex_pool, passfunc

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    if config.get("has_torch"):
        import torch
        import torch.nn as nn


class CNN2d(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,  # input channels (C_in, H, W)
        out_channels: int | None = None,  # output channels (C_out, H, W)
        start_filters: int = 16,
        num_layers: int = 3,  # num_layers
        num_per_layer: int = 2,  # number conv per layer
        use_skip_connections: bool = False,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = True,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(in_channels)
        self.start_filters = start_filters
        self.num_layers = num_layers
        self._num_per_layer = num_per_layer
        if use_skip_connections and num_per_layer < 2:
            raise ValueError(
                "If using skip connections, num_per_layer must be at least 2 to allow for "
                + "channel concatenation."
            )
        self.use_skip_connections = use_skip_connections
        self.dtype = dtype
        self.dropout = dropout
        self._use_batchnorm = use_batchnorm

        if self.dtype.is_complex:
            self.pool = complex_pool
        else:
            self.pool = passfunc
        self._pooler = nn.MaxPool2d(kernel_size=2, stride=2)

        self.concat = torch.cat
        self.flatten = nn.Flatten()

        if callable(activation):
            self._activation = activation
        else:
            self._activation = get_activation_function(activation, self.dtype)
        if callable(final_activation):
            self._final_activation = final_activation
        else:
            self._final_activation = get_activation_function(final_activation, self.dtype)
        if conv_kernel_size <=0:
            raise ValueError(f"Convolutional kernel size must be greater than 0. Got value {conv_kernel_size}")
        if conv_kernel_size % 2 == 0:
            raise ValueError(f"Convolutional kernel size must be an odd number. Got value {conv_kernel_size}")
        self._conv_kernel_size = int(conv_kernel_size)

        self._build()

    @property
    def activation(self) -> Callable:
        return self._activation

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @property
    def conv_kernel_size(self) -> int:
        return self._conv_kernel_size

    def _build(self):
        self.down_conv_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        in_channels = self.in_channels
        out_channels = self.start_filters
        for a0 in range(self.num_layers):
            if a0 != 0:
                out_channels = in_channels * 2
            self.down_conv_blocks.append(
                Conv2dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=0,
                    # dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self.activation,
                    kernel_size=self.conv_kernel_size,
                    padding=int(floor(self.conv_kernel_size/2)),
                )
            )
            in_channels = out_channels

        out_channels = in_channels * 2
        self.bottleneck = Conv2dBlock(
            nb_layers=self._num_per_layer,
            input_channels=in_channels,
            output_channels=out_channels,
            use_batchnorm=self._use_batchnorm,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self.activation,
            kernel_size=self.conv_kernel_size,
            padding=int(floor(self.conv_kernel_size/2)),
        )
        in_channels = out_channels

        for a0 in range(self.num_layers):
            out_channels = self.start_filters if a0 == self.num_layers - 1 else in_channels // 2

            in_channels2 = in_channels if self.use_skip_connections else out_channels

            self.upsample_blocks.append(
                Upsample2dBlock(
                    in_channels, out_channels, use_batchnorm=self._use_batchnorm, dtype=self.dtype
                )
            )

            self.up_conv_blocks.append(
                Conv2dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels2,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=0,
                    # dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self.activation,
                    kernel_size=self.conv_kernel_size,
                    padding=int(floor(self.conv_kernel_size/2)),
                )
            )

            in_channels = out_channels

        self.final_conv = Conv2dBlock(
            nb_layers=1,
            input_channels=self.start_filters,
            output_channels=self.out_channels,
            use_batchnorm=False,
            dropout=0,
            # dropout=self.dropout,
            dtype=self.dtype,
            activation=self.final_activation,
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down_block in self.down_conv_blocks:
            x = down_block(x)
            if self.use_skip_connections:
                skips.append(x)
            x = self.pool(x, self._pooler)

        x = self.bottleneck(x)
        for upsample_block, up_conv_block in zip(self.upsample_blocks, self.up_conv_blocks):
            x = upsample_block(x)
            if self.use_skip_connections:
                skip = skips.pop()
                x = torch.cat((x, skip), dim=1)
            x = up_conv_block(x)

        y = self.final_conv(x)

        return y

    def reset_weights(self):
        """
        Reset all weights.
        """

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)


class MultiChannelCNN2d(CNN2d):
    def __init__(
        self,
        in_channels=1,
        out_channels: int = 2,
        final_activations: list | tuple | None = None,
        **kwargs
    ):
        # Always use identity activation in base CNN, handle activations here
        super().__init__(in_channels=in_channels, out_channels=out_channels, final_activation="identity", **kwargs)
        self.final_activations = (
            ["sigmoid"] * out_channels if final_activations is None else final_activations
        )

    @property
    def final_activations(self): 
        return self._final_activations

    @final_activations.setter
    def final_activations(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != self.out_channels:
            raise ValueError(f"final_activations must be a list of length {self.out_channels}")
        self._final_activations = [get_activation_function(act, self.dtype) for act in value]

    def forward(self, x):
        out = super().forward(x) # B,C,H,W
        # Apply per-channel activation
        outs = []
        for i, fn in enumerate(self.final_activations):
            outs.append(fn(out[:, i:i+1]))
        return torch.cat(outs, dim=1)


# These aliases make the intentional separation from quantem.core.ml.cnn.CNN2d
# explicit while preserving the state-dict key layout of the paper checkpoint.
PolymerCNN2d = CNN2d
PolymerMultiChannelCNN2d = MultiChannelCNN2d

PAPER_MODEL_ID = "reference-v2"
PAPER_MODEL_VERSION = "2026-06-29"
PAPER_MODEL_SHA256 = "c2a4ed76cccd9313b4821d629b48767bbce6714b8466ff915357f5607b31c1a7"

_PAPER_SPEC: dict[str, Any] = {
    "schema_version": 1,
    "model_id": PAPER_MODEL_ID,
    "version": PAPER_MODEL_VERSION,
    "description": "Pinned paper polymer diffraction-peak detector.",
    "architecture": {
        "start_filters": 32,
        "num_layers": 4,
        "num_per_layer": 3,
        "kernel_size": 3,
        "input_channels": 1,
        "output_channels": 2,
        "dropout": 0.0,
    },
    "normalization": {
        "mode": "v1_global_percentile",
        "p_lower": 0.0418,
        "p_upper": 3.394,
    },
    "experimental_normalization": {
        "mode": "per_image_minmax_percentile",
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
    },
    "weights": {"filename": "best.pth", "sha256": PAPER_MODEL_SHA256},
}

# The DOI-backed URL is deliberately unset until the public, immutable Zenodo
# record exists. A local directory override remains available for private review.
DEFAULT_MODEL_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    PAPER_MODEL_ID: {
        PAPER_MODEL_VERSION: {
            "specification": _PAPER_SPEC,
            "weights_url": None,
        }
    }
}


@dataclass(frozen=True)
class PolymerModelResolution:
    """A verified, immutable polymer model artifact."""

    model_id: str
    version: str
    weights_path: Path
    specification: Mapping[str, Any]
    checksum: str


class PolymerModelError(RuntimeError):
    """Raised when a named polymer model cannot be resolved safely."""


def build_polymer_model(specification: Mapping[str, Any]) -> MultiChannelCNN2d:
    """Build the checkpoint-compatible network described by a model specification."""

    architecture = specification.get("architecture", {})
    return MultiChannelCNN2d(
        in_channels=int(architecture["input_channels"]),
        out_channels=int(architecture["output_channels"]),
        start_filters=int(architecture["start_filters"]),
        num_layers=int(architecture["num_layers"]),
        num_per_layer=int(architecture["num_per_layer"]),
        use_skip_connections=True,
        dropout=float(architecture.get("dropout", 0.0)),
        final_activations=["sigmoid"] * int(architecture["output_channels"]),
        conv_kernel_size=int(architecture["kernel_size"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_specification(
    specification: Mapping[str, Any], *, model_id: str, version: str, checksum: str
) -> None:
    if specification.get("schema_version") != 1:
        raise PolymerModelError(
            f"Model {model_id!r} version {version!r} uses an unsupported specification schema."
        )
    if specification.get("model_id") != model_id:
        raise PolymerModelError(
            f"Model specification identifies {specification.get('model_id')!r}, expected {model_id!r}."
        )
    architecture = specification.get("architecture", {})
    required = {
        "input_channels": 1,
        "output_channels": 2,
        "kernel_size": 3,
    }
    incompatible = {
        key: (architecture.get(key), expected)
        for key, expected in required.items()
        if architecture.get(key) != expected
    }
    if incompatible:
        raise PolymerModelError(
            f"Model {model_id!r} version {version!r} is incompatible with "
            f"BraggPeaksPolymer: {incompatible}."
        )
    declared = specification.get("weights", {}).get("sha256")
    if declared != checksum:
        raise PolymerModelError(
            f"Model specification checksum {declared!r} does not match registry checksum {checksum!r}."
        )


def _local_candidates(root: Path, model_id: str, version: str, filename: str):
    yield root / model_id / version / filename
    yield root / model_id / filename
    yield root / filename


def resolve_polymer_model(
    model_id: str = PAPER_MODEL_ID,
    version: str | None = None,
    *,
    latest: bool = False,
    local_model_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    registry: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    downloader: Callable[..., Any] = urlopen,
) -> PolymerModelResolution:
    """Resolve and verify a named polymer model.

    The paper model and version are pinned by default. ``latest=True`` is the
    only way to select a newer registered version. Private development can use
    ``local_model_dir`` or ``QUANTEM_POLYMER_MODEL_DIR`` without network access.
    """

    if latest and version is not None:
        raise ValueError("version and latest=True are mutually exclusive")
    selected_registry = DEFAULT_MODEL_REGISTRY if registry is None else registry
    versions = selected_registry.get(model_id)
    if not versions:
        raise PolymerModelError(f"Unknown polymer model {model_id!r}.")
    if latest:
        version = sorted(versions)[-1]
    elif version is None:
        version = PAPER_MODEL_VERSION if model_id == PAPER_MODEL_ID else sorted(versions)[0]
    entry = versions.get(version)
    if entry is None:
        raise PolymerModelError(
            f"Unknown version {version!r} for polymer model {model_id!r}."
        )

    specification = dict(entry["specification"])
    weights = specification.get("weights", {})
    checksum = str(weights.get("sha256", "")).lower()
    filename = str(weights.get("filename", "best.pth"))
    if len(checksum) != 64:
        raise PolymerModelError(f"Model {model_id!r} version {version!r} has no valid SHA-256.")
    _validate_specification(
        specification, model_id=model_id, version=version, checksum=checksum
    )

    local_root_value = local_model_dir or os.environ.get("QUANTEM_POLYMER_MODEL_DIR")
    if local_root_value:
        local_root = Path(local_root_value).expanduser()
        for candidate in _local_candidates(local_root, model_id, version, filename):
            if candidate.is_file():
                actual = _sha256(candidate)
                if actual != checksum:
                    raise PolymerModelError(
                        f"Checksum failure for local model {candidate}: expected {checksum}, got {actual}."
                    )
                return PolymerModelResolution(
                    model_id, version, candidate.resolve(), specification, actual
                )
        raise PolymerModelError(
            f"Model {model_id!r} version {version!r} was not found below local model "
            f"directory {local_root}. Expected {filename}."
        )

    cache_root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "quantem"
        / "polymer_models"
    )
    target = cache_root / model_id / version / checksum / filename
    if target.is_file():
        actual = _sha256(target)
        if actual == checksum:
            return PolymerModelResolution(model_id, version, target, specification, actual)
        raise PolymerModelError(
            f"Checksum failure for cached model {target}: expected {checksum}, got {actual}. "
            "Remove that file and resolve the model again."
        )

    weights_url = entry.get("weights_url")
    if not weights_url:
        raise PolymerModelError(
            f"Model {model_id!r} version {version!r} is not public yet and is absent from "
            "the local cache. Set QUANTEM_POLYMER_MODEL_DIR to the private model archive "
            "or pass local_model_dir explicitly."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as out:
            temporary = Path(out.name)
            try:
                response = downloader(str(weights_url))
                context = response if hasattr(response, "__enter__") else None
                source = context.__enter__() if context is not None else response
                try:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                finally:
                    if context is not None:
                        context.__exit__(None, None, None)
            except (HTTPError, URLError, OSError) as exc:
                raise PolymerModelError(
                    f"Could not download model {model_id!r} version {version!r} from "
                    f"{weights_url}: {exc}. An offline cache or local_model_dir may be used."
                ) from exc
        actual = _sha256(temporary)
        if actual != checksum:
            raise PolymerModelError(
                f"Checksum failure after downloading {model_id!r} version {version!r}: "
                f"expected {checksum}, got {actual}."
            )
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return PolymerModelResolution(model_id, version, target, specification, checksum)


__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "PAPER_MODEL_ID",
    "PAPER_MODEL_VERSION",
    "PolymerCNN2d",
    "PolymerModelError",
    "PolymerModelResolution",
    "PolymerMultiChannelCNN2d",
    "build_polymer_model",
    "resolve_polymer_model",
]
