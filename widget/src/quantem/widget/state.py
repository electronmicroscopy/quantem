import importlib.metadata
import json
import pathlib
from typing import Any

JSON_METADATA_VERSION = "1.0"


def resolve_widget_version() -> str:
    try:
        return importlib.metadata.version("quantem-widget")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_json_header(widget_name: str) -> dict[str, Any]:
    return {
        "metadata_version": JSON_METADATA_VERSION,
        "widget_name": widget_name,
        "widget_version": resolve_widget_version(),
    }


def wrap_state_dict(widget_name: str, state: dict[str, Any]) -> dict[str, Any]:
    envelope = build_json_header(widget_name)
    envelope["state"] = state
    return envelope


def unwrap_state_payload(payload: dict[str, Any], *, require_envelope: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a dict.")
    if "state" in payload:
        state = payload["state"]
        if not isinstance(state, dict):
            raise ValueError("State envelope field 'state' must be a dict.")
        return state
    if require_envelope:
        raise ValueError("State JSON file must be a versioned envelope with top-level 'state'.")
    return payload


def save_state_file(path: str | pathlib.Path, widget_name: str, state: dict[str, Any]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wrap_state_dict(widget_name, state), indent=2))
