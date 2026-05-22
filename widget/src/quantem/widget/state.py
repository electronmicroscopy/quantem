import importlib.metadata
import json
import pathlib
from typing import Any

JSON_METADATA_VERSION = "1.0"


def resolve_widget_version() -> str:
    for dist_name in ("quantem.widget", "quantem-widget"):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "0.0.0+local"


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


def unwrap_state_payload(
    payload: dict[str, Any],
    *,
    require_envelope: bool = False,
    expected_widget: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a dict.")
    if "state" in payload:
        state = payload["state"]
        if not isinstance(state, dict):
            raise ValueError("State envelope field 'state' must be a dict.")
        # If caller passed the widget name, refuse cross-widget loads
        # (Show2D state into Show3D would silently load wrong subset of traits).
        got = payload.get("widget_name")
        if expected_widget is not None and got is not None and got != expected_widget:
            raise ValueError(
                f"State envelope is for {got!r}, cannot load into {expected_widget!r}"
            )
        return state
    if require_envelope:
        raise ValueError("State JSON file must be a versioned envelope with top-level 'state'.")
    return payload


def _numpy_safe(o):
    # numpy scalars in ROI dicts (np.int64 from `arr.shape[0] // 2` etc.) used to
    # raise TypeError in json.dumps. Coerce via .item().
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_state_file(path: str | pathlib.Path, widget_name: str, state: dict[str, Any]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wrap_state_dict(widget_name, state), indent=2, default=_numpy_safe))
