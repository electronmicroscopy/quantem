import csv
from pathlib import Path
from typing import Optional, Union


def _read_csv_without_preamble(path: Union[Path, str]) -> list[str]:
    """Return CSV lines starting after leading citation/comment/blank lines."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.lower().startswith("citation:")
        ):
            return lines[index:]

    raise ValueError(f"{path} does not contain a CSV header")


def _parse_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


def load_xray_lines_database(path: Union[Path, str]) -> dict[str, dict[str, dict[str, float]]]:
    """Load X-ray lines CSV into the legacy element->line metadata mapping."""
    elements: dict[str, dict[str, dict[str, float]]] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}

    reader = csv.DictReader(_read_csv_without_preamble(path))
    for row in reader:
        element = str(row.get("element", "")).strip()
        line_name = str(row.get("line", "")).strip()
        if not element or not line_name:
            continue

        energy_kev = _parse_float(row, ("energy_keV", "energy (keV)", "energy"))
        if energy_kev is None:
            energy_ev = _parse_float(row, ("energy_eV", "energy (eV)"))
            if energy_ev is None:
                continue
            energy_kev = energy_ev / 1000.0

        weight = _parse_float(row, ("weight", "relative_intensity"))
        if weight is None:
            weight = 0.0

        element_lines = elements.setdefault(element, {})
        key = (element, line_name)
        if line_name in element_lines:
            duplicate_counts[key] = duplicate_counts.get(key, 1) + 1
            line_name = f"{line_name}__{duplicate_counts[key]}"

        element_lines[line_name] = {
            "energy (keV)": energy_kev,
            "weight": weight,
        }

    return elements


def load_eels_edges_database(path: Union[Path, str]) -> dict[str, dict[str, dict[str, object]]]:
    """Load EELS edge CSV into the legacy element->edge metadata mapping."""
    elements: dict[str, dict[str, dict[str, object]]] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}

    reader = csv.DictReader(_read_csv_without_preamble(path))
    fieldnames = set(reader.fieldnames or [])
    required_columns = ("symbol", "edge_label", "edge_energy_eV")
    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required EELS edge columns: {', '.join(missing_columns)}"
        )

    for row in reader:
        element_symbol = str(row.get("symbol", "")).strip()
        if not element_symbol:
            continue

        energy_ev = _parse_float(row, ("edge_energy_eV", "onset_energy (eV)", "energy_eV"))
        if energy_ev is None:
            continue

        edge_label = str(row.get("edge_label", "")).strip()
        element_edges = elements.setdefault(element_symbol, {})
        edge_name = f"{energy_ev:g} eV"
        key = (element_symbol, edge_name)
        if edge_name in element_edges:
            duplicate_counts[key] = duplicate_counts.get(key, 1) + 1
            edge_name = f"{edge_name}__{duplicate_counts[key]}"

        edge_info: dict[str, object] = {
            "onset_energy (eV)": energy_ev,
        }
        if edge_label:
            edge_info["edge_label"] = edge_label

        atomic_number = _parse_float(row, ("atomic_number",))
        if atomic_number is not None:
            edge_info["atomic_number"] = (
                int(atomic_number) if atomic_number.is_integer() else atomic_number
            )

        element_name = str(row.get("element", "")).strip()
        if element_name:
            edge_info["element"] = element_name

        element_edges[edge_name] = edge_info

    return elements
