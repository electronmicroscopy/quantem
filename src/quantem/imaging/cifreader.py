import os
import re
from pathlib import Path
from typing import Union

from quantem.core.io.serialize import AutoSerialize


class CIFReader(AutoSerialize):
    """A class to read and store data from a .cif file."""

    def __init__(self, filename: Union[str, Path]):
        self.filename = self._validate_cif_file(filename)
        self.data = {}
        self._atoms = []
        self._symmetry_ops_str = []  # raw strings like "x,y,z"
        self._symmetry_ops = []  # parsed as (R, t), where R is 3x3 and t is 3-vector

        # Initialize by reading all data
        self._read_file()

        # Validate cell parameters
        self._validate_cell_parameters()

    def _validate_cif_file(self, filename: Union[str, Path]):
        """Validate that the file is a .cif file and exists"""
        # Convert to string if Path object
        filename_str = str(filename)

        # Check if file exists
        if not os.path.exists(filename_str):
            raise FileNotFoundError(f"File not found: {filename_str}")

        # Check if it's a file (not a directory)
        if not os.path.isfile(filename_str):
            raise ValueError(f"Path is not a file: {filename_str}")

        # Check file extension
        if not filename_str.lower().endswith(".cif"):
            raise ValueError(f"File must be .cif file, got: {filename_str}")

        return filename_str

    def _validate_cell_parameters(self):
        """Validate presence and physical correctness of unit cell parameters."""
        cp = self.cell_params  # builds defaults for angles if missing
        values = cp["values"]
        # Required keys (now using short names)
        required = ["a", "b", "c", "alpha", "beta", "gamma"]
        missing = [k for k in required if k not in values]
        if missing:
            raise ValueError(
                f"Incomplete unit cell parameters: missing {missing}. Check .cif file"
            )

        # Convert to floats (in case any are strings)
        def as_float(k):
            v = values[k]
            if isinstance(v, (int, float)):
                return float(v)
            fv = self._to_float_if_possible(v)
            if fv is None:
                raise ValueError(f"Unit cell parameter {k} is not a valid number: {v}")
            return fv

        a = as_float("a")  # Changed from 'cell_length_a'
        b = as_float("b")  # Changed from 'cell_length_b'
        c = as_float("c")  # Changed from 'cell_length_c'
        alpha = as_float("alpha")  # Changed from 'cell_angle_alpha'
        beta = as_float("beta")  # Changed from 'cell_angle_beta'
        gamma = as_float("gamma")  # Changed from 'cell_angle_gamma'

        # Physical range checks
        if not (a > 0 and b > 0 and c > 0):
            raise ValueError(f"Unit cell lengths must be positive. Got a={a}, b={b}, c={c}")
        for name, angle in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
            if not (0.0 < angle < 180.0):
                raise ValueError(
                    f"Unit cell angle {name} must be between 0 and 180 degrees (exclusive). Got {angle}"
                )

        # Compute and validate volume
        import math

        alpha_rad = math.radians(alpha)
        beta_rad = math.radians(beta)
        gamma_rad = math.radians(gamma)
        cos_alpha = math.cos(alpha_rad)
        cos_beta = math.cos(beta_rad)
        cos_gamma = math.cos(gamma_rad)
        # General triclinic volume formula
        volume_sq = (
            1
            + 2 * cos_alpha * cos_beta * cos_gamma
            - cos_alpha * cos_alpha
            - cos_beta * cos_beta
            - cos_gamma * cos_gamma
        )
        if volume_sq <= 0:
            raise ValueError(
                "Unit cell geometry is invalid (non-positive Gram determinant). Check angles."
            )
        V = a * b * c * math.sqrt(volume_sq)
        if not (V > 0):
            raise ValueError(
                f"Unit cell volume must be positive. Computed {V} from a={a}, b={b}, c={c}, "
                f"alpha={alpha}, beta={beta}, gamma={gamma}"
            )

        # Store computed volume for convenience
        self.data["cell_volume"] = V

    def _read_file(self):
        """Read and parse the CIF file"""
        with open(self.filename, "r") as file:
            self.lines = [line.rstrip() for line in file.readlines()]

        self._parse_data()

    def _parse_data(self):
        """Parse CIF data"""
        i = 0
        while i < len(self.lines):
            line = self.lines[i].strip()

            if not line or line.startswith("#"):
                i += 1
                continue

            if line.startswith("_"):
                if line.startswith("_cell_"):
                    i = self._parse_cell_parameter(i)
                elif line.startswith("_space_group_name"):
                    i = self._parse_space_group(i)
                elif line.startswith("_chemical_formula"):
                    i = self._parse_formula(i)
                elif line.startswith("_space_group_symop_operation_xyz") or line.startswith(
                    "_symmetry_equiv_pos_as_xyz"
                ):
                    i = self._parse_symmetry_loop(i)
                elif line.startswith("_atom_site_"):
                    i = self._parse_atom_loop(i)
                else:
                    i += 1
            elif line == "loop_":
                i += 1
            else:
                i += 1

        # If no symmetry ops were found, default to identity operation
        if not self._symmetry_ops:
            self._symmetry_ops = [([1, 0, 0, 0, 1, 0, 0, 0, 1], (0.0, 0.0, 0.0))]

    def _parse_cell_parameter(self, line_idx: int):
        """Parse cell parameters"""
        line = self.lines[line_idx]
        parts = line.split(None, 1)  # split into tag and value (keep uncertainty)

        if len(parts) >= 2:
            param_name = parts[0][1:]  # Remove leading underscore
        raw_value = parts[1].strip()
        val = self._to_float_if_possible(raw_value)

        # Store float if possible; otherwise keep raw string
        self.data[param_name] = val if val is not None else raw_value
        return line_idx + 1

    def _parse_space_group(self, line_idx: int):
        """Parse space group information"""
        line = self.lines[line_idx]

        if "'" in line:
            space_group = line.split("'")[1]
        elif '"' in line:
            space_group = line.split('"')[1]
        else:
            parts = line.split()
            space_group = parts[1] if len(parts) > 1 else ""

        self.data["space_group"] = space_group
        return line_idx + 1

    def _parse_formula(self, line_idx: int):
        """Parse chemical formula"""
        line = self.lines[line_idx]

        if "'" in line:
            formula = line.split("'")[1]
        elif '"' in line:
            formula = line.split('"')[1]
        else:
            parts = line.split()
            formula = " ".join(parts[1:]) if len(parts) > 1 else ""

        self.data["formula"] = formula
        return line_idx + 1

    def _parse_symmetry_loop(self, line_idx: int):
        """Parse symmetry operations loop"""
        headers = []
        # Read headers for symmetry loop
        while line_idx < len(self.lines):
            s = self.lines[line_idx].strip()
            if s.startswith("_space_group_symop_") or s.startswith("_symmetry_equiv_pos_"):
                headers.append(s)
                line_idx += 1
            else:
                break

        # Determine which header column holds operation strings
        op_idx = -1
        for j, h in enumerate(headers):
            if h.endswith("operation_xyz") or h.endswith("as_xyz"):
                op_idx = j
                break

        # Read symmetry rows
        while line_idx < len(self.lines):
            s = self.lines[line_idx].strip()
            if not s or s.startswith("_") or s.startswith("#") or s == "loop_":
                break
            parts = s.split()
            if op_idx != -1 and len(parts) > op_idx:
                op_str = parts[op_idx].strip().strip('"').strip("'")
                # Accept commas with or without spaces
                self._symmetry_ops_str.append(op_str)
            line_idx += 1

        # Build numeric ops (R, t)
        self._build_symmetry_ops()

        return line_idx

    def _parse_atom_loop(self, line_idx: int):
        """Parse atom site loop"""
        headers = []

        # Read headers
        while line_idx < len(self.lines) and self.lines[line_idx].strip().startswith(
            "_atom_site_"
        ):
            headers.append(self.lines[line_idx].strip())
            line_idx += 1

        # Read atom data
        while line_idx < len(self.lines):
            line = self.lines[line_idx].strip()

            if not line or line.startswith("_") or line.startswith("#") or line == "loop_":
                break

            parts = line.split()
            if len(parts) >= len(headers):
                atom_data = {}
                for i, header in enumerate(headers):
                    if i < len(parts):
                        # Handle numeric fields including displacement parameters
                        if any(
                            field in header for field in ["fract_", "U_iso_or_equiv", "occupancy"]
                        ):
                            val = parts[i]
                            fv = self._to_float_if_possible(val)
                            atom_data[header] = fv if fv is not None else val
                        else:
                            atom_data[header] = parts[i]
                self._atoms.append(atom_data)

            line_idx += 1

        return line_idx

    def _to_float_if_possible(self, value):
        """Try to convert CIF numeric string (possibly with parentheses) to float"""
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return float(value)
        if isinstance(value, str):
            v = value.strip()
            # Trim uncertainty part like 0.123(4)
            if "(" in v:
                v = v.split("(")[0]
            try:
                return float(v)
            except ValueError:
                return None
        return None

    def _build_symmetry_ops(self):
        """Convert operation strings like '-x+1/2,y+1/2,z' to numeric (R, t)"""
        ops = []
        for op in self._symmetry_ops_str:
            # Normalize and split by commas
            op_clean = op.replace(" ", "")
            parts = op_clean.split(",")
            if len(parts) != 3:
                continue
            R_rows = []
            t_vals = []
            for comp in parts:
                r_row, t = self._parse_symop_component(comp)
                R_rows.append(r_row)
                t_vals.append(t)
            # Store R as flat 9 elements (row-major) to keep simple style consistent
            R_flat = [
                R_rows[0][0],
                R_rows[0][1],
                R_rows[0][2],
                R_rows[1][0],
                R_rows[1][1],
                R_rows[1][2],
                R_rows[2][0],
                R_rows[2][1],
                R_rows[2][2],
            ]
            ops.append((R_flat, (t_vals[0], t_vals[1], t_vals[2])))
        if ops:
            self._symmetry_ops = ops

    def _parse_symop_component(self, comp: str):
        """
        Parse a single component like '-x+1/2' into a row of R (rx, ry, rz) and a t shift.
        Allowed tokens: optional sign, x|y|z or integer or fraction n/d.
        """

        # Initialize row and translation
        r = [0, 0, 0]
        t = 0.0

        # Tokenize terms with optional signs
        # Examples matched: x, -x, +y, 1/2, -1/3, 1, -2
        for m in re.finditer(r"([+-]?)(x|y|z|\d+/\d+|\d+)", comp):
            sign_str, token = m.groups()
            sign = -1 if sign_str == "-" else 1
            if token in ("x", "y", "z"):
                idx = {"x": 0, "y": 1, "z": 2}[token]
                r[idx] += sign
            else:
                # number or fraction
                if "/" in token:
                    num, den = token.split("/")
                    try:
                        t += sign * (float(num) / float(den))
                    except ZeroDivisionError:
                        pass
                else:
                    t += sign * float(token)

        return r, t

    @property
    def cell_params(self):
        """Get unit cell parameters with values and units"""
        if not hasattr(self, "_cell_params"):
            vals = {}
            units = {}

            # Map from CIF keys to short keys
            length_params = {"cell_length_a": "a", "cell_length_b": "b", "cell_length_c": "c"}

            angle_params = {
                "cell_angle_alpha": ("alpha", 90.0),
                "cell_angle_beta": ("beta", 90.0),
                "cell_angle_gamma": ("gamma", 90.0),
            }

            # Process length parameters
            for cif_key, short_key in length_params.items():
                if cif_key in self.data:
                    vals[short_key] = self.data[cif_key]
                    units[short_key] = "Å"

            # Process angle parameters
            for cif_key, (short_key, default) in angle_params.items():
                if cif_key in self.data:
                    vals[short_key] = self.data[cif_key]
                    units[short_key] = "degrees"
                else:
                    vals[short_key] = default
                    units[short_key] = "degrees"

            self._cell_params = {
                "values": vals,
                "units": units,
                "complete": len(vals) == 6,  # Flag indicating if all params are present
            }

        return self._cell_params

    @property
    def atoms(self):
        """Get atomic coordinates"""
        return self._atoms

    @property
    def symmetry_operations(self):
        """Get raw symmetry operations strings"""
        return list(self._symmetry_ops_str)

    @property
    def spacegroup(self):
        """Get space group"""
        return self.data.get("space_group", "")

    @property
    def formula(self):
        """Get chemical formula"""
        return self.data.get("formula", "")


# End
