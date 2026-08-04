"""Versioned, reloadable PutPot semantic trajectory/controller parameters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 3
PROGRAM_NAME = "putpot_semantic_support_frames"


@dataclass(frozen=True)
class _Rule:
    kind: type
    minimum: float
    maximum: float


PARAMETER_RULES: dict[str, _Rule] = {
    "damping": _Rule(float, 1.0e-6, 1.0),
    "max_joint_delta": _Rule(float, 1.0e-6, 1.0),
    "max_position_step": _Rule(float, 1.0e-6, 0.1),
    "max_rotation_step": _Rule(float, 1.0e-6, 1.0),
    "support_clearance_m": _Rule(float, 0.0, 0.1),
    "transport_clearance_m": _Rule(float, 1.0e-6, 0.2),
    "collision_clearance_m": _Rule(float, 0.0, 0.2),
    "transport_steps": _Rule(int, 8, 2000),
    "lower_steps": _Rule(int, 1, 1000),
    "release_steps": _Rule(int, 1, 1000),
    "withdraw_steps": _Rule(int, 1, 1000),
    "settle_steps": _Rule(int, 1, 1000),
    "center_repair_steps": _Rule(int, 1, 1000),
    "missing_finger_contact_limit_m": _Rule(float, 0.001, 0.15),
    "receiving_jaw_center_translation_fraction": _Rule(float, 0.0, 1.0),
    "receiving_jaw_reorientation_fraction": _Rule(float, 0.0, 1.0),
}


@dataclass(frozen=True)
class PutPotProgramSpec:
    path: Path
    sha256: str
    parameters: Mapping[str, int | float]

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "program": PROGRAM_NAME,
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "parameters": dict(self.parameters),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_program_spec(path: str | Path) -> PutPotProgramSpec:
    """Load and fully validate one immutable program-spec file."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("PutPot program spec must be a JSON object")
    expected_top = {"schema_version", "program", "parameters"}
    if set(value) != expected_top:
        raise ValueError(
            "PutPot program spec keys must be exactly " + repr(sorted(expected_top))
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported PutPot program spec schema_version: "
            f"{value['schema_version']!r}"
        )
    if value["program"] != PROGRAM_NAME:
        raise ValueError(f"unsupported PutPot program: {value['program']!r}")
    parameters = value["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_RULES):
        raise ValueError(
            "PutPot program parameters must be exactly "
            + repr(sorted(PARAMETER_RULES))
        )
    validated: dict[str, int | float] = {}
    for name, rule in PARAMETER_RULES.items():
        parameter = parameters[name]
        if isinstance(parameter, bool):
            raise ValueError(f"PutPot program parameter {name} must be numeric")
        if rule.kind is int:
            if not isinstance(parameter, int):
                raise ValueError(f"PutPot program parameter {name} must be an integer")
            normalized: int | float = int(parameter)
        else:
            if not isinstance(parameter, (int, float)):
                raise ValueError(f"PutPot program parameter {name} must be numeric")
            normalized = float(parameter)
        if not rule.minimum <= normalized <= rule.maximum:
            raise ValueError(
                f"PutPot program parameter {name} must be in "
                f"[{rule.minimum}, {rule.maximum}]"
            )
        validated[name] = normalized
    return PutPotProgramSpec(source, _sha256(source), validated)


def apply_program_spec(namespace: Any, spec: PutPotProgramSpec) -> None:
    """Apply validated semantic parameters to an argparse-like namespace."""

    for name, value in spec.parameters.items():
        setattr(namespace, name, value)
