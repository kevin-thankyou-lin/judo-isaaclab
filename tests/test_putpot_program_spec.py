import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from judo_isaaclab.putpot_program_spec import (
    PARAMETER_RULES,
    apply_program_spec,
    load_program_spec,
)


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = REPO_ROOT / "configs/putpot_semantic_program_v3.json"


def _write_spec(tmp_path, mutate=None):
    value = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(value)
    path = tmp_path / "program.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_program_spec_validates_and_hashes_exact_file_bytes():
    spec = load_program_spec(DEFAULT_SPEC)
    expected = hashlib.sha256(DEFAULT_SPEC.read_bytes()).hexdigest()

    assert spec.sha256 == expected
    assert set(spec.parameters) == set(PARAMETER_RULES)
    assert spec.receipt()["sha256"] == expected
    assert spec.receipt()["path"] == str(DEFAULT_SPEC.resolve())

    namespace = SimpleNamespace()
    apply_program_spec(namespace, spec)
    assert namespace.transport_steps == spec.parameters["transport_steps"]
    assert namespace.missing_finger_contact_limit_m == pytest.approx(0.045)
    assert namespace.receiving_jaw_center_translation_fraction == pytest.approx(0.0)
    assert namespace.receiving_jaw_reorientation_fraction == pytest.approx(0.45)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.update(schema_version=4), "schema_version"),
        (lambda value: value.update(program="unknown"), "unsupported PutPot program"),
        (lambda value: value.update(extra=True), "keys must be exactly"),
        (
            lambda value: value["parameters"].pop("transport_steps"),
            "parameters must be exactly",
        ),
        (
            lambda value: value["parameters"].update(transport_steps=7),
            "transport_steps must be in",
        ),
        (
            lambda value: value["parameters"].update(settle_steps=1.5),
            "settle_steps must be an integer",
        ),
        (
            lambda value: value["parameters"].update(damping=True),
            "damping must be numeric",
        ),
        (
            lambda value: value["parameters"].update(
                receiving_jaw_center_translation_fraction=-0.01
            ),
            "receiving_jaw_center_translation_fraction must be in",
        ),
        (
            lambda value: value["parameters"].update(
                missing_finger_contact_limit_m=0.151
            ),
            "missing_finger_contact_limit_m must be in",
        ),
    ],
)
def test_program_spec_rejects_schema_shape_type_and_range(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_program_spec(_write_spec(tmp_path, mutate))
