"""Reloadable subprocess boundary for PutPot semantic controller programs.

The Isaac host owns simulation objects and Cartesian IK.  A controller child
owns semantic program logic and receives only JSON-serializable observations.
Replacing the child therefore reloads arbitrary Python without importing that
Python into, or restarting, the Isaac process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1
DEFAULT_TIMEOUT_S = 5.0


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any, *, nonfinite: str = "error") -> Any:
    """Convert controller data to JSON, optionally nulling sensor sentinels."""

    if nonfinite not in {"error", "null"}:
        raise ValueError("nonfinite policy must be error or null")

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            if nonfinite == "null":
                return None
            raise ValueError("controller IPC values must be finite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(item, nonfinite=nonfinite)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item, nonfinite=nonfinite) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return jsonable(value.tolist(), nonfinite=nonfinite)
    if hasattr(value, "item"):
        return jsonable(value.item(), nonfinite=nonfinite)
    raise TypeError(f"controller IPC value is not JSON serializable: {type(value)!r}")


def _finite_vector(value: Any, *, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"controller response {name} must contain {size} values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"controller response {name} must be numeric")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise ValueError(f"controller response {name} must be finite")
        result.append(normalized)
    return result


def validate_initialize_response(value: Any, base_steps: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("controller initialize response must be an object")
    expected = {"protocol_version", "program_name", "total_steps", "metadata"}
    if set(value) != expected:
        raise ValueError(
            "controller initialize response keys must be exactly " + repr(sorted(expected))
        )
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("controller protocol version mismatch")
    if not isinstance(value["program_name"], str) or not value["program_name"].strip():
        raise ValueError("controller program_name must be nonempty")
    total_steps = value["total_steps"]
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise ValueError("controller total_steps must be an integer")
    # Version 1 permits arbitrary phase/target logic but retains the host's
    # proven evidence horizon so strict endpoint predicates stay comparable.
    # A future protocol can negotiate a different evidence horizon explicitly.
    if total_steps != base_steps:
        raise ValueError(
            f"controller total_steps must equal evidence horizon {base_steps}"
        )
    if not isinstance(value["metadata"], dict):
        raise ValueError("controller metadata must be an object")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "program_name": value["program_name"],
        "total_steps": total_steps,
        "metadata": jsonable(value["metadata"]),
    }


def validate_command_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("controller command response must be an object")
    common = {"kind", "stage", "terminate"}
    kind = value.get("kind")
    if kind == "cartesian":
        expected = common | {"left_pose", "right_pose", "grippers"}
        if set(value) != expected:
            raise ValueError(
                "cartesian controller response keys must be exactly "
                + repr(sorted(expected))
            )
        result = {
            "kind": kind,
            "left_pose": _finite_vector(value["left_pose"], size=7, name="left_pose"),
            "right_pose": _finite_vector(value["right_pose"], size=7, name="right_pose"),
            "grippers": _finite_vector(value["grippers"], size=2, name="grippers"),
        }
    elif kind == "joint_action":
        expected = common | {"action"}
        if set(value) != expected:
            raise ValueError(
                "joint-action controller response keys must be exactly "
                + repr(sorted(expected))
            )
        result = {
            "kind": kind,
            "action": _finite_vector(value["action"], size=14, name="action"),
        }
    else:
        raise ValueError("controller response kind must be cartesian or joint_action")
    if not isinstance(value["stage"], str) or not value["stage"].strip():
        raise ValueError("controller response stage must be nonempty")
    if not isinstance(value["terminate"], bool):
        raise ValueError("controller response terminate must be boolean")
    result.update(stage=value["stage"], terminate=value["terminate"])
    return result


class ControllerPluginClient:
    """One controller child process with request/response JSONL IPC."""

    def __init__(
        self,
        plugin_path: str | os.PathLike[str],
        expected_sha256: str,
        *,
        runner_path: str | os.PathLike[str],
        log_path: str | os.PathLike[str],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        python: str = sys.executable,
    ) -> None:
        self.plugin_path = Path(plugin_path).resolve()
        self.sha256 = sha256_file(self.plugin_path)
        if self.sha256 != expected_sha256:
            raise ValueError("controller plugin hash does not match file bytes")
        if timeout_s <= 0.0:
            raise ValueError("controller timeout must be positive")
        self.timeout_s = float(timeout_s)
        self._sequence = 0
        self._log = open(log_path, "x", encoding="utf-8")
        self._process = subprocess.Popen(
            [python, str(Path(runner_path).resolve()), "--plugin", str(self.plugin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.initialized: dict[str, Any] | None = None
        hello = self._request("hello", {})
        if hello != {"protocol_version": PROTOCOL_VERSION}:
            self.close(force=True)
            raise ValueError("controller child handshake failed")

    @property
    def pid(self) -> int:
        return self._process.pid

    def _request(self, kind: str, payload: Mapping[str, Any]) -> Any:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"controller child exited with code {self._process.returncode}"
            )
        self._sequence += 1
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "id": self._sequence,
            "type": kind,
            # Simulator observations legitimately use NaN for unavailable pad
            # fractions before contact.  JSON null preserves missingness without
            # allowing non-finite controller commands back into the host.
            "payload": jsonable(payload, nonfinite="null"),
        }
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._process.stdin.flush()
        ready, _, _ = select.select(
            [self._process.stdout], [], [], self.timeout_s
        )
        if not ready:
            self.close(force=True)
            raise TimeoutError(f"controller child timed out handling {kind}")
        line = self._process.stdout.readline()
        if not line:
            returncode = self._process.poll()
            raise RuntimeError(
                f"controller child closed stdout while handling {kind}; "
                f"returncode={returncode}"
            )
        response = json.loads(line)
        expected = {"protocol_version", "id", "ok", "result", "error"}
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("controller child returned an invalid envelope")
        if response["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("controller child response protocol mismatch")
        if response["id"] != self._sequence:
            raise ValueError("controller child response id mismatch")
        if not response["ok"]:
            raise RuntimeError(f"controller plugin error: {response['error']}")
        if response["error"] is not None:
            raise ValueError("successful controller response included an error")
        return response["result"]

    def initialize(self, context: Mapping[str, Any], base_steps: int) -> dict[str, Any]:
        initialized = validate_initialize_response(
            self._request("initialize", context), base_steps
        )
        self.initialized = initialized
        return initialized

    def command(
        self,
        *,
        step: int,
        base_command: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.initialized is None:
            raise RuntimeError("controller child has not been initialized")
        return validate_command_response(
            self._request(
                "command",
                {
                    "step": step,
                    "base_command": base_command,
                    "observation": observation,
                },
            )
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "path": str(self.plugin_path),
            "sha256": self.sha256,
            "pid": self.pid,
            "program": None if self.initialized is None else self.initialized,
        }

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process.poll() is None and not force:
            try:
                self._request("shutdown", {})
            except Exception:
                force = True
        if process.poll() is None:
            if force:
                process.kill()
            else:
                process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self._log.close()

    def __enter__(self) -> "ControllerPluginClient":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close(force=_exc_type is not None)
