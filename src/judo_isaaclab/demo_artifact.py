"""Write replayable, provenance-backed demonstration HDF5 artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any, *, drop_env_axis: bool = True) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if drop_env_axis and result.ndim and result.shape[0] == 1:
        result = result[0]
    return result


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in tree.items():
        path = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            result.update(_flatten(value, path))
        else:
            result[path] = _array(value)
    return result


def _safe_observation(tree: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    """Keep numeric observations while omitting image-sized tensors.

    RGB evidence is stored in the independently verified MP4.  The HDF5 keeps
    policy/proprioceptive observations without multiplying campaign storage by
    hundreds of gigabytes.
    """

    if tree is None or not hasattr(tree, "items"):
        return {}
    result: dict[str, np.ndarray] = {}
    for name, value in _flatten(tree).items():
        if value.dtype.kind not in "biuf":
            continue
        if value.ndim >= 3 or value.size > 4096:
            continue
        result[name] = value
    return result


def _write_tree(group, values: Mapping[str, np.ndarray]) -> None:
    for path, value in values.items():
        parent = group
        parts = path.split("/")
        for name in parts[:-1]:
            parent = parent.require_group(name)
        parent.create_dataset(parts[-1], data=value, compression="gzip")


class DemonstrationRecorder:
    """Collect one uninterrupted rollout and write the project HDF schema."""

    def __init__(self) -> None:
        self._states: list[dict[str, np.ndarray]] = []
        self._actions: list[np.ndarray] = []
        self._observations: list[dict[str, np.ndarray]] = []
        self._semantic: list[dict[str, np.ndarray]] = []

    def start(self, state: Mapping[str, Any]) -> None:
        if self._states:
            raise RuntimeError("demonstration recorder already started")
        self._states.append(_flatten(state))

    def append(
        self,
        action: Any,
        state: Mapping[str, Any],
        *,
        observation: Mapping[str, Any] | None = None,
        semantic_observation: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._states:
            raise RuntimeError("call start() before append()")
        self._actions.append(_array(action))
        self._states.append(_flatten(state))
        self._observations.append(_safe_observation(observation))
        self._semantic.append(_safe_observation(semantic_observation))

    @staticmethod
    def _stack_common(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not rows:
            return {}
        common = set.intersection(*(set(row) for row in rows))
        result = {}
        for name in sorted(common):
            shapes = {row[name].shape for row in rows}
            if len(shapes) == 1:
                result[name] = np.stack([row[name] for row in rows])
        return result

    def write(
        self,
        path: str | Path,
        *,
        assets_instance_paths: Mapping[str, str],
        success: bool,
        metadata: Mapping[str, Any],
    ) -> None:
        if len(self._states) != len(self._actions) + 1:
            raise RuntimeError("a demo must contain exactly one more state than action")
        if not success:
            raise RuntimeError("refusing to write an unsuccessful demonstration")
        import h5py

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state_rows = self._stack_common(self._states)
        if not state_rows:
            raise RuntimeError("demonstration contains no common state fields")
        with h5py.File(destination, "w") as handle:
            data = handle.create_group("data")
            data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps(
                dict(assets_instance_paths), sort_keys=True
            )
            data.attrs["total"] = len(self._actions)
            data.attrs["generation_metadata"] = json.dumps(
                dict(metadata), sort_keys=True
            )
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = len(self._actions)
            demo.attrs["success"] = True
            demo.create_dataset(
                "actions", data=np.asarray(self._actions, dtype=np.float32), compression="gzip"
            )
            _write_tree(demo.create_group("states"), state_rows)
            _write_tree(
                demo.create_group("initial_state"),
                {name: value[:1] for name, value in state_rows.items()},
            )
            observations = self._stack_common(self._observations)
            observations.update(
                {
                    f"semantic/{name}": value
                    for name, value in self._stack_common(self._semantic).items()
                }
            )
            _write_tree(demo.create_group("obs"), observations)


def relative_asset_paths(
    absolute_paths: Mapping[str, str], objects_root: str | Path
) -> dict[str, str]:
    root = Path(objects_root).resolve()
    return {
        name: str(Path(path).resolve().relative_to(root))
        for name, path in absolute_paths.items()
    }
