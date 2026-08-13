"""Build deterministic pair-local PutPot HDF5 inputs with provenance receipts.

The source file is copied as a whole.  The only logical HDF5 mutation is
``data.attrs["ASSETS_INSTANCE_PATHS"]``; action, state, observation, metadata,
and demo ordering are preserved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import h5py
import numpy as np


PAIR_COUNT = 40
_DEMO_RE = re.compile(r"demo_(\d+)$")
_POT_RE = re.compile(r"cooking_pot_(\d{6})$")
_COOKTOP_RE = re.compile(r"induction_cooktop_(\d{6})$")
_ASSETS_ATTR = "ASSETS_INSTANCE_PATHS"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _dataset_digest(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(dataset.dtype), "shape": list(dataset.shape)},
            sort_keys=True,
        ).encode("utf-8")
    )
    value = np.asarray(dataset[()])
    if value.dtype.kind in "OUSV":
        for item in value.reshape(-1).tolist():
            encoded = json.dumps(_json_value(item), sort_keys=True).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    else:
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _logical_manifest(handle: h5py.File, *, omit_assets_attr: bool) -> dict[str, Any]:
    manifest: dict[str, Any] = {"groups": [], "attributes": {}, "datasets": {}}

    def record(path: str, obj: h5py.Group | h5py.Dataset) -> None:
        absolute = "/" + path if path else "/"
        attrs = {
            name: _json_value(value)
            for name, value in sorted(obj.attrs.items())
            if not (omit_assets_attr and absolute == "/data" and name == _ASSETS_ATTR)
        }
        if attrs:
            manifest["attributes"][absolute] = attrs
        if isinstance(obj, h5py.Dataset):
            manifest["datasets"][absolute] = {
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "sha256": _dataset_digest(obj),
            }
        else:
            manifest["groups"].append(absolute)

    record("", handle)
    handle.visititems(record)
    manifest["groups"].sort()
    return manifest


def _leaf_lengths(group: h5py.Group) -> dict[str, int]:
    lengths: dict[str, int] = {}

    def record(path: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            if obj.ndim < 1:
                raise ValueError(f"dataset {obj.name} is scalar and has no time axis")
            lengths[path] = int(obj.shape[0])

    group.visititems(record)
    return lengths


def validate_source_hdf5(
    source_hdf5: str | Path,
    *,
    acknowledge_nonstandard_alignment: bool = False,
) -> dict[str, Any]:
    """Validate demos from groups, deliberately ignoring stale ``data.total``."""

    source = Path(source_hdf5)
    if not source.is_file():
        raise FileNotFoundError(f"source HDF5 does not exist: {source}")

    demos: list[dict[str, Any]] = []
    with h5py.File(source, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ValueError("source HDF5 is missing the /data group")
        data = handle["data"]
        demo_names: list[tuple[int, str]] = []
        for name, obj in data.items():
            match = _DEMO_RE.fullmatch(name)
            if name.startswith("demo_") and not match:
                raise ValueError(f"invalid demo group name: /data/{name}")
            if match:
                if not isinstance(obj, h5py.Group):
                    raise ValueError(f"/data/{name} must be a group")
                demo_names.append((int(match.group(1)), name))
        demo_names.sort()
        if not demo_names:
            raise ValueError("source HDF5 has no demo_* groups")
        numeric_indices = [index for index, _ in demo_names]
        if len(numeric_indices) != len(set(numeric_indices)):
            raise ValueError("source HDF5 has duplicate numeric demo indices")

        for _, name in demo_names:
            demo = data[name]
            if "actions" not in demo or not isinstance(demo["actions"], h5py.Dataset):
                raise ValueError(f"/data/{name} is missing the actions dataset")
            if "states" not in demo or not isinstance(demo["states"], h5py.Group):
                raise ValueError(f"/data/{name} is missing the states group")
            actions = demo["actions"]
            if actions.ndim < 1 or int(actions.shape[0]) <= 0:
                raise ValueError(f"/data/{name}/actions must have a nonempty time axis")
            action_count = int(actions.shape[0])
            state_lengths = _leaf_lengths(demo["states"])
            if not state_lengths:
                raise ValueError(f"/data/{name}/states contains no datasets")
            unique_state_lengths = sorted(set(state_lengths.values()))
            if len(unique_state_lengths) != 1:
                raise ValueError(
                    f"/data/{name}/states has inconsistent time axes: {unique_state_lengths}"
                )
            state_count = unique_state_lengths[0]
            is_standard = state_count == action_count + 1
            is_acknowledgeable = state_count == action_count
            if not is_standard and not (
                acknowledge_nonstandard_alignment and is_acknowledgeable
            ):
                hint = (
                    "; pass --acknowledge-nonstandard-alignment only if this equal-length "
                    "source contract has been reviewed"
                    if is_acknowledgeable
                    else ""
                )
                raise ValueError(
                    f"/data/{name} has {action_count} actions and {state_count} states; "
                    f"expected actions + 1{hint}"
                )
            if "num_samples" in demo.attrs and int(demo.attrs["num_samples"]) != action_count:
                raise ValueError(
                    f"/data/{name} num_samples={demo.attrs['num_samples']} does not match "
                    f"{action_count} actions"
                )
            if "obs" in demo and isinstance(demo["obs"], h5py.Group):
                obs_lengths = _leaf_lengths(demo["obs"])
                bad_obs = sorted(
                    {length for length in obs_lengths.values() if length != action_count}
                )
                if bad_obs:
                    raise ValueError(
                        f"/data/{name}/obs has time axes {bad_obs}; expected {action_count}"
                    )
            if "processed_actions" in demo:
                processed = demo["processed_actions"]
                if not isinstance(processed, h5py.Dataset) or processed.ndim < 1:
                    raise ValueError(f"/data/{name}/processed_actions has no time axis")
                if int(processed.shape[0]) not in {action_count, action_count + 1}:
                    raise ValueError(
                        f"/data/{name}/processed_actions has {processed.shape[0]} rows; "
                        f"expected {action_count} or {action_count + 1}"
                    )
            demos.append(
                {
                    "name": name,
                    "actions": action_count,
                    "states": state_count,
                    "alignment": "actions_plus_one_state" if is_standard else "equal_length",
                }
            )

    return {
        "demo_count": len(demos),
        "computed_total_actions": sum(demo["actions"] for demo in demos),
        "demos": demos,
        "acknowledged_nonstandard_alignment": bool(acknowledge_nonstandard_alignment),
    }


def validate_asset_catalog(task_root: str | Path) -> dict[int, dict[str, str]]:
    task = Path(task_root)
    objects = task / "objects"
    specifications = (
        ("pot", objects / "PotSimple", _POT_RE, "cooking_pot"),
        ("cooktop", objects / "CooktopSimple", _COOKTOP_RE, "induction_cooktop"),
    )
    found: dict[str, dict[int, str]] = {}
    for role, root, pattern, stem in specifications:
        if not root.is_dir():
            raise FileNotFoundError(f"missing {role} asset root: {root}")
        indexed: dict[int, str] = {}
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            match = pattern.fullmatch(path.name)
            if not match:
                raise ValueError(f"unexpected directory in {root}: {path.name}")
            index = int(match.group(1))
            if index in indexed:
                raise ValueError(f"duplicate {role} pair index {index}")
            usd = path / f"{stem}_{index:06d}.usd"
            if not usd.is_file():
                raise FileNotFoundError(f"missing {role} USD for pair {index:06d}: {usd}")
            indexed[index] = str(path.relative_to(objects))
        expected = set(range(PAIR_COUNT))
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))
            extra = sorted(set(indexed) - expected)
            raise ValueError(
                f"{role} assets must be contiguous 000000-000039; "
                f"missing={missing}, extra={extra}"
            )
        found[role] = indexed
    return {
        index: {"pot": found["pot"][index], "cooktop": found["cooktop"][index]}
        for index in range(PAIR_COUNT)
    }


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.tmp."
    )
    os.close(descriptor)
    return Path(name)


def _publish(temp: Path, destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temp, destination)
        return
    try:
        os.link(temp, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output: {destination}") from error
    temp.unlink()


def build_target_hdf5(
    source_hdf5: str | Path,
    task_root: str | Path,
    pair_index: int,
    output_hdf5: str | Path,
    *,
    acknowledge_nonstandard_alignment: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if pair_index not in range(PAIR_COUNT):
        raise ValueError(f"pair_index must be in [0, {PAIR_COUNT - 1}], got {pair_index}")
    source = Path(source_hdf5).resolve()
    output = Path(output_hdf5).resolve()
    if output == source:
        raise ValueError("output_hdf5 must not be the source HDF5")
    receipt_path = output.with_suffix(output.suffix + ".provenance.json")
    receipt_hash_path = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    destinations = (output, receipt_path, receipt_hash_path)
    if not overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s): " + ", ".join(existing)
            )

    catalog = validate_asset_catalog(task_root)
    source_validation = validate_source_hdf5(
        source,
        acknowledge_nonstandard_alignment=acknowledge_nonstandard_alignment,
    )
    target_assets = catalog[pair_index]

    hdf_temp = _temporary_path(output)
    receipt_temp = _temporary_path(receipt_path)
    receipt_hash_temp = _temporary_path(receipt_hash_path)
    published: list[Path] = []
    try:
        shutil.copy2(source, hdf_temp)
        with h5py.File(source, "r") as source_handle:
            source_manifest = _logical_manifest(source_handle, omit_assets_attr=True)
            raw_source_assets = source_handle["data"].attrs.get(_ASSETS_ATTR)
            source_assets = json.loads(raw_source_assets) if raw_source_assets is not None else None
        with h5py.File(hdf_temp, "r+") as output_handle:
            output_handle["data"].attrs[_ASSETS_ATTR] = json.dumps(
                target_assets, sort_keys=True, separators=(",", ":")
            )
            output_handle.flush()
        with h5py.File(hdf_temp, "r") as output_handle:
            output_manifest = _logical_manifest(output_handle, omit_assets_attr=True)
            encoded_assets = output_handle["data"].attrs[_ASSETS_ATTR]
            if json.loads(encoded_assets) != target_assets:
                raise RuntimeError("target asset attribute did not round-trip")
        if source_manifest != output_manifest:
            raise RuntimeError("source and output differ outside ASSETS_INSTANCE_PATHS")

        source_manifest_sha = _sha256_bytes(_canonical_json(source_manifest))
        receipt_core = {
            "schema": "putpot-quality-target-v1",
            "pair_index": pair_index,
            "pair_id": f"{pair_index:06d}",
            "source_hdf5": str(source),
            "source_hdf5_sha256": sha256_file(source),
            "output_hdf5": str(output),
            "output_hdf5_sha256": sha256_file(hdf_temp),
            "source_assets_instance_paths": source_assets,
            "target_assets_instance_paths": target_assets,
            "logical_content_sha256_excluding_assets_attr": source_manifest_sha,
            "source_validation": source_validation,
        }
        receipt = dict(receipt_core)
        receipt["receipt_payload_sha256"] = _sha256_bytes(_canonical_json(receipt_core))
        receipt_bytes = _canonical_json(receipt)
        receipt_sha256 = _sha256_bytes(receipt_bytes)
        receipt_temp.write_bytes(receipt_bytes)
        receipt_hash_temp.write_text(
            f"{receipt_sha256}  {receipt_path.name}\n", encoding="utf-8"
        )
        for temp, destination in (
            (hdf_temp, output),
            (receipt_temp, receipt_path),
            (receipt_hash_temp, receipt_hash_path),
        ):
            _publish(temp, destination, overwrite=overwrite)
            published.append(destination)
        return receipt
    except Exception:
        for path in (hdf_temp, receipt_temp, receipt_hash_temp):
            path.unlink(missing_ok=True)
        if not overwrite:
            for path in published:
                path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hdf5", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--pair-index",
        action="append",
        type=int,
        help="zero-based asset pair; repeat as needed (default: all 40 pairs)",
    )
    parser.add_argument(
        "--acknowledge-nonstandard-alignment",
        action="store_true",
        help="accept reviewed sources with equal action/state lengths instead of states=actions+1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    pairs = args.pair_index if args.pair_index is not None else list(range(PAIR_COUNT))
    if len(set(pairs)) != len(pairs):
        parser.error("--pair-index values must be unique")
    output_root = Path(args.output_root)
    if not args.overwrite:
        existing: list[str] = []
        for pair_index in pairs:
            output = output_root / f"pot_{pair_index:06d}.hdf5"
            receipt = output.with_suffix(output.suffix + ".provenance.json")
            receipt_hash = receipt.with_suffix(receipt.suffix + ".sha256")
            existing.extend(str(path) for path in (output, receipt, receipt_hash) if path.exists())
        if existing:
            parser.error("refusing partial batch; output(s) already exist: " + ", ".join(existing))
    receipts = []
    for pair_index in pairs:
        receipts.append(
            build_target_hdf5(
                args.source_hdf5,
                args.task_root,
                pair_index,
                output_root / f"pot_{pair_index:06d}.hdf5",
                acknowledge_nonstandard_alignment=args.acknowledge_nonstandard_alignment,
                overwrite=args.overwrite,
            )
        )
    print(
        "PUTPOT_QUALITY_TARGETS="
        + json.dumps(
            {
                "count": len(receipts),
                "pairs": [receipt["pair_id"] for receipt in receipts],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
