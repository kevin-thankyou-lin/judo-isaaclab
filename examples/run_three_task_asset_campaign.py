"""Run resumable source-replay-first demo generation over three 40-pair tasks."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assets(path: str | Path) -> dict[str, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["data"].attrs["ASSETS_INSTANCE_PATHS"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _pair_id(assets: dict[str, str]) -> str:
    return "__".join(Path(assets[name]).name.lower() for name in sorted(assets))


def enumerate_pairs(task: dict[str, Any]) -> list[dict[str, str]]:
    files: list[str] = []
    for pattern in task["dataset_globs"]:
        files.extend(glob.glob(_expand(pattern)))
    files = sorted({os.path.realpath(path) for path in files})
    expected = int(task.get("expected_pairs", 40))
    if len(files) != expected:
        raise RuntimeError(
            f"{task['name']}: expected {expected} dataset files, found {len(files)}"
        )
    pairs = [
        {"dataset": path, "pair_id": _pair_id(_assets(path)), "assets": _assets(path)}
        for path in files
    ]
    ids = [pair["pair_id"] for pair in pairs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise RuntimeError(f"{task['name']}: duplicate asset pairs: {duplicates}")
    source = os.path.realpath(_expand(task["source_dataset"]))
    pairs.sort(key=lambda pair: (pair["dataset"] != source, pair["pair_id"]))
    if pairs[0]["dataset"] != source:
        raise RuntimeError(f"{task['name']}: source dataset is not in the 40-pair set")
    return pairs


def _run(command: list[str], log_path: Path, *, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("CAMPAIGN_COMMAND=" + json.dumps(command), flush=True)
    if dry_run:
        return 0
    with open(log_path, "w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return process.returncode


def _task_success(result: dict[str, Any]) -> bool:
    checks = result.get("checks", {})
    return bool(
        checks.get(
            "accepted_task_success",
            checks.get("coded_task_success", result.get("terminal", {}).get("task_success", False)),
        )
    )


def _reusable_classification(result: dict[str, Any], target_dataset: str) -> bool:
    if result.get("status") != "passed" or result.get("mode") != "replay":
        return False
    provenance = result.get("provenance", {})
    target = provenance.get("target_dataset", {})
    if target.get("sha256") != _sha256(target_dataset):
        return False
    for artifact in (result.get("video"), provenance.get("trace")):
        if not artifact or not os.path.isfile(artifact.get("path", "")):
            return False
        if artifact.get("sha256") != _sha256(artifact["path"]):
            return False
    return all(result.get("acceptance_checks", {}).values())


def validate_demo(path: str | Path, expected_assets: dict[str, str]) -> dict[str, Any]:
    import h5py

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing demonstration: {path}")
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = int(demo["actions"].shape[0])
        if actions <= 0 or int(demo.attrs.get("num_samples", -1)) != actions:
            raise RuntimeError(f"invalid action/sample count in {path}")
        state_lengths: list[int] = []
        demo["states"].visititems(
            lambda _, value: state_lengths.append(int(value.shape[0]))
            if isinstance(value, h5py.Dataset) else None
        )
        if not state_lengths or any(length != actions + 1 for length in state_lengths):
            raise RuntimeError(f"state/action alignment failed in {path}")
        observation_lengths: list[int] = []
        demo["obs"].visititems(
            lambda _, value: observation_lengths.append(int(value.shape[0]))
            if isinstance(value, h5py.Dataset) else None
        )
        if not observation_lengths or any(length != actions for length in observation_lengths):
            raise RuntimeError(f"observation/action alignment failed in {path}")
        raw_assets = handle["data"].attrs["ASSETS_INSTANCE_PATHS"]
        if isinstance(raw_assets, bytes):
            raw_assets = raw_assets.decode("utf-8")
        if json.loads(str(raw_assets)) != expected_assets:
            raise RuntimeError(f"demonstration asset provenance mismatch in {path}")
        if not bool(demo.attrs.get("success", False)):
            raise RuntimeError(f"demonstration success attribute is false in {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "actions": actions}


def _command(
    task: dict[str, Any],
    *,
    python: str,
    gear_repo: str,
    target: str,
    mode: str,
    output: Path,
    source_keyframes: Path | None,
    direct_replay_result: Path | None,
) -> list[str]:
    command = [
        python,
        str((REPO_ROOT / task["runner"]).resolve()),
        "--gear-repo", gear_repo,
        "--source-dataset", _expand(task["source_dataset"]),
        "--target-dataset", target,
        "--objects-root", _expand(task["objects_root"]),
        "--mode", mode,
        "--render",
        "--video", str(output / f"{mode}.mp4"),
        "--trace-npz", str(output / f"{mode}_trace.npz"),
        "--result-json", str(output / f"{mode}_result.json"),
        "--demo-hdf5", str(output / f"{mode}_demo.hdf5"),
    ]
    command.extend(_expand(str(value)) for value in task.get("runner_args", []))
    if mode == "replay":
        command.append("--classification-run")
        if source_keyframes is not None:
            command.extend(["--write-keyframes", str(source_keyframes)])
    else:
        if source_keyframes is not None:
            command.extend(["--source-keyframes", str(source_keyframes)])
        if direct_replay_result is not None:
            command.extend(["--direct-replay-result", str(direct_replay_result)])
    return command


def run_task(
    task: dict[str, Any],
    *,
    python: str,
    gear_repo: str,
    output_root: Path,
    dry_run: bool,
    max_pairs: int | None,
) -> dict[str, Any]:
    pairs = enumerate_pairs(task)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    task_root = output_root / task["name"]
    ledger_path = task_root / "ledger.json"
    ledger = _load(ledger_path) if ledger_path.is_file() else {
        "schema_version": 1,
        "task": task["name"],
        "expected_pairs": int(task.get("expected_pairs", 40)),
        "pairs": {},
    }
    keyframes = task_root / "source_keyframes.json" if task.get("needs_keyframes") else None

    for pair in pairs:
        pair_id = pair["pair_id"]
        existing = ledger["pairs"].get(pair_id, {})
        if existing.get("status") == "accepted":
            try:
                validate_demo(existing["demonstration"]["path"], pair["assets"])
                print(f"CAMPAIGN_RESUME_ACCEPTED={task['name']}:{pair_id}", flush=True)
                continue
            except Exception as error:
                print(f"CAMPAIGN_RESUME_INVALID={task['name']}:{pair_id}:{error}", flush=True)
        pair_root = task_root / pair_id
        pair_root.mkdir(parents=True, exist_ok=True)
        write_keyframes = keyframes if pair["dataset"] == os.path.realpath(_expand(task["source_dataset"])) else None
        replay_result_path = pair_root / "replay_result.json"
        prior_replay = _load(replay_result_path) if replay_result_path.is_file() else {}
        if not dry_run and _reusable_classification(prior_replay, pair["dataset"]):
            replay_rc = 0
            print(f"CAMPAIGN_REUSE_CLASSIFICATION={task['name']}:{pair_id}", flush=True)
        else:
            replay = _command(
                task,
                python=python,
                gear_repo=gear_repo,
                target=pair["dataset"],
                mode="replay",
                output=pair_root,
                source_keyframes=write_keyframes,
                direct_replay_result=None,
            )
            replay_rc = _run(replay, pair_root / "replay.log", dry_run=dry_run)
        if dry_run:
            continue
        replay_result = _load(replay_result_path) if replay_result_path.is_file() else {}
        replay_success = replay_rc == 0 and replay_result.get("status") == "passed" and _task_success(replay_result)
        if replay_success:
            demo = validate_demo(pair_root / "replay_demo.hdf5", pair["assets"])
            record = {
                "status": "accepted", "method": "source_action_replay",
                "dataset": pair["dataset"], "assets": pair["assets"],
                "result": str(replay_result_path.resolve()),
                "video": str((pair_root / "replay.mp4").resolve()),
                "demonstration": demo,
            }
        else:
            if replay_rc != 0 or replay_result.get("status") != "passed":
                record = {
                    "status": "infrastructure_failed", "method": "source_action_replay",
                    "dataset": pair["dataset"], "assets": pair["assets"],
                    "returncode": replay_rc, "result": str(replay_result_path.resolve()),
                }
            elif pair["dataset"] == os.path.realpath(_expand(task["source_dataset"])) and keyframes is not None and not keyframes.is_file():
                record = {
                    "status": "blocked", "reason": "source replay failed; no simulator-derived keyframes",
                    "dataset": pair["dataset"], "assets": pair["assets"],
                }
            else:
                repair_record = None
                repair_mode = task.get("repair_mode")
                if repair_mode and replay_result.get("checks", {}).get("coded_task_success"):
                    repair = _command(
                        task,
                        python=python,
                        gear_repo=gear_repo,
                        target=pair["dataset"],
                        mode=repair_mode,
                        output=pair_root,
                        source_keyframes=keyframes,
                        direct_replay_result=replay_result_path,
                    )
                    repair_rc = _run(repair, pair_root / f"{repair_mode}.log", dry_run=False)
                    repair_result_path = pair_root / f"{repair_mode}_result.json"
                    repair_result = _load(repair_result_path) if repair_result_path.is_file() else {}
                    if repair_rc == 0 and repair_result.get("status") == "passed" and _task_success(repair_result):
                        demo = validate_demo(pair_root / f"{repair_mode}_demo.hdf5", pair["assets"])
                        repair_record = {
                            "status": "accepted", "method": "source_action_prefix_with_supported_center_repair",
                            "dataset": pair["dataset"], "assets": pair["assets"],
                            "replay_result": str(replay_result_path.resolve()),
                            "result": str(repair_result_path.resolve()),
                            "video": str((pair_root / f"{repair_mode}.mp4").resolve()),
                            "demonstration": demo,
                        }
                if repair_record is not None:
                    record = repair_record
                    ledger["pairs"][pair_id] = record
                    ledger["summary"] = {
                        "accepted": sum(v.get("status") == "accepted" for v in ledger["pairs"].values()),
                        "replay_successes": sum(v.get("method") == "source_action_replay" and v.get("status") == "accepted" for v in ledger["pairs"].values()),
                        "adapted_successes": sum(v.get("method") != "source_action_replay" and v.get("status") == "accepted" for v in ledger["pairs"].values()),
                        "nonterminal": sum(v.get("status") != "accepted" for v in ledger["pairs"].values()),
                    }
                    _atomic_json(ledger_path, ledger)
                    print("CAMPAIGN_PAIR=" + json.dumps({"task": task["name"], "pair": pair_id, **record}, sort_keys=True), flush=True)
                    continue
                skill = _command(
                    task,
                    python=python,
                    gear_repo=gear_repo,
                    target=pair["dataset"],
                    mode="skill",
                    output=pair_root,
                    source_keyframes=keyframes,
                    direct_replay_result=replay_result_path,
                )
                skill_rc = _run(skill, pair_root / "skill.log", dry_run=False)
                skill_result_path = pair_root / "skill_result.json"
                skill_result = _load(skill_result_path) if skill_result_path.is_file() else {}
                if skill_rc == 0 and skill_result.get("status") == "passed" and _task_success(skill_result):
                    demo = validate_demo(pair_root / "skill_demo.hdf5", pair["assets"])
                    record = {
                        "status": "accepted", "method": "deterministic_semantic_skill",
                        "dataset": pair["dataset"], "assets": pair["assets"],
                        "replay_result": str(replay_result_path.resolve()),
                        "result": str(skill_result_path.resolve()),
                        "video": str((pair_root / "skill.mp4").resolve()),
                        "demonstration": demo,
                    }
                else:
                    record = {
                        "status": "adaptation_failed", "method": "deterministic_semantic_skill",
                        "dataset": pair["dataset"], "assets": pair["assets"],
                        "returncode": skill_rc, "replay_result": str(replay_result_path.resolve()),
                        "result": str(skill_result_path.resolve()),
                    }
        ledger["pairs"][pair_id] = record
        ledger["summary"] = {
            "accepted": sum(v.get("status") == "accepted" for v in ledger["pairs"].values()),
            "replay_successes": sum(v.get("method") == "source_action_replay" and v.get("status") == "accepted" for v in ledger["pairs"].values()),
            "adapted_successes": sum(v.get("method") != "source_action_replay" and v.get("status") == "accepted" for v in ledger["pairs"].values()),
            "nonterminal": sum(v.get("status") != "accepted" for v in ledger["pairs"].values()),
        }
        _atomic_json(ledger_path, ledger)
        print("CAMPAIGN_PAIR=" + json.dumps({"task": task["name"], "pair": pair_id, **record}, sort_keys=True), flush=True)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--task", action="append")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load(args.config)
    selected = set(args.task or [])
    tasks = [task for task in config["tasks"] if not selected or task["name"] in selected]
    if selected - {task["name"] for task in tasks}:
        raise ValueError(f"unknown tasks: {sorted(selected - {task['name'] for task in tasks})}")
    ledgers = [
        run_task(
            task,
            python=args.python,
            gear_repo=args.gear_repo,
            output_root=Path(args.output_root),
            dry_run=args.dry_run,
            max_pairs=args.max_pairs,
        )
        for task in tasks
    ]
    if args.dry_run:
        print("CAMPAIGN_DRY_RUN_PASSED")
        return
    accepted = sum(ledger.get("summary", {}).get("accepted", 0) for ledger in ledgers)
    expected = sum(int(ledger["expected_pairs"]) for ledger in ledgers)
    print(f"CAMPAIGN_FINAL={accepted}/{expected}", flush=True)
    if accepted != expected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
