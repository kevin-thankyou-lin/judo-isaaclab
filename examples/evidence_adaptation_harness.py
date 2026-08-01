"""Create, inspect, ingest, or execute evidence-led adaptation attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from judo_isaaclab.evidence_harness import EvidenceLedger, execute_attempt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workspace", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    for action in ("ingest", "run"):
        command = subparsers.add_parser(action)
        command.add_argument("--phase", required=True)
        command.add_argument("--result", required=True)
        command.add_argument("--log", required=True)
        command.add_argument("--trace")
        command.add_argument("--video")
        command.add_argument("--revision", required=True)
        command.add_argument("--source-id", required=True)
        command.add_argument("--target-id")
        if action == "ingest":
            command.add_argument("--returncode", type=int, default=0)
        else:
            command.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    workspace = Path(args.workspace).resolve()
    ledger = EvidenceLedger(workspace / "ledger.json", args.bundle)
    if args.action == "init":
        print(json.dumps(ledger.value, indent=2, sort_keys=True))
        return
    if args.action == "status":
        print(json.dumps(ledger.value, indent=2, sort_keys=True))
        return
    if args.action == "verify":
        proof = ledger.proof_status()
        print(json.dumps(proof, indent=2, sort_keys=True))
        if not proof["complete"]:
            raise SystemExit(2)
        return
    common = {
        "phase": args.phase,
        "result_path": args.result,
        "log_path": args.log,
        "revision": args.revision,
        "source_id": args.source_id,
        "target_id": args.target_id,
        "trace_path": args.trace,
        "video_path": args.video,
    }
    if args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise SystemExit("run requires a command after --")
        record = execute_attempt(ledger, command=command, **common)
    else:
        record = ledger.add_attempt(returncode=args.returncode, **common)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
