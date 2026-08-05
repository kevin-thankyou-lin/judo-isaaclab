"""Run one reloadable PutPot semantic controller plugin over JSONL IPC."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from judo_isaaclab.putpot_controller_protocol import PROTOCOL_VERSION, jsonable


def _load_plugin(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"putpot_controller_{path.stat().st_ino}_{path.stat().st_mtime_ns}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_controller", None)
    if not callable(factory):
        raise TypeError("controller plugin must define create_controller()")
    controller = factory()
    if not callable(getattr(controller, "initialize", None)):
        raise TypeError("controller must define initialize(context)")
    if not callable(getattr(controller, "command", None)):
        raise TypeError("controller must define command(request)")
    return controller


def _response(identifier, *, result=None, error=None):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "id": identifier,
        "ok": error is None,
        "result": jsonable(result) if error is None else None,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()
    controller = _load_plugin(Path(args.plugin).resolve())
    for line in sys.stdin:
        identifier = None
        should_stop = False
        try:
            request = json.loads(line)
            expected = {"protocol_version", "id", "type", "payload"}
            if not isinstance(request, dict) or set(request) != expected:
                raise ValueError("invalid controller request envelope")
            if request["protocol_version"] != PROTOCOL_VERSION:
                raise ValueError("controller request protocol mismatch")
            identifier = request["id"]
            kind = request["type"]
            payload = request["payload"]
            if kind == "hello":
                result = {"protocol_version": PROTOCOL_VERSION}
            elif kind == "initialize":
                result = controller.initialize(payload)
            elif kind == "command":
                result = controller.command(payload)
            elif kind == "shutdown":
                close = getattr(controller, "close", None)
                if callable(close):
                    close()
                result = {"closed": True}
                should_stop = True
            else:
                raise ValueError(f"unsupported controller request type: {kind!r}")
            response = _response(identifier, result=result)
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            response = _response(
                identifier,
                error=f"{type(exc).__name__}: {exc}",
            )
        print(json.dumps(response, sort_keys=True), flush=True)
        if should_stop:
            break


if __name__ == "__main__":
    main()
