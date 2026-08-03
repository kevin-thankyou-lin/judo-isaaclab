import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "examples/audit_putmarker_campaign.py"
    spec = importlib.util.spec_from_file_location("audit_putmarker_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_provenance_hash_audit_fails_closed(tmp_path):
    module = _module()
    asset = tmp_path / "asset.usd"
    asset.write_bytes(b"official")
    errors = []

    verified = module._check_provenance(
        {"usd": {"path": str(asset), "sha256": module._sha256(asset)}}, errors
    )
    assert verified == 1
    assert errors == []

    asset.write_bytes(b"changed")
    errors = []
    assert module._check_provenance(
        {"usd": {"path": str(asset), "sha256": "bad"}}, errors
    ) == 0
    assert "hash mismatch" in errors[0]


def test_video_receipt_requires_actual_full_decode(tmp_path, monkeypatch):
    module = _module()
    video = tmp_path / "skill.mp4"
    video.write_bytes(b"video")

    class Completed:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    calls = iter(
        [
            Completed(0, json.dumps({"streams": [{"codec_name": "h264", "nb_read_frames": "607", "width": 960, "height": 240}]})),
            Completed(2, stderr="decode error"),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(calls))
    receipt = module._video_receipt(video)

    assert receipt["codec"] == "h264"
    assert receipt["decoded_frames"] == 607
    assert receipt["full_decode_returncode"] == 2
    assert receipt["decode_stderr"] == "decode error"
