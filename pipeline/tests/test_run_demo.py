"""Local server launcher checks without starting a server or loading models."""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location("run_demo", Path(__file__).resolve().parents[1] / "scripts" / "run_demo.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def make_inputs(tmp_path):
    bundle, checkpoint = tmp_path / "bundle", tmp_path / "encoder.pt"
    bundle.mkdir()
    for name in ("manifest.json", "tiles.json", "embeddings.npy", "detections.geojson"):
        (bundle / name).write_bytes(b"unit-test-only")
    checkpoint.write_bytes(b"unit-test-only")
    return bundle, checkpoint


def test_existing_environment_overrides_are_preserved(tmp_path, monkeypatch):
    bundle, checkpoint = make_inputs(tmp_path)
    monkeypatch.setenv("BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("ENCODER_CHECKPOINT", str(checkpoint))
    assert launcher.configure_environment() == (bundle, checkpoint)
    assert os.environ["BUNDLE_DIR"] == str(bundle)
    assert os.environ["ENCODER_CHECKPOINT"] == str(checkpoint)


def test_defaults_do_not_depend_on_callers_directory(tmp_path, monkeypatch):
    bundle, checkpoint = make_inputs(tmp_path)
    monkeypatch.setattr(launcher, "DEFAULT_BUNDLE", bundle)
    monkeypatch.setattr(launcher, "DEFAULT_CHECKPOINT", checkpoint)
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    monkeypatch.delenv("ENCODER_CHECKPOINT", raising=False)
    monkeypatch.chdir(tmp_path.parent)
    assert launcher.configure_environment() == (bundle, checkpoint)


def test_missing_or_incomplete_assets_fail_before_server_import(tmp_path, monkeypatch):
    bundle, checkpoint = make_inputs(tmp_path)
    monkeypatch.setenv("BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("ENCODER_CHECKPOINT", str(checkpoint))
    (bundle / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest.json"):
        launcher.configure_environment()
    monkeypatch.setenv("BUNDLE_DIR", "")
    with pytest.raises(ValueError, match="BUNDLE_DIR is empty"):
        launcher.configure_environment()


def test_host_port_forwarding_without_server_start(tmp_path, monkeypatch):
    bundle, checkpoint = make_inputs(tmp_path)
    monkeypatch.setenv("BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("ENCODER_CHECKPOINT", str(checkpoint))
    server = Mock()
    monkeypatch.setitem(sys.modules, "uvicorn", server)
    assert launcher.main(["--host", "127.0.0.2", "--port", "8082"]) == 0
    server.run.assert_called_once_with("terralens.api:app", host="127.0.0.2", port=8082)


@pytest.mark.parametrize("port", ["0", "65536", "invalid"])
def test_invalid_ports_are_rejected(port):
    with pytest.raises(SystemExit):
        launcher.main(["--port", port])
