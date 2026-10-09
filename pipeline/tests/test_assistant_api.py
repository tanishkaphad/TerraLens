from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest

from fastapi import HTTPException
from fastapi.testclient import TestClient

from terralens import api
from terralens.jobs import JobError, JobManager


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    monkeypatch.setattr(api.assistant, "load_configuration", lambda: None)
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    api._CHAT_REQUESTS.clear()
    api._CHAT_REQUESTS_DAY.clear()


def test_assistant_status_reports_unconfigured_gemini_without_exposing_keys(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with TestClient(api.app) as client:
        status = client.get("/api/assistant/status")
        assert status.status_code == 200
        assert status.json()["provider"] == "gemini"
        assert status.json()["configured"] is False
        assert "API key" in status.json()["reason"]
        assert client.post("/api/chat", json={"message": "What is in this area?"}).status_code == 503


def test_chat_job_poll_and_export_use_the_saved_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-token")
    monkeypatch.setenv("JOBS_DB", str(tmp_path / "jobs.sqlite3"))
    manager = JobManager(tmp_path / "jobs.sqlite3", workers=1, max_pending=2)
    monkeypatch.setattr(api, "get_job_manager", lambda: manager)
    reference = {"type": "Feature", "id": "workflow_reference:0", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
        "properties": {"id": "roads:r-1", "layer": "workflow_reference", "original_feature_id": "roads:r-1",
            "source_attributes": {"surface": "unpaved"}}}
    observation = {"type": "Feature", "id": "detection:d-1", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0]]]},
        "properties": {"id": "d-1", "role": "visual_detection"}}
    subject = {"type": "Feature", "id": "building:b-1", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "properties": {"conditions": [{"condition": "within_distance", "reference_layer": "workflow_reference", "reference_feature_ids": ["0"], "status": "pass"}],
            "evidence_geometries": [observation]}}
    uncertain = {"type": "Feature", "id": "building:u-1", "geometry": subject["geometry"],
        "properties": {"conditions": [{"condition": "visual_detection", "status": "unknown"}], "evidence_geometries": []}}
    excluded = {"type": "Feature", "id": "building:x-1", "geometry": subject["geometry"],
        "properties": {"conditions": [{"condition": "within_distance", "status": "fail"}], "evidence_geometries": []}}
    run_calls = []

    def run(message, dataset_id, history, progress):
        run_calls.append(message)
        assert message == "Find buildings near roads"
        assert history == []
        progress("search", "Searching mapped evidence")
        result = {"status": "completed", "message": "One supported record.", "summary": "One supported record.",
            "workflow": {"title": "Mapped buildings near roads", "requirements": [], "steps": []},
            "artifacts": [{"id": "buildings", "kind": "map", "title": "Buildings",
                "result": {"features": [subject], "uncertain_candidates": [uncertain],
                    "verification_excluded": [excluded], "context_features": [reference],
                    "result_kind": "vector_features", "dataset_version": "frozen-v1",
                    "provenance": {"vector_layers": {"roads": {"source": "fixture"}}}}},
                {"id": "count", "kind": "chart", "title": "Count", "rows": [{"group": "All", "count": 1}],
                    "chart": {"labels": ["All"], "values": [1], "unit": "count"}},
                {"id": "inventory", "kind": "table", "title": "Inventory", "rows": [{"item": "Area", "value": "Fixture"}]}],
            "tool_trace": [], "validation": {"schema": "passed"}}
        fingerprint = json.dumps({"workflow": result["workflow"], "artifacts": result["artifacts"]},
            sort_keys=True, separators=(",", ":"))
        result["receipt"] = {"sha256": hashlib.sha256(fingerprint.encode()).hexdigest()}
        return result

    monkeypatch.setattr(api, "_assistant_run", run)
    with TestClient(api.app) as client:
        queued = client.post("/api/chat", json={"message": "Find buildings near roads", "dataset_id": "spacenet"})
        assert queued.status_code == 202
        identity = queued.json()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{identity['job_id']}", params={"conversation_id": identity["conversation_id"]}).json()
            if job["status"] in {"failed", "succeeded"}:
                break
            time.sleep(0.01)
        assert job["status"] == "succeeded"
        result = job["result"]
        assert result["conversation_id"] == identity["conversation_id"]
        assert result["export_url"].endswith(identity["conversation_id"])
        exported = client.get(result["export_url"])
        assert exported.status_code == 200
        collection = exported.json()
        assert collection["type"] == "FeatureCollection"
        assert {item["properties"]["workflow_status"] for item in collection["features"]} == {
            "supported", "uncertain", "verification_excluded", "evidence_geometry", "reference"}
        assert {item["id"] for item in collection["features"]} == {
            "building:b-1", "building:u-1", "building:x-1", "detection:d-1", "workflow_reference:0"}
        evidence = next(item for item in collection["features"] if item["id"] == "detection:d-1")
        assert evidence["properties"]["workflow_parent_feature_id"] == "building:b-1"
        saved = collection["metadata"]["workflow_result"]
        assert {item["kind"] for item in saved["artifacts"]} == {"map", "chart", "table"}
        fingerprint = json.dumps({"workflow": saved["workflow"], "artifacts": saved["artifacts"]},
            sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(fingerprint.encode()).hexdigest() == saved["receipt"]["sha256"]
        assert len(run_calls) == 1
        unauthorized = client.get(f"/api/jobs/{identity['job_id']}", params={"conversation_id": "0" * 32})
        assert unauthorized.status_code == 404
    manager.shutdown()


def test_rejected_job_does_not_consume_chat_capacity(monkeypatch):
    monkeypatch.setattr(api.assistant, "status", lambda: {"configured": True})
    monkeypatch.setenv("CHAT_REQUESTS_PER_HOUR", "1")
    monkeypatch.setenv("CHAT_REQUESTS_PER_DAY", "1")

    class RejectingManager:
        def submit(self, *args):
            raise JobError("Conversation not found.", 404)

    monkeypatch.setattr(api, "get_job_manager", lambda: RejectingManager())
    with TestClient(api.app) as client:
        rejected = client.post("/api/chat", json={"message": "Rejected request", "conversation_id": "f" * 32})
        assert rejected.status_code == 404
        assert len(api._CHAT_REQUESTS) == len(api._CHAT_REQUESTS_DAY) == 0

        class AcceptedManager:
            def submit(self, *args):
                return {"job_id": "a" * 32, "conversation_id": "b" * 32}

        monkeypatch.setattr(api, "get_job_manager", lambda: AcceptedManager())
        accepted = client.post("/api/chat", json={"message": "Accepted request"})
        assert accepted.status_code == 202
        assert len(api._CHAT_REQUESTS) == len(api._CHAT_REQUESTS_DAY) == 1


def test_capacity_check_and_successful_submission_are_atomic(monkeypatch):
    monkeypatch.setattr(api.assistant, "status", lambda: {"configured": True})
    monkeypatch.setenv("CHAT_REQUESTS_PER_HOUR", "1")
    monkeypatch.setenv("CHAT_REQUESTS_PER_DAY", "1")

    class SlowManager:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def submit(self, *args):
            with self.lock:
                self.calls += 1
            time.sleep(0.05)
            return {"job_id": "a" * 32, "conversation_id": "b" * 32}

    manager = SlowManager()
    monkeypatch.setattr(api, "get_job_manager", lambda: manager)
    barrier = threading.Barrier(3)

    def request():
        barrier.wait()
        try:
            api.chat(api.ChatRequest(message="Concurrent request"))
            return 202
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request) for _ in range(2)]
        barrier.wait()
        statuses = [future.result() for future in futures]
    assert sorted(statuses) == [202, 429]
    assert manager.calls == 1
    assert len(api._CHAT_REQUESTS) == len(api._CHAT_REQUESTS_DAY) == 1
