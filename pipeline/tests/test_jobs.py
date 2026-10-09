import threading
import time

import pytest

from terralens.jobs import JobError, JobManager
from terralens.workflow import WorkflowError


def wait_for(manager, submitted, status="succeeded", timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = manager.get(submitted["job_id"], submitted["conversation_id"])
        if current["status"] == status:
            return current
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {status}: {current}")


def test_persists_result_progress_and_compact_last_eight_history(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager(path, workers=1, max_pending=2)
    histories = []
    conversation_id = None

    def run(message, dataset_id, history, progress):
        histories.append(history)
        progress("search", f"Searching {dataset_id}")
        return {
            "message": f"answer for {message}",
            "workflow": {"title": message, "steps": [{"id": "search"}]},
            "artifacts": [{"large": "x" * 20_000}],
        }

    for index in range(10):
        submitted = manager.submit(f"question {index}", "demo", conversation_id, run)
        conversation_id = submitted["conversation_id"]
        completed = wait_for(manager, submitted)
        assert completed["stage"] == {"step_id": "search", "label": "Searching demo"}
    assert len(histories[-1]) == 8
    assert [item["message"] for item in histories[-1]] == [f"question {index}" for index in range(1, 9)]
    assert all(set(item["result"]) == {"summary", "workflow"} for item in histories[-1])
    assert histories[-1][0]["result"]["summary"] == "answer for question 1"
    snapshot = completed["result"]
    snapshot["message"] = "mutated client copy"
    assert manager.get(completed["job_id"], conversation_id)["result"]["message"] == "answer for question 9"
    manager.shutdown()

    restarted = JobManager(path)
    persisted = restarted.get(completed["job_id"], conversation_id)
    assert persisted["status"] == "succeeded"
    assert persisted["result"]["artifacts"][0]["large"] == "x" * 20_000
    restarted.shutdown()


def test_owner_capability_unknown_conversation_and_one_active_job(tmp_path):
    manager = JobManager(tmp_path / "jobs.sqlite3", workers=2, max_pending=3)
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return {"summary": "done", "workflow": {}}

    first = manager.submit("first", None, None, blocked)
    assert entered.wait(2)
    with pytest.raises(JobError) as wrong_owner:
        manager.get(first["job_id"], "0" * 32)
    assert wrong_owner.value.status_code == 404
    with pytest.raises(JobError) as unknown:
        manager.submit("unknown", None, "f" * 32, blocked)
    assert unknown.value.status_code == 404
    with pytest.raises(JobError) as active:
        manager.submit("second", None, first["conversation_id"], blocked)
    assert active.value.status_code == 409
    release.set()
    wait_for(manager, first)
    follow_up = manager.submit("second", None, first["conversation_id"], lambda *args: {"summary": "ok"})
    assert wait_for(manager, follow_up)["status"] == "succeeded"
    manager.shutdown()


def test_admission_is_bounded_before_job_insert(tmp_path):
    manager = JobManager(tmp_path / "jobs.sqlite3", workers=1, max_pending=1)
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return {"summary": "done"}

    first = manager.submit("first", None, None, blocked)
    assert entered.wait(2)
    with pytest.raises(JobError) as overloaded:
        manager.submit("second", None, None, blocked)
    assert overloaded.value.status_code == 429
    release.set()
    wait_for(manager, first)
    manager.shutdown()


def test_startup_marks_queued_and_running_jobs_interrupted(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    original = JobManager(path, workers=1, max_pending=2)
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return {"summary": "must not overwrite restart state"}

    running = original.submit("running", None, None, blocked)
    assert entered.wait(2)
    queued = original.submit("queued", None, None, lambda *args: {"summary": "queued"})

    restarted = JobManager(path)
    for submitted in (running, queued):
        recovered = restarted.get(submitted["job_id"], submitted["conversation_id"])
        assert recovered["status"] == "failed"
        assert recovered["message"] == "Execution was interrupted before completion."
    release.set()
    original.shutdown()
    assert restarted.get(running["job_id"], running["conversation_id"])["status"] == "failed"
    restarted.shutdown()


def test_progress_is_visible_and_failures_are_normalized(tmp_path):
    manager = JobManager(tmp_path / "jobs.sqlite3", workers=1, max_pending=2)
    progressed = threading.Event()
    release = threading.Event()

    def running(message, dataset_id, history, progress):
        progress("join", "Joining mapped evidence")
        progressed.set()
        assert release.wait(3)
        return {"summary": "done"}

    submitted = manager.submit("show progress", None, None, running)
    assert progressed.wait(2)
    current = manager.get(submitted["job_id"], submitted["conversation_id"])
    assert current["status"] == "running"
    assert current["stage"] == {"step_id": "join", "label": "Joining mapped evidence"}
    release.set()
    wait_for(manager, submitted)

    useful = manager.submit(
        "bad workflow",
        None,
        submitted["conversation_id"],
        lambda *args: (_ for _ in ()).throw(WorkflowError("Requested evidence is unavailable.")),
    )
    failed = wait_for(manager, useful, "failed")
    assert failed["message"] == "Requested evidence is unavailable."
    hidden = manager.submit(
        "internal failure",
        None,
        submitted["conversation_id"],
        lambda *args: (_ for _ in ()).throw(RuntimeError("secret=/private/key")),
    )
    failed = wait_for(manager, hidden, "failed")
    assert failed["message"] == "Execution failed; no geographic result was committed."
    manager.shutdown()
