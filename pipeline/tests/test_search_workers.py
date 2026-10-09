import threading
import time

import pytest

from terralens.search_store import SearchStore, SearchStoreError
from terralens.search_workers import SearchWorkerRunner, WorkUnitResult


def record(identifier):
    return {"id": str(identifier), "group": "matches", "feature": {
        "type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {},
    }}


def test_configurable_pool_progressively_persists_all_outputs_and_dependencies(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    ids = [f"part-{n}" for n in range(6)]
    run = store.create_run({}, {}, [{"id": identifier} for identifier in ids] + [{"id": "final", "depends_on": ids}])
    lock = threading.Lock()
    active, peak = 0, 0

    def process(unit, checkpoint):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        checkpoint({"prepared": True})
        time.sleep(0.02)
        if unit["id"] == "final":
            assert store.get_run(run["run_id"])["unit_counts"]["completed"] == 6
            assert len(store.get_records(run["run_id"])) == 246
        with lock:
            active -= 1
        return WorkUnitResult([record(f"{unit['id']}-{n}") for n in range(41)], {"complete": True})

    runner = SearchWorkerRunner(store, workers=3)
    try:
        result = runner.submit(run["run_id"], process).result(timeout=10)
        assert result["status"] == "completed"
        assert peak == 3
        assert len(store.export(run["run_id"])["features"]) == 287
        assert len(store.results(run["run_id"], page_size=15)["records"]) == 15
        assert all(unit["checkpoint"] == {"prepared": True} for unit in store.units(run["run_id"]))
    finally:
        runner.shutdown()


def test_work_budget_pauses_without_changing_plan_or_losing_results(tmp_path):
    path = tmp_path / "search.sqlite3"
    store = SearchStore(path)
    plan = {"query": "a previously unseen concept", "scope": {"type": "dataset"}}
    run = store.create_run(plan, {}, [{"id": str(n)} for n in range(5)])
    processor = lambda unit, checkpoint: {"records": [record(f"{unit['id']}-{n}") for n in range(121)], "coverage": {"processed": True}}
    runner = SearchWorkerRunner(store, workers=2, max_units_per_run=2)
    try:
        paused = runner.submit(run["run_id"], processor).result(timeout=10)
        assert paused["status"] == "paused"
        assert paused["pause_reason"] == "work_unit_budget"
        assert paused["plan"] == plan
        assert paused["unit_counts"]["completed"] == 2
        assert len(store.export(run["run_id"])["features"]) == 242
    finally:
        runner.shutdown()
    reopened = SearchStore(path)
    resumed_runner = SearchWorkerRunner(reopened, workers=3)
    try:
        complete = resumed_runner.resume(run["run_id"], processor).result(timeout=10)
        assert complete["status"] == "completed"
        assert len(reopened.export(run["run_id"])["features"]) == 605
        assert all(unit["attempts"] == 1 for unit in reopened.units(run["run_id"]))
    finally:
        resumed_runner.shutdown()


def test_failed_work_preserves_checkpoints_and_resumes_only_unfinished(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = store.create_run({}, {}, [{"id": "successful"}, {"id": "incomplete"}])
    visits = []

    def initial(unit, checkpoint):
        visits.append(unit["id"])
        checkpoint({"part": 2})
        if unit["id"] == "incomplete":
            raise RuntimeError("provider credential must not appear in API state")
        return {"records": [record(unit["id"])], "coverage": {"complete": True}}

    runner = SearchWorkerRunner(store, workers=2)
    try:
        failed = runner.submit(run["run_id"], initial).result(timeout=10)
        assert failed["status"] == "needs_attention"
        assert failed["counts"]["matches"] == 1
        assert failed["coverage"]["failed_units"] == 1
        assert "credential" not in str(failed)
        recovered = []

        def continuation(unit, checkpoint):
            recovered.append(unit["id"])
            assert unit["checkpoint"] == {"part": 2}
            return {"records": [record(unit["id"])], "coverage": {"complete": True}}

        complete = runner.resume(run["run_id"], continuation).result(timeout=10)
        assert recovered == ["incomplete"]
        assert complete["status"] == "completed"
        assert complete["counts"]["matches"] == 2
    finally:
        runner.shutdown()


def test_bounded_admission_idempotent_submission_and_lease_heartbeat(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    first = store.create_run({}, {}, [{"id": "first"}])
    other = store.create_run({}, {}, [{"id": "other"}])
    entered, release = threading.Event(), threading.Event()

    def blocked(unit, checkpoint):
        entered.set()
        assert release.wait(5)
        return {"records": [record(unit["id"])], "coverage": {}}

    runner = SearchWorkerRunner(store, workers=1, max_pending=1, lease_seconds=0.15)
    try:
        future = runner.submit(first["run_id"], blocked)
        assert entered.wait(2)
        assert runner.submit(first["run_id"], blocked) is future
        with pytest.raises(SearchStoreError) as failure:
            runner.submit(other["run_id"], blocked)
        assert failure.value.status_code == 429
        time.sleep(0.35)
        assert store.claim_unit(first["run_id"]) is None
        assert store.get_run(other["run_id"])["status"] == "queued"
        release.set()
        assert future.result(timeout=5)["status"] == "completed"
    finally:
        release.set()
        runner.shutdown()


def test_runtime_pause_prevents_stale_output_and_keeps_checkpoint(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = store.create_run({}, {}, [{"id": "slow"}])
    entered, release = threading.Event(), threading.Event()

    def blocked(unit, checkpoint):
        checkpoint({"step": "in_progress"})
        entered.set()
        assert release.wait(5)
        return {"records": [record("stale")], "coverage": {}}

    runner = SearchWorkerRunner(store, workers=1, max_runtime_seconds=0.15, poll_interval=0.01)
    try:
        future = runner.submit(run["run_id"], blocked)
        assert entered.wait(2)
        paused = future.result(timeout=3)
        assert paused["status"] == "paused"
        assert paused["pause_reason"] == "runtime_budget"
        release.set()
    finally:
        release.set()
        runner.shutdown()
    assert store.get_records(run["run_id"]) == []
    assert store.units(run["run_id"])[0]["checkpoint"] == {"step": "in_progress"}


def test_bounded_retries_do_not_duplicate_results(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = store.create_run({}, {}, [{"id": "retried"}])

    def flaky(unit, checkpoint):
        if unit["attempts"] == 1:
            checkpoint({"prepared": True})
            raise RuntimeError("temporary provider failure")
        assert unit["checkpoint"] == {"prepared": True}
        return WorkUnitResult([record("once")], {})

    runner = SearchWorkerRunner(store, max_attempts=2)
    try:
        complete = runner.submit(run["run_id"], flaky).result(timeout=5)
        assert complete["status"] == "completed"
        assert complete["counts"]["matches"] == 1
        assert store.units(run["run_id"])[0]["attempts"] == 2
    finally:
        runner.shutdown()
