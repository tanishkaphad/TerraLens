import concurrent.futures
import base64
import json
import time

import pytest

from terralens.search_store import SearchStore, SearchStoreError


def record(identifier, group="matches", **properties):
    return {
        "id": str(identifier), "group": group,
        "feature": {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "properties": properties},
        "decisions": [{"clause_id": "unprepared_attribute", "state": "pass" if group == "matches" else "unknown"}],
    }


def make_run(store, units):
    return store.create_run({"query": "unprepared description", "scope": {"type": "loaded_data"}}, {"source": "version-a"}, units)


def test_all_results_before_pagination_full_map_and_saved_export(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "first"}, {"id": "remaining"}])
    first = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], first["id"], first["lease_token"], [record(f"a-{n:04d}") for n in range(257)], {"processed": 1})
    page = store.results(run["run_id"], page_size=31)
    revision = page["revision"]
    assert page["total"] == 257
    assert page["partial"] is True
    second = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], second["id"], second["lease_token"], [record("new-result")], {"processed": 1})
    ids = [item["id"] for item in page["records"]]
    while page["next_cursor"]:
        page = store.results(run["run_id"], cursor=page["next_cursor"], page_size=31)
        assert page["revision"] == revision
        ids.extend(item["id"] for item in page["records"])
    assert len(ids) == len(set(ids)) == 257
    original_export = store.export(run["run_id"], revision)
    assert {feature["id"] for feature in original_export["features"]} == set(ids)
    assert original_export["metadata"]["partial"] is True
    assert original_export["features"][0]["properties"]["decisions"][0]["clause_id"] == "unprepared_attribute"
    assert len(store.map_data(run["run_id"])["features"]) == 258
    assert store.export(run["run_id"])["metadata"]["complete"] is True
    reopened = SearchStore(store.path)
    assert reopened.export(run["run_id"], revision) == original_export
    assert len(reopened.get_records(run["run_id"])) == 258


def test_reclassification_deduplication_and_retry_commit_are_revision_safe(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "a"}, {"id": "b"}])
    first = store.claim_unit(run["run_id"])
    candidate = record("stable", "needs_review")
    committed = store.commit_unit(run["run_id"], first["id"], first["lease_token"], [candidate], {"processed": True})
    first_revision = committed["revision"]
    repeated = store.commit_unit(run["run_id"], first["id"], first["lease_token"], [candidate], {"processed": True})
    assert repeated["revision"] == first_revision
    with pytest.raises(SearchStoreError, match="different evidence"):
        store.commit_unit(run["run_id"], first["id"], first["lease_token"], [record("stable")], {"processed": True})
    second = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], second["id"], second["lease_token"], [record("stable")], {})
    assert len(store.get_records(run["run_id"])) == 1
    assert store.results(run["run_id"], first_revision, groups=["needs_review"])["total"] == 1
    assert store.results(run["run_id"], groups=["needs_review"])["total"] == 0
    assert store.export(run["run_id"], first_revision)["features"][0]["properties"]["result_group"] == "needs_review"
    assert store.export(run["run_id"])["features"][0]["properties"]["result_group"] == "matches"


def test_cursor_is_bound_to_run_revision_and_groups(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "unit"}])
    unit = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], [record("a"), record("b")], {})
    page = store.results(run["run_id"], groups=["matches"], page_size=1)
    other = make_run(store, [])
    for kwargs in [
        {"run_id": other["run_id"], "groups": ["matches"]},
        {"run_id": run["run_id"], "groups": ["needs_review"]},
        {"run_id": run["run_id"], "groups": ["matches"], "revision": 0},
    ]:
        with pytest.raises(SearchStoreError):
            store.results(cursor=page["next_cursor"], **kwargs)
    with pytest.raises(SearchStoreError):
        store.results(run["run_id"], cursor="not-a-cursor")
    assert store.results(run["run_id"], groups=[], page_size=1)["total"] == 0


def test_pause_resume_checkpoint_reclaim_and_cancel_survive_restart(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "first"}, {"id": "second"}])
    first = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], first["id"], first["lease_token"], [record("preserved")], {})
    second = store.claim_unit(run["run_id"])
    store.checkpoint_unit(run["run_id"], second["id"], second["lease_token"], {"window": 17})
    paused = store.pause(run["run_id"], "configured_budget")
    assert paused["status"] == "paused"
    assert store.claim_unit(run["run_id"]) is None
    with pytest.raises(SearchStoreError):
        store.commit_unit(run["run_id"], second["id"], second["lease_token"], [record("stale")], {})
    reopened = SearchStore(store.path)
    assert reopened.get_run(run["run_id"])["pause_reason"] == "configured_budget"
    reopened.resume(run["run_id"])
    reclaimed = reopened.claim_unit(run["run_id"])
    assert reclaimed["id"] == "second"
    assert reclaimed["checkpoint"] == {"window": 17}
    assert reclaimed["lease_token"] != second["lease_token"]
    cancelled = reopened.cancel(run["run_id"])
    assert cancelled["status"] == "cancelled"
    assert len(reopened.export(run["run_id"])["features"]) == 1
    with pytest.raises(SearchStoreError):
        reopened.resume(run["run_id"])
    assert reopened.export(run["run_id"], paused["revision"])["metadata"]["status"] == "paused"


def test_expired_claim_is_recovered_without_invalidating_live_worker(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "work"}])
    first = store.claim_unit(run["run_id"], lease_seconds=0.15)
    reopened = SearchStore(store.path)
    assert reopened.claim_unit(run["run_id"]) is None
    time.sleep(0.2)
    recovered = reopened.claim_unit(run["run_id"])
    assert recovered["id"] == first["id"]
    assert recovered["attempts"] == 2
    with pytest.raises(SearchStoreError):
        store.commit_unit(run["run_id"], first["id"], first["lease_token"], [record("stale")], {})
    reopened.commit_unit(run["run_id"], recovered["id"], recovered["lease_token"], [record("current")], {})
    assert [item["id"] for item in reopened.get_records(run["run_id"])] == ["current"]


def test_concurrent_claims_commit_once_and_preserve_complete_set(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": f"unit-{n}"} for n in range(35)])

    def worker():
        claimed = []
        while unit := store.claim_unit(run["run_id"]):
            claimed.append(unit["id"])
            store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], [record(unit["id"])], {"checked": True})
        return claimed

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        claimed = [identifier for identifiers in executor.map(lambda _: worker(), range(8)) for identifier in identifiers]
    assert len(claimed) == len(set(claimed)) == 35
    assert store.get_run(run["run_id"])["complete"] is True
    assert all(unit["attempts"] == 1 for unit in store.units(run["run_id"]))
    assert len(store.export(run["run_id"])["features"]) == 35


def test_dependency_graph_failure_recovery_and_internal_evidence(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "final", "depends_on": ["prepare"]}, {"id": "prepare"}])
    unit = store.claim_unit(run["run_id"])
    assert unit["id"] == "prepare"
    store.fail_unit(run["run_id"], unit["id"], unit["lease_token"], "source failed")
    assert store.get_run(run["run_id"])["status"] == "needs_attention"
    assert store.claim_unit(run["run_id"]) is None
    store.resume(run["run_id"])
    prepare = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], prepare["id"], prepare["lease_token"], [record("raw", "evidence")], {})
    assert len(store.get_records(run["run_id"], groups=["evidence"])) == 1
    assert store.map_data(run["run_id"])["features"] == []
    final = store.claim_unit(run["run_id"])
    assert final["id"] == "final"
    store.commit_unit(run["run_id"], final["id"], final["lease_token"], [record("qualified"), record("excluded", "excluded")], {})
    assert len(store.export(run["run_id"])["features"]) == 1
    assert len(store.export(run["run_id"], groups=["evidence", "excluded"])["features"]) == 2
    assert store.list_runs(statuses=["completed"])[0]["run_id"] == run["run_id"]
    with pytest.raises(SearchStoreError, match="acyclic"):
        make_run(store, [{"id": "a", "depends_on": ["b"]}, {"id": "b", "depends_on": ["a"]}])
    with pytest.raises(SearchStoreError, match="existing"):
        make_run(store, [{"id": "a", "depends_on": ["missing"]}])


def test_commit_validation_is_atomic_and_empty_export_is_explanatory(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "unit"}])
    unit = store.claim_unit(run["run_id"])
    with pytest.raises(SearchStoreError):
        store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], [record("valid"), record("bad", value=float("nan"))], {})
    projected = record("projected")
    projected["feature"]["geometry"]["coordinates"] = [500000, 1500000]
    with pytest.raises(SearchStoreError, match="WGS84"):
        store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], [record("valid"), projected], {})
    assert store.get_records(run["run_id"]) == []
    assert store.units(run["run_id"])[0]["status"] == "running"
    exported = store.export(run["run_id"])
    assert exported["features"] == []
    assert exported["metadata"]["coverage"]["running_units"] == 1
    assert exported["metadata"]["dataset_versions"] == {"source": "version-a"}
    assert exported["metadata"]["partial"] is True


def test_startup_recovers_only_abandoned_leases_and_preserves_checkpoint(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "abandoned"}])
    abandoned = store.claim_unit(run["run_id"], lease_seconds=0.02)
    store.checkpoint_unit(run["run_id"], abandoned["id"], abandoned["lease_token"], {"part": 7}, lease_seconds=0.02)
    time.sleep(0.03)
    reopened = SearchStore(store.path)
    assert reopened.get_run(run["run_id"])["status"] == "paused"
    assert reopened.get_run(run["run_id"])["pause_reason"] == "interrupted_worker"
    assert reopened.units(run["run_id"])[0]["status"] == "pending"
    reopened.resume(run["run_id"])
    recovered = reopened.claim_unit(run["run_id"])
    assert recovered["checkpoint"] == {"part": 7}
    assert recovered["lease_token"] != abandoned["lease_token"]


def test_finished_workers_with_missing_data_export_partial_and_visible_counts(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = make_run(store, [{"id": "partial-data"}])
    unit = store.claim_unit(run["run_id"])
    saved = store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], [
        record("visible", "needs_review"), record("gap", "coverage_gaps"),
        record("raw", "evidence"), record("rejected", "excluded"),
    ], {"status": "missing_evidence"})
    assert saved["status"] == "completed"
    assert saved["execution_complete"] is True
    assert saved["complete"] is False
    assert saved["partial"] is True
    assert saved["data_coverage_complete"] is False
    assert saved["has_coverage_gaps"] is True
    assert saved["total_results"] == 2
    assert saved["total_records"] == 4
    assert store.export(run["run_id"])["metadata"]["partial"] is True


def test_dynamic_property_sort_pages_entire_saved_revision_and_export_with_missing_values(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    field = 'sensor.value "measured"'
    run = store.create_run({"output": {"sort_by": "-" + field}}, {}, [{"id": "initial"}, {"id": "later"}])
    unit = store.claim_unit(run["run_id"])
    records = [record(f"item-{n:03d}", **{field: n % 19}) for n in range(127)]
    records += [record("missing"), record("null", **{field: None}), record("non-scalar", **{field: [100]})]
    store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], records, {})
    page = store.results(run["run_id"], groups=["matches"], page_size=7)
    revision = page["revision"]
    assert page["sort"]["property"] == field
    assert page["sort"]["direction"] == "desc"
    assert page["total"] == 130
    next_unit = store.claim_unit(run["run_id"])
    store.commit_unit(run["run_id"], next_unit["id"], next_unit["lease_token"], [record("item-000", **{field: 10000})], {})
    actual = []
    while True:
        actual.extend(item["id"] for item in page["records"])
        if not page["next_cursor"]:
            break
        page = store.results(run["run_id"], groups=["matches"], cursor=page["next_cursor"], page_size=7)
        assert page["revision"] == revision
    expected = [item["id"] for item in sorted(records[:127], key=lambda item: (-item["feature"]["properties"][field], item["id"]))]
    expected.extend(["missing", "non-scalar", "null"])
    assert actual == expected
    saved = store.export(run["run_id"], revision, groups=["matches"])
    assert [feature["id"] for feature in saved["features"]] == expected
    assert saved["metadata"]["selected_counts"] == {"matches": 130}
    assert saved["metadata"]["exported_features"] == 130
    assert store.results(run["run_id"], groups=["matches"], page_size=1)["records"][0]["id"] == "item-000"


def test_numeric_sort_preserves_large_integer_precision_and_cursor_sort_binding(tmp_path):
    store = SearchStore(tmp_path / "search.sqlite3")
    run = store.create_run({"output": {"sort_by": "+measurement"}}, {}, [{"id": "unit"}])
    unit = store.claim_unit(run["run_id"])
    values = [record("z", measurement=2 ** 70), record("a", measurement=2 ** 70 + 1), record("missing")]
    store.commit_unit(run["run_id"], unit["id"], unit["lease_token"], values, {})
    page = store.results(run["run_id"], page_size=1)
    assert page["records"][0]["id"] == "z"
    decoded = json.loads(base64.urlsafe_b64decode(page["next_cursor"]))
    decoded["sort"]["direction"] = "desc"
    tampered = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode()
    with pytest.raises(SearchStoreError, match="sorting policy"):
        store.results(run["run_id"], cursor=tampered)
    next_page = store.results(run["run_id"], cursor=page["next_cursor"], page_size=1)
    assert next_page["records"][0]["id"] == "a"
    assert store.results(run["run_id"], cursor=next_page["next_cursor"])["records"][0]["id"] == "missing"
