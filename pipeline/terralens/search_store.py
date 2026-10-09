"""Durable search work, complete results and immutable GeoJSON revisions.

Result pagination is a delivery operation. It never limits records committed by
workers or features included in an export. Geometric evidence is supplied by the
spatial executor in RFC 7946 coordinates; this module does not infer geography.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


RESULT_GROUPS = frozenset({
    "matches", "needs_review", "visual_candidates", "related_results",
    "excluded", "coverage_gaps", "evidence",
})
DEFAULT_RESULT_GROUPS = tuple(sorted(RESULT_GROUPS - {"evidence", "excluded"}))
_MISSING_SORT_KEY = "[3,null]"


class SearchStoreError(ValueError):
    """An expected persistence or lease failure with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SearchStoreError("Search evidence must contain finite JSON values.", 422) from exc


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SearchStoreError(f"{name} must be positive and finite.", 422)
    return float(value)


def _geometry(value: object) -> None:
    """Validate the coordinate contract before labelling exports RFC 7946."""
    if value is None:
        return  # RFC 7946 permits a Feature with an unknown/null geometry.
    if not isinstance(value, dict):
        raise SearchStoreError("Result geometry must be a GeoJSON geometry object or null.", 422)
    kind = value.get("type")
    if kind == "GeometryCollection":
        children = value.get("geometries")
        if not isinstance(children, list):
            raise SearchStoreError("A geometry collection must contain a geometries list.", 422)
        for child in children:
            if child is None:
                raise SearchStoreError("Geometry collection members must be geometry objects.", 422)
            _geometry(child)
        return
    depths = {"Point": 0, "MultiPoint": 1, "LineString": 1, "MultiLineString": 2, "Polygon": 2, "MultiPolygon": 3}
    if not isinstance(kind, str) or kind not in depths:
        raise SearchStoreError("The GeoJSON geometry type is invalid.", 422)

    def coordinates(values: object, depth: int) -> None:
        if not isinstance(values, (list, tuple)):
            raise SearchStoreError("GeoJSON coordinates must be arrays.", 422)
        if depth:
            for child in values:
                coordinates(child, depth - 1)
            return
        if (len(values) < 2 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                                    not math.isfinite(item) for item in values) or
                not -180 <= values[0] <= 180 or not -90 <= values[1] <= 90):
            raise SearchStoreError("Result positions must contain finite WGS84 longitude/latitude coordinates.", 422)

    coordinates(value.get("coordinates"), depths[kind])


def _sort_value(record_json: str, property_name: str) -> str:
    """An exact property key, including punctuation, never becomes SQL text."""
    value = json.loads(record_json)["feature"].get("properties", {}).get(property_name)
    if isinstance(value, bool):
        key = [2, value]
    elif isinstance(value, (int, float)):
        key = [0, value]
    elif isinstance(value, str):
        key = [1, value]
    else:
        key = [3, None]
    return _json(key)


def _compare_sort_values(left: str, right: str) -> int:
    # Python retains arbitrary JSON integer precision. Encoding a number as SQL
    # TEXT and sorting it lexically, or coercing it to REAL, loses that contract.
    left_value, right_value = json.loads(left), json.loads(right)
    return (left_value > right_value) - (left_value < right_value)


class SearchStore:
    """SQLite store safe for multiple threads and independently started workers.

    A claim token authorizes exactly one work attempt. Reopening the database
    preserves completed units and checkpoints. Expired claims are reclaimed on
    demand; opening another store never invalidates another live worker's lease.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection(transaction=False) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS search_runs (
                    run_id TEXT PRIMARY KEY,
                    parent_run_id TEXT REFERENCES search_runs(run_id),
                    plan_json TEXT NOT NULL,
                    versions_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    pause_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS search_units (
                    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
                    unit_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    coverage_json TEXT NOT NULL DEFAULT '{}',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_token TEXT,
                    lease_until REAL,
                    completed_token TEXT,
                    commit_hash TEXT,
                    committed_revision INTEGER,
                    error TEXT,
                    PRIMARY KEY(run_id, unit_id),
                    UNIQUE(run_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS search_units_claim
                    ON search_units(run_id, status, lease_until, ordinal);
                CREATE TABLE IF NOT EXISTS search_records (
                    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
                    result_id TEXT NOT NULL,
                    valid_from INTEGER NOT NULL,
                    valid_to INTEGER,
                    result_group TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    PRIMARY KEY(run_id, result_id, valid_from)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS search_records_current
                    ON search_records(run_id, result_id) WHERE valid_to IS NULL;
                CREATE INDEX IF NOT EXISTS search_records_page
                    ON search_records(run_id, result_group, result_id, valid_from, valid_to);
                CREATE TABLE IF NOT EXISTS search_revisions (
                    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
                    revision INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(run_id, revision)
                );
            """)
        self.recover_expired_units()

    @contextmanager
    def _connection(self, *, write: bool = False, transaction: bool = True) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.create_function("search_sort_value", 2, _sort_value, deterministic=True)
            connection.create_collation("search_sort_values", _compare_sort_values)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA synchronous = FULL")
            if transaction:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _run_row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM search_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise SearchStoreError("Search run not found.", 404)
        return row

    @staticmethod
    def _unit_row(connection: sqlite3.Connection, run_id: str, unit_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM search_units WHERE run_id = ? AND unit_id = ?", (run_id, unit_id),
        ).fetchone()
        if row is None:
            raise SearchStoreError("Search work unit not found.", 404)
        return row

    def _summary(self, connection: sqlite3.Connection, run_id: str) -> dict:
        run = self._run_row(connection, run_id)
        unit_counts = dict.fromkeys(("pending", "running", "completed", "failed", "cancelled"), 0)
        for row in connection.execute(
            "SELECT status, COUNT(*) AS n FROM search_units WHERE run_id = ? GROUP BY status", (run_id,),
        ):
            unit_counts[row["status"]] = row["n"]
        counts = {group: 0 for group in sorted(RESULT_GROUPS)}
        for row in connection.execute(
            "SELECT result_group, COUNT(*) AS n FROM search_records WHERE run_id = ? AND valid_to IS NULL GROUP BY result_group",
            (run_id,),
        ):
            counts[row["result_group"]] = row["n"]
        units = connection.execute(
            "SELECT unit_id, payload_json, status, coverage_json, error, attempts FROM search_units WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        total_units = sum(unit_counts.values())
        execution_complete = run["status"] == "completed"
        coverage_details = [json.loads(row["coverage_json"]) for row in units]
        explicit_gaps = bool(counts["coverage_gaps"]) or any(
            details.get("status") in {"missing_evidence", "unsuitable", "unsuitable_date", "partial", "failed"}
            or details.get("complete") is False for details in coverage_details
        )
        declarations = [details.get("complete") for details in coverage_details]
        data_coverage_complete = False if explicit_gaps else (
            True if execution_complete and declarations and all(value is True for value in declarations) else None
        )
        complete = execution_complete and not explicit_gaps
        return {
            "run_id": run_id, "parent_run_id": run["parent_run_id"],
            "plan": json.loads(run["plan_json"]), "dataset_versions": json.loads(run["versions_json"]),
            "status": run["status"], "revision": run["revision"],
            "created_at": run["created_at"], "updated_at": run["updated_at"],
            "pause_reason": run["pause_reason"], "counts": counts,
            "total_results": sum(counts[group] for group in DEFAULT_RESULT_GROUPS),
            "total_records": sum(counts.values()), "unit_counts": unit_counts,
            "total_units": total_units, "complete": complete,
            "execution_complete": execution_complete, "data_coverage_complete": data_coverage_complete,
            "has_coverage_gaps": explicit_gaps, "partial": not complete,
            "coverage": {
                "total_units": total_units, "completed_units": unit_counts["completed"],
                "pending_units": unit_counts["pending"], "running_units": unit_counts["running"],
                "failed_units": unit_counts["failed"], "cancelled_units": unit_counts["cancelled"],
                "units": [{"unit_id": row["unit_id"], "status": row["status"],
                           "scope": json.loads(row["payload_json"]).get("scope"),
                           "coverage": json.loads(row["coverage_json"]),
                           "attempts": row["attempts"], "error": row["error"]} for row in units],
            },
        }

    def _revision(self, connection: sqlite3.Connection, run_id: str, *, initial: bool = False) -> int:
        if not initial:
            connection.execute(
                "UPDATE search_runs SET revision = revision + 1, updated_at = ? WHERE run_id = ?", (time.time(), run_id),
            )
        metadata = self._summary(connection, run_id)
        connection.execute(
            "INSERT INTO search_revisions(run_id, revision, metadata_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, metadata["revision"], _json(metadata), time.time()),
        )
        return metadata["revision"]

    def create_run(self, plan: dict, dataset_versions: dict, units: list[dict], parent_run_id: str | None = None) -> dict:
        if not isinstance(plan, dict) or not isinstance(dataset_versions, dict) or not isinstance(units, list):
            raise SearchStoreError("Plan, dataset versions and work units have invalid types.", 422)
        plan_json, versions_json = _json(plan), _json(dataset_versions)
        prepared = []
        seen = set()
        for ordinal, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise SearchStoreError("Every search work unit must be an object.", 422)
            unit_id = unit.get("unit_id", unit.get("id"))
            if unit_id is None:
                unit_id = secrets.token_hex(16)
            if not isinstance(unit_id, str) or not unit_id or unit_id in seen:
                raise SearchStoreError("Work unit IDs must be distinct non-empty strings.", 422)
            seen.add(unit_id)
            prepared.append((unit_id, ordinal, _json(unit)))
        dependencies = {}
        for unit_id, _, payload_json in prepared:
            depends_on = json.loads(payload_json).get("depends_on", [])
            if (not isinstance(depends_on, list) or any(not isinstance(item, str) or item not in seen for item in depends_on)
                    or len(set(depends_on)) != len(depends_on)):
                raise SearchStoreError("Work dependencies must be distinct existing unit IDs.", 422)
            dependencies[unit_id] = depends_on
        # Iterative topological validation avoids imposing Python's recursion
        # depth as a hidden limit on long processing graphs.
        remaining = {unit_id: len(parents) for unit_id, parents in dependencies.items()}
        children = {unit_id: [] for unit_id in dependencies}
        for unit_id, parents in dependencies.items():
            for dependency in parents:
                children[dependency].append(unit_id)
        ready = [unit_id for unit_id, count in remaining.items() if count == 0]
        visited = 0
        while ready:
            unit_id = ready.pop()
            visited += 1
            for child in children[unit_id]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
        if visited != len(dependencies):
            raise SearchStoreError("Work dependencies must form an acyclic graph.", 422)
        run_id, now = secrets.token_hex(16), time.time()
        with self._connection(write=True) as connection:
            if parent_run_id is not None:
                self._run_row(connection, parent_run_id)
            connection.execute(
                "INSERT INTO search_runs(run_id, parent_run_id, plan_json, versions_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, parent_run_id, plan_json, versions_json, "queued" if units else "completed", now, now),
            )
            connection.executemany(
                "INSERT INTO search_units(run_id, unit_id, ordinal, payload_json) VALUES (?, ?, ?, ?)",
                [(run_id, *unit) for unit in prepared],
            )
            self._revision(connection, run_id, initial=True)
            return self._summary(connection, run_id)

    def get_run(self, run_id: str) -> dict:
        with self._connection() as connection:
            return self._summary(connection, run_id)

    def list_runs(self, statuses: list[str] | tuple[str, ...] | set[str] | None = None) -> list[dict]:
        if statuses is not None and (not isinstance(statuses, (list, tuple, set)) or any(
                not isinstance(status, str) or status not in {"queued", "running", "paused", "completed", "needs_attention", "cancelled"} for status in statuses)):
            raise SearchStoreError("Selected run statuses are invalid.", 422)
        with self._connection() as connection:
            if statuses is None:
                rows = connection.execute("SELECT run_id FROM search_runs ORDER BY created_at, run_id").fetchall()
            else:
                placeholders = ",".join("?" for _ in statuses)
                rows = connection.execute(
                    f"SELECT run_id FROM search_runs WHERE status IN ({placeholders}) ORDER BY created_at, run_id", list(statuses),
                ).fetchall()
            return [self._summary(connection, row["run_id"]) for row in rows]

    def recover_expired_units(self) -> list[str]:
        """Recover abandoned work on startup, preserving evidence/checkpoints.

        A run with no other live workers pauses for explicit continuation. An
        independently opened store never invalidates unexpired claims.
        """
        with self._connection(write=True) as connection:
            now = time.time()
            rows = connection.execute(
                "SELECT DISTINCT run_id FROM search_units WHERE status = 'running' AND lease_until <= ?", (now,),
            ).fetchall()
            recovered = []
            for row in rows:
                run_id = row["run_id"]
                connection.execute(
                    """UPDATE search_units SET status = 'pending', lease_token = NULL, lease_until = NULL
                       WHERE run_id = ? AND status = 'running' AND lease_until <= ?""", (run_id, now),
                )
                active = connection.execute(
                    "SELECT 1 FROM search_units WHERE run_id = ? AND status = 'running' LIMIT 1", (run_id,),
                ).fetchone()
                if active is None:
                    connection.execute(
                        "UPDATE search_runs SET status = 'paused', pause_reason = 'interrupted_worker' WHERE run_id = ? AND status IN ('queued', 'running')",
                        (run_id,),
                    )
                self._revision(connection, run_id)
                recovered.append(run_id)
            return recovered

    @staticmethod
    def _unit_dict(row: sqlite3.Row) -> dict:
        return {
            "run_id": row["run_id"], "id": row["unit_id"], "unit_id": row["unit_id"],
            "payload": json.loads(row["payload_json"]), "ordinal": row["ordinal"],
            "status": row["status"], "checkpoint": json.loads(row["checkpoint_json"]),
            "coverage": json.loads(row["coverage_json"]), "attempts": row["attempts"],
            "lease_token": row["lease_token"], "lease_expires_at": row["lease_until"],
            "error": row["error"], "committed_revision": row["committed_revision"],
        }

    def units(self, run_id: str) -> list[dict]:
        with self._connection() as connection:
            self._run_row(connection, run_id)
            return [self._unit_dict(row) for row in connection.execute(
                "SELECT * FROM search_units WHERE run_id = ? ORDER BY ordinal", (run_id,),
            )]

    def claim_unit(self, run_id: str, lease_seconds: float = 300) -> dict | None:
        lease_seconds = _positive(lease_seconds, "Lease duration")
        with self._connection(write=True) as connection:
            run = self._run_row(connection, run_id)
            if run["status"] not in {"queued", "running"}:
                return None
            now = time.time()
            candidates = connection.execute(
                """SELECT * FROM search_units WHERE run_id = ?
                   AND (status = 'pending' OR (status = 'running' AND lease_until <= ?))
                   ORDER BY ordinal""", (run_id, now),
            ).fetchall()
            states = {item["unit_id"]: item["status"] for item in connection.execute(
                "SELECT unit_id, status FROM search_units WHERE run_id = ?", (run_id,),
            )}
            row = next((candidate for candidate in candidates if all(
                states[dependency] == "completed" for dependency in json.loads(candidate["payload_json"]).get("depends_on", [])
            )), None)
            if row is None:
                self._finish_state(connection, run_id)
                if self._run_row(connection, run_id)["status"] != run["status"]:
                    self._revision(connection, run_id)
                return None
            token = secrets.token_hex(24)
            connection.execute(
                """UPDATE search_units SET status = 'running', attempts = attempts + 1,
                   lease_token = ?, lease_until = ?, error = NULL WHERE run_id = ? AND unit_id = ?""",
                (token, now + lease_seconds, run_id, row["unit_id"]),
            )
            connection.execute("UPDATE search_runs SET status = 'running' WHERE run_id = ?", (run_id,))
            self._revision(connection, run_id)
            return self._unit_dict(self._unit_row(connection, run_id, row["unit_id"]))

    def _leased(self, connection: sqlite3.Connection, run_id: str, unit_id: str, token: str) -> sqlite3.Row:
        run = self._run_row(connection, run_id)
        row = self._unit_row(connection, run_id, unit_id)
        if (run["status"] != "running" or row["status"] != "running" or
                not token or row["lease_token"] != token or row["lease_until"] <= time.time()):
            raise SearchStoreError("The work lease expired or was revoked; this attempt cannot change the run.", 409)
        return row

    def checkpoint_unit(self, run_id: str, unit_id: str, lease_token: str, checkpoint: dict, lease_seconds: float = 300) -> dict:
        if not isinstance(checkpoint, dict):
            raise SearchStoreError("A checkpoint must be an object.", 422)
        encoded, seconds = _json(checkpoint), _positive(lease_seconds, "Lease duration")
        with self._connection(write=True) as connection:
            self._leased(connection, run_id, unit_id, lease_token)
            connection.execute(
                "UPDATE search_units SET checkpoint_json = ?, lease_until = ? WHERE run_id = ? AND unit_id = ?",
                (encoded, time.time() + seconds, run_id, unit_id),
            )
            return self._unit_dict(self._unit_row(connection, run_id, unit_id))

    def renew_unit(self, run_id: str, unit_id: str, lease_token: str, lease_seconds: float = 300) -> None:
        seconds = _positive(lease_seconds, "Lease duration")
        with self._connection(write=True) as connection:
            self._leased(connection, run_id, unit_id, lease_token)
            connection.execute(
                "UPDATE search_units SET lease_until = ? WHERE run_id = ? AND unit_id = ?",
                (time.time() + seconds, run_id, unit_id),
            )

    @staticmethod
    def _records(records: list[dict]) -> list[tuple[str, str, str]]:
        if not isinstance(records, list):
            raise SearchStoreError("Unit results must be a list of records.", 422)
        prepared, seen = [], set()
        for record in records:
            if not isinstance(record, dict):
                raise SearchStoreError("Every result must be an object.", 422)
            result_id, group, feature = record.get("id"), record.get("group"), record.get("feature")
            if not isinstance(result_id, str) or not result_id or result_id in seen:
                raise SearchStoreError("Result IDs must be distinct non-empty strings within a commit.", 422)
            if not isinstance(group, str) or group not in RESULT_GROUPS:
                raise SearchStoreError("The result evidence group is invalid.", 422)
            if (not isinstance(feature, dict) or feature.get("type") != "Feature" or
                    "geometry" not in feature or not isinstance(feature.get("properties", {}), dict)):
                raise SearchStoreError("Each result must contain a GeoJSON Feature and object properties.", 422)
            _geometry(feature["geometry"])
            seen.add(result_id)
            prepared.append((result_id, group, _json(record)))
        return prepared

    def _finish_state(self, connection: sqlite3.Connection, run_id: str) -> None:
        counts = {row["status"]: row["n"] for row in connection.execute(
            "SELECT status, COUNT(*) AS n FROM search_units WHERE run_id = ? GROUP BY status", (run_id,),
        )}
        if not counts.get("pending") and not counts.get("running"):
            status = "needs_attention" if counts.get("failed") else "completed"
            connection.execute("UPDATE search_runs SET status = ? WHERE run_id = ?", (status, run_id))
        elif counts.get("pending") and not counts.get("running") and counts.get("failed"):
            units = connection.execute("SELECT unit_id, status, payload_json FROM search_units WHERE run_id = ?", (run_id,)).fetchall()
            states = {unit["unit_id"]: unit["status"] for unit in units}
            ready = any(unit["status"] == "pending" and all(
                states[dependency] == "completed" for dependency in json.loads(unit["payload_json"]).get("depends_on", [])
            ) for unit in units)
            if not ready:
                connection.execute("UPDATE search_runs SET status = 'needs_attention' WHERE run_id = ?", (run_id,))

    def commit_unit(self, run_id: str, unit_id: str, lease_token: str, records: list[dict], coverage: dict) -> dict:
        prepared = self._records(records)
        if not isinstance(coverage, dict):
            raise SearchStoreError("Coverage must be an object.", 422)
        coverage_json = _json(coverage)
        commit_hash = hashlib.sha256(_json({"records": records, "coverage": coverage}).encode()).hexdigest()
        with self._connection(write=True) as connection:
            run, unit = self._run_row(connection, run_id), self._unit_row(connection, run_id, unit_id)
            if unit["status"] == "completed" and lease_token and unit["completed_token"] == lease_token:
                if unit["commit_hash"] != commit_hash:
                    raise SearchStoreError("This attempt was already committed with different evidence.", 409)
                response = self._summary(connection, run_id)
                response["committed_revision"] = unit["committed_revision"]
                return response
            self._leased(connection, run_id, unit_id, lease_token)
            revision = run["revision"] + 1
            for result_id, group, encoded in prepared:
                previous = connection.execute(
                    "SELECT record_json FROM search_records WHERE run_id = ? AND result_id = ? AND valid_to IS NULL",
                    (run_id, result_id),
                ).fetchone()
                if previous is not None and previous["record_json"] == encoded:
                    continue
                connection.execute(
                    "UPDATE search_records SET valid_to = ? WHERE run_id = ? AND result_id = ? AND valid_to IS NULL",
                    (revision, run_id, result_id),
                )
                connection.execute(
                    "INSERT INTO search_records(run_id, result_id, valid_from, result_group, record_json, unit_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, result_id, revision, group, encoded, unit_id),
                )
            connection.execute(
                """UPDATE search_units SET status = 'completed', coverage_json = ?, completed_token = ?,
                   commit_hash = ?, committed_revision = ?, lease_token = NULL, lease_until = NULL, error = NULL
                   WHERE run_id = ? AND unit_id = ?""",
                (coverage_json, lease_token, commit_hash, revision, run_id, unit_id),
            )
            self._finish_state(connection, run_id)
            self._revision(connection, run_id)
            response = self._summary(connection, run_id)
            response["committed_revision"] = revision
            return response

    def fail_unit(self, run_id: str, unit_id: str, lease_token: str, error: str, retryable: bool = False) -> dict:
        if not isinstance(error, str) or not error.strip():
            raise SearchStoreError("A work failure must have an explanation.", 422)
        with self._connection(write=True) as connection:
            self._leased(connection, run_id, unit_id, lease_token)
            connection.execute(
                """UPDATE search_units SET status = ?, error = ?, lease_token = NULL, lease_until = NULL
                   WHERE run_id = ? AND unit_id = ?""",
                ("pending" if retryable else "failed", error, run_id, unit_id),
            )
            self._finish_state(connection, run_id)
            self._revision(connection, run_id)
            return self._summary(connection, run_id)

    def pause(self, run_id: str, reason: str | None = None) -> dict:
        with self._connection(write=True) as connection:
            run = self._run_row(connection, run_id)
            if run["status"] in {"completed", "cancelled"}:
                return self._summary(connection, run_id)
            connection.execute(
                "UPDATE search_runs SET status = 'paused', pause_reason = ? WHERE run_id = ?", (reason, run_id),
            )
            connection.execute(
                "UPDATE search_units SET status = 'pending', lease_token = NULL, lease_until = NULL WHERE run_id = ? AND status = 'running'",
                (run_id,),
            )
            self._revision(connection, run_id)
            return self._summary(connection, run_id)

    def resume(self, run_id: str) -> dict:
        with self._connection(write=True) as connection:
            run = self._run_row(connection, run_id)
            if run["status"] == "cancelled":
                raise SearchStoreError("Cancelled runs are immutable; create a new run to search again.", 409)
            if run["status"] in {"completed", "running", "queued"}:
                return self._summary(connection, run_id)
            connection.execute("UPDATE search_runs SET status = 'queued', pause_reason = NULL WHERE run_id = ?", (run_id,))
            connection.execute("UPDATE search_units SET status = 'pending' WHERE run_id = ? AND status = 'failed'", (run_id,))
            self._revision(connection, run_id)
            return self._summary(connection, run_id)

    def cancel(self, run_id: str) -> dict:
        with self._connection(write=True) as connection:
            run = self._run_row(connection, run_id)
            if run["status"] in {"completed", "cancelled"}:
                return self._summary(connection, run_id)
            connection.execute("UPDATE search_runs SET status = 'cancelled', pause_reason = NULL WHERE run_id = ?", (run_id,))
            connection.execute(
                "UPDATE search_units SET status = 'cancelled', lease_token = NULL, lease_until = NULL WHERE run_id = ? AND status != 'completed'",
                (run_id,),
            )
            self._revision(connection, run_id)
            return self._summary(connection, run_id)

    def snapshot(self, run_id: str) -> int:
        with self._connection(write=True) as connection:
            self._run_row(connection, run_id)
            return self._revision(connection, run_id)

    @staticmethod
    def _groups(groups: list[str] | tuple[str, ...] | set[str] | str | None) -> tuple[str, ...]:
        if groups is None:
            return DEFAULT_RESULT_GROUPS
        if isinstance(groups, str):
            groups = [groups]
        if not isinstance(groups, (list, tuple, set, frozenset)) or any(not isinstance(group, str) or group not in RESULT_GROUPS for group in groups):
            raise SearchStoreError("Selected result groups are invalid.", 422)
        return tuple(sorted(set(groups)))

    def _snapshot_metadata(self, connection: sqlite3.Connection, run_id: str, revision: int | None) -> dict:
        run = self._run_row(connection, run_id)
        revision = run["revision"] if revision is None else revision
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise SearchStoreError("Result revision must be a non-negative integer.", 422)
        row = connection.execute(
            "SELECT metadata_json FROM search_revisions WHERE run_id = ? AND revision = ?", (run_id, revision),
        ).fetchone()
        if row is None:
            raise SearchStoreError("Result revision not found.", 404)
        return json.loads(row["metadata_json"])

    @staticmethod
    def _where(run_id: str, revision: int, groups: tuple[str, ...]) -> tuple[str, list]:
        placeholders = ",".join("?" for _ in groups)
        where = f"run_id = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?) AND result_group IN ({placeholders})"
        return where, [run_id, revision, revision, *groups]

    @staticmethod
    def _sort_policy(metadata: dict) -> dict:
        output = metadata["plan"].get("output") or {}
        if not isinstance(output, dict):
            raise SearchStoreError("The saved output specification must be an object.", 422)
        requested = output.get("sort_by")
        direction, property_name = "asc", None
        if requested is not None:
            if not isinstance(requested, str) or not requested:
                raise SearchStoreError("Result sorting requires a non-empty feature property key.", 422)
            direction = "desc" if requested.startswith("-") else "asc"
            property_name = requested[1:] if requested.startswith(("-", "+")) else requested
            if not property_name:
                raise SearchStoreError("Result sorting requires a feature property key after its direction.", 422)
        return {
            "property": property_name, "direction": direction, "missing": "last",
            "tie_breaker": "result_id_ascending", "value_order": "number_string_boolean",
            "non_scalar_values": "missing", "version": 1,
        }

    def _ordered_rows(self, connection: sqlite3.Connection, run_id: str, metadata: dict,
                      groups: tuple[str, ...], sort_policy: dict, after: dict | None = None,
                      limit: int | None = None) -> sqlite3.Cursor:
        where, parameters = self._where(run_id, metadata["revision"], groups)
        if sort_policy["property"] is None:
            if after is not None:
                where += " AND result_id > ?"
                parameters.append(after["id"])
            query = f"SELECT result_id, record_json, NULL AS sort_key FROM search_records WHERE {where} ORDER BY result_id"
        else:
            # Every row in the selected saved revision participates in ordering.
            # The SQL UDF extracts a literal dictionary key; user-provided field
            # names cannot alter the query or restrict result membership.
            query = ("SELECT result_id, record_json, sort_key FROM (SELECT result_id, record_json, "
                     f"search_sort_value(record_json, ?) AS sort_key FROM search_records WHERE {where})")
            parameters = [sort_policy["property"], *parameters]
            if after is not None:
                comparison = "<" if sort_policy["direction"] == "desc" else ">"
                key = _json(after["sort_key"])
                missing = int(key == _MISSING_SORT_KEY)
                query += (" WHERE ((sort_key = ?) > ? OR ((sort_key = ?) = ? AND ("
                          f"sort_key COLLATE search_sort_values {comparison} ? COLLATE search_sort_values OR "
                          "(sort_key COLLATE search_sort_values = ? COLLATE search_sort_values AND result_id > ?))))")
                parameters.extend([_MISSING_SORT_KEY, missing, _MISSING_SORT_KEY, missing, key, key, after["id"]])
            query += f" ORDER BY (sort_key = ?) ASC, sort_key COLLATE search_sort_values {sort_policy['direction'].upper()}, result_id ASC"
            parameters.append(_MISSING_SORT_KEY)
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        return connection.execute(query, parameters)

    def results(self, run_id: str, revision: int | None = None, groups=None, cursor: str | None = None, page_size: int = 50) -> dict:
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
            raise SearchStoreError("Page size must be a positive integer.", 422)
        chosen = self._groups(groups)
        after = None
        decoded = None
        if cursor is not None:
            if not isinstance(cursor, str):
                raise SearchStoreError("Result cursor is invalid.", 422)
            try:
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            except (ValueError, TypeError, UnicodeError) as exc:
                raise SearchStoreError("Result cursor is invalid.", 422) from exc
            if (not isinstance(decoded, dict) or decoded.get("run_id") != run_id or
                    decoded.get("groups") != list(chosen) or not isinstance(decoded.get("after"), dict) or
                    not isinstance(decoded["after"].get("id"), str) or
                    isinstance(decoded.get("revision"), bool) or not isinstance(decoded.get("revision"), int)):
                raise SearchStoreError("Result cursor does not belong to this run and group selection.", 422)
            if revision is not None and revision != decoded["revision"]:
                raise SearchStoreError("Result cursor belongs to a different revision.", 422)
            revision, after = decoded["revision"], decoded["after"]
        with self._connection() as connection:
            metadata = self._snapshot_metadata(connection, run_id, revision)
            revision = metadata["revision"]
            sort_policy = self._sort_policy(metadata)
            if decoded is not None:
                if decoded.get("sort") != sort_policy:
                    raise SearchStoreError("Result cursor belongs to a different sorting policy.", 422)
                if sort_policy["property"] is not None:
                    key = after.get("sort_key")
                    if (not isinstance(key, list) or len(key) != 2 or isinstance(key[0], bool) or
                            key[0] not in (0, 1, 2, 3) or
                            (key[0] == 0 and (isinstance(key[1], bool) or not isinstance(key[1], (int, float)))) or
                            (key[0] == 1 and not isinstance(key[1], str)) or
                            (key[0] == 2 and not isinstance(key[1], bool)) or
                            (key[0] == 3 and key[1] is not None)):
                        raise SearchStoreError("Result cursor contains an invalid sorting value.", 422)
                    _json(key)
            total = sum(metadata["counts"].get(group, 0) for group in chosen)
            rows = self._ordered_rows(connection, run_id, metadata, chosen, sort_policy, after, page_size + 1).fetchall()
            more = len(rows) > page_size
            rows = rows[:page_size]
            next_cursor = None
            if more:
                next_cursor = base64.urlsafe_b64encode(_json({
                    "run_id": run_id, "revision": revision, "groups": chosen, "sort": sort_policy,
                    "after": {"id": rows[-1]["result_id"], "sort_key": json.loads(rows[-1]["sort_key"]) if rows[-1]["sort_key"] is not None else None},
                }).encode()).decode()
            return {
                "run_id": run_id, "revision": revision, "groups": list(chosen),
                "records": [json.loads(row["record_json"]) for row in rows],
                "counts": metadata["counts"], "total": total, "page_size": page_size,
                "selected_counts": {group: metadata["counts"].get(group, 0) for group in chosen},
                "sort": sort_policy,
                "next_cursor": next_cursor, "status": metadata["status"],
                "complete": metadata["complete"], "partial": metadata["partial"],
                "execution_complete": metadata["execution_complete"],
                "has_coverage_gaps": metadata["has_coverage_gaps"],
                "coverage": metadata["coverage"], "metadata": metadata,
            }

    def export(self, run_id: str, revision: int | None = None, groups=None) -> dict:
        chosen = self._groups(groups)
        with self._connection() as connection:
            metadata = self._snapshot_metadata(connection, run_id, revision)
            sort_policy = self._sort_policy(metadata)
            features = []
            for row in self._ordered_rows(connection, run_id, metadata, chosen, sort_policy):
                record = json.loads(row["record_json"])
                feature = copy.deepcopy(record["feature"])
                feature["id"] = record["id"]
                properties = feature.setdefault("properties", {})
                for key, value in record.items():
                    if key not in {"id", "group", "feature"}:
                        properties[key] = copy.deepcopy(value)
                properties.update({
                    "result_id": record["id"], "result_group": record["group"],
                    "run_id": run_id, "revision": metadata["revision"],
                    "run_status": metadata["status"], "partial": metadata["partial"],
                    "execution_complete": metadata["execution_complete"],
                    "data_coverage_complete": metadata["data_coverage_complete"],
                    "dataset_versions": metadata["dataset_versions"],
                })
                properties.setdefault("feature_role", "coverage" if record["group"] == "coverage_gaps" else "result")
                features.append(feature)
            metadata["selected_groups"] = list(chosen)
            metadata["selected_counts"] = {group: metadata["counts"].get(group, 0) for group in chosen}
            metadata["exported_features"] = len(features)
            metadata["sort"] = sort_policy
            metadata["coordinate_reference_system"] = "WGS84 longitude/latitude (RFC 7946)"
            return {"type": "FeatureCollection", "features": features, "metadata": metadata}

    def get_records(self, run_id: str, revision: int | None = None, groups=None) -> list[dict]:
        """Read all underlying records, including internal evidence by default."""
        chosen = tuple(sorted(RESULT_GROUPS)) if groups is None else self._groups(groups)
        with self._connection() as connection:
            metadata = self._snapshot_metadata(connection, run_id, revision)
            where, parameters = self._where(run_id, metadata["revision"], chosen)
            return [json.loads(row["record_json"]) for row in connection.execute(
                f"SELECT record_json FROM search_records WHERE {where} ORDER BY result_id", parameters,
            )]

    all_records = get_records

    def map_data(self, run_id: str, revision: int | None = None, groups=None) -> dict:
        """Return the complete saved layer independent of viewport or list page."""
        return self.export(run_id, revision, groups)
