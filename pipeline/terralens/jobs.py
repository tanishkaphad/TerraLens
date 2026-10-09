"""Durable, bounded background jobs for assistant workflows."""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .workflow import WorkflowError


_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_MAX_RESULT_BYTES = 32 * 1024 * 1024
_MAX_MESSAGE_BYTES = 64 * 1024
_MAX_ERROR_CHARS = 500
_MAX_JOBS = 100
_RETENTION_SECONDS = 24 * 60 * 60
_GENERIC_FAILURE = "Execution failed; no geographic result was committed."
_INTERRUPTED_FAILURE = "Execution was interrupted before completion."


class JobError(ValueError):
    """An expected job API failure with an HTTP-compatible status code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class _ResultError(Exception):
    pass


class JobManager:
    """Run bounded jobs while persisting immutable results and compact context."""

    def __init__(self, path: Path, workers: int = 2, max_pending: int = 8):
        if (
            not isinstance(workers, int)
            or isinstance(workers, bool)
            or not isinstance(max_pending, int)
            or isinstance(max_pending, bool)
            or workers < 1
            or max_pending < 1
        ):
            raise ValueError("workers and max_pending must be positive integers")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._slots = threading.BoundedSemaphore(max_pending)
        self._state_lock = threading.Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="terralens-job")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA synchronous = NORMAL")
            return connection
        except Exception:
            connection.close()
            raise

    def _initialize(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    request_message TEXT NOT NULL,
                    dataset_id TEXT,
                    status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'succeeded', 'failed')),
                    stage_json TEXT NOT NULL,
                    error_message TEXT,
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_conversation_history
                    ON jobs(conversation_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_conversation
                    ON jobs(conversation_id) WHERE status IN ('queued', 'running');
                """
            )
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE jobs
                   SET status = 'failed', stage_json = ?, error_message = ?, result_json = NULL, updated_at = ?
                   WHERE status IN ('queued', 'running')""",
                (self._stage_json("interrupted", "Interrupted"), _INTERRUPTED_FAILURE, now),
            )
            connection.commit()
        except Exception:
            if connection is not None:
                connection.rollback()
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _stage_json(step_id: str, label: str) -> str:
        return json.dumps({"step_id": step_id, "label": label}, separators=(",", ":"))

    @staticmethod
    def _validate_submit(message: str, dataset_id: str | None, conversation_id: str | None) -> None:
        if not isinstance(message, str) or not message.strip():
            raise JobError("A non-empty assistant message is required.", 422)
        if len(message.encode("utf-8")) > _MAX_MESSAGE_BYTES:
            raise JobError("The assistant message is too large.", 422)
        if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id or len(dataset_id) > 160):
            raise JobError("The dataset identifier is invalid.", 422)
        if conversation_id is not None and (not isinstance(conversation_id, str) or not _ID_PATTERN.fullmatch(conversation_id)):
            raise JobError("Conversation not found.", 404)

    def _cleanup_locked(self, connection: sqlite3.Connection, now: float) -> None:
        """Bound every cleanup statement; normal operation never retains over 100 rows."""
        connection.execute(
            """DELETE FROM jobs WHERE job_id IN (
                   SELECT job_id FROM jobs
                   WHERE status IN ('succeeded', 'failed') AND created_at < ?
                   ORDER BY created_at LIMIT 100
               )""",
            (now - _RETENTION_SECONDS,),
        )
        count = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
        excess = max(0, count - (_MAX_JOBS - 1))
        if excess:
            connection.execute(
                """DELETE FROM jobs WHERE job_id IN (
                       SELECT job_id FROM jobs WHERE status IN ('succeeded', 'failed')
                       ORDER BY created_at LIMIT ?
                   )""",
                (min(excess, 100),),
            )
        count = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
        if count >= _MAX_JOBS:
            raise JobError("The background job store is at capacity; try again after an active job finishes.", 429)
        connection.execute(
            """DELETE FROM conversations WHERE conversation_id IN (
                   SELECT c.conversation_id FROM conversations c
                   LEFT JOIN jobs j ON j.conversation_id = c.conversation_id
                   WHERE j.job_id IS NULL LIMIT 100
               )"""
        )

    def submit(
        self,
        message: str,
        dataset_id: str | None,
        conversation_id: str | None,
        run: Callable[[str, str | None, list[dict], Callable[[str, str], None]], dict],
    ) -> dict:
        self._validate_submit(message, dataset_id, conversation_id)
        if not callable(run):
            raise JobError("The background job callback is invalid.", 500)
        with self._state_lock:
            if self._closed:
                raise JobError("The background job manager is unavailable.", 503)
        if not self._slots.acquire(blocking=False):
            raise JobError("Too many background jobs are already pending.", 429)

        job_id = secrets.token_hex(16)
        owner_id = conversation_id or secrets.token_hex(16)
        connection: sqlite3.Connection | None = None
        inserted = False
        try:
            connection = self._connect()
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            self._cleanup_locked(connection, now)
            if conversation_id is None:
                connection.execute(
                    "INSERT INTO conversations(conversation_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (owner_id, now, now),
                )
            elif connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?", (owner_id,)
            ).fetchone() is None:
                raise JobError("Conversation not found.", 404)
            if connection.execute(
                "SELECT 1 FROM jobs WHERE conversation_id = ? AND status IN ('queued', 'running')",
                (owner_id,),
            ).fetchone() is not None:
                raise JobError("This conversation already has an active job.", 409)
            connection.execute(
                """INSERT INTO jobs(
                       job_id, conversation_id, request_message, dataset_id, status, stage_json,
                       error_message, result_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 'queued', ?, NULL, NULL, ?, ?)""",
                (job_id, owner_id, message, dataset_id, self._stage_json("queued", "Waiting to start"), now, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (now, owner_id)
            )
            connection.commit()
            inserted = True
        except JobError:
            connection.rollback()
            raise
        except sqlite3.IntegrityError as exc:
            if connection is not None:
                connection.rollback()
            if connection.execute(
                "SELECT 1 FROM jobs WHERE conversation_id = ? AND status IN ('queued', 'running')",
                (owner_id,),
            ).fetchone() is not None:
                raise JobError("This conversation already has an active job.", 409) from exc
            raise JobError("The background job could not be admitted.", 503) from exc
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise JobError("The background job store is unavailable.", 503) from exc
        finally:
            if connection is not None:
                connection.close()
            if not inserted:
                self._slots.release()

        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("manager closed")
                self._executor.submit(self._execute, job_id, run)
        except Exception as exc:
            try:
                self._fail(job_id, _GENERIC_FAILURE)
            except sqlite3.Error:
                pass
            finally:
                self._slots.release()
            raise JobError("The background worker is unavailable.", 503) from exc
        return {"job_id": job_id, "conversation_id": owner_id}

    def _start(self, job_id: str) -> sqlite3.Row | None:
        connection = self._connect()
        try:
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """UPDATE jobs SET status = 'running', stage_json = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'queued'""",
                (self._stage_json("running", "Starting analysis"), now, job_id),
            ).rowcount
            row = connection.execute(
                "SELECT request_message, dataset_id, conversation_id FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            connection.commit()
            return row if changed else None
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _compact_history_result(result: dict) -> dict:
        compact: dict = {}
        if "workflow" in result:
            compact["workflow"] = result["workflow"]
        summary = result.get("summary", result.get("message"))
        if summary is not None:
            compact["summary"] = summary[:4000] if isinstance(summary, str) else summary
        return compact

    def _history(self, conversation_id: str, current_job_id: str) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT request_message, result_json FROM jobs
                   WHERE conversation_id = ? AND job_id != ? AND status = 'succeeded'
                   ORDER BY created_at DESC, rowid DESC LIMIT 8""",
                (conversation_id, current_job_id),
            ).fetchall()
        finally:
            connection.close()
        history = []
        for row in reversed(rows):
            result = json.loads(row["result_json"])
            history.append({
                "message": row["request_message"],
                "result": self._compact_history_result(result),
            })
        return history

    def _progress(self, job_id: str, step_id: str, label: str) -> None:
        safe_step = str(step_id).strip()[:64] or "working"
        safe_label = str(label).strip()[:300] or "Working"
        connection = self._connect()
        try:
            connection.execute(
                """UPDATE jobs SET stage_json = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'running'""",
                (self._stage_json(safe_step, safe_label), time.time(), job_id),
            )
        finally:
            connection.close()

    @staticmethod
    def _serialize_result(result: dict) -> str:
        if not isinstance(result, dict):
            raise _ResultError("job callbacks must return dictionaries")
        try:
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise _ResultError("job result is not valid JSON") from exc
        if len(encoded.encode("utf-8")) > _MAX_RESULT_BYTES:
            raise _ResultError("job result exceeds the persistence limit")
        return encoded

    @staticmethod
    def _failure_message(exc: Exception) -> str:
        useful = isinstance(exc, WorkflowError) or (
            isinstance(exc, ValueError)
            and exc.__class__.__module__.startswith("terralens.")
        )
        if not useful:
            return _GENERIC_FAILURE
        message = " ".join(str(exc).split())
        return message[:_MAX_ERROR_CHARS] if message else _GENERIC_FAILURE

    def _succeed(self, job_id: str, result_json: str) -> None:
        connection = self._connect()
        try:
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT conversation_id FROM jobs WHERE job_id = ? AND status = 'running'", (job_id,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    """UPDATE jobs SET status = 'succeeded', result_json = ?, error_message = NULL, updated_at = ?
                       WHERE job_id = ? AND status = 'running'""",
                    (result_json, now, job_id),
                )
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (now, row["conversation_id"]),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _fail(self, job_id: str, message: str) -> None:
        connection = self._connect()
        try:
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT conversation_id FROM jobs WHERE job_id = ? AND status IN ('queued', 'running')", (job_id,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    """UPDATE jobs SET status = 'failed', error_message = ?, result_json = NULL, updated_at = ?
                       WHERE job_id = ? AND status IN ('queued', 'running')""",
                    (message[:_MAX_ERROR_CHARS], now, job_id),
                )
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (now, row["conversation_id"]),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _execute(self, job_id: str, run: Callable) -> None:
        try:
            row = self._start(job_id)
            if row is None:
                return
            history = self._history(row["conversation_id"], job_id)
            result = run(
                row["request_message"],
                row["dataset_id"],
                history,
                lambda step_id, label: self._progress(job_id, step_id, label),
            )
            self._succeed(job_id, self._serialize_result(result))
        except Exception as exc:
            try:
                self._fail(job_id, self._failure_message(exc))
            except sqlite3.Error:
                pass
        finally:
            self._slots.release()

    def get(self, job_id: str, conversation_id: str) -> dict:
        if not isinstance(job_id, str) or not _ID_PATTERN.fullmatch(job_id):
            raise JobError("Job not found.", 404)
        if not isinstance(conversation_id, str) or not _ID_PATTERN.fullmatch(conversation_id):
            raise JobError("Job not found.", 404)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                """SELECT job_id, conversation_id, status, stage_json, error_message, result_json
                   FROM jobs WHERE job_id = ? AND conversation_id = ?""",
                (job_id, conversation_id),
            ).fetchone()
        except sqlite3.Error as exc:
            raise JobError("The background job store is unavailable.", 503) from exc
        finally:
            if connection is not None:
                connection.close()
        if row is None:
            raise JobError("Job not found.", 404)
        response = {
            "job_id": row["job_id"],
            "conversation_id": row["conversation_id"],
            "status": row["status"],
            "stage": json.loads(row["stage_json"]),
        }
        if row["status"] == "failed":
            response["message"] = row["error_message"] or _GENERIC_FAILURE
        elif row["status"] == "succeeded":
            response["result"] = json.loads(row["result_json"])
        return response

    def shutdown(self, wait: bool = True) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    close = shutdown
