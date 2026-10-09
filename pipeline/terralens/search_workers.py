"""Bounded concurrent execution of durable, dependency-aware search work.

Processors receive a claimed unit and a checkpoint callback. They return every
produced record and a coverage dictionary. Worker count and processing budgets
control resource use without imposing a result cap or changing the search plan.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable

from .search_store import SearchStore, SearchStoreError


class WorkInterrupted(Exception):
    """The run was paused, cancelled or reclaimed by another worker."""


@dataclass(frozen=True)
class WorkUnitResult:
    records: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


Processor = Callable[[dict, Callable[[dict], None]], WorkUnitResult | dict]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


class SearchWorkerRunner:
    """Process persisted units using one globally bounded pool.

    ``submit`` returns a Future for a run; ``run`` is the synchronous equivalent.
    Multiple service processes can share the store: leases prevent simultaneous
    work commits and checkpoints survive service restarts. Automatic retries are
    bounded by max_attempts; explicitly resuming a failed run starts a new attempt.

    max_units_per_run and max_runtime_seconds are per execution budgets. Reaching
    either pauses remaining work and records the reason; resume gives the stored
    plan a fresh execution budget. Result counts are never used as a budget.
    """

    def __init__(
        self, store: SearchStore, workers: int = 2, max_pending: int = 8,
        lease_seconds: float = 300, max_units_per_run: int | None = None,
        max_runtime_seconds: float | None = None, max_attempts: int = 1,
        poll_interval: float = 0.1,
    ):
        self.store = store
        self.workers = _integer(workers, "workers")
        self.max_pending = _integer(max_pending, "max_pending")
        self.lease_seconds = _seconds(lease_seconds, "lease_seconds")
        self.max_units_per_run = None if max_units_per_run is None else _integer(max_units_per_run, "max_units_per_run")
        self.max_runtime_seconds = None if max_runtime_seconds is None else _seconds(max_runtime_seconds, "max_runtime_seconds")
        self.max_attempts = _integer(max_attempts, "max_attempts")
        self.poll_interval = _seconds(poll_interval, "poll_interval")
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="terralens-search-unit")
        self._coordinators = ThreadPoolExecutor(max_workers=max_pending, thread_name_prefix="terralens-search-run")
        self._work_slots = threading.BoundedSemaphore(workers)
        self._run_slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.RLock()
        self._closed = False
        self._stopping = threading.Event()
        self._active: dict[str, Future] = {}

    def submit(self, run_id: str, processor: Processor) -> Future:
        if not callable(processor):
            raise ValueError("processor must be callable")
        self.store.get_run(run_id)
        with self._lock:
            if self._closed:
                raise SearchStoreError("Search workers are unavailable.", 503)
            current = self._active.get(run_id)
            if current is not None and not current.done():
                return current
            if not self._run_slots.acquire(blocking=False):
                raise SearchStoreError("The search queue is full; the run remains saved for later continuation.", 429)
            try:
                future = self._coordinators.submit(self.run, run_id, processor)
                self._active[run_id] = future
                future.add_done_callback(lambda completed: self._finish_submission(run_id, completed))
                return future
            except Exception:
                self._run_slots.release()
                raise

    def _finish_submission(self, run_id: str, future: Future) -> None:
        with self._lock:
            if self._active.get(run_id) is future:
                self._active.pop(run_id, None)
        self._run_slots.release()

    def resume(self, run_id: str, processor: Processor) -> Future:
        self.store.resume(run_id)
        return self.submit(run_id, processor)

    def run(self, run_id: str, processor: Processor) -> dict:
        """Execute all currently eligible units and commit results progressively."""
        if not callable(processor):
            raise ValueError("processor must be callable")
        self.store.get_run(run_id)
        started, admitted = time.monotonic(), 0
        pending: set[Future] = set()
        try:
            while True:
                current = self.store.get_run(run_id)
                if current["status"] not in {"queued", "running"}:
                    break
                if self._stopping.is_set():
                    self.store.pause(run_id, "worker_shutdown")
                    break
                if self.max_runtime_seconds is not None and time.monotonic() - started >= self.max_runtime_seconds:
                    self.store.pause(run_id, "runtime_budget")
                    break
                budget_used = self.max_units_per_run is not None and admitted >= self.max_units_per_run
                if budget_used:
                    if not pending:
                        self.store.pause(run_id, "work_unit_budget")
                        break
                else:
                    while len(pending) < self.workers and self._work_slots.acquire(blocking=False):
                        if self.max_units_per_run is not None and admitted >= self.max_units_per_run:
                            self._work_slots.release()
                            break
                        try:
                            unit = self.store.claim_unit(run_id, self.lease_seconds)
                        except Exception:
                            self._work_slots.release()
                            raise
                        if unit is None:
                            self._work_slots.release()
                            break
                        try:
                            future = self._executor.submit(self._process, unit, processor)
                        except Exception:
                            self._work_slots.release()
                            self.store.pause(run_id, "worker_unavailable")
                            raise
                        future.add_done_callback(lambda _: self._work_slots.release())
                        pending.add(future)
                        admitted += 1
                if pending:
                    done, pending = wait(pending, timeout=self.poll_interval, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                else:
                    current = self.store.get_run(run_id)
                    if current["status"] not in {"queued", "running"}:
                        break
                    # Other processes may hold valid leases. Wait for their
                    # commits or expiry rather than claiming completion early.
                    self._stopping.wait(self.poll_interval)
            return self.store.get_run(run_id)
        except Exception:
            # Never strand a claimed unit because its coordinator stopped.
            current = self.store.get_run(run_id)
            if current["status"] in {"queued", "running"}:
                self.store.pause(run_id, "coordinator_failure")
            raise

    def _process(self, unit: dict, processor: Processor) -> None:
        run_id, unit_id, token = unit["run_id"], unit["unit_id"], unit["lease_token"]
        finished, interrupted = threading.Event(), threading.Event()

        def checkpoint(value: dict) -> None:
            if interrupted.is_set():
                raise WorkInterrupted("The search work lease was revoked.")
            try:
                self.store.checkpoint_unit(run_id, unit_id, token, value, self.lease_seconds)
            except SearchStoreError as exc:
                if exc.status_code == 409:
                    interrupted.set()
                    raise WorkInterrupted(str(exc)) from exc
                raise

        def heartbeat() -> None:
            while not finished.wait(self.lease_seconds / 3):
                try:
                    self.store.renew_unit(run_id, unit_id, token, self.lease_seconds)
                except Exception:
                    interrupted.set()
                    return

        renewer = threading.Thread(target=heartbeat, name="terralens-search-lease", daemon=True)
        renewer.start()
        try:
            output = processor(unit, checkpoint)
            if interrupted.is_set():
                return
            if isinstance(output, WorkUnitResult):
                records, coverage = output.records, output.coverage
            elif isinstance(output, dict) and "records" in output and "coverage" in output:
                records, coverage = output["records"], output["coverage"]
            else:
                raise ValueError("A search processor must return records and coverage.")
            self.store.commit_unit(run_id, unit_id, token, records, coverage)
        except WorkInterrupted:
            pass
        except Exception as exc:
            # Internal exceptions can contain provider credentials or paths.
            # Expected user-facing processor errors use the same status contract.
            expected = isinstance(exc, SearchStoreError) or (
                exc.__class__.__module__.startswith("terralens.") and
                isinstance(getattr(exc, "status_code", None), int)
            )
            message = " ".join(str(exc).split())[:1000] if expected else "Work processing failed; coverage remains incomplete."
            if not message:
                message = "Work processing failed; coverage remains incomplete."
            try:
                self.store.fail_unit(
                    run_id, unit_id, token, message,
                    retryable=unit["attempts"] < self.max_attempts,
                )
            except SearchStoreError as lease_error:
                if lease_error.status_code != 409:
                    raise
        finally:
            finished.set()
            renewer.join()

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stopping.set()
        self._coordinators.shutdown(wait=wait, cancel_futures=False)
        self._executor.shutdown(wait=wait, cancel_futures=False)

    close = shutdown
