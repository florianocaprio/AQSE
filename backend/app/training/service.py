from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, RLock, Thread
from uuid import uuid4

from app.quantum.admission import heavy_quantum_slot
from app.training.run_models import TrainingJobState, TrainingJobView
from app.training.workflow import TrainingJobOutcome, execute_canonical_training

MAXIMUM_JOB_RECORDS = 16


class TrainingJobBusyError(RuntimeError):
    pass


class TrainingJobCapacityError(RuntimeError):
    pass


@dataclass
class _JobRecord:
    job_id: str
    state: TrainingJobState
    created_at_utc: str
    updated_at_utc: str
    cancellation: Event
    accepted_update_count: int = 0
    stop_reason: str | None = None
    run_artifact_id: str | None = None
    error: str | None = None
    cancellation_requested: bool = False

    def view(self) -> TrainingJobView:
        return TrainingJobView(
            job_id=self.job_id,
            state=self.state,
            created_at_utc=self.created_at_utc,
            updated_at_utc=self.updated_at_utc,
            accepted_update_count=self.accepted_update_count,
            stop_reason=self.stop_reason,
            run_artifact_id=self.run_artifact_id,
            error=self.error,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrainingJobService:
    def __init__(
        self,
        runner: Callable[[Event], TrainingJobOutcome] = execute_canonical_training,
        *,
        maximum_records: int = MAXIMUM_JOB_RECORDS,
    ) -> None:
        self._runner = runner
        self.maximum_records = maximum_records
        self._records: dict[str, _JobRecord] = {}
        self._active_job_id: str | None = None
        self._lock = RLock()

    def start(self) -> TrainingJobView:
        with self._lock:
            if self._active_job_id is not None:
                raise TrainingJobBusyError("an AQSE QNG training job is already active")
            if not heavy_quantum_slot.acquire(blocking=False):
                raise TrainingJobBusyError("another heavy quantum operation is active")
            try:
                self._ensure_capacity_locked()
                now = _utc_now()
                record = _JobRecord(
                    job_id=f"qng-job-{uuid4().hex}",
                    state=TrainingJobState.CREATED,
                    created_at_utc=now,
                    updated_at_utc=now,
                    cancellation=Event(),
                )
                self._records[record.job_id] = record
                self._active_job_id = record.job_id
                worker = Thread(
                    target=self._run,
                    args=(record.job_id,),
                    name=f"aqse-qng-{record.job_id[-8:]}",
                    daemon=True,
                )
                worker.start()
                return record.view()
            except Exception:
                heavy_quantum_slot.release()
                raise

    def get(self, job_id: str) -> TrainingJobView | None:
        with self._lock:
            record = self._records.get(job_id)
            return None if record is None else record.view()

    def cancel(self, job_id: str) -> TrainingJobView | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            if record.state in {TrainingJobState.CREATED, TrainingJobState.RUNNING}:
                record.cancellation_requested = True
                record.cancellation.set()
                record.updated_at_utc = _utc_now()
            return record.view()

    def _ensure_capacity_locked(self) -> None:
        if len(self._records) < self.maximum_records:
            return
        terminal = [
            record
            for record in self._records.values()
            if record.state
            in {
                TrainingJobState.COMPLETED,
                TrainingJobState.CANCELLED,
                TrainingJobState.FAILED,
            }
        ]
        if not terminal:
            raise TrainingJobCapacityError("bounded QNG job registry is full")
        oldest = min(terminal, key=lambda item: item.created_at_utc)
        del self._records[oldest.job_id]

    def _run(self, job_id: str) -> None:
        try:
            with self._lock:
                record = self._records[job_id]
                record.state = TrainingJobState.RUNNING
                record.updated_at_utc = _utc_now()
            outcome = self._runner(record.cancellation)
            with self._lock:
                record.accepted_update_count = outcome.accepted_update_count
                record.stop_reason = outcome.stop_reason
                record.run_artifact_id = outcome.run_artifact_id
                record.state = (
                    TrainingJobState.CANCELLED
                    if outcome.stop_reason == "CANCELLED"
                    else TrainingJobState.COMPLETED
                )
                record.updated_at_utc = _utc_now()
        except Exception as exc:
            with self._lock:
                record = self._records[job_id]
                record.state = TrainingJobState.FAILED
                record.error = f"{type(exc).__name__}: {exc}"[:500]
                record.updated_at_utc = _utc_now()
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
            heavy_quantum_slot.release()


training_jobs = TrainingJobService()
