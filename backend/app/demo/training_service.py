from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from app.demo.bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    DemoTaskPartition,
    TaskId,
)
from app.demo.bundle_storage import write_bundle, write_selection_freeze
from app.demo.evaluation import (
    DemoTrainingCancelled,
    fit_demo_task_candidates,
    freeze_model_selection,
)
from app.demo.knowledge_storage import load_approved_train_partition
from app.demo.provisioning import find_current_study, load_task_partitions
from app.demo.runtime_models import (
    DemoTrainingIntent,
    DemoTrainingJobView,
    DemoTrainingStepView,
)
from app.demo.training import DemoQngStep
from app.quantum.admission import heavy_quantum_slot
from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel
from app.training.storage import artifact_root

MAX_TRAINING_JOB_BYTES = 2 * 1024 * 1024


def _knowledge_training_partitions(
    intent: DemoTrainingIntent,
    local_validation: DemoTaskPartition,
    network_validation: DemoTaskPartition,
) -> tuple[
    DemoTaskPartition,
    DemoTaskPartition,
    DemoTaskPartition,
    DemoTaskPartition,
] | None:
    """Bind two reviewed TRAIN collections to the frozen demo VALIDATION set."""

    if (
        intent.local_train_collection_id is None
        or intent.network_train_collection_id is None
    ):
        return None
    local_train = load_approved_train_partition(intent.local_train_collection_id)
    network_train = load_approved_train_partition(intent.network_train_collection_id)
    if (
        local_train.task_id != LOCAL_TASK_ID
        or network_train.task_id != NETWORK_TASK_ID
    ):
        raise ValueError("approved knowledge collections are not ordered local/network")
    from app.demo.bundle_models import canonical_digest

    digest = canonical_digest(
        {
            "schema_version": "aqse.network-demo.knowledge-training-pair.v1",
            "canonical_validation_study_id": intent.study_artifact_id,
            "canonical_validation_study_digest": local_validation.study_content_digest,
            "local_train_collection_id": intent.local_train_collection_id,
            "local_train_collection_digest": local_train.study_content_digest,
            "network_train_collection_id": intent.network_train_collection_id,
            "network_train_collection_digest": network_train.study_content_digest,
        }
    )
    study_id = f"aqse-knowledge-study-{digest[:16]}"
    updates = {"study_artifact_id": study_id, "study_content_digest": digest}
    return (
        local_train.model_copy(update=updates),
        network_train.model_copy(update=updates),
        local_validation.model_copy(update=updates),
        network_validation.model_copy(update=updates),
    )


class DemoTrainingBusyError(RuntimeError):
    pass


class DemoTrainingStoreError(RuntimeError):
    pass


class _StoredDemoTrainingJob(FrozenModel):
    schema_version: str = "aqse.demo-training-store.v1"
    intent: DemoTrainingIntent
    job: DemoTrainingJobView


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _job_id(intent: DemoTrainingIntent) -> str:
    from app.demo.bundle_models import canonical_digest

    digest = canonical_digest(intent.model_dump(mode="json"))
    return f"aqse-demo-training-{digest[:16]}"


def _training_root() -> Path:
    root = artifact_root() / "network-demo" / "training-jobs"
    if root.exists() and root.is_symlink():
        raise DemoTrainingStoreError("demo training-job root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_path(job_id: str) -> Path:
    if not job_id.startswith("aqse-demo-training-") or len(job_id) != 35:
        raise ValueError("invalid demo training job identifier")
    return _training_root() / f"{job_id}.json"


def _read_stored(path: Path) -> _StoredDemoTrainingJob:
    if path.is_symlink() or not path.is_file():
        raise DemoTrainingStoreError("demo training job is missing or unsafe")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_TRAINING_JOB_BYTES:
        raise DemoTrainingStoreError("demo training job exceeds the bounded reader")
    try:
        return _StoredDemoTrainingJob.model_validate(json.loads(path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DemoTrainingStoreError("demo training job is invalid") from exc


def _write_stored(value: _StoredDemoTrainingJob) -> None:
    path = _job_path(value.job.job_id)
    payload = canonical_json_bytes(value.model_dump(mode="json")) + b"\n"
    if len(payload) > MAX_TRAINING_JOB_BYTES:
        raise DemoTrainingStoreError("demo training job exceeds the storage limit")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DemoTrainingJobService:
    def __init__(self) -> None:
        self._active_job_id: str | None = None
        self._intents: dict[str, DemoTrainingIntent] = {}
        self._jobs: dict[str, DemoTrainingJobView] = {}
        self._cancellations: dict[str, Event] = {}
        self._threads: dict[str, Thread] = {}
        self._lock = RLock()

    def start(self, intent: DemoTrainingIntent) -> tuple[DemoTrainingJobView, bool]:
        job_id = _job_id(intent)
        path = _job_path(job_id)
        lock_path = _training_root() / ".training-admission.lock"
        with lock_path.open("a+b") as admission:
            fcntl.flock(admission.fileno(), fcntl.LOCK_EX)
            with self._lock:
                known = self._jobs.get(job_id)
                if known is not None:
                    if self._intents[job_id] != intent:
                        raise DemoTrainingStoreError("training intent digest collision")
                    return known, False
                if path.exists():
                    stored = _read_stored(path)
                    if stored.intent != intent:
                        raise DemoTrainingStoreError("stored training intent differs")
                    job = stored.job
                    if job.state in {"created", "running"}:
                        job = job.model_copy(
                            update={
                                "state": "failed",
                                "updated_at_utc": _utc_now(),
                                "current_stage": "interrupted",
                                "error": (
                                    "Previous backend process ended before this job "
                                    "reached a terminal state; use a new intent to retry."
                                ),
                            }
                        )
                        _write_stored(_StoredDemoTrainingJob(intent=intent, job=job))
                    self._remember(intent, job)
                    return job, False
                if self._active_job_id is not None:
                    raise DemoTrainingBusyError("another bounded demo training job is active")
                if not heavy_quantum_slot.acquire(blocking=False):
                    raise DemoTrainingBusyError("another heavy quantum operation is active")
                now = _utc_now()
                job = DemoTrainingJobView(
                    job_id=job_id,
                    intent_id=intent.intent_id,
                    study_artifact_id=intent.study_artifact_id,
                    state="created",
                    created_at_utc=now,
                    updated_at_utc=now,
                    current_stage="admitted",
                    progress=(),
                )
                cancellation = Event()
                self._remember(intent, job)
                self._cancellations[job_id] = cancellation
                self._active_job_id = job_id
                _write_stored(_StoredDemoTrainingJob(intent=intent, job=job))
                worker = Thread(
                    target=self._run,
                    args=(job_id,),
                    name=f"aqse-demo-training-{job_id[-8:]}",
                    daemon=True,
                )
                self._threads[job_id] = worker
                worker.start()
                return job, True

    def get(self, job_id: str) -> DemoTrainingJobView | None:
        with self._lock:
            known = self._jobs.get(job_id)
            if known is not None:
                return known
        path = _job_path(job_id)
        if not path.exists():
            return None
        stored = _read_stored(path)
        with self._lock:
            self._remember(stored.intent, stored.job)
        return stored.job

    def cancel(self, job_id: str) -> DemoTrainingJobView | None:
        with self._lock:
            job = self._jobs.get(job_id)
            cancellation = self._cancellations.get(job_id)
            if job is None:
                return self.get(job_id)
            if cancellation is not None and job.state in {"created", "running"}:
                cancellation.set()
                self._update_locked(job_id, current_stage="cancellation requested")
            return self._jobs[job_id]

    def clear_for_tests(self) -> None:
        with self._lock:
            cancellations = tuple(self._cancellations.values())
            threads = tuple(self._threads.values())
        for cancellation in cancellations:
            cancellation.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

    def _remember(
        self,
        intent: DemoTrainingIntent,
        job: DemoTrainingJobView,
    ) -> None:
        self._intents[job.job_id] = intent
        self._jobs[job.job_id] = job

    def _update_locked(self, job_id: str, **updates: Any) -> DemoTrainingJobView:
        job = self._jobs[job_id].model_copy(
            update={"updated_at_utc": _utc_now(), **updates}
        )
        self._jobs[job_id] = job
        _write_stored(
            _StoredDemoTrainingJob(intent=self._intents[job_id], job=job)
        )
        return job

    def _set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            self._update_locked(job_id, state="running", current_stage=stage)

    def _progress(self, job_id: str, task_id: TaskId, step: DemoQngStep) -> None:
        with self._lock:
            current = [
                item for item in self._jobs[job_id].progress if item.task_id != task_id
            ]
            current.append(
                DemoTrainingStepView(
                    task_id=task_id,
                    candidate_name="protected_qng",
                    accepted_updates=step.step_index + 1,
                    current_loss=step.loss_after,
                )
            )
            ordered = tuple(
                sorted(
                    current,
                    key=lambda item: 0 if item.task_id == LOCAL_TASK_ID else 1,
                )
            )
            self._update_locked(job_id, progress=ordered)

    def _run(self, job_id: str) -> None:
        try:
            with self._lock:
                intent = self._intents[job_id]
                cancellation = self._cancellations[job_id]
            located = find_current_study()
            if located is None or located[1].artifact_id != intent.study_artifact_id:
                raise ValueError("requested immutable demo study is unavailable")
            study_path, manifest = located
            self._set_stage(job_id, "loading TRAIN and VALIDATION")
            local_train, network_train = load_task_partitions(
                study_path,
                manifest,
                "train",
            )
            local_validation, network_validation = load_task_partitions(
                study_path,
                manifest,
                "validation",
            )
            knowledge_partitions = _knowledge_training_partitions(
                intent,
                local_validation,
                network_validation,
            )
            if knowledge_partitions is not None:
                (
                    local_train,
                    network_train,
                    local_validation,
                    network_validation,
                ) = knowledge_partitions
                self._set_stage(
                    job_id,
                    "loaded approved knowledge TRAIN collections and frozen VALIDATION",
                )
            self._set_stage(job_id, "local protected QNG and downstream fit")
            local = fit_demo_task_candidates(
                local_train,
                local_validation,
                cancellation=cancellation,
                progress=lambda task, step: self._progress(job_id, task, step),
            )
            if cancellation.is_set():
                raise DemoTrainingCancelled("demo training cancelled after local task")
            self._set_stage(job_id, "network protected QNG and downstream fit")
            network = fit_demo_task_candidates(
                network_train,
                network_validation,
                cancellation=cancellation,
                progress=lambda task, step: self._progress(job_id, task, step),
            )
            if cancellation.is_set():
                raise DemoTrainingCancelled("demo training cancelled after network task")
            self._set_stage(job_id, "publishing candidate bundles and selection freeze")
            for bundle in (*local.bundles, *network.bundles):
                write_bundle(bundle)
            freeze = freeze_model_selection(local, network)
            write_selection_freeze(freeze)
            selections = {item.task_id: item.selected_bundle_id for item in freeze.selections}
            with self._lock:
                self._update_locked(
                    job_id,
                    state="completed",
                    current_stage="completed; explicit bundle application required",
                    selection_freeze_id=freeze.freeze_id,
                    selected_local_bundle_id=selections[LOCAL_TASK_ID],
                    selected_network_bundle_id=selections[NETWORK_TASK_ID],
                )
        except DemoTrainingCancelled:
            with self._lock:
                self._update_locked(
                    job_id,
                    state="cancelled",
                    current_stage="cancelled without publishing a new selection",
                )
        except Exception as exc:
            with self._lock:
                self._update_locked(
                    job_id,
                    state="failed",
                    current_stage="failed",
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
            heavy_quantum_slot.release()


demo_training_jobs = DemoTrainingJobService()
