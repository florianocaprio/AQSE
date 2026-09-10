from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock, Thread
from time import perf_counter
from typing import Literal

from pydantic import Field

from app.quantum.admission import heavy_quantum_slot
from app.training.assembly import assemble_training_input
from app.training.execution_intent import EngineeringBenchmarkIntent
from app.training.models import FrozenModel
from app.training.run_models import EvaluationCounters
from app.training.runner import run_bounded_qng
from app.training.seal import verify_canonical_test_seal
from app.training.storage import artifact_root
from app.training.workflow import canonical_paths, validate_training_input_arrays


class EngineeringBenchmarkState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EngineeringBenchmarkResult(FrozenModel):
    schema_version: Literal["aqse.qng-responsiveness-result.v1"] = (
        "aqse.qng-responsiveness-result.v1"
    )
    purpose: Literal["engineering_benchmark"] = "engineering_benchmark"
    candidate_artifact_published: Literal[False] = False
    accepted_update_count: int = Field(ge=0, le=10)
    compute_wall_time_ms: float = Field(ge=0.0)
    end_to_end_wall_time_ms: float = Field(ge=0.0)
    peak_process_rss_bytes: int = Field(ge=0)
    counters: EvaluationCounters


class EngineeringBenchmarkView(FrozenModel):
    intent: EngineeringBenchmarkIntent
    state: EngineeringBenchmarkState
    created_at_utc: str
    updated_at_utc: str
    result: EngineeringBenchmarkResult | None = None
    error: str | None = Field(default=None, max_length=500)


@dataclass(frozen=True)
class EngineeringBenchmarkStart:
    benchmark: EngineeringBenchmarkView
    created: bool


class EngineeringBenchmarkBusyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute_real_qng_responsiveness_benchmark(
    root: Path | None = None,
) -> EngineeringBenchmarkResult:
    dataset_path, encoder_path, bank_path, _ = canonical_paths(root or artifact_root())
    seal_before = verify_canonical_test_seal(dataset_path)
    training_input = assemble_training_input(
        dataset_path=dataset_path,
        encoder_path=encoder_path,
        train_bank_path=bank_path,
    )
    validate_training_input_arrays(training_input)
    started = perf_counter()
    run = run_bounded_qng(training_input)
    elapsed_ms = (perf_counter() - started) * 1_000.0
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during engineering benchmark")
    return EngineeringBenchmarkResult(
        accepted_update_count=run.accepted_update_count,
        compute_wall_time_ms=run.total_wall_time_ms,
        end_to_end_wall_time_ms=elapsed_ms,
        peak_process_rss_bytes=run.peak_process_rss_bytes,
        counters=run.counters,
    )


class EngineeringBenchmarkService:
    def __init__(
        self,
        runner: Callable[[], EngineeringBenchmarkResult] = (
            execute_real_qng_responsiveness_benchmark
        ),
    ) -> None:
        self._runner = runner
        self._view: EngineeringBenchmarkView | None = None
        self._lock = RLock()

    def start(
        self,
        intent: EngineeringBenchmarkIntent,
    ) -> EngineeringBenchmarkStart:
        with self._lock:
            if self._view is not None:
                return EngineeringBenchmarkStart(benchmark=self._view, created=False)
            if not heavy_quantum_slot.acquire(blocking=False):
                raise EngineeringBenchmarkBusyError(
                    "another heavy quantum operation is active"
                )
            now = _utc_now()
            self._view = EngineeringBenchmarkView(
                intent=intent,
                state=EngineeringBenchmarkState.CREATED,
                created_at_utc=now,
                updated_at_utc=now,
            )
            try:
                worker = Thread(
                    target=self._run,
                    name="aqse-qng-responsiveness-benchmark",
                    daemon=True,
                )
                worker.start()
            except Exception:
                self._view = None
                heavy_quantum_slot.release()
                raise
            return EngineeringBenchmarkStart(benchmark=self._view, created=True)

    def get(self) -> EngineeringBenchmarkView | None:
        with self._lock:
            return self._view

    def _run(self) -> None:
        try:
            with self._lock:
                if self._view is None:
                    raise RuntimeError("engineering benchmark state is missing")
                self._view = self._view.model_copy(
                    update={
                        "state": EngineeringBenchmarkState.RUNNING,
                        "updated_at_utc": _utc_now(),
                    }
                )
            result = self._runner()
            with self._lock:
                self._view = self._view.model_copy(
                    update={
                        "state": EngineeringBenchmarkState.COMPLETED,
                        "updated_at_utc": _utc_now(),
                        "result": result,
                    }
                )
        except Exception as exc:
            with self._lock:
                if self._view is not None:
                    self._view = self._view.model_copy(
                        update={
                            "state": EngineeringBenchmarkState.FAILED,
                            "updated_at_utc": _utc_now(),
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        }
                    )
        finally:
            heavy_quantum_slot.release()


qng_responsiveness_benchmarks = EngineeringBenchmarkService()
