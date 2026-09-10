from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.training.canonical import canonical_json_bytes
from app.training.execution_intent import (
    ExecutionIntentClaim,
    ExecutionIntentResult,
    StoredExecutionIntent,
    TrainingExecutionIntent,
    build_execution_intent_claim,
    build_execution_intent_result,
    validate_execution_intent_claim,
    validate_execution_intent_result,
)
from app.training.run_models import TrainingJobView
from app.training.storage import artifact_limit_bytes, artifact_root

MAX_INTENT_JSON_BYTES = 1_048_576


class ExecutionIntentConflictError(RuntimeError):
    pass


class ExecutionIntentStore(Protocol):
    def load(self, intent: TrainingExecutionIntent) -> StoredExecutionIntent | None: ...

    def claim(
        self,
        intent: TrainingExecutionIntent,
        *,
        job_id: str,
        claimed_at_utc: str,
    ) -> tuple[StoredExecutionIntent, bool]: ...

    def complete(
        self,
        intent: TrainingExecutionIntent,
        job: TrainingJobView,
    ) -> StoredExecutionIntent: ...


def _intent_key(intent_id: str) -> str:
    return hashlib.sha256(intent_id.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError("execution-intent file is missing or unsafe")
    if path.stat().st_size > MAX_INTENT_JSON_BYTES:
        raise ValueError("execution-intent file exceeds the bounded size")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("execution-intent JSON is invalid") from exc


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


class FileExecutionIntentStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or artifact_root()).resolve()
        self.base = self.root / "execution-intents"
        self.lock_path = self.base / ".registry.lock"

    def _path(self, intent: TrainingExecutionIntent) -> Path:
        path = self.base / _intent_key(intent.intent_id)
        if self.root not in path.resolve(strict=False).parents:
            raise ValueError("execution-intent path escapes the artifact root")
        return path

    def _load_unlocked(
        self,
        intent: TrainingExecutionIntent,
    ) -> StoredExecutionIntent | None:
        path = self._path(intent)
        if not path.exists():
            return None
        entries = {item.name for item in path.iterdir()}
        if not entries.issubset({"claim.json", "result.json"}) or "claim.json" not in entries:
            raise ValueError("execution-intent record contains unexpected entries")
        claim = ExecutionIntentClaim.model_validate(_read_json(path / "claim.json"))
        validate_execution_intent_claim(claim)
        if claim.intent != intent:
            raise ExecutionIntentConflictError(
                "execution intent ID is already bound to different content"
            )
        result = None
        if "result.json" in entries:
            result = ExecutionIntentResult.model_validate(_read_json(path / "result.json"))
            validate_execution_intent_result(result, claim)
        return StoredExecutionIntent(claim=claim, result=result)

    def load(self, intent: TrainingExecutionIntent) -> StoredExecutionIntent | None:
        return self._load_unlocked(intent)

    def claim(
        self,
        intent: TrainingExecutionIntent,
        *,
        job_id: str,
        claimed_at_utc: str,
    ) -> tuple[StoredExecutionIntent, bool]:
        self.base.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = self._load_unlocked(intent)
            if existing is not None:
                return existing, False
            if _root_usage(self.root) >= artifact_limit_bytes():
                raise OSError("AQSE artifact root has reached its configured byte limit")
            claim = build_execution_intent_claim(
                intent,
                job_id=job_id,
                claimed_at_utc=claimed_at_utc,
            )
            final = self._path(intent)
            staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
            staging.mkdir()
            try:
                target = staging / "claim.json"
                target.write_bytes(
                    canonical_json_bytes(claim.model_dump(mode="json")) + b"\n"
                )
                target.chmod(0o444)
                os.replace(staging, final)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
            return StoredExecutionIntent(claim=claim, result=None), True

    def complete(
        self,
        intent: TrainingExecutionIntent,
        job: TrainingJobView,
    ) -> StoredExecutionIntent:
        self.base.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = self._load_unlocked(intent)
            if existing is None:
                raise ValueError("execution intent must be claimed before completion")
            result = build_execution_intent_result(existing.claim, job)
            if existing.result is not None:
                if existing.result != result:
                    raise ExecutionIntentConflictError(
                        "execution intent already has another terminal result"
                    )
                return existing
            final = self._path(intent) / "result.json"
            staging = self.base / f".result-{_intent_key(intent.intent_id)}-{uuid4().hex}.tmp"
            try:
                staging.write_bytes(
                    canonical_json_bytes(result.model_dump(mode="json")) + b"\n"
                )
                staging.chmod(0o444)
                os.replace(staging, final)
            finally:
                if staging.exists():
                    staging.unlink()
            return StoredExecutionIntent(claim=existing.claim, result=result)


class MemoryExecutionIntentStore:
    def __init__(self) -> None:
        self._records: dict[str, StoredExecutionIntent] = {}

    def load(self, intent: TrainingExecutionIntent) -> StoredExecutionIntent | None:
        existing = self._records.get(intent.intent_id)
        if existing is not None and existing.claim.intent != intent:
            raise ExecutionIntentConflictError(
                "execution intent ID is already bound to different content"
            )
        return existing

    def claim(
        self,
        intent: TrainingExecutionIntent,
        *,
        job_id: str,
        claimed_at_utc: str,
    ) -> tuple[StoredExecutionIntent, bool]:
        existing = self.load(intent)
        if existing is not None:
            return existing, False
        claim = build_execution_intent_claim(
            intent,
            job_id=job_id,
            claimed_at_utc=claimed_at_utc,
        )
        record = StoredExecutionIntent(claim=claim, result=None)
        self._records[intent.intent_id] = record
        return record, True

    def complete(
        self,
        intent: TrainingExecutionIntent,
        job: TrainingJobView,
    ) -> StoredExecutionIntent:
        existing = self.load(intent)
        if existing is None:
            raise ValueError("execution intent must be claimed before completion")
        result = build_execution_intent_result(existing.claim, job)
        if existing.result is not None and existing.result != result:
            raise ExecutionIntentConflictError(
                "execution intent already has another terminal result"
            )
        record = StoredExecutionIntent(claim=existing.claim, result=result)
        self._records[intent.intent_id] = record
        return record
