from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel
from app.training.run_models import TrainingJobState, TrainingJobView


class TrainingExecutionIntent(FrozenModel):
    schema_version: Literal["aqse.qng-execution-intent.v1"] = (
        "aqse.qng-execution-intent.v1"
    )
    intent_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._:-]*$",
    )
    purpose: Literal["candidate_training"] = "candidate_training"


class EngineeringBenchmarkIntent(FrozenModel):
    schema_version: Literal["aqse.qng-execution-intent.v1"] = (
        "aqse.qng-execution-intent.v1"
    )
    intent_id: Literal["aqse-1d3-g4-real-load-benchmark"] = (
        "aqse-1d3-g4-real-load-benchmark"
    )
    purpose: Literal["engineering_benchmark"] = "engineering_benchmark"


class ExecutionIntentClaim(FrozenModel):
    schema_version: Literal["aqse.qng-execution-claim.v1"] = (
        "aqse.qng-execution-claim.v1"
    )
    claim_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    intent: TrainingExecutionIntent
    intent_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    job_id: str
    claimed_at_utc: str


class ExecutionIntentResult(FrozenModel):
    schema_version: Literal["aqse.qng-execution-result.v1"] = (
        "aqse.qng-execution-result.v1"
    )
    result_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    intent_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    job: TrainingJobView


@dataclass(frozen=True)
class StoredExecutionIntent:
    claim: ExecutionIntentClaim
    result: ExecutionIntentResult | None


@dataclass(frozen=True)
class TrainingJobStart:
    job: TrainingJobView
    created: bool


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def execution_intent_digest(intent: TrainingExecutionIntent) -> str:
    return _digest(intent.model_dump(mode="json"))


def build_execution_intent_claim(
    intent: TrainingExecutionIntent,
    *,
    job_id: str,
    claimed_at_utc: str,
) -> ExecutionIntentClaim:
    intent_digest = execution_intent_digest(intent)
    payload = {
        "schema_version": "aqse.qng-execution-claim.v1",
        "intent": intent.model_dump(mode="json"),
        "intent_digest": intent_digest,
        "job_id": job_id,
        "claimed_at_utc": claimed_at_utc,
    }
    return ExecutionIntentClaim(**payload, claim_digest=_digest(payload))


def validate_execution_intent_claim(claim: ExecutionIntentClaim) -> None:
    if claim.intent_digest != execution_intent_digest(claim.intent):
        raise ValueError("execution-intent digest is invalid")
    payload = claim.model_dump(mode="json", exclude={"claim_digest"})
    if claim.claim_digest != _digest(payload):
        raise ValueError("execution-intent claim digest is invalid")


def build_execution_intent_result(
    claim: ExecutionIntentClaim,
    job: TrainingJobView,
) -> ExecutionIntentResult:
    if job.job_id != claim.job_id:
        raise ValueError("execution-intent result job identity is inconsistent")
    if job.state not in {
        TrainingJobState.COMPLETED,
        TrainingJobState.CANCELLED,
        TrainingJobState.FAILED,
    }:
        raise ValueError("execution-intent result must be terminal")
    payload = {
        "schema_version": "aqse.qng-execution-result.v1",
        "intent_digest": claim.intent_digest,
        "job": job.model_dump(mode="json"),
    }
    return ExecutionIntentResult(**payload, result_digest=_digest(payload))


def validate_execution_intent_result(
    result: ExecutionIntentResult,
    claim: ExecutionIntentClaim,
) -> None:
    if result.intent_digest != claim.intent_digest or result.job.job_id != claim.job_id:
        raise ValueError("execution-intent result is incompatible with its claim")
    payload = result.model_dump(mode="json", exclude={"result_digest"})
    if result.result_digest != _digest(payload):
        raise ValueError("execution-intent result digest is invalid")
