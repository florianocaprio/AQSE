from __future__ import annotations

import hashlib

from app.training.canonical import canonical_json_bytes
from app.training.evaluation_models import EvaluationProtocol

DATASET_DIGEST = "064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7"
PROFILE_FINGERPRINT = "cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17"
ENCODER_DIGEST = "f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51"
TRAIN_BANK_DIGEST = "32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd"
VALIDATION_BANK_DIGEST = (
    "105467fa5a4a5cc7c867f7f77d92eaa61d231949e7c2356e8f3817c541c0f047"
)
SEED_SCHEDULE = (1_001_005, 1_001_006, 1_001_007, 1_001_008, 1_001_009)
METHODS = (
    "snr_threshold",
    "rbf_svc_circular",
    "fixed_theta_tqk",
    "gradient_tqk",
    "qng_tqk",
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_evaluation_protocol() -> EvaluationProtocol:
    scientific = {
        "schema_version": "aqse.comparative-protocol.v1",
        "milestone": "1D.4a",
        "dataset_id": "aqse-development-064acca20fc788c6",
        "dataset_digest": DATASET_DIGEST,
        "feature_profile_fingerprint": PROFILE_FINGERPRINT,
        "encoder_id": "aqse-encoder-f9cf4bc12a767410",
        "encoder_digest": ENCODER_DIGEST,
        "train_bank_id": "aqse-train-bank-32b5f93897676535",
        "train_bank_digest": TRAIN_BANK_DIGEST,
        "validation_bank_id": "aqse-validation-bank-105467fa5a4a5cc7",
        "validation_bank_digest": VALIDATION_BANK_DIGEST,
        "seed_schedule": SEED_SCHEDULE,
        "methods": METHODS,
        "snr_rule": "predict +1 when snr <= TRAIN-selected threshold",
        "snr_training_tie_break": (
            "max TRAIN balanced_accuracy then macro_f1 then lowest threshold"
        ),
        "rbf_c_grid": (0.1, 1.0, 10.0),
        "rbf_gamma_grid": (0.01, 0.1, 1.0),
        "quantum_svc_c_grid": (0.1, 1.0, 10.0),
        "maximum_updates": 10,
        "qng_learning_rate": 0.2,
        "qng_damping": 0.001,
        "qng_maximum_step_norm": 0.4,
        "qng_armijo": "protected-backtracking-unchanged",
        "gradient_learning_rate": 0.2,
        "gradient_maximum_step_norm": 0.4,
        "gradient_line_search": "none",
        "selection_primary": "validation_balanced_accuracy",
        "selection_secondary": "validation_macro_f1",
        "selection_tie_break": (
            "earliest_checkpoint_then_lowest_C_then_lowest_gamma_then_lowest_seed"
        ),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 1_001_010,
        "bootstrap_policy": "stratified-independent-lineage-percentile-95",
        "validation_role": "selection-only",
        "final_fit_policy": "TRAIN-only; no refit after VALIDATION selection",
        "future_test_policy": (
            "separate-G5-authorization; one fixed central window per TEST lineage"
        ),
        "test_state_required": "sealed",
        "create_test_bank": False,
    }
    digest = _digest(scientific)
    return EvaluationProtocol.model_validate(
        {
            **scientific,
            "protocol_id": f"aqse-comparative-protocol-{digest[:16]}",
            "content_digest": digest,
        }
    )


def validate_evaluation_protocol(protocol: EvaluationProtocol) -> None:
    scientific = protocol.model_dump(
        mode="json",
        exclude={"protocol_id", "content_digest"},
    )
    expected = _digest(scientific)
    if protocol.content_digest != expected:
        raise ValueError("comparative protocol digest is invalid")
    if protocol.protocol_id != f"aqse-comparative-protocol-{expected[:16]}":
        raise ValueError("comparative protocol identity is invalid")
