from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from app.quantum.user_pipeline.tqk8 import StateEngine
from app.training.assembly import LABEL_POLICY, TrainingInput
from app.training.canonical import array_identity, canonical_json_bytes, file_sha256
from app.training.encoding import PhaseDirectEncoder, input_contract_for
from app.training.evaluation import _circular_features
from app.training.evaluation_assembly import ComparisonInput
from app.training.evaluation_models import (
    ComparativeEvaluationArtifact,
    EvaluationProtocol,
    MethodName,
    MethodSelection,
    ModelSelectionFreeze,
)
from app.training.held_out_models import (
    FinalHeldOutArtifact,
    G5AuthorizationArtifact,
    HeldOutClassificationMetrics,
    HeldOutInputIdentity,
    HeldOutMethodResult,
    HeldOutMethodRuntime,
    HeldOutMetricInterval,
    LedgerEventIdentity,
    MethodComputeCost,
    SemanticTestLoad,
    TestBankArtifact,
    TestBankRowReference,
)
from app.training.models import (
    DatasetManifest,
    DatasetPartition,
    EpisodeLabel,
    TestAccessLedgerEntry,
)
from app.training.storage import ObservablePartition

ENTRY_COMMIT = "81045762d69c6a82c63e47196a2ae36899786233"
DATASET_ID = "aqse-development-064acca20fc788c6"
DATASET_DIGEST = "064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7"
ENCODER_ID = "aqse-encoder-f9cf4bc12a767410"
ENCODER_DIGEST = "f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51"
PROFILE_FINGERPRINT = "cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17"
TRAIN_BANK_ID = "aqse-train-bank-32b5f93897676535"
TRAIN_BANK_DIGEST = "32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd"
PROTOCOL_ID = "aqse-comparative-protocol-a671952c35f0ffe8"
PROTOCOL_DIGEST = "a671952c35f0ffe80707a4f6fed48ff3ff17c679a3392ce4c1fc697f087eff9d"
EVALUATION_ID = "aqse-comparative-evaluation-5b0aeed2cdd9aeba"
EVALUATION_DIGEST = "5b0aeed2cdd9aeba51bf99ef43ff304a2770c2c422b127d9e07ee66a4c3c4fb0"
FREEZE_ID = "aqse-model-selection-freeze-433ca4c0cae8fabf"
FREEZE_DIGEST = "433ca4c0cae8fabf06120f7e929c0e0471d338f48f120aad15f18611f7b4016f"
INITIAL_LEDGER_SHA256 = (
    "210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6"
)
ACTOR = "AQSE Milestone 1D.4b evaluator"
OBSERVATION_REASON = f"freeze={FREEZE_ID}; fixed-test-bank-and-observations"
LABEL_REASON = f"freeze={FREEZE_ID}; final-held-out-labels"
WINDOW_ORDINAL = 9
METHODS: tuple[MethodName, ...] = (
    "snr_threshold",
    "rbf_svc_circular",
    "fixed_theta_tqk",
    "gradient_tqk",
    "qng_tqk",
)
PROTECTED_HASHES = {
    "app/quantum/user_pipeline/tqk8.py": (
        "cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689"
    ),
    "app/quantum/user_pipeline/sampler_qng.py": (
        "7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6"
    ),
    "tests/quantum/test_tqk8.py": (
        "c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09"
    ),
    "../docs/notebooks/TQK8_walkthrough.ipynb": (
        "9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be"
    ),
}


@dataclass(frozen=True)
class HeldOutInput:
    raw: NDArray[np.float64]
    encoded: NDArray[np.float64]
    labels: NDArray[np.int8]
    identity: HeldOutInputIdentity


@dataclass(frozen=True)
class _PreparedMethod:
    selection: MethodSelection
    model: SVC | None
    theta: NDArray[np.float64] | None
    train_kernel: NDArray[np.float64] | None
    preparation_wall_time_ms: float


@dataclass(frozen=True)
class PreparedFrozenMethods:
    train_encoded: NDArray[np.float64]
    train_labels: NDArray[np.int8]
    rbf_mean: NDArray[np.float64]
    rbf_scale: NDArray[np.float64]
    methods: tuple[_PreparedMethod, ...]


@dataclass(frozen=True)
class HeldOutEvaluationOutcome:
    artifact: FinalHeldOutArtifact
    method_runtimes: tuple[HeldOutMethodRuntime, ...]
    total_wall_time_ms: float


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _scientific_payload(value: Any, *, id_field: str) -> dict[str, Any]:
    return value.model_dump(mode="json", exclude={id_field, "content_digest"})


def validate_g5_authorization(value: G5AuthorizationArtifact) -> None:
    expected = _digest(_scientific_payload(value, id_field="authorization_id"))
    if value.content_digest != expected:
        raise ValueError("G5 authorization digest is invalid")
    if value.authorization_id != f"aqse-g5-authorization-{expected[:16]}":
        raise ValueError("G5 authorization identity is invalid")
    if (
        value.entry_commit != ENTRY_COMMIT
        or value.dataset_id != DATASET_ID
        or value.dataset_digest != DATASET_DIGEST
        or value.encoder_id != ENCODER_ID
        or value.encoder_digest != ENCODER_DIGEST
        or value.train_bank_id != TRAIN_BANK_ID
        or value.train_bank_digest != TRAIN_BANK_DIGEST
        or value.protocol_id != PROTOCOL_ID
        or value.protocol_digest != PROTOCOL_DIGEST
        or value.evaluation_id != EVALUATION_ID
        or value.evaluation_digest != EVALUATION_DIGEST
        or value.freeze_id != FREEZE_ID
        or value.freeze_digest != FREEZE_DIGEST
        or value.initial_ledger_sha256 != INITIAL_LEDGER_SHA256
    ):
        raise ValueError("G5 authorization compatibility tuple is invalid")
    if tuple(item.reason for item in value.expected_semantic_loads) != (
        OBSERVATION_REASON,
        LABEL_REASON,
    ):
        raise ValueError("G5 authorization reasons differ from the frozen access plan")


def build_g5_authorization(
    protocol: EvaluationProtocol,
    evaluation: ComparativeEvaluationArtifact,
    freeze: ModelSelectionFreeze,
    *,
    initial_ledger_sha256: str,
) -> G5AuthorizationArtifact:
    if (
        protocol.protocol_id != PROTOCOL_ID
        or protocol.content_digest != PROTOCOL_DIGEST
        or evaluation.evaluation_id != EVALUATION_ID
        or evaluation.content_digest != EVALUATION_DIGEST
        or freeze.freeze_id != FREEZE_ID
        or freeze.content_digest != FREEZE_DIGEST
        or initial_ledger_sha256 != INITIAL_LEDGER_SHA256
        or evaluation.selections != freeze.selections
    ):
        raise ValueError("approved 1D.4a freeze identities are incompatible")
    scientific = {
        "schema_version": "aqse.g5-authorization.v1",
        "milestone": "1D.4b",
        "gate_state": "G5 closed before TEST opening",
        "entry_commit": ENTRY_COMMIT,
        "purpose": "milestone-1d4b-final-held-out-evaluation",
        "dataset_id": DATASET_ID,
        "dataset_digest": DATASET_DIGEST,
        "encoder_id": ENCODER_ID,
        "encoder_digest": ENCODER_DIGEST,
        "train_bank_id": TRAIN_BANK_ID,
        "train_bank_digest": TRAIN_BANK_DIGEST,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.content_digest,
        "evaluation_id": evaluation.evaluation_id,
        "evaluation_digest": evaluation.content_digest,
        "freeze_id": freeze.freeze_id,
        "freeze_digest": freeze.content_digest,
        "initial_ledger_sha256": initial_ledger_sha256,
        "initial_ledger_event_count": 1,
        "test_lineage_count": 24,
        "window_ordinal": WINDOW_ORDINAL,
        "cohort_rule": "all-TEST-lineages-fixed-window-ordinal-9",
        "selection_independent_of_test_semantics": True,
        "no_fallback": True,
        "common_eligible_cohort": True,
        "selections": [item.model_dump(mode="json") for item in freeze.selections],
        "metrics": freeze.final_evaluation_procedure.metrics,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 1_001_010,
        "bootstrap_policy": "stratified-independent-test-lineage-percentile-95",
        "expected_semantic_loads": [
            SemanticTestLoad(
                sequence=1,
                operation="load_observations(TEST)",
                reason=OBSERVATION_REASON,
            ).model_dump(mode="json"),
            SemanticTestLoad(
                sequence=2,
                operation="load_labels(TEST)",
                reason=LABEL_REASON,
            ).model_dump(mode="json"),
        ],
        "generation_plan_access_allowed": False,
        "no_refit": True,
        "no_training_rerun": True,
        "no_winner_predeclared": True,
        "contains_test_values": False,
    }
    digest = _digest(scientific)
    value = G5AuthorizationArtifact.model_validate(
        {
            **scientific,
            "authorization_id": f"aqse-g5-authorization-{digest[:16]}",
            "content_digest": digest,
        }
    )
    validate_g5_authorization(value)
    return value


def validate_test_bank(bank: TestBankArtifact) -> None:
    expected = _digest(_scientific_payload(bank, id_field="artifact_id"))
    if bank.content_digest != expected:
        raise ValueError("TEST bank digest is invalid")
    if bank.artifact_id != f"aqse-test-bank-{expected[:16]}":
        raise ValueError("TEST bank identity is invalid")


def build_test_bank(
    observations: ObservablePartition,
    manifest: DatasetManifest,
    authorization: G5AuthorizationArtifact,
) -> TestBankArtifact:
    validate_g5_authorization(authorization)
    if (
        observations.partition is not DatasetPartition.TEST
        or observations.dataset_id != DATASET_ID
        or manifest.dataset_id != DATASET_ID
        or manifest.scientific_digest != DATASET_DIGEST
        or len(observations.episode_ids) != 24
        or len(set(observations.lineage_ids)) != 24
        or observations.valid_mask.shape != (24, 19)
    ):
        raise ValueError("TEST observations are incompatible with the fixed bank rule")
    index_by_episode = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    rows: list[TestBankRowReference] = []
    for episode_id in sorted(observations.episode_ids):
        index = index_by_episode[episode_id]
        window = observations.windows[index][WINDOW_ORDINAL]
        eligible = bool(observations.valid_mask[index, WINDOW_ORDINAL])
        rows.append(
            TestBankRowReference(
                episode_id=episode_id,
                lineage_id=observations.lineage_ids[index],
                window_id=window.window_id,
                valid_for_quantum=eligible,
                abstention_reason=(
                    None if eligible else "fixed_ordinal_9_not_valid_for_quantum"
                ),
            )
        )
    eligible_count = sum(item.valid_for_quantum for item in rows)
    scientific = {
        "schema_version": "aqse.test-bank.v1",
        "authorization_id": authorization.authorization_id,
        "authorization_digest": authorization.content_digest,
        "source_dataset_id": DATASET_ID,
        "source_dataset_digest": DATASET_DIGEST,
        "encoder_id": ENCODER_ID,
        "encoder_digest": ENCODER_DIGEST,
        "source_partition": "test",
        "selection_policy": "all-TEST-lineages-fixed-window-ordinal-9",
        "selection_independent_of_labels": True,
        "no_fallback": True,
        "window_ordinal": WINDOW_ORDINAL,
        "row_count": 24,
        "lineage_count": 24,
        "eligible_count": eligible_count,
        "abstained_count": 24 - eligible_count,
        "rows": [item.model_dump(mode="json") for item in rows],
    }
    digest = _digest(scientific)
    bank = TestBankArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-test-bank-{digest[:16]}",
            "content_digest": digest,
        }
    )
    validate_test_bank(bank)
    return bank


def _episode_observation_digests(
    observations: ObservablePartition,
) -> dict[str, str]:
    result: dict[str, str] = {}
    arrays = (
        observations.time_s,
        observations.measured_field_T,
        observations.temperature_K,
        observations.saturation_mask,
        observations.features,
        observations.valid_mask,
    )
    for index, episode_id in enumerate(observations.episode_ids):
        payload = {"arrays": [array_identity(np.asarray(value[index])) for value in arrays]}
        result[episode_id] = _digest(payload)
    return result


def _freeze_array(value: NDArray[Any], dtype: Any) -> NDArray[Any]:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def assemble_held_out_input(
    *,
    observations: ObservablePartition,
    labels: tuple[EpisodeLabel, ...],
    bank: TestBankArtifact,
    encoder: PhaseDirectEncoder,
    training_input: TrainingInput,
    comparison: ComparisonInput,
    reference_observations: tuple[ObservablePartition, ObservablePartition],
) -> HeldOutInput:
    validate_test_bank(bank)
    if bank.authorization_id == "" or bank.source_dataset_id != observations.dataset_id:
        raise ValueError("TEST bank and observations are incompatible")
    label_by_episode = {item.episode_id: item for item in labels}
    if len(label_by_episode) != 24 or len(labels) != 24:
        raise ValueError("held-out evaluation requires all 24 separate TEST labels")
    observation_index = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    all_raw: list[NDArray[np.float64]] = []
    eligible_raw: list[NDArray[np.float64]] = []
    all_labels: list[int] = []
    eligible_labels: list[int] = []
    eligible_lineages: list[str] = []
    eligible_windows: list[str] = []
    abstained_lineages: list[str] = []
    abstained_windows: list[str] = []
    for reference in bank.rows:
        index = observation_index.get(reference.episode_id)
        label = label_by_episode.get(reference.episode_id)
        if index is None or label is None:
            raise ValueError("TEST bank row is absent from observations or labels")
        if (
            observations.lineage_ids[index] != reference.lineage_id
            or label.lineage_id != reference.lineage_id
            or label.origin_policy != LABEL_POLICY
            or observations.windows[index][WINDOW_ORDINAL].window_id
            != reference.window_id
        ):
            raise ValueError("TEST observation, label and bank identities differ")
        raw = np.asarray(observations.features[index, WINDOW_ORDINAL], dtype=np.float64)
        all_raw.append(raw)
        all_labels.append(label.target)
        if reference.valid_for_quantum:
            eligible_raw.append(raw)
            eligible_labels.append(label.target)
            eligible_lineages.append(reference.lineage_id)
            eligible_windows.append(reference.window_id)
        else:
            abstained_lineages.append(reference.lineage_id)
            abstained_windows.append(reference.window_id)

    if set(bank.rows[index].lineage_id for index in range(24)) & (
        set(comparison.identity.train_lineage_ids)
        | set(comparison.identity.validation_lineage_ids)
    ):
        raise ValueError("TEST lineages overlap TRAIN or VALIDATION")
    test_observation_digests = _episode_observation_digests(observations)
    reference_digests = {
        digest
        for partition in reference_observations
        for digest in _episode_observation_digests(partition).values()
    }
    if set(test_observation_digests.values()) & reference_digests:
        raise ValueError("duplicate observation content crosses dataset partitions")

    raw_matrix = np.asarray(all_raw, dtype=np.float64)
    eligible_raw_matrix = np.asarray(eligible_raw, dtype=np.float64).reshape(-1, 8)
    encoded = np.asarray(
        encoder.transform(eligible_raw_matrix, context=input_contract_for(encoder.artifact)),
        dtype=np.float64,
    )
    all_label_array = np.asarray(all_labels, dtype=np.int8)
    eligible_label_array = np.asarray(eligible_labels, dtype=np.int8)
    if (
        raw_matrix.shape != (24, 8)
        or encoded.shape != (bank.eligible_count, 8)
        or eligible_label_array.shape != (bank.eligible_count,)
        or training_input.X_train.shape != (32, 8)
    ):
        raise ValueError("held-out matrix shapes are incompatible")
    observation_content_digest = _digest(
        {
            "ordered_episode_digests": [
                test_observation_digests[item.episode_id] for item in bank.rows
            ]
        }
    )
    identity_payload = {
        "schema_version": "aqse.held-out-input.v1",
        "train_encoded_identity": array_identity(training_input.X_train),
        "train_label_identity": array_identity(training_input.y_train),
        "test_observation_tensor_identity": array_identity(observations.features),
        "test_observation_content_digest": observation_content_digest,
        "test_raw_row_identity": array_identity(raw_matrix),
        "test_encoded_row_identity": array_identity(encoded),
        "test_label_identity": array_identity(all_label_array),
        "eligible_test_label_identity": array_identity(eligible_label_array),
        "eligible_lineage_ids": eligible_lineages,
        "eligible_window_ids": eligible_windows,
        "abstained_lineage_ids": abstained_lineages,
        "abstained_window_ids": abstained_windows,
        "quantum_test_train_kernel_shape": (bank.eligible_count, 32),
    }
    digest = _digest(identity_payload)
    identity = HeldOutInputIdentity.model_validate(
        {
            **identity_payload,
            "fingerprint_id": f"aqse-held-out-input-{digest[:16]}",
            "content_digest": digest,
        }
    )
    return HeldOutInput(
        raw=_freeze_array(eligible_raw_matrix, np.float64),
        encoded=_freeze_array(encoded, np.float64),
        labels=_freeze_array(eligible_label_array, np.int8),
        identity=identity,
    )


def _selection_map(freeze: ModelSelectionFreeze) -> dict[MethodName, MethodSelection]:
    if tuple(item.method for item in freeze.selections) != METHODS:
        raise ValueError("model-selection freeze method order is incompatible")
    return {item.method: item for item in freeze.selections}


def _validate_frozen_parameters(selections: dict[MethodName, MethodSelection]) -> None:
    snr = selections["snr_threshold"]
    rbf = selections["rbf_svc_circular"]
    quantum = [selections[name] for name in METHODS[2:]]
    if snr.selected_snr_threshold != 11.433898484324114:
        raise ValueError("frozen SNR threshold changed")
    if rbf.selected_svc_c != 10.0 or rbf.selected_rbf_gamma != 0.01:
        raise ValueError("frozen RBF hyperparameters changed")
    if any(
        item.selected_seed != 1_001_008
        or item.selected_checkpoint_index != 0
        or item.selected_svc_c != 1.0
        or item.selected_theta is None
        for item in quantum
    ):
        raise ValueError("frozen quantum selection changed")
    if quantum[0].selected_theta != quantum[1].selected_theta:
        raise ValueError("fixed and gradient frozen theta differ")
    if quantum[0].selected_theta != quantum[2].selected_theta:
        raise ValueError("fixed and QNG frozen theta differ")


def prepare_frozen_methods(
    comparison: ComparisonInput,
    evaluation: ComparativeEvaluationArtifact,
    freeze: ModelSelectionFreeze,
) -> PreparedFrozenMethods:
    if (
        evaluation.evaluation_id != EVALUATION_ID
        or evaluation.content_digest != EVALUATION_DIGEST
        or freeze.freeze_id != FREEZE_ID
        or freeze.content_digest != FREEZE_DIGEST
        or evaluation.selections != freeze.selections
        or comparison.identity != evaluation.comparison_input
    ):
        raise ValueError("held-out reconstruction does not match the 1D.4a freeze")
    selections = _selection_map(freeze)
    _validate_frozen_parameters(selections)
    train_circular = _circular_features(comparison.X_train_raw)
    snapshot = evaluation.classical_preprocessing
    mean = np.asarray(snapshot.fitted_mean, dtype=np.float64)
    scale = np.asarray(snapshot.fitted_scale, dtype=np.float64)
    observed_mean = train_circular.mean(axis=0)
    observed_scale = train_circular.std(axis=0)
    observed_scale = np.where(observed_scale > 0.0, observed_scale, 1.0)
    if not np.allclose(mean, observed_mean, rtol=0.0, atol=1.0e-14):
        raise ValueError("frozen RBF mean differs from TRAIN-only reconstruction")
    if not np.allclose(scale, observed_scale, rtol=0.0, atol=1.0e-14):
        raise ValueError("frozen RBF scale differs from TRAIN-only reconstruction")
    transformed_train = (train_circular - mean) / scale

    prepared: list[_PreparedMethod] = []
    for method in METHODS:
        selection = selections[method]
        started = perf_counter()
        if method == "snr_threshold":
            model = None
            theta = None
            train_kernel = None
        elif method == "rbf_svc_circular":
            model = SVC(
                kernel="rbf",
                C=selection.selected_svc_c,
                gamma=selection.selected_rbf_gamma,
            ).fit(transformed_train, comparison.y_train)
            theta = None
            train_kernel = None
        else:
            theta = np.asarray(selection.selected_theta, dtype=np.float64)
            train_kernel = np.asarray(
                StateEngine("numpy").gram(comparison.X_train_encoded, theta),
                dtype=np.float64,
            )
            if train_kernel.shape != (32, 32):
                raise ValueError("frozen quantum TRAIN kernel shape is invalid")
            model = SVC(kernel="precomputed", C=selection.selected_svc_c).fit(
                train_kernel,
                comparison.y_train,
            )
        prepared.append(
            _PreparedMethod(
                selection=selection,
                model=model,
                theta=theta,
                train_kernel=train_kernel,
                preparation_wall_time_ms=(perf_counter() - started) * 1_000.0,
            )
        )
    return PreparedFrozenMethods(
        train_encoded=_freeze_array(comparison.X_train_encoded, np.float64),
        train_labels=_freeze_array(comparison.y_train, np.int8),
        rbf_mean=_freeze_array(mean, np.float64),
        rbf_scale=_freeze_array(scale, np.float64),
        methods=tuple(prepared),
    )


def _held_out_metrics(
    labels: NDArray[np.int8],
    predictions: NDArray[np.int8],
    scores: NDArray[np.float64],
) -> HeldOutClassificationMetrics:
    y = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(predictions, dtype=np.int8)
    decision = np.asarray(scores, dtype=np.float64)
    if y.shape != predicted.shape or y.shape != decision.shape or y.ndim != 1:
        raise ValueError("held-out classification arrays are not aligned")
    matrix = confusion_matrix(y, predicted, labels=[-1, 1])
    negative_support = int(np.count_nonzero(y == -1))
    positive_support = int(np.count_nonzero(y == 1))
    both_classes = negative_support > 0 and positive_support > 0
    recalls = recall_score(
        y,
        predicted,
        labels=[-1, 1],
        average=None,
        zero_division=0.0,
    )
    return HeldOutClassificationMetrics(
        balanced_accuracy=(
            float(balanced_accuracy_score(y, predicted)) if both_classes else None
        ),
        macro_f1=(
            float(f1_score(y, predicted, labels=[-1, 1], average="macro"))
            if both_classes
            else None
        ),
        roc_auc=float(roc_auc_score(y, decision)) if both_classes else None,
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
        negative_support=negative_support,
        positive_support=positive_support,
        negative_recall=float(recalls[0]) if negative_support else None,
        positive_recall=float(recalls[1]) if positive_support else None,
    )


def _bootstrap(
    labels: NDArray[np.int8],
    predictions: NDArray[np.int8],
    metrics: HeldOutClassificationMetrics,
) -> tuple[HeldOutMetricInterval, HeldOutMetricInterval]:
    negative = np.flatnonzero(labels == -1)
    positive = np.flatnonzero(labels == 1)
    if len(negative) == 0 or len(positive) == 0:
        return (
            HeldOutMetricInterval(metric="balanced_accuracy"),
            HeldOutMetricInterval(metric="macro_f1"),
        )
    rng = np.random.default_rng(1_001_010)
    balanced: list[float] = []
    macro: list[float] = []
    for _ in range(2000):
        indices = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        observed = labels[indices]
        predicted = predictions[indices]
        balanced.append(float(balanced_accuracy_score(observed, predicted)))
        macro.append(
            float(f1_score(observed, predicted, labels=[-1, 1], average="macro"))
        )

    def interval(
        metric: str,
        values: list[float],
        point: float | None,
    ) -> HeldOutMetricInterval:
        lower, upper = np.quantile(np.asarray(values), (0.025, 0.975))
        return HeldOutMetricInterval(
            metric=metric,
            lower=float(lower),
            point=point,
            upper=float(upper),
        )

    return (
        interval("balanced_accuracy", balanced, metrics.balanced_accuracy),
        interval("macro_f1", macro, metrics.macro_f1),
    )


def _protected_hashes(backend_root: Path) -> dict[str, str]:
    observed = {
        relative: file_sha256((backend_root / relative).resolve())
        for relative in PROTECTED_HASHES
    }
    if observed != PROTECTED_HASHES:
        raise ValueError("protected scientific source identity changed")
    return observed


def _ledger_identities(
    entries: tuple[TestAccessLedgerEntry, ...],
) -> tuple[LedgerEventIdentity, LedgerEventIdentity, LedgerEventIdentity]:
    if len(entries) != 3:
        raise ValueError("final TEST ledger must contain exactly three entries")
    return tuple(  # type: ignore[return-value]
        LedgerEventIdentity(
            sequence=item.sequence,
            event=item.event,
            actor=item.actor,
            reason=item.reason,
            fixture_only=item.fixture_only,
            entry_sha256=item.entry_sha256,
        )
        for item in entries
    )


def validate_final_held_out(value: FinalHeldOutArtifact) -> None:
    expected = _digest(_scientific_payload(value, id_field="artifact_id"))
    if value.content_digest != expected:
        raise ValueError("final held-out artifact digest is invalid")
    if value.artifact_id != f"aqse-final-held-out-{expected[:16]}":
        raise ValueError("final held-out artifact identity is invalid")
    if (
        value.authorization_digest == ""
        or value.dataset_id != DATASET_ID
        or value.dataset_digest != DATASET_DIGEST
        or value.protocol_id != PROTOCOL_ID
        or value.protocol_digest != PROTOCOL_DIGEST
        or value.evaluation_id != EVALUATION_ID
        or value.evaluation_digest != EVALUATION_DIGEST
        or value.freeze_id != FREEZE_ID
        or value.freeze_digest != FREEZE_DIGEST
        or value.encoder_id != ENCODER_ID
        or value.encoder_digest != ENCODER_DIGEST
        or value.train_bank_id != TRAIN_BANK_ID
        or value.train_bank_digest != TRAIN_BANK_DIGEST
        or value.initial_ledger_sha256 != INITIAL_LEDGER_SHA256
        or value.protected_source_hashes != PROTECTED_HASHES
    ):
        raise ValueError("final held-out compatibility tuple is invalid")
    if tuple(item.reason for item in value.ledger_events[1:]) != (
        OBSERVATION_REASON,
        LABEL_REASON,
    ):
        raise ValueError("final TEST ledger reasons differ from the authorized plan")


def evaluate_held_out(
    *,
    prepared: PreparedFrozenMethods,
    held_out: HeldOutInput,
    authorization: G5AuthorizationArtifact,
    protocol: EvaluationProtocol,
    evaluation: ComparativeEvaluationArtifact,
    freeze: ModelSelectionFreeze,
    bank: TestBankArtifact,
    ledger_entries: tuple[TestAccessLedgerEntry, ...],
    final_ledger_sha256: str,
    backend_root: Path,
) -> HeldOutEvaluationOutcome:
    validate_g5_authorization(authorization)
    validate_test_bank(bank)
    if (
        protocol.protocol_id != PROTOCOL_ID
        or evaluation.evaluation_id != EVALUATION_ID
        or freeze.freeze_id != FREEZE_ID
        or authorization.freeze_digest != freeze.content_digest
        or bank.authorization_id != authorization.authorization_id
        or bank.authorization_digest != authorization.content_digest
    ):
        raise ValueError("final evaluation artifact graph is incompatible")
    if tuple(item.selection for item in prepared.methods) != freeze.selections:
        raise ValueError("prepared methods differ from the selection freeze")

    started = perf_counter()
    method_results: list[HeldOutMethodResult] = []
    method_runtimes: list[HeldOutMethodRuntime] = []
    quantum_outputs: list[
        tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.float64]]
    ] = []
    for prepared_method in prepared.methods:
        method_started = perf_counter()
        method = prepared_method.selection.method
        if method == "snr_threshold":
            threshold = prepared_method.selection.selected_snr_threshold
            if threshold is None:
                raise ValueError("frozen SNR threshold is absent")
            scores = threshold - held_out.raw[:, 5]
            predictions = np.where(held_out.raw[:, 5] <= threshold, 1, -1).astype(
                np.int8
            )
            train_coordinates = 0
            test_coordinates = 0
            fit_count = 0
        elif method == "rbf_svc_circular":
            if prepared_method.model is None:
                raise ValueError("frozen RBF model is absent")
            transformed = (
                _circular_features(held_out.raw) - prepared.rbf_mean
            ) / prepared.rbf_scale
            predictions = np.asarray(
                prepared_method.model.predict(transformed),
                dtype=np.int8,
            )
            scores = np.asarray(
                prepared_method.model.decision_function(transformed),
                dtype=np.float64,
            )
            train_coordinates = 0
            test_coordinates = 0
            fit_count = 1
        else:
            if prepared_method.model is None or prepared_method.theta is None:
                raise ValueError("frozen quantum reconstruction is incomplete")
            test_kernel = np.asarray(
                StateEngine("numpy").gram(
                    held_out.encoded,
                    prepared_method.theta,
                    prepared.train_encoded,
                ),
                dtype=np.float64,
            )
            if test_kernel.shape != (bank.eligible_count, 32):
                raise ValueError("quantum TEST/TRAIN kernel orientation is invalid")
            predictions = np.asarray(
                prepared_method.model.predict(test_kernel),
                dtype=np.int8,
            )
            scores = np.asarray(
                prepared_method.model.decision_function(test_kernel),
                dtype=np.float64,
            )
            quantum_outputs.append((test_kernel, predictions, scores))
            train_coordinates = 32 * 32
            test_coordinates = bank.eligible_count * 32
            fit_count = 1
        metrics = _held_out_metrics(held_out.labels, predictions, scores)
        intervals = _bootstrap(held_out.labels, predictions, metrics)
        method_results.append(
            HeldOutMethodResult(
                method=method,
                frozen_selection=prepared_method.selection,
                metrics=metrics,
                confidence_intervals=intervals,
                eligible_count=bank.eligible_count,
                abstained_count=bank.abstained_count,
                coverage=bank.eligible_count / 24.0,
                prediction_digest=array_identity(predictions)["content_sha256"],
                score_digest=array_identity(scores)["content_sha256"],
                compute_cost=MethodComputeCost(
                    test_rows=bank.eligible_count,
                    model_fit_count=fit_count,
                    prediction_count=bank.eligible_count,
                    train_kernel_coordinates=train_coordinates,
                    test_train_kernel_coordinates=test_coordinates,
                ),
            )
        )
        method_runtimes.append(
            HeldOutMethodRuntime(
                method=method,
                wall_time_ms=(perf_counter() - method_started) * 1_000.0
                + prepared_method.preparation_wall_time_ms,
            )
        )

    if len(quantum_outputs) != 3:
        raise ValueError("exactly three frozen quantum results are required")
    first_kernel, first_prediction, first_score = quantum_outputs[0]
    for kernel, prediction, score in quantum_outputs[1:]:
        if not np.allclose(kernel, first_kernel, rtol=0.0, atol=1.0e-12):
            raise ValueError("fixed/GD/QNG TEST kernels differ")
        if not np.array_equal(prediction, first_prediction):
            raise ValueError("fixed/GD/QNG TEST predictions differ")
        if not np.allclose(score, first_score, rtol=0.0, atol=1.0e-12):
            raise ValueError("fixed/GD/QNG TEST scores differ")

    scientific = {
        "schema_version": "aqse.final-held-out-evaluation.v1",
        "milestone": "1D.4b",
        "authorization_id": authorization.authorization_id,
        "authorization_digest": authorization.content_digest,
        "dataset_id": DATASET_ID,
        "dataset_digest": DATASET_DIGEST,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.content_digest,
        "evaluation_id": evaluation.evaluation_id,
        "evaluation_digest": evaluation.content_digest,
        "freeze_id": freeze.freeze_id,
        "freeze_digest": freeze.content_digest,
        "encoder_id": ENCODER_ID,
        "encoder_digest": ENCODER_DIGEST,
        "train_bank_id": TRAIN_BANK_ID,
        "train_bank_digest": TRAIN_BANK_DIGEST,
        "test_bank_id": bank.artifact_id,
        "test_bank_digest": bank.content_digest,
        "held_out_input": held_out.identity.model_dump(mode="json"),
        "total_test_lineages": 24,
        "eligible_count": bank.eligible_count,
        "abstained_count": bank.abstained_count,
        "coverage": bank.eligible_count / 24.0,
        "selections": [item.model_dump(mode="json") for item in freeze.selections],
        "results": [item.model_dump(mode="json") for item in method_results],
        "initial_ledger_sha256": INITIAL_LEDGER_SHA256,
        "final_ledger_sha256": final_ledger_sha256,
        "final_ledger_event_count": 3,
        "ledger_events": [
            item.model_dump(mode="json") for item in _ledger_identities(ledger_entries)
        ],
        "semantic_test_load_count": 2,
        "generation_plan_access_count": 0,
        "no_winner_predeclared": True,
        "test_results_not_used_for_model_selection": True,
        "no_test_refit": True,
        "fixed_gradient_qng_identical": True,
        "protected_source_hashes": _protected_hashes(backend_root),
        "limitations": (
            "This simulated task is directly SNR-separable and is not representative of broad sensor diagnosis.",
            "Exact NumPy statevectors are engineering simulation, not physical-QPU execution.",
            "Held-out results do not establish quantum advantage and cannot change the frozen selections.",
        ),
    }
    digest = _digest(scientific)
    artifact = FinalHeldOutArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-final-held-out-{digest[:16]}",
            "content_digest": digest,
        }
    )
    validate_final_held_out(artifact)
    return HeldOutEvaluationOutcome(
        artifact=artifact,
        method_runtimes=tuple(method_runtimes),
        total_wall_time_ms=(perf_counter() - started) * 1_000.0
        + sum(item.preparation_wall_time_ms for item in prepared.methods),
    )
