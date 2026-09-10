from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MLP_MODEL_ID = "aqse.classical.mlp-32x16-tanh-lbfgs.v1"
UNCERTAIN_TOP_SCORE_THRESHOLD = 0.70
UNCERTAIN_MARGIN_THRESHOLD = 0.15

ModelRole = Literal["afse_classifier", "raw_feature_baseline"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class StandardizerArtifact(FrozenModel):
    schema_version: Literal["aqse.classical.standardizer.v1"] = (
        "aqse.classical.standardizer.v1"
    )
    fitted_on_partition: Literal["TRAIN"] = "TRAIN"
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @model_validator(mode="after")
    def validate_statistics(self) -> StandardizerArtifact:
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("standardizer mean and scale must have equal non-zero size")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("standardizer scales must be positive")
        return self


class MLPClassifierArtifact(FrozenModel):
    """JSON-safe parameters for the frozen compact sklearn MLP contract."""

    schema_version: Literal["aqse.classical-mlp-artifact.v1"] = (
        "aqse.classical-mlp-artifact.v1"
    )
    artifact_id: str = Field(pattern=r"^aqse-mlp-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: Literal["aqse.classical.mlp-32x16-tanh-lbfgs.v1"] = MLP_MODEL_ID
    model_role: ModelRole
    task_id: str
    input_space_id: str
    fitted_on_partition: Literal["TRAIN"] = "TRAIN"
    fitted_on_dataset_id: str
    fitted_on_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_sample_ids: tuple[str, ...]
    feature_count: int = Field(gt=0)
    feature_order: tuple[str, ...]
    classes: tuple[str, ...] = Field(min_length=2)
    standardizer: StandardizerArtifact
    hidden_layer_sizes: tuple[Literal[32], Literal[16]] = (32, 16)
    activation: Literal["tanh"] = "tanh"
    output_activation: Literal["logistic", "softmax"]
    solver: Literal["lbfgs"] = "lbfgs"
    alpha: Literal[0.001] = 1.0e-3
    max_iter: Literal[500] = 500
    max_fun: Literal[15000] = 15_000
    random_state: Literal[2001005] = 2_001_005
    early_stopping: Literal[False] = False
    coefficients: tuple[tuple[tuple[float, ...], ...], ...]
    intercepts: tuple[tuple[float, ...], ...]
    n_iter: int = Field(gt=0, le=500)
    terminal_loss: float = Field(ge=0.0)
    convergence_warnings: tuple[str, ...]
    uncertain_top_score_threshold: Literal[0.7] = UNCERTAIN_TOP_SCORE_THRESHOLD
    uncertain_margin_threshold: Literal[0.15] = UNCERTAIN_MARGIN_THRESHOLD
    score_semantics: Literal["model-score;not-probability-calibrated"] = (
        "model-score;not-probability-calibrated"
    )
    persistence_format: Literal["bounded-json-numeric-arrays;no-pickle"] = (
        "bounded-json-numeric-arrays;no-pickle"
    )
    sklearn_version: str
    numpy_reference_max_abs_error: float = Field(ge=0.0, le=1.0e-9)
    effective_source_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_network_shape(self) -> MLPClassifierArtifact:
        if len(self.feature_order) != self.feature_count:
            raise ValueError("MLP feature order must match its input dimension")
        if len(self.standardizer.mean) != self.feature_count:
            raise ValueError("MLP standardizer must match its input dimension")
        if len(set(self.train_sample_ids)) != len(self.train_sample_ids):
            raise ValueError("MLP TRAIN sample identifiers must be distinct")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("MLP class order must be distinct")
        if len(self.coefficients) != 3 or len(self.intercepts) != 3:
            raise ValueError("MLP must contain exactly three affine layers")
        output_size = 1 if len(self.classes) == 2 else len(self.classes)
        expected_shapes = (
            (self.feature_count, 32),
            (32, 16),
            (16, output_size),
        )
        for matrix, (rows, columns) in zip(
            self.coefficients,
            expected_shapes,
            strict=True,
        ):
            if len(matrix) != rows or any(len(row) != columns for row in matrix):
                raise ValueError("MLP coefficient matrix has an incompatible shape")
        if tuple(len(values) for values in self.intercepts) != (32, 16, output_size):
            raise ValueError("MLP intercept vector has an incompatible shape")
        expected_activation = "logistic" if len(self.classes) == 2 else "softmax"
        if self.output_activation != expected_activation:
            raise ValueError("MLP output activation differs from its class count")
        if not self.effective_source_hashes:
            raise ValueError("MLP effective source provenance is required")
        return self


class MLPQueryContext(FrozenModel):
    model_id: Literal["aqse.classical.mlp-32x16-tanh-lbfgs.v1"] = MLP_MODEL_ID
    model_role: ModelRole
    task_id: str
    input_space_id: str
    feature_count: int = Field(gt=0)
    feature_order: tuple[str, ...]


class MLPScoreBatch(FrozenModel):
    sample_ids: tuple[str, ...]
    class_order: tuple[str, ...]
    scores: tuple[tuple[float, ...], ...]
    predicted_classes: tuple[str, ...]
    displayed_classes: tuple[str, ...]
    uncertain: tuple[bool, ...]
    top_scores: tuple[float, ...]
    top_two_margins: tuple[float, ...]
    score_semantics: Literal["model-score;not-probability-calibrated"] = (
        "model-score;not-probability-calibrated"
    )

    @model_validator(mode="after")
    def validate_rows(self) -> MLPScoreBatch:
        size = len(self.sample_ids)
        row_fields = (
            self.scores,
            self.predicted_classes,
            self.displayed_classes,
            self.uncertain,
            self.top_scores,
            self.top_two_margins,
        )
        if any(len(values) != size for values in row_fields):
            raise ValueError("MLP score output fields must contain the same rows")
        if any(len(row) != len(self.class_order) for row in self.scores):
            raise ValueError("MLP score rows must match the frozen class order")
        return self
