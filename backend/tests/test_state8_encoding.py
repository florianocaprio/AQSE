from __future__ import annotations

import numpy as np
import pytest

from app.features.state8 import state8_profile
from app.features.state8_encoding import (
    STATE8_ENCODING_POLICY_ID,
    fit_state8_encoder,
    load_state8_encoder,
    validate_state8_encoder_artifact,
)
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)
from app.quantum.user_pipeline.tqk8 import AngleScaler


def _training_matrix() -> np.ndarray:
    return np.asarray(
        (
            (-5.0, 0.2, -0.4, -12.0, -1.0, 0.1, -0.8, 0.91),
            (-1.0, 0.8, -0.1, -3.0, 0.0, 0.5, -0.2, 0.96),
            (2.0, 1.4, 0.3, 4.0, 1.0, 1.2, 0.4, 0.98),
            (8.0, 2.2, 0.9, 11.0, 2.0, 2.5, 0.9, 1.0),
        ),
        dtype=np.float64,
    )


def test_state8_encoder_is_train_only_deterministic_and_versions_all_inputs() -> None:
    profile = state8_profile(LOCAL_STATE8_PROFILE_ID)
    training = _training_matrix()

    first = fit_state8_encoder(
        training,
        profile=profile,
        partition="train",
        training_identity="network-study/train/v1",
    )
    second = fit_state8_encoder(
        training.copy(),
        profile=profile,
        partition="train",
        training_identity="network-study/train/v1",
    )

    assert first.artifact == second.artifact
    assert first.artifact.encoding_policy_id == STATE8_ENCODING_POLICY_ID
    assert first.artifact.all_coordinates_scaled
    assert first.artifact.feature_order == profile.feature_names
    assert first.artifact.fit_row_count == 4
    assert first.artifact.encoder_id == second.artifact.encoder_id
    validate_state8_encoder_artifact(first.artifact)
    assert not first.scaler.mean.flags.writeable
    assert not first.scaler.scale.flags.writeable

    with pytest.raises(ValueError, match="TRAIN only"):
        fit_state8_encoder(
            training,
            profile=profile,
            partition="validation",
            training_identity="network-study/validation/v1",
        )


def test_state8_encoding_uses_protected_angle_scaler_on_all_eight_coordinates() -> None:
    profile = state8_profile(LOCAL_STATE8_PROFILE_ID)
    training = _training_matrix()
    query = np.asarray(
        ((0.5, -2.8, 0.0, 1.0, 0.5, 0.8, 0.0, 0.97),),
        dtype=np.float64,
    )
    encoder = fit_state8_encoder(
        training,
        profile=profile,
        partition="train",
        training_identity="all-eight-coordinate-check",
    )

    encoded = encoder.transform(query, profile=profile)
    expected = AngleScaler.fit(training).transform(query)

    np.testing.assert_array_equal(encoded, expected)
    assert encoded[0, 1] != pytest.approx(query[0, 1])
    assert not encoded.flags.writeable


def test_state8_encoder_rejects_cross_profile_reuse() -> None:
    local = state8_profile(LOCAL_STATE8_PROFILE_ID)
    network = state8_profile(NETWORK_STATE8_PROFILE_ID)
    encoder = fit_state8_encoder(
        _training_matrix(),
        profile=local,
        partition="train",
        training_identity="local-model-family",
    )

    with pytest.raises(ValueError, match="incompatible"):
        encoder.transform(_training_matrix(), profile=network)


def test_state8_encoding_is_query_order_and_batch_invariant() -> None:
    profile = state8_profile(NETWORK_STATE8_PROFILE_ID)
    encoder = fit_state8_encoder(
        _training_matrix(),
        profile=profile,
        partition="train",
        training_identity="network-model-family",
    )
    query = np.asarray(
        (
            (1.0, 0.5, 0.1, 2.0, 0.2, 0.4, 0.7, 1.0),
            (-2.0, 1.0, -0.2, -1.0, -0.1, 0.7, -0.3, 0.95),
        ),
        dtype=np.float64,
    )

    together = encoder.transform(query, profile=profile)
    reversed_batch = encoder.transform(query[::-1].copy(), profile=profile)
    singles = np.vstack(
        [encoder.transform(row[None, :], profile=profile)[0] for row in query]
    )

    np.testing.assert_array_equal(together, reversed_batch[::-1])
    np.testing.assert_array_equal(together, singles)
    reloaded = load_state8_encoder(encoder.artifact)
    np.testing.assert_array_equal(
        together,
        reloaded.transform(query, profile=profile),
    )


@pytest.mark.parametrize(
    "invalid",
    (
        np.ones((2, 7)),
        np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan]]),
    ),
)
def test_state8_encoder_rejects_incomplete_or_nonfinite_rows(invalid: np.ndarray) -> None:
    profile = state8_profile(LOCAL_STATE8_PROFILE_ID)

    with pytest.raises(ValueError, match="shape|NaN"):
        fit_state8_encoder(
            invalid,
            profile=profile,
            partition="train",
            training_identity="invalid-fixture",
        )
