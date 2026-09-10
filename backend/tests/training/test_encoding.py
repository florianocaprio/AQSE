from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
import pytest

from app.quantum.user_pipeline.tqk8 import StateEngine
from app.training import encoding
from app.training.encoding import (
    PHASE_INDEX,
    fit_phase_direct_encoder,
    input_contract_for,
    validate_encoder_artifact,
    wrap_phase_direct,
)
from app.training.encoding_storage import (
    load_encoder_artifact,
    write_encoder_artifact,
)
from app.training.models import DatasetPartition
from app.training.storage import load_observations, write_dataset


@pytest.fixture(scope="module")
def encoding_dataset(tmp_path_factory, small_development_build):
    root = tmp_path_factory.mktemp("encoding-dataset")
    return write_dataset(small_development_build, root=root)[:2]


def _approve_fixture_identity(manifest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(encoding, "CANONICAL_DEVELOPMENT_DATASET_ID", manifest.dataset_id)
    monkeypatch.setattr(
        encoding,
        "CANONICAL_DEVELOPMENT_DIGEST",
        manifest.scientific_digest,
    )
    monkeypatch.setattr(
        encoding,
        "CANONICAL_FEATURE_PROFILE_FINGERPRINT",
        manifest.feature_profile_fingerprint,
    )


def _encoder_fixture(encoding_dataset, monkeypatch: pytest.MonkeyPatch):
    path, manifest = encoding_dataset
    _approve_fixture_identity(manifest, monkeypatch)
    train = load_observations(path, DatasetPartition.TRAIN)
    return fit_phase_direct_encoder(train, manifest), path, manifest, train


def test_scaler_fitting_is_train_only_and_repeatable(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, path, manifest, train = _encoder_fixture(encoding_dataset, monkeypatch)
    repeated = fit_phase_direct_encoder(train, manifest)
    validation = load_observations(path, DatasetPartition.VALIDATION)

    assert encoder.artifact == repeated.artifact
    assert encoder.artifact.fitting_population.episode_count == len(train.episode_ids)
    assert encoder.artifact.fitting_population.accepted_window_count == int(
        np.count_nonzero(train.valid_mask)
    )
    with pytest.raises(ValueError, match="TRAIN only"):
        fit_phase_direct_encoder(validation, manifest)


def test_validation_transform_never_mutates_or_refits_encoder(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, path, _, _ = _encoder_fixture(encoding_dataset, monkeypatch)
    validation = load_observations(path, DatasetPartition.VALIDATION)
    context = input_contract_for(encoder.artifact)
    raw = validation.features[validation.valid_mask]
    mean_before = encoder.scaler.mean.copy()
    scale_before = encoder.scaler.scale.copy()
    identity_before = encoder.artifact.artifact_id

    encoded = encoder.transform(raw, context=context)
    perturbed = raw.copy()
    perturbed[:, 0] += 500.0
    encoder.transform(perturbed, context=context)

    assert encoded.shape == raw.shape
    np.testing.assert_array_equal(encoder.scaler.mean, mean_before)
    np.testing.assert_array_equal(encoder.scaler.scale, scale_before)
    assert encoder.artifact.artifact_id == identity_before


def test_query_row_order_preserves_each_encoded_row(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _, _, train = _encoder_fixture(encoding_dataset, monkeypatch)
    context = input_contract_for(encoder.artifact)
    raw = train.features[train.valid_mask][:8]
    order = np.asarray([6, 1, 4, 0, 7, 2, 5, 3])

    expected = encoder.transform(raw, context=context)
    reordered = encoder.transform(raw[order], context=context)

    np.testing.assert_array_equal(reordered, expected[order])


def test_non_phase_coordinates_delegate_exactly_to_protected_scaler(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _, _, train = _encoder_fixture(encoding_dataset, monkeypatch)
    raw = train.features[train.valid_mask][:12]
    protected = encoder.scaler.transform(raw)
    actual = encoder.transform(raw, context=input_contract_for(encoder.artifact))
    non_phase = [0, 2, 3, 4, 5, 6, 7]

    np.testing.assert_array_equal(actual[:, non_phase], protected[:, non_phase])
    np.testing.assert_array_equal(actual[:, PHASE_INDEX], wrap_phase_direct(raw[:, 1]))


def test_phase_wrap_canonical_cases() -> None:
    source = np.asarray([0.0, pi, -pi, 3 * pi, -3 * pi, 8 * pi + 0.3, -6 * pi + 0.3])
    actual = wrap_phase_direct(source)

    np.testing.assert_allclose(
        actual,
        np.asarray([0.0, -pi, -pi, -pi, -pi, 0.3, 0.3]),
        rtol=0.0,
        atol=4.0e-15,
    )
    assert np.all(actual >= -pi)
    assert np.all(actual < pi)
    with pytest.raises(ValueError, match="NaN or infinity"):
        wrap_phase_direct([np.inf])


def test_encoder_rejects_every_declared_compatibility_mismatch(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _, _, train = _encoder_fixture(encoding_dataset, monkeypatch)
    context = input_contract_for(encoder.artifact)
    raw = train.features[train.valid_mask][:1]
    profile = dict(context.feature_profile)
    wrong_order = dict(profile)
    wrong_order["feature_names"] = list(reversed(profile["feature_names"]))
    wrong_units = dict(profile)
    wrong_units["feature_units"] = ["wrong", *profile["feature_units"][1:]]
    incompatible = (
        context.model_copy(update={"source_dataset_id": "aqse-development-wrong"}),
        context.model_copy(update={"feature_profile_fingerprint": "0" * 64}),
        context.model_copy(update={"feature_profile": wrong_order}),
        context.model_copy(update={"feature_profile": wrong_units}),
        context.model_copy(update={"phase_reference": "global_clock"}),
        context.model_copy(update={"encoding_policy_id": "legacy-preview"}),
        context.model_copy(update={"scaler_id": "aqse-angle-scaler-0000000000000000"}),
        context.model_copy(
            update={
                "tqk_compatibility": context.tqk_compatibility.model_copy(
                    update={"tqk8_source_sha256": "0" * 64}
                )
            }
        ),
        context.model_copy(
            update={
                "tqk_compatibility": context.tqk_compatibility.model_copy(
                    update={"input_count": 7}
                )
            }
        ),
    )

    for wrong_context in incompatible:
        with pytest.raises(ValueError, match="incompatible"):
            encoder.transform(raw, context=wrong_context)


def test_encoder_rejects_corruption_and_invalid_numeric_inputs(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _, _, train = _encoder_fixture(encoding_dataset, monkeypatch)
    context = input_contract_for(encoder.artifact)
    raw = train.features[train.valid_mask][:2]

    with pytest.raises(ValueError, match="content digest"):
        validate_encoder_artifact(
            encoder.artifact.model_copy(update={"content_digest": "0" * 64})
        )
    for invalid in (
        np.asarray([[1.0] * 7]),
        np.asarray([[1.0] * 7 + [np.nan]]),
        np.asarray([[1.0] * 7 + [np.inf]]),
        np.asarray([[*raw[0, :7], "not-a-number"]], dtype=object),
    ):
        with pytest.raises((TypeError, ValueError)):
            encoder.transform(invalid, context=context)


def test_encoder_artifact_is_immutable_compatible_and_contains_no_theta(
    tmp_path: Path,
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _, manifest, _ = _encoder_fixture(encoding_dataset, monkeypatch)
    path, execution = write_encoder_artifact(encoder, root=tmp_path)
    loaded = load_encoder_artifact(
        path,
        expected_artifact_id=encoder.artifact.artifact_id,
        expected_dataset_id=manifest.dataset_id,
        expected_dataset_digest=manifest.scientific_digest,
        expected_profile_fingerprint=manifest.feature_profile_fingerprint,
    )

    assert loaded.artifact == encoder.artifact
    assert execution.artifact_id == encoder.artifact.artifact_id
    assert '"theta"' not in (path / "encoder.json").read_text(encoding="utf-8")
    assert not loaded.scaler.mean.flags.writeable
    assert not loaded.scaler.scale.flags.writeable
    with pytest.raises(FileExistsError, match="already exists"):
        write_encoder_artifact(encoder, root=tmp_path)
    with pytest.raises(ValueError, match="dataset identity"):
        load_encoder_artifact(
            path,
            expected_artifact_id=encoder.artifact.artifact_id,
            expected_dataset_id="aqse-development-wrong",
            expected_dataset_digest=manifest.scientific_digest,
            expected_profile_fingerprint=manifest.feature_profile_fingerprint,
        )


def test_actual_vqc_periodicity_seam_and_cross_engine_agreement(
    encoding_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _, _, train = _encoder_fixture(encoding_dataset, monkeypatch)
    context = input_contract_for(encoder.artifact)
    base = train.features[train.valid_mask][0].copy()
    periodic_raw = np.stack([base, base])
    periodic_raw[0, 1] = 0.37
    periodic_raw[1, 1] = 0.37 + 2.0 * pi
    epsilon = 1.0e-8
    seam_raw = np.stack([base, base])
    seam_raw[0, 1] = -pi + epsilon
    seam_raw[1, 1] = pi - epsilon
    encoded_periodic = encoder.transform(periodic_raw, context=context)
    encoded_seam = encoder.transform(seam_raw, context=context)
    theta = np.linspace(-0.45, 0.55, 16, dtype=np.float64)

    engines = (StateEngine("numpy"), StateEngine("qiskit"))
    kernels: list[np.ndarray] = []
    for engine in engines:
        periodic_states = engine.states(encoded_periodic, theta)
        periodic_density = np.asarray(
            [np.outer(state, state.conj()) for state in periodic_states]
        )
        np.testing.assert_allclose(
            periodic_density[0], periodic_density[1], rtol=0.0, atol=1.0e-12
        )
        seam_kernel = engine.gram(encoded_seam, theta)
        assert seam_kernel[0, 1] >= 1.0 - 1.0e-12
        kernels.append(engine.gram(np.vstack([encoded_periodic, encoded_seam]), theta))

    np.testing.assert_allclose(kernels[0], kernels[1], rtol=0.0, atol=1.0e-12)
