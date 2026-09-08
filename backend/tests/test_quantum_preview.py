from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from test_windowed_features import measured_series, request_for

from app.features import extract_windowed_magnetometer_features
from app.features.provenance import sign_window_record
from app.quantum.preview import (
    MAX_PREVIEW_SAMPLES,
    QuantumPreviewRequest,
    describe_tqk8_circuit,
    run_quantum_preview,
)
from app.quantum.user_pipeline.tqk8 import AngleScaler


def preview_request(*, backend: str = "numpy", count: int = 4) -> QuantumPreviewRequest:
    extracted = extract_windowed_magnetometer_features(request_for(measured_series()))
    return QuantumPreviewRequest(
        backend=backend,
        feature_profile=extracted.profile,
        reference_dataset_id="reference-acquisition-001",
        reference_windows=extracted.windows[:count],
        theta=tuple(np.linspace(-0.6, 0.6, 16)),
    )


def test_preview_reuses_existing_scaler_and_keeps_theta_fixed() -> None:
    request = preview_request()
    theta_before = tuple(request.theta)
    raw = np.asarray([window.features.values for window in request.reference_windows])
    expected_scaler = AngleScaler.fit(raw)

    response = run_quantum_preview(request)

    assert tuple(request.theta) == theta_before
    assert response.executed_theta == theta_before
    np.testing.assert_allclose(response.scaler.mean, expected_scaler.mean)
    np.testing.assert_allclose(response.scaler.scale, expected_scaler.scale)
    np.testing.assert_allclose(
        response.encoded_reference_angles,
        expected_scaler.transform(raw),
    )


def test_reference_kernel_has_fidelity_invariants() -> None:
    response = run_quantum_preview(preview_request())
    kernel = np.asarray(response.reference_kernel)

    assert kernel.shape == (4, 4)
    np.testing.assert_allclose(kernel, kernel.T, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1.0e-12)
    assert np.min(kernel) >= -1.0e-12
    assert np.max(kernel) <= 1.0 + 1.0e-12
    assert response.diagnostics.minimum_eigenvalue >= -1.0e-12
    assert response.scientific_scope.endswith("no QNG training")


def test_query_uses_reference_scaler_and_returns_cross_kernel() -> None:
    extracted = extract_windowed_magnetometer_features(request_for(measured_series()))
    request = QuantumPreviewRequest(
        mode="reference_query",
        backend="numpy",
        feature_profile=extracted.profile,
        reference_dataset_id="reference-first-four",
        reference_windows=extracted.windows[:2],
        query_windows=extracted.windows[4:6],
        theta=(0.0,) * 16,
    )

    response = run_quantum_preview(request)

    assert np.asarray(response.query_reference_kernel).shape == (2, 2)
    reference_raw = np.asarray(response.raw_reference_features)
    query_raw = np.asarray(response.raw_query_features)
    expected_scaler = AngleScaler.fit(reference_raw)
    np.testing.assert_allclose(
        response.encoded_query_angles,
        expected_scaler.transform(query_raw),
    )


def test_qiskit_and_numpy_previews_agree() -> None:
    numpy_response = run_quantum_preview(preview_request(backend="numpy", count=2))
    qiskit_response = run_quantum_preview(preview_request(backend="qiskit", count=2))

    np.testing.assert_allclose(
        qiskit_response.reference_kernel,
        numpy_response.reference_kernel,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_preview_kernel_depends_on_trainable_theta() -> None:
    baseline_request = preview_request(backend="numpy", count=4)
    shifted_request = baseline_request.model_copy(
        update={"theta": tuple(np.linspace(0.9, -0.9, 16))}
    )

    baseline = np.asarray(run_quantum_preview(baseline_request).reference_kernel)
    shifted = np.asarray(run_quantum_preview(shifted_request).reference_kernel)

    assert np.max(np.abs(baseline - shifted)) > 1.0e-6


def test_preview_backend_contract_excludes_physical_qpu_modes() -> None:
    payload = preview_request(count=2).model_dump()
    payload["backend"] = "ibm_qpu"

    with pytest.raises(ValidationError):
        QuantumPreviewRequest.model_validate(payload)


def test_invalid_windows_and_oversized_previews_are_rejected() -> None:
    invalid = extract_windowed_magnetometer_features(
        request_for(measured_series(constant=True))
    )
    with pytest.raises(ValidationError, match="only valid windows"):
        QuantumPreviewRequest(
            feature_profile=invalid.profile,
            reference_dataset_id="invalid",
            reference_windows=invalid.windows[:2],
            theta=(0.0,) * 16,
        )

    valid_request = preview_request(count=2).model_dump()
    valid_request["reference_windows"] = [
        valid_request["reference_windows"][index % 2]
        | {"window_id": f"window-{index}"}
        for index in range(MAX_PREVIEW_SAMPLES + 1)
    ]
    with pytest.raises(ValidationError, match="too_long"):
        QuantumPreviewRequest.model_validate(valid_request)


def test_reference_query_rejects_noncausal_or_overlapping_windows() -> None:
    extracted = extract_windowed_magnetometer_features(request_for(measured_series()))

    with pytest.raises(ValidationError, match="strictly causal"):
        QuantumPreviewRequest(
            mode="reference_query",
            backend="numpy",
            feature_profile=extracted.profile,
            reference_dataset_id="overlap",
            reference_windows=extracted.windows[:2],
            query_windows=extracted.windows[2:3],
            theta=(0.0,) * 16,
        )

    touching_query = extracted.windows[4].model_copy(
        update={"start_time_s": extracted.windows[1].end_time_s}
    )
    with pytest.raises(ValidationError, match="strictly causal"):
        QuantumPreviewRequest(
            mode="reference_query",
            backend="numpy",
            feature_profile=extracted.profile,
            reference_dataset_id="touching-boundary",
            reference_windows=extracted.windows[:2],
            query_windows=[touching_query],
            theta=(0.0,) * 16,
        )

    with pytest.raises(ValidationError, match="strictly causal"):
        QuantumPreviewRequest(
            mode="reference_query",
            backend="numpy",
            feature_profile=extracted.profile,
            reference_dataset_id="future-reference",
            reference_windows=extracted.windows[4:6],
            query_windows=extracted.windows[:2],
            theta=(0.0,) * 16,
        )


def test_preview_rejects_client_modified_feature_provenance() -> None:
    request = preview_request(count=2)
    forged = request.reference_windows[0].model_copy(
        update={
            "features": request.reference_windows[0].features.model_copy(
                update={
                    "values": (
                        request.reference_windows[0].features.values[0] + 1.0,
                        *request.reference_windows[0].features.values[1:],
                    )
                }
            )
        }
    )
    forged_request = request.model_copy(
        update={"reference_windows": [forged, request.reference_windows[1]]}
    )

    with pytest.raises(ValueError, match="provenance validation failed"):
        run_quantum_preview(forged_request)


def test_preview_contract_requires_feature_provenance_token() -> None:
    payload = preview_request(count=2).model_dump()
    del payload["reference_windows"][0]["provenance_token"]

    with pytest.raises(ValidationError, match="provenance_token"):
        QuantumPreviewRequest.model_validate(payload)


def test_preview_rejects_degenerate_reference_dataset() -> None:
    request = preview_request(count=2)
    first = request.reference_windows[0]
    duplicate = first.model_copy(
        update={
            "window_id": "duplicate-window",
            "start_index": first.end_index,
            "end_index": first.end_index + (first.end_index - first.start_index),
            "start_time_s": first.end_time_s + 0.01,
            "end_time_s": first.end_time_s + 1.0,
            "center_time_s": first.end_time_s + 0.5,
            "provenance_token": "",
        }
    )
    duplicate = duplicate.model_copy(
        update={
            "provenance_token": sign_window_record(request.feature_profile, duplicate)
        }
    )
    degenerate = request.model_copy(update={"reference_windows": [first, duplicate]})

    with pytest.raises(ValueError, match="reference dataset is degenerate"):
        run_quantum_preview(degenerate)


def test_circuit_description_is_generated_from_double_upload_vqc() -> None:
    description = describe_tqk8_circuit()

    assert description.qubits == 8
    assert description.trainable_parameters == 16
    assert description.feature_uploads_per_feature == 2
    assert description.cz_edges == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (1, 2),
        (3, 4),
        (5, 6),
    )
    assert [operation.stage for operation in description.operations].count(
        "feature_upload_1"
    ) == 8
    assert [operation.stage for operation in description.operations].count(
        "feature_upload_2"
    ) == 8
