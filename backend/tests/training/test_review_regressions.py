from __future__ import annotations

import hashlib
import json
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.training.canonical import array_identity, file_sha256
from app.training.generation import DatasetBuild
from app.training.models import DatasetPartition, SplitAssignment
from app.training.models import TestAccessAuthorization as AccessAuthorization
from app.training.storage import (
    LEDGER_NAME,
    load_labels,
    load_observations,
    load_partition,
    verify_archive_opaque,
    write_dataset,
)


def _refresh_manifest_record(dataset_path: Path, relative_path: str) -> None:
    manifest_path = dataset_path / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = dataset_path / relative_path
    record = next(item for item in manifest["files"] if item["relative_path"] == relative_path)
    record["byte_count"] = target.stat().st_size
    record["file_sha256"] = file_sha256(target)
    record["content_sha256"] = hashlib.sha256(target.read_bytes().rstrip(b"\n")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _replace_array_and_refresh_manifest(
    dataset_path: Path, relative_path: str, value: np.ndarray
) -> None:
    target = dataset_path / relative_path
    target.chmod(0o644)
    with target.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    manifest_path = dataset_path / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == relative_path)
    identity = array_identity(value)
    record.update(
        byte_count=target.stat().st_size,
        file_sha256=file_sha256(target),
        content_sha256=identity["content_sha256"],
        dtype=identity["dtype"],
        shape=identity["shape"],
        endianness=identity["endianness"],
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_writer_rejects_identical_observations_renamed_across_partitions(
    tmp_path: Path,
    generated_episodes,
) -> None:
    original = generated_episodes[0]
    cloned_plan = original.plan.model_copy(
        update={
            "episode_id": "episode-ffffffffffffffff",
            "lineage_id": "lineage-ffffffffffffffff",
        }
    )
    cloned_label = original.label.model_copy(
        update={
            "episode_id": cloned_plan.episode_id,
            "lineage_id": cloned_plan.lineage_id,
        }
    )
    clone = replace(original, plan=cloned_plan, label=cloned_label)
    assert clone.observation_content_digest == original.observation_content_digest
    assert clone.observation_binding_digest != original.observation_binding_digest
    independent = generated_episodes[1]
    assignments = (
        SplitAssignment(
            episode_id=original.plan.episode_id,
            lineage_id=original.plan.lineage_id,
            partition=DatasetPartition.TRAIN,
        ),
        SplitAssignment(
            episode_id=clone.plan.episode_id,
            lineage_id=clone.plan.lineage_id,
            partition=DatasetPartition.TEST,
        ),
        SplitAssignment(
            episode_id=independent.plan.episode_id,
            lineage_id=independent.plan.lineage_id,
            partition=DatasetPartition.VALIDATION,
        ),
    )
    build = DatasetBuild(
        kind="development",
        master_seed=7,
        split_seed=8,
        episodes=(original, clone, independent),
        assignments=assignments,
    )

    with pytest.raises(ValueError, match="duplicate observation"):
        write_dataset(build, root=tmp_path)


def test_writer_accepts_independent_equal_time_axes_with_different_data(
    tmp_path: Path,
    small_development_build,
) -> None:
    assert all(
        np.array_equal(
            small_development_build.episodes[0].time_s,
            episode.time_s,
        )
        for episode in small_development_build.episodes[1:]
    )
    assert not np.array_equal(
        small_development_build.episodes[0].measured_field_T,
        small_development_build.episodes[1].measured_field_T,
    )
    path, manifest, _ = write_dataset(small_development_build, root=tmp_path)
    assert path.is_dir()
    assert manifest.schema_version == "aqse.dataset-manifest.v2"


def test_writer_aligns_reordered_assignments_by_episode_identity(
    tmp_path: Path,
    small_development_build,
) -> None:
    reordered = replace(
        small_development_build,
        assignments=tuple(reversed(small_development_build.assignments)),
    )
    path, _, _ = write_dataset(reordered, root=tmp_path)
    assert path.is_dir()


def test_writer_rejects_missing_and_incoherent_identity_references(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    missing = replace(
        small_pilot_build,
        assignments=small_pilot_build.assignments[:-1],
    )
    with pytest.raises(ValueError, match="exactly one split assignment"):
        write_dataset(missing, root=tmp_path / "missing")

    wrong = small_pilot_build.assignments[0].model_copy(
        update={"lineage_id": "lineage-ffffffffffffffff"}
    )
    incoherent = replace(
        small_pilot_build,
        assignments=(wrong, *small_pilot_build.assignments[1:]),
    )
    with pytest.raises(ValueError, match="lineage does not match"):
        write_dataset(incoherent, root=tmp_path / "incoherent")


def test_writer_rejects_raw_source_interval_outside_observation(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    episode = small_pilot_build.episodes[0]
    invalid = replace(
        episode,
        raw_source=episode.raw_source.model_copy(update={"end_index": len(episode.time_s) + 1}),
    )
    build = replace(small_pilot_build, episodes=(invalid, *small_pilot_build.episodes[1:]))
    with pytest.raises(ValueError, match="raw source interval exceeds"):
        write_dataset(build, root=tmp_path)


def test_train_load_never_semantically_loads_test_arrays(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_development_build, root=tmp_path)
    loaded_paths: list[Path] = []
    from app.training import storage

    original = storage._safe_load_array

    def recording_loader(array_path: Path, *args):
        loaded_paths.append(array_path)
        return original(array_path, *args)

    monkeypatch.setattr(storage, "_safe_load_array", recording_loader)
    load_partition(path, DatasetPartition.TRAIN)
    assert loaded_paths
    assert all("/partitions/train/" in item.as_posix() for item in loaded_paths)


def test_train_load_never_decodes_test_metadata_labels_or_plans(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_development_build, root=tmp_path)
    decoded_paths: list[Path] = []
    from app.training import storage

    original = storage._read_json

    def recording_reader(json_path: Path, *args, **kwargs):
        decoded_paths.append(json_path)
        return original(json_path, *args, **kwargs)

    monkeypatch.setattr(storage, "_read_json", recording_reader)
    load_observations(path, DatasetPartition.TRAIN)
    decoded = {item.relative_to(path).as_posix() for item in decoded_paths}
    assert "partitions/test/metadata.json" not in decoded
    assert "labels/test.json" not in decoded
    assert "generation/test.json" not in decoded


def test_denied_test_open_does_not_deserialize_arrays(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_development_build, root=tmp_path)
    loaded_paths: list[Path] = []
    from app.training import storage

    original = storage._safe_load_array

    def recording_loader(array_path: Path, *args):
        loaded_paths.append(array_path)
        return original(array_path, *args)

    monkeypatch.setattr(storage, "_safe_load_array", recording_loader)
    with pytest.raises(PermissionError, match="sealed by default"):
        load_partition(path, DatasetPartition.TEST)
    assert loaded_paths == []


def test_opaque_verification_never_calls_numpy_load(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_development_build, root=tmp_path)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("opaque verification must not call numpy.load")

    monkeypatch.setattr(np, "load", forbidden_load)
    verify_archive_opaque(path)


def test_loader_rejects_unmanifested_numeric_payload(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    np.save(path / "partitions" / "pilot" / "extra.npy", np.asarray([1.0]))
    with pytest.raises(ValueError, match="unexpected scientific file"):
        load_partition(path, DatasetPartition.PILOT)


def test_observable_loader_does_not_return_generation_plans(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    loaded = load_partition(path, DatasetPartition.PILOT)
    assert not hasattr(loaded, "plans")
    assert not hasattr(loaded, "labels")
    assert not hasattr(loaded, "target")
    assert set(vars(loaded)).isdisjoint(
        {"plans", "labels", "target", "white_noise_std_nt", "generation_seed", "signal_seed"}
    )
    metadata = json.loads(
        (path / "partitions" / "pilot" / "metadata.json").read_text(encoding="utf-8")
    )
    assert "features" not in metadata["windows"][0][0]


def test_window_quality_mutation_cannot_be_hidden_by_refreshing_file_checksums(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    relative_path = "partitions/pilot/metadata.json"
    target = path / relative_path
    target.chmod(0o644)
    metadata = json.loads(target.read_text(encoding="utf-8"))
    metadata["windows"][0][0]["quality"]["valid_for_quantum"] = not metadata["windows"][0][0][
        "quality"
    ]["valid_for_quantum"]
    target.write_text(json.dumps(metadata), encoding="utf-8")
    _refresh_manifest_record(path, relative_path)

    with pytest.raises(ValueError, match="scientific|window|quality"):
        load_observations(path, DatasetPartition.PILOT)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("windows", 0, 0, "start_index"), 1),
        (("windows", 0, 0, "sensor_id"), "M2"),
        (("observable_provenance", "field_unit"), "nT"),
        (("observable_provenance", "feature_profile_fingerprint"), "0" * 64),
    ],
)
def test_window_ledger_profile_and_units_mutations_are_rejected(
    tmp_path: Path,
    small_pilot_build,
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    relative = "partitions/pilot/metadata.json"
    target = path / relative
    target.chmod(0o644)
    payload = json.loads(target.read_text(encoding="utf-8"))
    cursor = payload
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = replacement
    target.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_manifest_record(path, relative)
    with pytest.raises((ValueError, TypeError)):
        load_observations(path, DatasetPartition.PILOT)


@pytest.mark.parametrize("mutation", ["feature", "shape", "nonfinite"])
def test_numeric_mutations_fail_even_with_refreshed_manifest_checksums(
    tmp_path: Path,
    small_pilot_build,
    mutation: str,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    relative = "partitions/pilot/features.npy"
    value = np.load(path / relative, allow_pickle=False)
    if mutation == "feature":
        value[0, 0, 0] += 1.0
    elif mutation == "shape":
        value = value[:, :-1, :]
    else:
        value[0, 0, 0] = np.inf
    _replace_array_and_refresh_manifest(path, relative, value)
    with pytest.raises(ValueError):
        load_observations(path, DatasetPartition.PILOT)


def test_npy_header_shape_limit_is_checked_before_allocation(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    relative = "partitions/pilot/features.npy"
    target = path / relative
    target.chmod(0o644)
    with target.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle,
            {"descr": "<f8", "fortran_order": False, "shape": (50_000_001,)},
        )
    manifest_path = path / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == relative)
    record["byte_count"] = target.stat().st_size
    record["file_sha256"] = file_sha256(target)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="element limit"):
        load_observations(path, DatasetPartition.PILOT)


def test_manifested_symlink_is_rejected_on_actual_read(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    target = path / "partitions" / "pilot" / "features.npy"
    source = path / "partitions" / "pilot" / "time_s.npy"
    target.chmod(0o644)
    target.unlink()
    target.symlink_to(source.name)
    with pytest.raises(ValueError, match="symlink"):
        verify_archive_opaque(path)


def test_json_limits_are_enforced_before_semantic_decode(
    tmp_path: Path,
    small_pilot_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    monkeypatch.setattr("app.training.storage.MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="bounded load limit"):
        load_observations(path, DatasetPartition.PILOT)


def test_fixture_test_ledger_appends_are_serialized_and_identity_bound(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, manifest, _ = write_dataset(small_development_build, root=tmp_path)
    monkeypatch.setenv("AQSE_TEST_FIXTURE_DATASET_ID", manifest.dataset_id)
    authorization = AccessAuthorization(
        actor="pytest-concurrent",
        reason="concurrency fixture",
        fixture_only=True,
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                lambda _: load_labels(path, DatasetPartition.TEST, authorization=authorization),
                range(3),
            )
        )
    assert all(results)
    entries = [json.loads(line) for line in (path / LEDGER_NAME).read_text().splitlines()]
    assert [item["sequence"] for item in entries] == list(range(4))
    assert all(item["scientific_digest"] == manifest.scientific_digest for item in entries)
    assert [item["event"] for item in entries] == ["sealed", "opened", "opened", "opened"]


def test_candidate_failure_is_not_published_and_cleans_tracemalloc(
    tmp_path: Path,
    small_pilot_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_candidate(*args, **kwargs):
        raise ValueError("candidate fixture rejected")

    monkeypatch.setattr("app.training.storage._validate_archive_candidate", fail_candidate)
    with pytest.raises(ValueError, match="candidate fixture rejected"):
        write_dataset(small_pilot_build, root=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert not tracemalloc.is_tracing()


def test_observation_content_digest_excludes_operational_and_truth_fields(
    generated_episodes,
) -> None:
    episode = generated_episodes[0]
    changed = replace(
        episode,
        plan=episode.plan.model_copy(
            update={
                "generation_seed": episode.plan.generation_seed + 1,
                "signal_seed": episode.plan.signal_seed + 1,
                "white_noise_std_nt": episode.plan.white_noise_std_nt + 0.01,
            }
        ),
        label=episode.label.model_copy(update={"target": -episode.label.target}),
    )
    assert changed.observation_content_digest == episode.observation_content_digest
    assert changed.observation_binding_digest == episode.observation_binding_digest
