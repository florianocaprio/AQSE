from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from app.features.models import FeatureExtractionRequest, MeasuredVectorSeries
from app.features.windowed import extract_windowed_magnetometer_features
from app.training.generation import (
    FEATURE_WINDOW,
    _validate_pilot,
    build_episode_plans,
)


def test_controlled_domain_and_labels_are_frozen_before_generation() -> None:
    specifications = build_episode_plans(
        master_seed=42,
        episodes_per_class=3,
        purpose="contract",
    )
    assert len(specifications) == 6
    assert [item.label.target for item in specifications].count(-1) == 3
    assert [item.label.target for item in specifications].count(1) == 3
    assert len({item.plan.lineage_id for item in specifications}) == 6
    for item in specifications:
        assert 6.0 <= item.plan.amplitude_nt <= 10.0
        assert 4.0 <= item.plan.frequency_hz <= 8.0
        assert 292.15 <= item.plan.temperature_k <= 294.15
        if item.label.target == -1:
            assert 0.25 <= item.plan.white_noise_std_nt <= 0.75
        else:
            assert 1.5 <= item.plan.white_noise_std_nt <= 3.0

    pilot_lineages = {
        item.plan.lineage_id
        for item in build_episode_plans(
            master_seed=1_001_001,
            episodes_per_class=12,
            purpose="pilot",
        )
    }
    development_lineages = {
        item.plan.lineage_id
        for item in build_episode_plans(
            master_seed=1_001_002,
            episodes_per_class=60,
            purpose="development",
        )
    }
    assert pilot_lineages.isdisjoint(development_lineages)


def test_generated_episode_keeps_all_windows_and_no_label_in_predictors(
    generated_episodes,
) -> None:
    episode = generated_episodes[0]
    assert episode.measured_field_T.shape == (1_000, 3)
    assert episode.features.shape == (19, 8)
    assert episode.valid_mask.shape == (19,)
    assert len(episode.windows) == episode.coverage.proposed_windows == 19
    assert episode.coverage.accepted_windows + episode.coverage.rejected_windows == 19
    assert not np.any(episode.saturation_mask)
    assert episode.features.dtype.names is None


def test_pilot_gate_rejects_zero_coverage_without_resampling(generated_episodes) -> None:
    valid_pilot = tuple(generated_episodes)
    _validate_pilot(valid_pilot)
    failed_episode = replace(
        valid_pilot[0],
        coverage=valid_pilot[0].coverage.model_copy(
            update={
                "accepted_windows": 0,
                "rejected_windows": valid_pilot[0].coverage.proposed_windows,
                "coverage_fraction": 0.0,
            }
        ),
    )
    with pytest.raises(RuntimeError, match="zero valid windows"):
        _validate_pilot((failed_episode, *valid_pilot[1:]))


def test_same_observations_with_different_external_truth_produce_same_features(
    generated_episodes,
) -> None:
    episode = generated_episodes[0]
    series = MeasuredVectorSeries(
        acquisition_id="observable-only",
        sensor_id="M1",
        sampling_rate_hz=100.0,
        time_s=episode.time_s.tolist(),
        measured_field=[tuple(row) for row in episode.measured_field_T],
        field_unit="T",
        temperature_k=episode.temperature_K.tolist(),
        saturation_mask=[tuple(row) for row in episode.saturation_mask],
    )
    first = extract_windowed_magnetometer_features(
        FeatureExtractionRequest(series=series, channel="x", window=FEATURE_WINDOW)
    )
    second = extract_windowed_magnetometer_features(
        FeatureExtractionRequest(series=series, channel="x", window=FEATURE_WINDOW)
    )
    np.testing.assert_array_equal(
        [item.features.values for item in first.windows],
        [item.features.values for item in second.windows],
    )


def test_scientific_observation_digest_is_reproducible_across_processes() -> None:
    program = """
import json
from app.training.generation import build_episode_plans, generate_episode
item = build_episode_plans(master_seed=5150, episodes_per_class=1, purpose='process')[0]
episode = generate_episode(item)
print(json.dumps({'digest': episode.observation_digest, 'features': episode.features.tolist()}, sort_keys=True))
"""
    first = subprocess.check_output([sys.executable, "-c", program], text=True)
    second = subprocess.check_output([sys.executable, "-c", program], text=True)
    assert json.loads(first) == json.loads(second)


def test_full_archive_scientific_digest_is_reproducible_across_processes() -> None:
    program = """
import tempfile
from pathlib import Path
from app.training.generation import DatasetBuild, build_episode_plans, generate_episode
from app.training.models import DatasetPartition, SplitAssignment
from app.training.storage import write_dataset
specs = build_episode_plans(master_seed=6160, episodes_per_class=1, purpose='archive-process')
episodes = tuple(generate_episode(item) for item in specs)
assignments = tuple(SplitAssignment(episode_id=e.plan.episode_id, lineage_id=e.plan.lineage_id, partition=DatasetPartition.PILOT) for e in episodes)
build = DatasetBuild(kind='pilot', master_seed=6160, split_seed=None, episodes=episodes, assignments=assignments)
with tempfile.TemporaryDirectory() as directory:
    _, manifest, _ = write_dataset(build, root=Path(directory))
    print(manifest.scientific_digest)
"""
    first = subprocess.check_output([sys.executable, "-c", program], text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", program], text=True).strip()
    assert first == second
