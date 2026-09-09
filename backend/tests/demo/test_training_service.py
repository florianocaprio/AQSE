from __future__ import annotations

from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from app.demo.evaluation import DemoTrainingCancelled
from app.demo.runtime_models import DemoTrainingIntent
from app.demo.training_service import (
    DemoTrainingBusyError,
    DemoTrainingJobService,
)


def _intent(suffix: str) -> DemoTrainingIntent:
    return DemoTrainingIntent(
        intent_id=f"aqse-demo-intent-{suffix}",
        study_artifact_id="aqse-network-study-fixture",
    )


def test_training_intent_is_idempotent_busy_guarded_and_cancellable(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AQSE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "app.demo.training_service.find_current_study",
        lambda: (
            tmp_path / "study",
            SimpleNamespace(artifact_id="aqse-network-study-fixture"),
        ),
    )
    monkeypatch.setattr(
        "app.demo.training_service.load_task_partitions",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def cancellable_fit(
        _train,
        _validation,
        *,
        cancellation,
        progress,
    ):
        del progress
        while not cancellation.is_set():
            sleep(0.01)
        raise DemoTrainingCancelled("fixture cancellation")

    monkeypatch.setattr(
        "app.demo.training_service.fit_demo_task_candidates",
        cancellable_fit,
    )
    service = DemoTrainingJobService()
    intent = _intent("same-click")

    first, created = service.start(intent)
    duplicate, duplicate_created = service.start(intent)

    assert created
    assert not duplicate_created
    assert duplicate.job_id == first.job_id
    with pytest.raises(DemoTrainingBusyError):
        service.start(_intent("different-click"))

    cancelled = service.cancel(first.job_id)
    assert cancelled is not None
    deadline = monotonic() + 2.0
    terminal = service.get(first.job_id)
    while terminal is not None and terminal.state != "cancelled" and monotonic() < deadline:
        sleep(0.01)
        terminal = service.get(first.job_id)
    assert terminal is not None
    assert terminal.state == "cancelled"
    assert terminal.selection_freeze_id is None

    restarted = DemoTrainingJobService()
    loaded, loaded_created = restarted.start(intent)
    assert not loaded_created
    assert loaded == terminal
    service.clear_for_tests()
