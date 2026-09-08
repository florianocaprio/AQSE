from __future__ import annotations

from datetime import datetime, timezone
from threading import Thread
from time import sleep

import pytest

import app.network.session as session_module
from app.network.defaults import default_network_configuration
from app.network.models import NetworkSessionConfiguration, SessionState
from app.network.session import NetworkSession
from app.network.simulation import NetworkSimulator


def test_ui_refresh_rate_batches_frames_without_changing_physical_samples() -> None:
    base = default_network_configuration(1)
    slow_refresh = NetworkSessionConfiguration.model_validate(
        {**base.model_dump(), "ui_refresh_rate_Hz": 5.0}
    )
    fast_refresh = NetworkSessionConfiguration.model_validate(
        {**base.model_dump(), "ui_refresh_rate_Hz": 20.0}
    )
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    slow_simulator = NetworkSimulator(slow_refresh)
    fast_simulator = NetworkSimulator(fast_refresh)
    slow_frames = [
        slow_simulator.generate("same", index + 1, index / 100.0, epoch, ())
        for index in range(30)
    ]
    fast_frames = [
        fast_simulator.generate("same", index + 1, index / 100.0, epoch, ())
        for index in range(30)
    ]
    assert [frame.observation for frame in slow_frames] == [
        frame.observation for frame in fast_frames
    ]

    session = NetworkSession(slow_refresh)
    session.start()
    try:
        batch = session.wait_for_observations(0, timeout_s=1.0, limit=250)
    finally:
        session.stop()
    assert len(batch.frames) >= 20


def test_pause_and_stop_remain_responsive_under_accelerated_overload() -> None:
    base = default_network_configuration(8)
    overloaded = NetworkSessionConfiguration.model_validate(
        {
            **base.model_dump(),
            "sampling_rate_Hz": 2_000.0,
            "ui_refresh_rate_Hz": 5.0,
            "time_scale": 100.0,
            "buffer_duration_s": 1.0,
        }
    )
    session = NetworkSession(overloaded)
    session.start()
    sleep(0.02)
    pause_thread = Thread(target=session.pause)
    pause_thread.start()
    pause_thread.join(timeout=1.0)

    assert not pause_thread.is_alive()
    assert session.view().status.state.value == "paused"
    assert session.view().status.simulation_lag_s == 0.0

    stop_thread = Thread(target=session.stop)
    stop_thread.start()
    stop_thread.join(timeout=1.0)
    assert not stop_thread.is_alive()
    assert session.view().status.state.value == "stopped"
    assert session.view().status.simulation_lag_s == 0.0


def test_worker_lag_returns_to_zero_after_a_batch_catches_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = NetworkSessionConfiguration.model_validate(
        {
            **default_network_configuration(1).model_dump(),
            "sampling_rate_Hz": 1.0,
            "ui_refresh_rate_Hz": 1.0,
            "time_scale": 1.0,
        }
    )
    session = NetworkSession(configuration)
    session._state = SessionState.RUNNING
    ticks = iter((0.0, 1.25, 1.30))
    monkeypatch.setattr(session_module, "monotonic", lambda: next(ticks))

    def finish_after_one_batch(frames: int) -> None:
        assert frames == 1
        session._worker_should_exit = True

    monkeypatch.setattr(session, "_generate_locked", finish_after_one_batch)

    session._run()

    assert session.view().status.simulation_lag_s == 0.0
