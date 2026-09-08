from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from threading import Condition, RLock, Thread
from time import monotonic, sleep
from uuid import uuid4

from app.network.models import (
    BufferedFrame,
    BufferStatus,
    FrameBatch,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    ObservationSnapshot,
    ScheduledEventResponse,
    SessionState,
    SessionStatus,
    SessionView,
    TruthFrameBatch,
    TruthSnapshot,
)
from app.network.simulation import NetworkSimulator

MAX_SYNCHRONOUS_REPLAY_FRAMES = 5_000
MAX_WORKER_BATCH_FRAMES = 64


class SessionConflictError(RuntimeError):
    """Raised when a control is incompatible with the current session state."""


class SessionCapacityError(RuntimeError):
    """Raised when the bounded in-memory session registry is full."""


@dataclass(frozen=True)
class _ScheduledEventRecord:
    event: NetworkEventConfiguration
    effective_frame_id: int
    configuration_version: int


class NetworkSession:
    """Single authoritative simulation job and bounded frame store."""

    def __init__(self, configuration: NetworkSessionConfiguration) -> None:
        self.session_id = uuid4().hex
        self._configuration = configuration
        self._configuration_version = 1
        self._events = list(configuration.events)
        self._event_records = [
            _ScheduledEventRecord(
                event=event,
                effective_frame_id=1,
                configuration_version=1,
            )
            for event in configuration.events
        ]
        self._epoch_utc = datetime.now(timezone.utc)
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at
        self._state = SessionState.CREATED
        self._latest_frame_id = 0
        self._sim_time_s = 0.0
        self._simulation_lag_s = 0.0
        self._overwritten_frames = 0
        self._buffer: deque[BufferedFrame] = deque(
            maxlen=configuration.buffer_capacity_frames
        )
        self._simulator = NetworkSimulator(configuration)
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._worker: Thread | None = None
        self._worker_should_exit = False

    def view(self) -> SessionView:
        with self._lock:
            return SessionView(
                status=self._status_locked(),
                configuration=self._configuration,
            )

    def start(self) -> SessionView:
        with self._condition:
            if self._state is SessionState.STOPPED:
                raise SessionConflictError("reset a stopped session before starting it")
            if self._latest_frame_id == 0:
                self._generate_locked(1)
            self._state = SessionState.RUNNING
            self._updated_at = datetime.now(timezone.utc)
            self._ensure_worker_locked()
            self._condition.notify_all()
            return self.view()

    def pause(self) -> SessionView:
        with self._condition:
            if self._state is SessionState.STOPPED:
                return self.view()
            self._state = SessionState.PAUSED
            self._simulation_lag_s = 0.0
            self._updated_at = datetime.now(timezone.utc)
            self._condition.notify_all()
            return self.view()

    def resume(self) -> SessionView:
        return self.start()

    def stop(self) -> SessionView:
        with self._condition:
            self._state = SessionState.STOPPED
            self._worker_should_exit = True
            self._simulation_lag_s = 0.0
            self._updated_at = datetime.now(timezone.utc)
            self._condition.notify_all()
            return self.view()

    def reset(self) -> SessionView:
        self._halt_worker()
        with self._condition:
            self._reset_locked()
            return self.view()

    def replay(self) -> SessionView:
        with self._lock:
            frame_count = self._latest_frame_id
            if frame_count > MAX_SYNCHRONOUS_REPLAY_FRAMES:
                raise SessionConflictError(
                    "synchronous replay is limited to "
                    f"{MAX_SYNCHRONOUS_REPLAY_FRAMES} frames; reset the session "
                    "and replay it incrementally"
                )
        self._halt_worker()
        with self._condition:
            # A running worker may have completed a frame between the initial
            # bound check and being halted. Recheck before doing synchronous work.
            frame_count = self._latest_frame_id
            if frame_count > MAX_SYNCHRONOUS_REPLAY_FRAMES:
                self._state = SessionState.PAUSED
                raise SessionConflictError(
                    "synchronous replay is limited to "
                    f"{MAX_SYNCHRONOUS_REPLAY_FRAMES} frames; reset the session "
                    "and replay it incrementally"
                )
            self._reset_locked()
            if frame_count:
                self._generate_locked(frame_count)
                self._state = SessionState.PAUSED
            return self.view()

    def step(self, frames: int) -> FrameBatch:
        with self._condition:
            if self._state not in {SessionState.CREATED, SessionState.PAUSED}:
                raise SessionConflictError("step is allowed only while created or paused")
            previous = self._latest_frame_id
            self._generate_locked(frames)
            return self._observation_batch_locked(previous, frames)

    def observation_snapshot(self) -> ObservationSnapshot:
        with self._lock:
            latest = self._buffer[-1].observation if self._buffer else None
            return ObservationSnapshot(status=self._status_locked(), observation=latest)

    def truth_snapshot(self) -> TruthSnapshot:
        with self._lock:
            latest = self._buffer[-1].truth if self._buffer else None
            return TruthSnapshot(status=self._status_locked(), truth=latest)

    def observation_batch(self, after_frame_id: int, limit: int) -> FrameBatch:
        with self._lock:
            return self._observation_batch_locked(after_frame_id, limit)

    def truth_batch(self, after_frame_id: int, limit: int) -> TruthFrameBatch:
        with self._lock:
            selected, gap = self._select_locked(after_frame_id, limit)
            return TruthFrameBatch(
                session_id=self.session_id,
                from_frame_id=(selected[0].truth.frame_id if selected else None),
                to_frame_id=(selected[-1].truth.frame_id if selected else None),
                gap_detected=gap,
                frames=tuple(frame.truth for frame in selected),
                status=self._status_locked(),
            )

    def wait_for_observations(
        self,
        after_frame_id: int,
        timeout_s: float,
        limit: int,
    ) -> FrameBatch:
        with self._condition:
            target_frames = min(
                limit,
                max(
                    1,
                    ceil(
                        self._configuration.sampling_rate_Hz
                        / self._configuration.ui_refresh_rate_Hz
                    ),
                ),
            )
            if self._state is not SessionState.STOPPED:
                self._condition.wait_for(
                    lambda: self._latest_frame_id - after_frame_id >= target_frames
                    or self._state is not SessionState.RUNNING,
                    timeout=timeout_s,
                )
            return self._observation_batch_locked(after_frame_id, limit)

    @property
    def stream_refresh_interval_s(self) -> float:
        return 1.0 / self._configuration.ui_refresh_rate_Hz

    def schedule_event(
        self,
        event: NetworkEventConfiguration,
    ) -> ScheduledEventResponse:
        with self._condition:
            if len(self._events) >= 128:
                raise SessionCapacityError("a session may schedule at most 128 events")
            if any(existing.event_id == event.event_id for existing in self._events):
                raise SessionConflictError(f"event_id already exists: {event.event_id}")
            if self._latest_frame_id > 0 and event.start_time_s <= self._sim_time_s:
                raise SessionConflictError(
                    "event start_time_s must be after the latest generated simulated time"
                )
            known = {node.sensor_id for node in self._configuration.nodes}
            unknown = set(event.target_sensor_ids) - known
            if unknown:
                raise SessionConflictError(
                    f"event targets unknown sensors: {', '.join(sorted(unknown))}"
                )
            self._events.append(event)
            self._configuration = NetworkSessionConfiguration.model_validate(
                {
                    **self._configuration.model_dump(),
                    "events": [item.model_dump() for item in self._events],
                }
            )
            self._configuration_version += 1
            self._event_records.append(
                _ScheduledEventRecord(
                    event=event,
                    effective_frame_id=self._latest_frame_id + 1,
                    configuration_version=self._configuration_version,
                )
            )
            self._updated_at = datetime.now(timezone.utc)
            effective = self._latest_frame_id + 1
            self._condition.notify_all()
            return ScheduledEventResponse(
                event=event,
                configuration_version=self._configuration_version,
                effective_frame_id=effective,
            )

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker_should_exit = False
        self._worker = Thread(
            target=self._run,
            name=f"aqse-network-{self.session_id[:8]}",
            daemon=True,
        )
        self._worker.start()

    def _run(self) -> None:
        interval_s = self._configuration.sample_interval_s / self._configuration.time_scale
        deadline = monotonic() + interval_s
        while True:
            generated = False
            with self._condition:
                if self._worker_should_exit:
                    return
                if self._state is not SessionState.RUNNING:
                    self._simulation_lag_s = 0.0
                    self._condition.wait(timeout=0.25)
                    deadline = monotonic() + interval_s
                    continue
                remaining = deadline - monotonic()
                if remaining > 0.0:
                    self._simulation_lag_s = 0.0
                    self._condition.wait(timeout=min(remaining, 0.25))
                    continue
                overdue = max(0.0, -remaining)
                due_frames = min(MAX_WORKER_BATCH_FRAMES, 1 + int(overdue / interval_s))
                self._generate_locked(due_frames)
                deadline += due_frames * interval_s
                self._simulation_lag_s = max(
                    0.0,
                    (monotonic() - deadline) * self._configuration.time_scale,
                )
                generated = True
            if generated:
                # Yield after every bounded batch so controls and snapshots cannot
                # be starved when accelerated simulation remains behind schedule.
                sleep(0)

    def _halt_worker(self) -> None:
        with self._condition:
            worker = self._worker
            self._worker_should_exit = True
            self._condition.notify_all()
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        if worker is not None and worker.is_alive():
            raise SessionConflictError(
                "simulation worker did not stop within the bounded control timeout"
            )
        with self._condition:
            self._worker = None

    def _reset_locked(self) -> None:
        self._state = SessionState.CREATED
        self._worker_should_exit = False
        self._latest_frame_id = 0
        self._sim_time_s = 0.0
        self._simulation_lag_s = 0.0
        self._overwritten_frames = 0
        self._buffer.clear()
        self._simulator = NetworkSimulator(self._configuration)
        self._updated_at = datetime.now(timezone.utc)
        self._condition.notify_all()

    def _generate_locked(self, frames: int) -> None:
        for _ in range(frames):
            frame_id = self._latest_frame_id + 1
            sim_time_s = (frame_id - 1) * self._configuration.sample_interval_s
            eligible_records = tuple(
                record
                for record in self._event_records
                if frame_id >= record.effective_frame_id
            )
            frame_configuration_version = max(
                (record.configuration_version for record in eligible_records),
                default=1,
            )
            frame = self._simulator.generate(
                session_id=self.session_id,
                frame_id=frame_id,
                sim_time_s=sim_time_s,
                epoch_utc=self._epoch_utc,
                events=(record.event for record in eligible_records),
                configuration_version=frame_configuration_version,
            )
            if len(self._buffer) == self._buffer.maxlen:
                self._overwritten_frames += 1
            self._buffer.append(frame)
            self._latest_frame_id = frame_id
            self._sim_time_s = sim_time_s
        self._updated_at = datetime.now(timezone.utc)
        self._condition.notify_all()

    def _select_locked(
        self,
        after_frame_id: int,
        limit: int,
    ) -> tuple[list[BufferedFrame], bool]:
        if not self._buffer:
            return [], False
        oldest = self._buffer[0].observation.frame_id
        gap = after_frame_id < oldest - 1
        selected = [
            frame
            for frame in self._buffer
            if frame.observation.frame_id > after_frame_id
        ][:limit]
        return selected, gap

    def _observation_batch_locked(
        self,
        after_frame_id: int,
        limit: int,
    ) -> FrameBatch:
        selected, gap = self._select_locked(after_frame_id, limit)
        return FrameBatch(
            session_id=self.session_id,
            from_frame_id=(selected[0].observation.frame_id if selected else None),
            to_frame_id=(selected[-1].observation.frame_id if selected else None),
            gap_detected=gap,
            frames=tuple(frame.observation for frame in selected),
            status=self._status_locked(),
        )

    def _status_locked(self) -> SessionStatus:
        oldest = self._buffer[0].observation.frame_id if self._buffer else None
        newest = self._buffer[-1].observation.frame_id if self._buffer else None
        return SessionStatus(
            session_id=self.session_id,
            state=self._state,
            configuration_version=self._configuration_version,
            sim_time_s=self._sim_time_s,
            latest_frame_id=self._latest_frame_id,
            created_at=self._created_at,
            updated_at=self._updated_at,
            simulation_lag_s=self._simulation_lag_s,
            buffer=BufferStatus(
                size=len(self._buffer),
                capacity=self._buffer.maxlen or 1,
                oldest_frame_id=oldest,
                newest_frame_id=newest,
                overwritten_frames=self._overwritten_frames,
            ),
        )


class NetworkSessionRegistry:
    def __init__(self, maximum_sessions: int = 16) -> None:
        self.maximum_sessions = maximum_sessions
        self._sessions: dict[str, NetworkSession] = {}
        self._lock = RLock()

    def create(self, configuration: NetworkSessionConfiguration) -> NetworkSession:
        with self._lock:
            if len(self._sessions) >= self.maximum_sessions:
                raise SessionCapacityError(
                    f"in-memory registry is limited to {self.maximum_sessions} sessions"
                )
            session = NetworkSession(configuration)
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> NetworkSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.stop()
        return True

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.stop()


network_sessions = NetworkSessionRegistry()
