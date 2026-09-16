from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.training.canonical import canonical_json_bytes

from .datasets import ReplicatedDataset


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _npy_bytes(value: NDArray[Any]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


class ReplicaStore:
    """Single-use TEST store for one replica.

    The append-only sequence prevents accidental re-evaluation in this local
    workflow.  It does not provide distributed consensus or protection against
    a malicious filesystem administrator.
    """

    def __init__(self, root: Path):
        self.root = root
        self.ledger_path = root / "test-access-ledger.jsonl"

    def publish(self, dataset: ReplicatedDataset, protocol: dict[str, Any]) -> None:
        if self.root.exists():
            raise FileExistsError(f"replica artifact directory already exists: {self.root}")
        self.root.mkdir(parents=True)
        test_observations = _npy_bytes(dataset.X_test)
        test_labels = _npy_bytes(dataset.y_test)
        _atomic_write(self.root / "test-observations.npy", test_observations)
        _atomic_write(self.root / "test-labels.npy", test_labels)
        _atomic_write(self.root / "protocol.json", _json_bytes(protocol))
        self._append_event(
            "sealed",
            {
                "test_observation_sha256": _sha256(test_observations),
                "test_label_sha256": _sha256(test_labels),
                "protocol_sha256": _sha256(_json_bytes(protocol)),
            },
        )

    def freeze_selection(self, selection: dict[str, Any]) -> None:
        entries = self.read_ledger()
        if [entry["event"] for entry in entries] != ["sealed"]:
            raise PermissionError("model selection must freeze before TEST access")
        path = self.root / "model-selection-freeze.json"
        if path.exists():
            raise FileExistsError("model selection is already frozen")
        _atomic_write(path, _json_bytes(selection))

    def open_test_observations(self) -> NDArray[np.float64]:
        if not (self.root / "model-selection-freeze.json").is_file():
            raise PermissionError("TEST observations require a model-selection freeze")
        entries = self.read_ledger()
        if [entry["event"] for entry in entries] != ["sealed"]:
            raise PermissionError("TEST observations may be opened exactly once")
        self._append_event("observations_opened", {})
        result = np.asarray(
            np.load(self.root / "test-observations.npy", allow_pickle=False),
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    def open_test_labels(self) -> NDArray[np.int8]:
        entries = self.read_ledger()
        if [entry["event"] for entry in entries] != ["sealed", "observations_opened"]:
            raise PermissionError("TEST labels require the single observation access")
        self._append_event("labels_opened", {})
        result = np.asarray(
            np.load(self.root / "test-labels.npy", allow_pickle=False),
            dtype=np.int8,
        )
        result.setflags(write=False)
        return result

    def publish_evaluation(self, result: dict[str, Any]) -> None:
        entries = self.read_ledger()
        if [entry["event"] for entry in entries] != [
            "sealed",
            "observations_opened",
            "labels_opened",
        ]:
            raise PermissionError("evaluation publication requires the declared TEST sequence")
        payload = _json_bytes(result)
        _atomic_write(self.root / "result.json", payload)
        self._append_event("evaluation_published", {"result_sha256": _sha256(payload)})

    def verify_closed(self) -> tuple[dict[str, Any], ...]:
        entries = self.read_ledger()
        if [entry["event"] for entry in entries] != [
            "sealed",
            "observations_opened",
            "labels_opened",
            "evaluation_published",
        ]:
            raise ValueError("replica TEST ledger is not closed")
        for index, entry in enumerate(entries):
            scientific = {
                key: value for key, value in entry.items() if key != "entry_sha256"
            }
            if entry["entry_sha256"] != _sha256(canonical_json_bytes(scientific)):
                raise ValueError("replica TEST ledger entry digest is invalid")
            expected_previous = None if index == 0 else entries[index - 1]["entry_sha256"]
            if entry["previous_entry_sha256"] != expected_previous:
                raise ValueError("replica TEST ledger chain is invalid")
        return entries

    def read_ledger(self) -> tuple[dict[str, Any], ...]:
        if not self.ledger_path.exists():
            return ()
        entries = tuple(
            json.loads(line)
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines()
            if line
        )
        return entries

    def _append_event(self, event: str, details: dict[str, Any]) -> None:
        entries = self.read_ledger()
        scientific = {
            "schema_version": "aqse.paper-replica-test-ledger.v1",
            "sequence": len(entries),
            "event": event,
            "previous_entry_sha256": (
                None if not entries else entries[-1]["entry_sha256"]
            ),
            "details": details,
        }
        entry = {
            **scientific,
            "entry_sha256": _sha256(canonical_json_bytes(scientific)),
        }
        payload = _json_bytes(entry)
        descriptor = os.open(
            self.ledger_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
