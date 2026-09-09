from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.training.canonical import canonical_json_bytes
from app.training.models import DatasetManifest, LegacyDatasetManifest, TestAccessLedgerEntry
from app.training.storage import LEDGER_NAME, MAX_JSON_BYTES, verify_archive_opaque

CANONICAL_DATASET_ID = "aqse-development-064acca20fc788c6"
CANONICAL_DATASET_DIGEST = (
    "064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7"
)
CANONICAL_LEDGER_SHA256 = (
    "210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6"
)


def verify_canonical_test_seal(dataset_path: Path) -> tuple[DatasetManifest, str]:
    if os.environ.get("AQSE_ALLOW_TEST_OPEN") == "1":
        raise PermissionError("AQSE_ALLOW_TEST_OPEN must remain disabled during 1D.3")
    resolved = dataset_path.resolve()
    manifest = verify_archive_opaque(resolved)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("AQSE 1D.3 requires an archive-v2 dataset")
    if (
        manifest.dataset_id != CANONICAL_DATASET_ID
        or manifest.scientific_digest != CANONICAL_DATASET_DIGEST
        or manifest.test_state != "sealed"
    ):
        raise ValueError("canonical dataset or TEST seal identity is incompatible")
    ledger_path = resolved / LEDGER_NAME
    if (
        ledger_path.is_symlink()
        or not ledger_path.is_file()
        or ledger_path.stat().st_size > MAX_JSON_BYTES
    ):
        raise ValueError("canonical TEST ledger is missing, unsafe or oversized")
    payload = ledger_path.read_bytes()
    lines = [line for line in payload.splitlines() if line]
    if len(lines) != 1:
        raise ValueError("canonical TEST ledger must contain one genesis event")
    entry = TestAccessLedgerEntry.model_validate_json(lines[0])
    entry_digest = hashlib.sha256(
        canonical_json_bytes(entry.model_dump(mode="json", exclude={"entry_sha256"}))
    ).hexdigest()
    if (
        entry.sequence != 0
        or entry.event != "sealed"
        or entry.dataset_id != manifest.dataset_id
        or entry.scientific_digest != manifest.scientific_digest
        or entry.previous_entry_sha256 is not None
        or entry.fixture_only
        or entry.entry_sha256 != entry_digest
    ):
        raise ValueError("canonical TEST ledger genesis event is invalid")
    ledger_digest = hashlib.sha256(payload).hexdigest()
    if ledger_digest != CANONICAL_LEDGER_SHA256:
        raise ValueError("canonical TEST ledger SHA-256 changed")
    return manifest, ledger_digest
