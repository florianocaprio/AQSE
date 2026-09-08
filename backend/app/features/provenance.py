from __future__ import annotations

import hashlib
import hmac
import json
import secrets

from app.features.models import FeatureProfile, WindowFeatureRecord

# Ephemeral by design: browser-session artifacts must be recomputed after backend restart.
_PROVENANCE_KEY = secrets.token_bytes(32)


def _payload(profile: FeatureProfile, record: WindowFeatureRecord) -> bytes:
    return json.dumps(
        {
            "profile": profile.model_dump(mode="json"),
            "record": record.model_dump(
                mode="json",
                exclude={"provenance_token"},
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_window_record(
    profile: FeatureProfile,
    record: WindowFeatureRecord,
) -> str:
    digest = hmac.new(_PROVENANCE_KEY, _payload(profile, record), hashlib.sha256)
    return "feature-window-v1." + digest.hexdigest()


def verify_window_record(
    profile: FeatureProfile,
    record: WindowFeatureRecord,
) -> bool:
    expected = sign_window_record(profile, record)
    return hmac.compare_digest(expected, record.provenance_token)
