from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.training.assembly import assemble_training_input  # noqa: E402
from app.training.runner import (  # noqa: E402
    canonical_theta0,
    cross_engine_one_step_gate,
    wrapper_equivalence,
)
from app.training.seal import verify_canonical_test_seal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate mandatory AQSE 1D.3 QNG gates without publishing a run."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--train-bank-path", type=Path, required=True)
    arguments = parser.parse_args()
    dataset_path = arguments.dataset_path.resolve()
    seal_before = verify_canonical_test_seal(dataset_path)
    training_input = assemble_training_input(
        dataset_path=dataset_path,
        encoder_path=arguments.encoder_path.resolve(),
        train_bank_path=arguments.train_bank_path.resolve(),
    )
    theta0 = canonical_theta0()
    equivalence = wrapper_equivalence(
        training_input.X_train,
        training_input.y_train,
        theta0,
        steps=3,
    )
    cross_engine = cross_engine_one_step_gate(
        training_input.X_train,
        training_input.y_train,
        theta0,
    )
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during QNG preflight")
    print(
        json.dumps(
            {
                "training_input_fingerprint": training_input.identity.fingerprint_id,
                "training_input_digest": training_input.identity.content_digest,
                "theta0": theta0.tolist(),
                "wrapper_equivalence": equivalence,
                "cross_engine": cross_engine,
                "test_state": seal_before[0].test_state,
                "test_ledger_sha256": seal_before[1],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
