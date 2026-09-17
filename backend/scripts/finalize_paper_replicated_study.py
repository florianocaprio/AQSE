from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.paper_study.recovery import recover_completed_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize one completed AQSE study from persisted results only."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reporting-source-commit", required=True)
    arguments = parser.parse_args()
    result = recover_completed_study(
        source=arguments.source,
        output=arguments.output,
        reporting_source_commit=arguments.reporting_source_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
