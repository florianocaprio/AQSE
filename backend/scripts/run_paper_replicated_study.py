from __future__ import annotations

import argparse
from pathlib import Path

from app.paper_study.execution import execute_preregistered_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the frozen AQSE N=12/N=12/N=30 offline study."
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("/artifacts"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-document-sha256", required=True)
    args = parser.parse_args()
    result = execute_preregistered_study(
        artifact_root=args.artifact_root,
        source_commit=args.source_commit,
        protocol_document_sha256=args.protocol_document_sha256,
    )
    print(result)


if __name__ == "__main__":
    main()
