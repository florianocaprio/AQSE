"""Print the bounded Milestone 1D.0 audit evidence as concise JSON."""

from __future__ import annotations

import json
from dataclasses import asdict

from research_audit.probes import run_all_probes


def main() -> None:
    """Run the audit probes without writing datasets or other artifacts."""

    print(json.dumps(asdict(run_all_probes()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
