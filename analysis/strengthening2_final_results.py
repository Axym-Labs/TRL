"""Summarize the frozen strengthening-2 MNIST matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from terel.resubmission.strengthening2_analysis import (
    analyze_capture24_order,
    analyze_final_results,
    analyze_online_continuation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--failure-ledger", type=Path)
    parser.add_argument("--online-continuation-results", type=Path)
    parser.add_argument("--capture24-order-results", type=Path)
    parser.add_argument("--capture24-order-seeds", type=int, nargs="+")
    parser.add_argument(
        "--capture24-order-role",
        choices=("validation", "heldout"),
        default="validation",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    analysis = {
        "mnist": analyze_final_results(
            arguments.results,
            failure_ledger=arguments.failure_ledger,
        )
    }
    if arguments.online_continuation_results is not None:
        analysis["mnist"]["online_continuation"] = analyze_online_continuation(
            arguments.online_continuation_results
        )
    if arguments.capture24_order_results is not None:
        analysis["capture24_order"] = analyze_capture24_order(
            arguments.capture24_order_results,
            expected_seeds=(
                tuple(arguments.capture24_order_seeds)
                if arguments.capture24_order_seeds is not None
                else (501, 502, 503)
            ),
            role=arguments.capture24_order_role,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
