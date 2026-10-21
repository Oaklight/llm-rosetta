"""Command line entry point: ``python -m tools.conformance``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections.abc import Sequence

from .runner import run_suite
from .suites import SUITES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.conformance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a probe suite")
    run.add_argument("--suite", required=True, choices=sorted(SUITES))
    run.add_argument(
        "--out",
        default="-",
        help="report path; '-' writes to stdout (default)",
    )
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--retries", type=int, default=0)
    run.add_argument("--max-workers", type=int, default=None)
    run.add_argument(
        "--secret-env",
        action="append",
        default=[],
        metavar="NAME",
        help="environment variable whose value must be redacted (repeatable)",
    )

    args = parser.parse_args(argv)

    probes, environment = SUITES[args.suite]()
    secrets = [os.environ[name] for name in args.secret_env if name in os.environ]

    report = run_suite(
        args.suite,
        probes,
        timeout=args.timeout,
        retries=args.retries,
        max_workers=args.max_workers,
        environment=environment,
        secrets=secrets,
    )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)

    if args.out == "-":
        print(payload)
    else:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n")

    return 0 if report.ok else 1
