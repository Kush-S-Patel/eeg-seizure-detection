"""Audit selected local EDFs and sidecars."""

from __future__ import annotations

import argparse

from seizure_detector.audit import audit_dataset, print_report, save_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Check files without opening EDF headers")
    args = parser.parse_args()
    report = audit_dataset(read_headers=not args.fast)
    path = save_report(report)
    ok = print_report(report)
    print(f"\nAudit written to {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
