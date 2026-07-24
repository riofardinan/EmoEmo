"""Assert this folder's BERT settings still match the verified reference.

    python check_config.py                        # checks defaults
    python check_config.py exps/lwf_B0-I7.json    # checks one run
    python check_config.py --against ../bertgo    # also re-check the snapshot

Exits non-zero on any mismatch. `run_all.sh` calls it before every sweep.

The reference lives in `utils/reference.py` as literals, so this folder runs
standalone — `../bertgo` does not need to exist. Pass `--against <path>` to a
bertgo checkout to additionally confirm the snapshot has not drifted from it.
"""

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config, load_config  # noqa: E402
from utils.reference import (  # noqa: E402
    BERT_REFERENCE,
    INTENTIONAL_DIFFERENCES,
)


def check_snapshot(cfg, label: str) -> int:
    print(f"Checking emogo [{label}] against utils/reference.py\n")
    print(f"{'field':<30}{'this run':<40}{'reference':<40}")
    print("-" * 110)

    bad = []
    for field, (expected, _, _) in BERT_REFERENCE.items():
        actual = getattr(cfg, field)
        ok = actual == expected
        print(f"{field:<30}{str(actual):<40}{str(expected):<40}"
              f"{'' if ok else '  <-- MISMATCH'}")
        if not ok:
            bad.append((field, actual, expected))

    print("\nIntentional differences from bertgo (not asserted):")
    for key, why in INTENTIONAL_DIFFERENCES.items():
        print(f"  - {key}: {why}")

    if bad:
        print(f"\nFAIL — {len(bad)} mismatch(es):")
        for field, actual, expected in bad:
            print(f"  {field} = {actual!r}, reference says {expected!r}")
            print(f"      ({BERT_REFERENCE[field][2]})")
        return 1

    print(f"\nOK — all {len(BERT_REFERENCE)} BERT settings match the reference.")
    return 0


def check_against_bertgo(path: str) -> int:
    """Confirm the frozen snapshot still agrees with a live bertgo checkout."""
    config_py = os.path.join(path, "config.py")
    if not os.path.isfile(config_py):
        print(f"\n--against: no config.py under {path}, skipping.")
        return 0

    spec = importlib.util.spec_from_file_location("bertgo_config", config_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = module.Config()

    print(f"\nCross-checking utils/reference.py against {config_py}")
    drift = []
    for field, (expected, their_field, _) in BERT_REFERENCE.items():
        if not hasattr(ref, their_field):
            drift.append((field, their_field, expected, "<missing>"))
            continue
        theirs = getattr(ref, their_field)
        if theirs != expected:
            drift.append((field, their_field, expected, theirs))

    if drift:
        print(f"DRIFT — {len(drift)} field(s) differ from the live bertgo:")
        for field, their_field, expected, theirs in drift:
            print(f"  reference.{field} = {expected!r} but "
                  f"bertgo.{their_field} = {theirs!r}")
        print("Update utils/reference.py if the bertgo change was intended.")
        return 1

    print(f"OK — the snapshot matches bertgo on all {len(BERT_REFERENCE)} fields.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", nargs="?", default=None,
                    help="Config JSON to check. Defaults to Config() defaults.")
    ap.add_argument("--against", metavar="PATH", default=None,
                    help="Path to a bertgo checkout, to re-verify the snapshot.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    label = args.config or "Config() defaults"

    status = check_snapshot(cfg, label)
    if args.against:
        status |= check_against_bertgo(args.against)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
