"""Assert that emogo's BERT settings match the verified ../bertgo replication.

Run this before any experiment, and after touching either config:

    python check_config.py                       # checks defaults
    python check_config.py exps/lwf_B0-I7.json   # checks a specific run

Exits non-zero on any mismatch. The point is that ../bertgo is the reference
configuration (and the Upper-bound row for these tables); if the incremental
runs quietly drift away from it, every comparison against that baseline stops
meaning anything.
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "bertgo"))

from utils.config import Config, load_config  # noqa: E402

# emogo field -> bertgo field. Same name unless stated.
SHARED = {
    "model_name": "model_name",
    "do_lower_case": "do_lower_case",
    "max_seq_length": "max_seq_length",
    "classifier_dropout": "classifier_dropout",
    "batch_size": "train_batch_size",
    "eval_batch_size": "eval_batch_size",
    "learning_rate": "learning_rate",
    "warmup_proportion": "warmup_proportion",
    "weight_decay_rate": "weight_decay_rate",
    "adam_beta1": "adam_beta1",
    "adam_beta2": "adam_beta2",
    "adam_epsilon": "adam_epsilon",
    "max_grad_norm": "max_grad_norm",
    "exclude_from_weight_decay": "exclude_from_weight_decay",
    "drop_last": "drop_last",
    "fp16": "fp16",
}

# Differences that are intentional, with the reason. Printed, never asserted.
EXPECTED_DIFFERENCES = {
    "epochs": "bertgo trains 4.0 epochs once; emogo trains 4 epochs *per task* "
              "(init_epochs/epochs), each with its own warmup+decay cycle.",
    "seed": "EmoGrowth/PyCIL convention is 1993; bertgo used 42. Affects init "
            "and shuffling only — keep it fixed across methods within a table.",
    "threshold": "bertgo binarises at probability > 0.3 (GoEmotions Table 4); "
                 "emogo binarises at logit > 0 i.e. probability > 0.5, the "
                 "EmoGrowth utils/metrics.py convention. Raw logits are saved "
                 "per task so either can be recomputed.",
    "data": "bertgo uses train/dev/test; emogo uses train/test only, since the "
            "incremental protocol has no per-task model selection.",
}


def main() -> int:
    import config as bertgo_config  # noqa: E402  (needs sys.path above)

    ref = bertgo_config.Config()
    cfg = load_config(sys.argv[1]) if len(sys.argv) > 1 else Config()
    label = sys.argv[1] if len(sys.argv) > 1 else "Config() defaults"

    print(f"Comparing emogo [{label}] against ../bertgo/config.py\n")
    print(f"{'field':<32}{'emogo':<38}{'bertgo':<38}")
    print("-" * 108)

    bad = []
    for mine, theirs in SHARED.items():
        a, b = getattr(cfg, mine), getattr(ref, theirs)
        ok = a == b
        flag = "" if ok else "  <-- MISMATCH"
        name = mine if mine == theirs else f"{mine} = {theirs}"
        print(f"{name:<32}{str(a):<38}{str(b):<38}{flag}")
        if not ok:
            bad.append((mine, a, theirs, b))

    print("\nIntentional differences (not asserted):")
    for key, why in EXPECTED_DIFFERENCES.items():
        print(f"  - {key}: {why}")

    if bad:
        print(f"\nFAIL — {len(bad)} mismatch(es):")
        for mine, a, theirs, b in bad:
            print(f"  emogo.{mine} = {a!r} but bertgo.{theirs} = {b!r}")
        return 1

    print(f"\nOK — all {len(SHARED)} shared BERT settings match ../bertgo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
