#!/bin/bash
# Class-order robustness check, B0-I7.
#
# EmoGrowth's own Limitations (Appendix D) says: "the impact of the order of
# learning emotion categories and the number of emotion categories in each
# task on the experimental results needs to be further explored." Every run
# in this project so far uses one alphabetical order, so the reported +/-sd
# carries initialisation noise only.
#
# The five orders in exps/class_orders.json are drawn once from a dedicated
# RNG and passed as explicit lists, so cfg.seed stays at 1993 and the spread
# across runs is class-order variance alone, not order mixed with init.
#
#   ./run_class_order.sh                     # 4 methods x 5 orders = 20 runs
#   ./run_class_order.sh "lwf aesl"          # a subset
set -e

METHODS=${1:-"lwf agcn aesl prs"}
SEED=${2:-1993}

for M in ${METHODS}; do
  for K in 1 2 3 4 5; do
    echo "=== ${M} / order${K} / B0-I7 / seed ${SEED} ==="
    python main.py --config "exps/${M}-order${K}_B0-I7.json" --seed "${SEED}"
  done
done

echo
echo "=== last mAP by class order ==="
python - <<'PY'
import glob, json, os, re
import statistics as st

rows = {}
for d in sorted(glob.glob("results/*-order*/B0-I7/seed*")):
    f = os.path.join(d, "summary.json")
    if not os.path.isfile(f):
        continue
    m = re.search(r"results/(\w+)-order(\d)", d)
    rows.setdefault(m.group(1), {})[m.group(2)] = 100 * json.load(open(f))["last_acc"]["map"]

if not rows:
    print("no runs found under results/")
else:
    ks = sorted({k for v in rows.values() for k in v})
    print(f"{'method':<10}" + "".join(f"{'order'+k:>9}" for k in ks) + f"{'sd':>8}")
    for m, v in rows.items():
        vals = [v[k] for k in ks if k in v]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{m:<10}" + "".join(f"{v.get(k, float('nan')):>9.1f}" for k in ks)
              + f"{sd:>8.1f}")
    print("\nCompare each sd against the seed spread in Table 2. If the order "
          "spread is larger, the reported +/-sd understates the uncertainty.")
PY
