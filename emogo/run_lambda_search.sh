#!/bin/bash
# AESL lambda search on the grids EmoGrowth states in Appendix B.4.
#
# Eq. 15 of the paper is
#     L = L_ce + lambda1 * L_kd_model + lambda2 * L_kd_aff + lambda3 * L_le
# and B.4 says lambda1 = 1, while lambda2 and lambda3 are *searched per
# dataset*:
#     lambda2 in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
#     lambda3 in {0.001, 0.01, 0.1, 1, 2, 5, 10}
#
# The released multi_label.json ships lambda2 = 0.1 and lambda3 = 0.005 —
# neither is on the paper's grid — and those are the values this project
# inherited. GoEmotions is a new dataset, so by the reference's own protocol
# the search has to be repeated here.
#
# Coordinate-wise, not the full 49-cell grid: sweep lambda2 with lambda3 at
# its inherited value, then sweep lambda3 at the best lambda2. 14 runs.
#
#   ./run_lambda_search.sh            # both sweeps, seed 1993
#   ./run_lambda_search.sh lam2       # lambda2 only
#   ./run_lambda_search.sh lam3 1994
set -e

WHICH=${1:-both}
SEED=${2:-1993}

L2="0.2 0.3 0.4 0.5 0.6 0.7 0.8"
L3="0.001 0.01 0.1 1 2 5 10"

run () {
  echo "=== aesl $1 / B0-I7 / seed ${SEED} ==="
  python main.py --config "exps/aesl-$1_B0-I7.json" --seed "${SEED}"
}

if [ "${WHICH}" = "both" ] || [ "${WHICH}" = "lam2" ]; then
  for V in ${L2}; do run "lam2-${V}"; done
fi

if [ "${WHICH}" = "both" ] || [ "${WHICH}" = "lam3" ]; then
  for V in ${L3}; do run "lam3-${V}"; done
fi

echo
echo "=== last mAP per setting ==="
python - <<'PY'
import glob, json, os, re
rows = []
for d in sorted(glob.glob("results/aesl-lam*/B0-I7/seed*")):
    f = os.path.join(d, "summary.json")
    if not os.path.isfile(f):
        continue
    s = json.load(open(f))
    tag = re.search(r"aesl-(lam\d-[\d.]+)", d).group(1)
    rows.append((tag, 100 * s["last_acc"]["map"], 100 * s["avg_acc"]["map"]))
if not rows:
    print("no runs found under results/")
else:
    print(f"{'setting':<16}{'last mAP':>10}{'avg mAP':>10}")
    for t, last, avg in rows:
        print(f"{t:<16}{last:>10.1f}{avg:>10.1f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest last mAP: {best[0]} at {best[1]:.1f}")
    print("inherited setting for comparison: lamda2=0.1, lamda3=0.005 "
          "(neither on the paper's grid)")
PY
