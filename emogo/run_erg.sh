#!/bin/bash
# Experiments on how the emotional relation graph is *estimated*.
#
# Motivation, measured on this corpus (analysis in the paper's Section 3):
# of the 378 label pairs, 11% never co-occur, 36% co-occur fewer than five
# times, 50% fewer than ten. The median relative standard error of P(i|j) is
# 30%. Per task it is worse — under B16-I2 three of the six increments have
# *zero* co-occurrences, so the block the graph branch propagates over is
# identically empty, while under B0-I7 the base task has 1013 co-occurrences
# over 21 pairs with no empty cell. The LwF-AGCN gap tracks that: 1.2 mAP at
# B0-I7, 10.6 at B16-I2.
#
# Three experiments:
#
#   subsample  Causal test. Thin the rows the graph is estimated from while
#              the classifier still trains on the full task, on B0-I7 where
#              the graph currently works. If estimation quality is what drives
#              the damage, AGCN should degrade toward its B16-I2 behaviour as
#              the fraction falls. If it does not, the hypothesis is wrong and
#              the shrinkage experiments below have no basis.
#
#   alpha      Shrinkage sweep on B16-I2, where the adjacency is worst, to
#              locate a working alpha. alpha = 0 is the raw estimator, i.e.
#              the runs already in results/.
#
#   confirm    The chosen alpha across all four protocols and three seeds,
#              plus the pooled variant. The prediction is differential: large
#              recovery at B16-I2 and B16-I3, little change at B0-I7. A
#              uniform improvement everywhere would mean the mechanism as
#              stated is wrong, and that has to be reported.
#
#   ./run_erg.sh subsample
#   ./run_erg.sh alpha
#   ./run_erg.sh confirm 10          # alpha chosen from the sweep
set -e

WHAT=${1:-subsample}
ALPHA=${2:-10}
METHODS="agcn aesl"

case "${WHAT}" in

  subsample)
    for M in ${METHODS}; do
      for F in 0.5 0.25 0.1 0.05; do
        for S in 1993 1994 1995; do
          echo "=== ${M} / B0-I7 / graph from ${F} of rows / seed ${S} ==="
          python main.py --config "exps/${M}_B0-I7.json" --seed "${S}" \
            --adj_subsample "${F}" --tag "sub${F}"
        done
      done
    done
    ;;

  alpha)
    for M in ${METHODS}; do
      for A in 1 5 10 25 50; do
        echo "=== ${M} / B16-I2 / shrink alpha=${A} / seed 1993 ==="
        python main.py --config "exps/${M}_B16-I2.json" --seed 1993 \
          --adj_estimator shrink --adj_alpha "${A}" --tag "shrink${A}"
      done
    done
    ;;

  confirm)
    for M in ${METHODS}; do
      for P in B0-I7 B0-I4 B16-I3 B16-I2; do
        for S in 1993 1994 1995; do
          echo "=== ${M} / ${P} / shrink alpha=${ALPHA} / seed ${S} ==="
          python main.py --config "exps/${M}_${P}.json" --seed "${S}" \
            --adj_estimator shrink --adj_alpha "${ALPHA}" --tag "shrink"
          echo "=== ${M} / ${P} / shrink_pool alpha=${ALPHA} / seed ${S} ==="
          python main.py --config "exps/${M}_${P}.json" --seed "${S}" \
            --adj_estimator shrink_pool --adj_alpha "${ALPHA}" --tag "pool"
        done
      done
    done
    ;;

  *)
    echo "usage: $0 {subsample|alpha|confirm [alpha]}" >&2
    exit 1
    ;;
esac

echo
echo "=== last mAP ==="
python - <<'PY'
import glob, json, os, re
import statistics as st

rows = {}
for d in sorted(glob.glob("results/*/*/seed*")):
    f = os.path.join(d, "summary.json")
    if not os.path.isfile(f):
        continue
    m = re.match(r"results/([^/]+)/([^/]+)/seed(\d+)", d)
    name, proto = m.group(1), m.group(2)
    if not re.match(r"(agcn|aesl|lwf)", name):
        continue
    rows.setdefault((name, proto), []).append(
        100 * json.load(open(f))["last_acc"]["map"])

if not rows:
    print("no runs found under results/")
else:
    print(f"{'variant':<22}{'protocol':<10}{'n':>3}{'last mAP':>11}{'sd':>7}")
    for (name, proto), v in sorted(rows.items()):
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"{name:<22}{proto:<10}{len(v):>3}{st.mean(v):>11.1f}{sd:>7.1f}")
    print("\nRead the shrink rows against the plain agcn/aesl rows for the same "
          "protocol, and against lwf as the no-graph reference.")
PY
