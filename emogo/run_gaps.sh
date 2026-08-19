#!/bin/bash
# The gaps the review panel named that are still open, in priority order.
# Each block is independent; run what you need.
#
#   ./run_gaps.sh buffer      12 runs  — equalise ER's memory budget
#   ./run_gaps.sh ablation    12 runs  — AGCN/AESL with the graph branch off
#   ./run_gaps.sh batch        3 runs  — AESL at the reference's batch size
#   ./run_gaps.sh emobank     12 runs  — the second VAD source, already configured
#   ./run_gaps.sh order       60 runs  — class order at 3 seeds, 6 methods
#   ./run_gaps.sh seeds      180 runs  — extend the main grid from 3 to 8 seeds
set -e

WHAT=${1:-}
PROTOCOLS="B0-I7 B0-I4 B16-I3 B16-I2"
SEEDS="1993 1994 1995"

case "${WHAT}" in

  # ER runs on memory_per_class (20/class, so 140 -> 280 -> 420) while RS, PRS
  # and OCDM hit the 500 cap. fixed_memory=false routes ER through
  # memory_size//total_classes instead, giving it ~500 throughout.
  buffer)
    for P in ${PROTOCOLS}; do for S in ${SEEDS}; do
      echo "=== er / ${P} / equal budget / seed ${S} ==="
      python main.py --config "exps/er_${P}.json" --seed "${S}" \
        --fixed_memory false --tag "eq"
    done; done
    ;;

  # AGCN is LwF's loss evaluated through the graph. With no edges the
  # normaliser returns exactly I, so this isolates what the graph contributes.
  # If AGCN-empty lands on LwF, every AGCN-LwF gap is the graph branch; if it
  # does not, the gap is architecture and that has to be said instead.
  ablation)
    for M in agcn aesl; do for P in B0-I7 B16-I2; do for S in ${SEEDS}; do
      echo "=== ${M} / ${P} / graph branch off / seed ${S} ==="
      python main.py --config "exps/${M}_${P}.json" --seed "${S}" \
        --adj_estimator empty --tag "nograph"
    done; done; done
    ;;

  # AESL's relation-KD is the only loss here defined over batch statistics.
  # EmoGrowth uses 128; this project uses 16, i.e. 240 off-diagonal entries
  # per matrix instead of 16256.
  batch)
    for S in ${SEEDS}; do
      echo "=== aesl / B0-I7 / batch 128 / seed ${S} ==="
      python main.py --config "exps/aesl_B0-I7.json" --seed "${S}" \
        --batch_size 128 --tag "b128"
    done
    ;;

  # Run precompute_vad.py --source emobank first.
  emobank)
    for P in ${PROTOCOLS}; do for S in ${SEEDS}; do
      echo "=== aesl-emobank / ${P} / seed ${S} ==="
      python main.py --config "exps/aesl-emobank_${P}.json" --seed "${S}"
    done; done
    ;;

  # Class-order variance is currently one seed per order, which cannot
  # separate order variance from seed variance. Six methods covers every
  # family: none, distillation, replay, balanced replay, graph, graph+losses.
  order)
    for M in finetune lwf er prs agcn aesl; do
      for K in 1 2 3 4 5; do for S in ${SEEDS}; do
        echo "=== ${M} / order${K} / B0-I7 / seed ${S} ==="
        python main.py --config "exps/${M}-order${K}_B0-I7.json" --seed "${S}"
      done; done
    done
    ;;

  # n=3 gives an exact permutation test a floor of p=0.100, so no
  # distribution-free version of any significance claim exists. n=8 brings the
  # floor to 0.00016.
  seeds)
    for M in finetune ewc lwf er rs ocdm prs agcn aesl; do
      for P in ${PROTOCOLS}; do
        for S in 1996 1997 1998 1999 2000; do
          echo "=== ${M} / ${P} / seed ${S} ==="
          python main.py --config "exps/${M}_${P}.json" --seed "${S}"
        done
      done
    done
    ;;

  *)
    sed -n '2,12p' "$0" >&2
    exit 1
    ;;
esac
