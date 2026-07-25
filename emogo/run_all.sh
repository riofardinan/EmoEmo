#!/bin/bash
# Run methods across the four 28-class protocols.
#   ./run_all.sh                    -> all 9 methods (36 runs)
#   ./run_all.sh lwf                -> lwf only, 4 runs
#   ./run_all.sh "ocdm aesl"        -> a subset
#   ./run_all.sh "finetune lwf" 1994
#
# AESL needs its affective cache built first (see precompute_vad.py). The
# aesl-emobank variant is a separate method name: ./run_all.sh aesl-emobank
set -e

METHODS=${1:-"finetune ewc lwf er rs ocdm prs agcn aesl"}
SEED=${2:-1993}
PROTOCOLS="B0-I7 B0-I4 B16-I3 B16-I2"

# Fail fast if the BERT settings have drifted from the verified ../bertgo
# replication — every table here is read against that baseline.
for M in ${METHODS}; do
  python check_config.py "exps/${M}_B0-I7.json" > /dev/null
done
echo "config check passed for: ${METHODS}"

for M in ${METHODS}; do
  for P in ${PROTOCOLS}; do
    echo "=== ${M} / ${P} / seed ${SEED} ==="
    python main.py --config "exps/${M}_${P}.json" --seed "${SEED}"
  done
done

echo
echo "=== summary ==="
for P in ${PROTOCOLS}; do
  python compare.py --protocol "${P}" --seed "${SEED}"
done
