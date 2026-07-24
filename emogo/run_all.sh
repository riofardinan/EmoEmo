#!/bin/bash
# Run methods across the four 28-class protocols.
#   ./run_all.sh                    -> finetune, ewc, lwf, er, rs (20 runs)
#   ./run_all.sh lwf                -> lwf only, 4 runs
#   ./run_all.sh "finetune lwf" 1994
set -e

METHODS=${1:-"finetune ewc lwf er rs"}
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
