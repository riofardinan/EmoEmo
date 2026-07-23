#!/bin/bash
# Run one method across all four 28-class protocols.
#   ./run_all.sh              -> finetune, seed 1993
#   ./run_all.sh lwf          -> lwf, seed 1993
#   ./run_all.sh lwf 1994     -> lwf, seed 1994
set -e

METHOD=${1:-finetune}
SEED=${2:-1993}

# Fail fast if the BERT settings have drifted from the verified ../bertgo
# replication — every table here is read against that baseline.
python check_config.py "exps/${METHOD}_B0-I7.json"

for P in B0-I7 B0-I4 B16-I3 B16-I2; do
  echo "=== ${METHOD} / ${P} / seed ${SEED} ==="
  python main.py --config "exps/${METHOD}_${P}.json" --seed "${SEED}"
done
