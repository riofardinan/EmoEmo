#!/bin/bash
# Run one method across all four 28-class protocols.
# Usage: ./run_all.sh [method] [seed]
set -e
METHOD=${1:-finetune}
SEED=${2:-1993}
for P in B0-I7 B0-I4 B16-I3 B16-I2; do
  echo "=== ${METHOD} / ${P} / seed ${SEED} ==="
  python main.py --config "exps/finetune_${P}.json" \
    --method "${METHOD}" --protocol "${P}" --seed "${SEED}"
done
