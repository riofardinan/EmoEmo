#!/usr/bin/env bash
# HS2 experiment on a cloud GPU, then one archive to download for local analysis.
# Run from the emocil/ folder:   bash hs/cloud_run.sh            # seed 0
#                                bash hs/cloud_run.sh 1,2        # more seeds later
# Resumable: finished runs are skipped, so the same command continues after an interruption.
set -euo pipefail
SEEDS="${1:-0}"
PAIR=ekfine
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"
python -m hs.run --pair "$PAIR" --seeds "$SEEDS"
python -m hs.frozen --pair "$PAIR"
if grep -q FAILED "results/hs_${PAIR}/progress.log"; then echo "WARNING: some runs failed, see results/hs_${PAIR}/progress.log"; fi
tar czf "hs_${PAIR}_results.tgz" --exclude='*.pt' -C results "hs_${PAIR}"
echo "Done. Download emocil/hs_${PAIR}_results.tgz ($(du -h "hs_${PAIR}_results.tgz" | cut -f1))."
