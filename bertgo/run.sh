#!/bin/bash
# Replicate the GoEmotions BERT baseline (Table 4, macro-F1 ~.46).
set -e
python train.py --output_dir ./output/run1 --fp16 true "$@"
