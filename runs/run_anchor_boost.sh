#!/bin/bash
# Where the anchor can actually be seen: MMStar perception, boost 50.
#
#   screen -dmS anchor_boost bash runs/run_anchor_boost.sh
#
# tools/anchor_leverage.py shows the paper's own strength (1.5) moves the
# next-token logits by a fraction of what it takes to flip a token the model
# emits with probability 0.96, so a table at 1.5 measures nothing. 50 is 33x
# that and does move the reasoning, which is what makes this a test of the
# region source rather than of the intervention's size.
#
# Both rows use the CoT prompt, so the only difference between them is where
# the region comes from. Plain `cot` on the same samples is the control, and it
# already exists in the same folder. The existing files hold the full 1500
# sample run, so compare with `tools/report_metrics.py --intersect`.
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
TYPES="coarse perception,fine-grained perception"
LIMIT=200
DEVICE=${DEVICE:-1}

mkdir -p "$LOGS"

# The POPE pilot is using GPU 1 first; wait for its screen rather than for any
# run_qwen_eval2.py process, because the MMStar full run is one of those too.
while screen -ls 2>/dev/null | grep -qE "pope_anchor|pope_cg"; do sleep 60; done
sleep 30

echo "===== anchor boost sweep started $(date) ====="

$PY scripts/run_qwen_eval2.py \
    --data_name mmstar \
    --types "$TYPES" \
    --limit $LIMIT \
    --prompt_modes anchor_cot,vrg \
    --anchor_multiply 50 \
    --max_new_tokens 2000 \
    --device "$DEVICE" \
    > "$LOGS/anchor_boost50_mmstar.log" 2>&1

echo "===== anchor boost sweep finished $(date) ====="
