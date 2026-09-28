#!/bin/bash
# POPE pilot: the object-centric benchmark, where a region anchor has the best
# chance of doing something. Every mode gets the same token budget, because a
# row generated with a different budget is not a comparable row, and that
# budget has to be large enough that the two reasoning modes finish their
# answer rather than being cut off -- a truncated CoT is not a fair CoT.
#
# Rows: direct and cot are the controls, anchor and anchor_cot use tool located
# regions, vrg uses the attention derived regions the paper's own source gives.
#
#   screen -dmS pope_anchor bash runs/run_pope_anchor.sh
#
# POPE is stored in category order, so --limit on the whole set would return
# only the popular split; each split is run separately, which is also how POPE
# tables are normally reported (popular / random / adversarial).
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
LIMIT=${LIMIT:-60}
TOKENS=${TOKENS:-256}
DEVICE=${DEVICE:-1}

mkdir -p "$LOGS"

# GPU 1 is running the MMStar cg chain; wait for it rather than fighting for
# memory, which already cost one OOM warning in that job.
while pgrep -f "run_cgr.py" > /dev/null; do sleep 60; done
sleep 30

echo "===== POPE anchor pilot started $(date) ====="

for category in popular random adversarial; do

  echo "----- $category -----"

  $PY scripts/run_qwen_eval2.py \
      --data_name POPE \
      --types "$category" \
      --limit "$LIMIT" \
      --prompt_modes direct,cot,anchor,anchor_cot,vrg \
      --max_new_tokens "$TOKENS" \
      --device "$DEVICE" \
      > "$LOGS/pope_anchor_${category}.log" 2>&1

  echo "[$category] done $(date +%H:%M)"

done

echo "===== POPE anchor pilot finished $(date) ====="
