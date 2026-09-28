#!/bin/bash
# POPE with the grounding tool allowed to abstain, on top of the pilot run.
#
#   screen -dmS anchor_abstain bash runs/run_anchor_abstain.sh
#
# The plain grounding prompt returns a box for every sample, including all of
# POPE's negative half where the question's object is absent by construction,
# so those samples get anchored on a hallucinated region. Asked to abstain it
# answers with an empty list on 18 of 20 absent objects and on only 1 of 20
# present ones, which makes the negative items a real test rather than a
# foregone one.
#
# Same samples, same budget, same strength as the pilot; only the grounding
# prompt differs, and the run writes its own files.
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
LIMIT=${LIMIT:-60}
TOKENS=${TOKENS:-256}
DEVICE=${DEVICE:-1}

mkdir -p "$LOGS"

while screen -ls 2>/dev/null | grep -qE "pope_anchor|anchor_boost"; do sleep 60; done
sleep 30

echo "===== abstaining anchor run started $(date) ====="

for category in popular random adversarial; do

  echo "----- $category -----"

  $PY scripts/run_qwen_eval2.py \
      --data_name POPE \
      --types "$category" \
      --limit "$LIMIT" \
      --prompt_modes anchor,anchor_cot \
      --anchor_grounding_abstain \
      --max_new_tokens "$TOKENS" \
      --device "$DEVICE" \
      > "$LOGS/pope_abstain_${category}.log" 2>&1

  echo "[$category] done $(date +%H:%M)"

done

echo "===== abstaining anchor run finished $(date) ====="
