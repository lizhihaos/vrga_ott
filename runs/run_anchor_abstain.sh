#!/bin/bash
# POPE at a strength where the anchor can move an answer, plain against
# abstaining.
#
#   screen -dmS anchor_abstain bash runs/run_anchor_abstain.sh
#
# Two questions, both on the same 180 samples and both under a boost that has
# leverage (tools/anchor_leverage.py):
#
#   anchor@50   the plain tool prompt returns a box for every sample,
#               including POPE's negative half where the object is absent, so
#               those samples are anchored on a hallucinated region
#   anchor@50+abstain
#               the tool is allowed to answer with an empty list, which it does
#               on 18 of 20 absent objects and 1 of 20 present ones, leaving
#               the intervention inert on those samples instead
#
# The pair is what makes the negative half a measurement rather than a
# foregone conclusion, and it writes its own files so the 1.5x pilot is not
# touched.
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
LIMIT=${LIMIT:-60}
TOKENS=${TOKENS:-256}
BOOST=${BOOST:-50}
DEVICE=${DEVICE:-1}

mkdir -p "$LOGS"

while screen -ls 2>/dev/null | grep -qE "pope_anchor|pope_cg|anchor_boost"; do sleep 60; done
sleep 30

echo "===== POPE boost ${BOOST} plain/abstain started $(date) ====="

for category in popular random adversarial; do

  echo "----- $category -----"

  $PY scripts/run_qwen_eval2.py \
      --data_name POPE \
      --types "$category" \
      --limit "$LIMIT" \
      --prompt_modes anchor \
      --anchor_multiply "$BOOST" \
      --max_new_tokens "$TOKENS" \
      --device "$DEVICE" \
      > "$LOGS/pope_boost50_${category}.log" 2>&1

  echo "[$category] plain done $(date +%H:%M)"

  $PY scripts/run_qwen_eval2.py \
      --data_name POPE \
      --types "$category" \
      --limit "$LIMIT" \
      --prompt_modes anchor \
      --anchor_multiply "$BOOST" \
      --anchor_grounding_abstain \
      --max_new_tokens "$TOKENS" \
      --device "$DEVICE" \
      > "$LOGS/pope_boost50_abstain_${category}.log" 2>&1

  echo "[$category] abstain done $(date +%H:%M)"

done

echo "===== POPE boost ${BOOST} plain/abstain finished $(date) ====="
