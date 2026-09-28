#!/bin/bash
# 原有的 cg（grounded CCoT / CG-VR）在 POPE 上跑一份，样本与预告的试点完全一致。
#
#   screen -dmS pope_cg bash runs/run_pope_cg.sh
#
# 和 direct/cot/anchor/vrg 用同一批 3 类各 60 条、同一个 token 预算（256），
# 这样这张表里的每一行都是可比的。cg 每个样本要 3-4 次调用，比单次生成慢。
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
LIMIT=${LIMIT:-60}
TOKENS=${TOKENS:-256}
DEVICE=${DEVICE:-1}

mkdir -p "$LOGS"

# 别和正在跑的锚点实验抢显存；它们跑完这个才启动。
while screen -ls 2>/dev/null | grep -qE "pope_anchor|anchor_leverage"; do sleep 60; done
sleep 20

echo "===== POPE cg started $(date) ====="

for category in popular random adversarial; do

  echo "----- $category -----"

  $PY scripts/run_cgr.py \
      --data_name POPE \
      --types "$category" \
      --limit "$LIMIT" \
      --pipeline cg \
      --answer_tokens "$TOKENS" \
      --locate_backend dino \
      --device "$DEVICE" \
      > "$LOGS/pope_cg_${category}.log" 2>&1

  echo "[$category] done $(date +%H:%M)"

done

echo "===== POPE cg finished $(date) ====="
