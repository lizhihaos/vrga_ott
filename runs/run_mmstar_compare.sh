#!/bin/bash
# MMStar perception 子集：把所有方法在同一批 200 条上跑齐。
#
#   screen -dmS mmcmp bash run_mmstar_compare.sh
#
# GPU 0: direct/cot（续跑）→ ccot（原版 CCoT，用于和 grounded CCoT 对照）
# GPU 1: cg（grounded CCoT）→ react
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
TYPES="coarse perception,fine-grained perception"
LIMIT=200
LOGS=rebuttal/logs
mkdir -p "$LOGS"

echo "===== v2 comparison started $(date) ====="

(
  $PY scripts/run_qwen_eval2.py --data_name mmstar --types "$TYPES" --limit $LIMIT \
      --prompt_modes direct,cot,ccot --max_new_tokens 2000 --device 0 \
      > "$LOGS/mm_gpu0.log" 2>&1
  echo "[gpu0] done $(date +%H:%M)"
) &

(
  $PY scripts/run_cgr.py --data_name mmstar --types "$TYPES" --limit $LIMIT --device 1 \
      --pipeline cg --answer_tokens 2000 --locate_backend dino \
      > "$LOGS/mm_cg.log" 2>&1
  echo "[gpu1] cg done $(date +%H:%M)"

  $PY scripts/run_cgr.py --data_name mmstar --types "$TYPES" --limit $LIMIT --device 1 \
      --pipeline react --answer_tokens 2000 --locate_backend dino \
      > "$LOGS/mm_react.log" 2>&1
  echo "[gpu1] react done $(date +%H:%M)"
) &

wait
echo "===== v2 comparison finished $(date) ====="
