#!/bin/bash
# haloquest 上的 cg（原方法）：3B 和 7B，各自一张卡并行。
#
#   screen -dmS hq_cg bash runs/run_haloquest_cg.sh
#
# 600 条全跑，token 预算 2000（和表里 direct/cot/ccot/icot 那几行一致），
# 区域后端用 dino。每个模型单独一张卡。
#
# 7B 用 --vision_attn sdpa：eager 的视觉注意力在 haloquest 这种大图上峰值
# 超过 23GB，24GB 卡直接 OOM，答案会退化成 fallback；而且表里 7B 那一列本来
# 就是 sdpa 跑的（runs/run_all_datasets.sh 第 133 行），这样才一致。
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
mkdir -p "$LOGS"

echo "===== haloquest cg started $(date) ====="

(
  $PY scripts/run_cgr.py \
      --data_name haloquest \
      --model_name Qwen2.5-VL-3B-Instruct \
      --pipeline cg \
      --locate_backend dino \
      --answer_tokens 2000 \
      --device 1 \
      > "$LOGS/hq_cg_3b.log" 2>&1
  echo "[3B] done $(date +%H:%M)"
) &

(
  $PY scripts/run_cgr.py \
      --data_name haloquest \
      --model_name Qwen2.5-VL-7B-Instruct \
      --pipeline cg \
      --locate_backend dino \
      --answer_tokens 2000 \
      --vision_attn sdpa \
      --device 0 \
      > "$LOGS/hq_cg_7b.log" 2>&1
  echo "[7B] done $(date +%H:%M)"
) &

wait
echo "===== haloquest cg finished $(date) ====="
