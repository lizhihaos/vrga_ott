#!/bin/bash
# MMStar 全集（1500 条）：四方法同批对比。
#
#   screen -dmS mmfull bash runs/run_mmstar_full.sh
#
# 已跑过的 200 条 perception 样本是全集的有效子集，续跑按 id 跳过。
# GPU 0: direct,cot,ccot（三种模式串行，约 8 小时）
# GPU 1: cg（约 2 小时）
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
mkdir -p "$LOGS"

# 等当前 mmstar 对比链（react 还有剩余）结束，避免抢显存
while pgrep -f "run_cgr.py" > /dev/null; do sleep 60; done
sleep 30

echo "===== MMStar full set started $(date) ====="

(
  $PY scripts/run_qwen_eval2.py --data_name mmstar \
      --prompt_modes direct,cot,ccot --max_new_tokens 2000 --device 0 \
      > "$LOGS/mmfull_gpu0.log" 2>&1
  echo "[gpu0] done $(date +%H:%M)"
) &

(
  $PY scripts/run_cgr.py --data_name mmstar --device 1 \
      --pipeline cg --answer_tokens 2000 --locate_backend dino \
      > "$LOGS/mmfull_cg.log" 2>&1
  echo "[gpu1] cg done $(date +%H:%M)"
) &

wait
echo "===== MMStar full set finished $(date) ====="
