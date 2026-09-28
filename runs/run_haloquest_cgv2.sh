#!/bin/bash
# cgv2（严格版）在 haloquest 上：先 19 条做前后对比，再续跑满 600 条。
#
#   screen -dmS hq_cgv2 bash runs/run_haloquest_cgv2.sh
#
# 先跑前 19 条是因为那 19 条正是暴露问题的样本（id 8/15/17/18），跑完立刻能看出
# verdict 有没有真的约束住答案。两次调用用的是同一个文件，所以第二步会按 id 续跑，
# 不会重复前 19 条。
#
# 等 GPU1 上的 cg 跑完再启动：两张卡现在都只剩 6-10GB，硬挤会让正在跑的进程 OOM，
# 白天已经发生过一次。
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
MODEL=${MODEL:-Qwen2.5-VL-3B-Instruct}
DEVICE=${DEVICE:-1}
BACKEND=${BACKEND:-dino}
FALLBACK=${FALLBACK:-refuse}

mkdir -p "$LOGS"

while screen -ls 2>/dev/null | grep -qE "hq_cg"; do sleep 60; done
sleep 30

echo "===== cgv2 haloquest started $(date) ====="

# 1) 前 19 条：和旧 cg 同一批样本，直接对比
$PY scripts/run_cgr.py \
    --data_name haloquest \
    --model_name "$MODEL" \
    --pipeline cgv2 \
    --locate_backend "$BACKEND" \
    --answer_tokens 2000 \
    --fallback "$FALLBACK" \
    --limit 19 \
    --device "$DEVICE" \
    > "$LOGS/hq_cgv2_19.log" 2>&1

echo "[19] done $(date +%H:%M)"

# 2) 全量 600 条，续跑
$PY scripts/run_cgr.py \
    --data_name haloquest \
    --model_name "$MODEL" \
    --pipeline cgv2 \
    --locate_backend "$BACKEND" \
    --answer_tokens 2000 \
    --fallback "$FALLBACK" \
    --device "$DEVICE" \
    > "$LOGS/hq_cgv2_full.log" 2>&1

echo "[full] done $(date +%H:%M)"
echo "===== cgv2 haloquest finished $(date) ====="
