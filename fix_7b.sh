#!/bin/bash
# Fix the Qwen2.5-VL-7B haloquest gap.
#
# The first pass ran the 7B on one card and the vision encoder ran out of
# memory on the largest images: ccot lost 183 samples, cot 12, direct 1. The
# empty records were already stripped, so the generator will regenerate
# exactly those, and this run shards the model over both cards
# (--device auto) which leaves the vision tower several GB of headroom.
#
# Writes, scores and rebuilds the table; safe to re-run, everything resumes.

set -u

cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
DATA=haloquest
MODEL=Qwen2.5-VL-7B-Instruct
MODES="direct,cot,ccot"
MAXTOK=2000
LOGS=rebuttal/logs

mkdir -p "$LOGS"

echo "===== 7B fix started $(date) ====="

$PY run_qwen_eval2.py \
    --data_name "$DATA" \
    --model_name "$MODEL" \
    --device auto \
    --vision_attn sdpa \
    --prompt_modes "$MODES" \
    --max_new_tokens "$MAXTOK" \
    > "$LOGS/fix_7b_gen.log" 2>&1

echo "===== generation done $(date +%H:%M), exit=$? ====="

for mode in ${MODES//,/ }; do

    echo "[score] $mode"

    $PY score_parallel.py \
        --input "rebuttal/${DATA}/${MODEL}/${mode}_maxNew${MAXTOK}.jsonl" \
        --mode "$mode" \
        --output "rebuttal/scores/${DATA}/${MODEL}/${mode}.json" \
        --workers 12 >> "$LOGS/fix_7b_score.log" 2>&1

    tail -1 "$LOGS/fix_7b_score.log"
done

echo "===== table $(date +%H:%M) ====="

$PY build_table.py \
    --scores rebuttal/scores \
    --datasets "$DATA" \
    --models Qwen2.5-VL-3B-Instruct,Qwen2.5-VL-7B-Instruct,R1-OneVision-7B \
    --modes direct,cot,ccot,icot \
    --markdown rebuttal/baseline_tables.md

echo "===== 7B fix finished $(date) ====="
