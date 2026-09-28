#!/bin/bash
# HaloQuest baseline table: models x prompt modes, then scoring, then the table.
#
# Designed for a detached screen session, e.g.
#     screen -dmS baseline bash run_baseline_table.sh
#     screen -r baseline          # attach
#     tail -f rebuttal/logs/driver.log    # or just watch the log
#
# Phases:
#   1. 3B and 7B run in parallel, one per GPU
#   2. 30B runs alone, sharded across both GPUs
#   3. every finished run is scored with DeepSeek, in parallel
#   4. the accuracy table is written to rebuttal/baseline_table.md
#
# Everything is resumable: the generator skips samples already in its file
# and the scorer skips ids already judged, so re-running this script after an
# interruption continues instead of starting over.

set -u

cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
MODES="direct,cot,ccot,icot"
MAXTOK=2000
LOGS=rebuttal/logs
SCORES=rebuttal/scores
MODELS_DIR=/home/lizhihao/.cache/huggingface/models

mkdir -p "$LOGS" "$SCORES"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "DEEPSEEK_API_KEY is not set: scoring will fail." >&2
fi

score_model() {
    # tag model modes
    local tag=$1 model=$2 modes=$3
    local m file

    for m in ${modes//,/ }; do

        file="rebuttal/haloquest_${model}_${modes}_maxNew${MAXTOK}.jsonl"

        if [ ! -f "$file" ]; then
            echo "[score] skip $tag/$m: $file missing"
            continue
        fi

        echo "[score] $tag/$m <- $file"
        $PY score_parallel.py --input "$file" --mode "$m" \
            --output "$SCORES/${tag}_${m}.json" --workers 12 \
            >> "$LOGS/${tag}_score.log" 2>&1
    done
}

run_model() {
    # tag model device extra_flags
    local tag=$1 model=$2 device=$3 extra=$4

    echo "[run] $tag ($model, device=$device) start $(date +%H:%M)"

    $PY run_qwen_eval2.py \
        --model_name "$model" \
        --device "$device" \
        --data_name haloquest \
        --prompt_modes "$MODES" \
        --max_new_tokens "$MAXTOK" \
        $extra \
        > "$LOGS/${tag}_gen.log" 2>&1

    echo "[run] $tag done $(date +%H:%M) exit=$?"

    score_model "$tag" "$model" "$MODES"
}

echo "===== baseline table started $(date) ====="

# ------------------------------------------------------------
# Phase 1: one model per GPU
# ------------------------------------------------------------

run_model qwen25vl-3b Qwen2.5-VL-3B-Instruct 0 "" &
PID_3B=$!

run_model qwen25vl-7b Qwen2.5-VL-7B-Instruct 1 "--vision_attn sdpa" &
PID_7B=$!

wait $PID_3B $PID_7B

# ------------------------------------------------------------
# Phase 2: 30B, both GPUs, CPU offload for whatever does not fit
#
# Qwen3-VL has no compute_3d_position_ids / get_placeholder_mask, which the
# ICoT injection needs, so that mode is left out of this row.
# ------------------------------------------------------------

if [ -d "$MODELS_DIR/Qwen3-VL-30B-A3B-Instruct" ]; then
    MODES="direct,cot,ccot"
    run_model qwen3vl-30b Qwen3-VL-30B-A3B-Instruct auto "--vision_attn sdpa"
else
    echo "[run] skip 30B: weights not present"
fi

# ------------------------------------------------------------
# Table
# ------------------------------------------------------------

echo "===== table ====="
$PY build_table.py --scores "$SCORES" \
    --modes direct,cot,ccot,icot \
    --markdown rebuttal/baseline_table.md
