#!/bin/bash
# All datasets x prompt modes for the Qwen models, plus the R1 reasoning model.
#
# Scale warning: the datasets hold 42k samples in total, and a Qwen run costs
# five generations per sample (direct, cot, ccot twice, icot). Full coverage
# is weeks of GPU time, so every dataset is capped with --limit here; raise
# CAP to buy more coverage. Runs are resumable, so changing CAP and re-running
# continues from what is already saved.
#
#     screen -dmS alldata bash run_all_datasets.sh
#     tail -f rebuttal/logs/driver_all.log
#
# Phases:
#   1. 3B on GPU 0 and 7B on GPU 1, in parallel
#   2. R1-OneVision on GPU 0 (reasoning model: one mode, not four)
#   3. scoring and the per-dataset tables

set -u

cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
LOGS=rebuttal/logs
SCORES=rebuttal/scores
MODES="direct,cot,ccot,icot"
MAXTOK=2000
CAP=${CAP:-0}          # 0 = run every sample of the dataset

# --limit is only passed when a cap is set: "${CAP:+--limit $CAP}" would
# expand for CAP=0 and then process nothing.
LIMIT_ARG=""
if [ "${CAP:-0}" != "0" ]; then LIMIT_ARG="--limit $CAP"; fi

# The three large sets (POPE 9000, mmbench 11095, amber 14216) are left out:
# with five generations per sample they dominate the runtime. Add them back
# through DATASETS=... when there is time for them.
DATASETS=${DATASETS:-"haloquest mmhal HallusionBench RH_halu RH_reason mmstar MathVistaMini"}

mkdir -p "$LOGS" "$SCORES"

# score_parallel.py creates the per-dataset folder it writes into.

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "DEEPSEEK_API_KEY is not set: scoring will fail." >&2
fi

tag_of() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

score_dataset() {
    # model_tag dataset modes
    local tag=$1 dataset=$2 modes=$3 m file

    for m in ${modes//,/ }; do

        # One folder per dataset, one file per mode.
        file="rebuttal/${dataset}/${MODEL_FULL}/${m}_maxNew${MAXTOK}.jsonl"

        if [ ! -f "$file" ]; then
            echo "[score] skip $tag/$dataset/$m: no file"
            continue
        fi

        $PY scripts/score_parallel.py --input "$file" --mode "$m" \
            --output "$SCORES/${dataset}/${MODEL_FULL}/${m}.json" --workers 12 \
            >> "$LOGS/${tag}_score.log" 2>&1
    done
}

run_qwen() {
    # model_tag model_name device extra
    local tag=$1 model=$2 device=$3 extra=$4 dataset

    for dataset in $DATASETS; do

        MODEL_FULL="$model"

        echo "[run] $tag on $dataset (cap $CAP) start $(date +%H:%M)"

        $PY scripts/run_qwen_eval2.py \
            --model_name "$model" \
            --device "$device" \
            --data_name "$dataset" \
            --prompt_modes "$MODES" \
            --max_new_tokens "$MAXTOK" \
            $LIMIT_ARG \
            $extra \
            >> "$LOGS/${tag}_gen.log" 2>&1

        echo "[run] $tag on $dataset done $(date +%H:%M)"

        score_dataset "$tag" "$dataset" "$MODES"
    done
}

run_r1() {
    # R1 reasons by default, so it has a single mode rather than four.
    local dataset

    for dataset in $DATASETS; do

        echo "[run] r1-onevision on $dataset (cap $CAP) start $(date +%H:%M)"

        $PY scripts/run_r1_eval.py \
            --data_name "$dataset" \
            --max_new_tokens "$MAXTOK" \
            $LIMIT_ARG \
            --device 0 \
            >> "$LOGS/r1_gen.log" 2>&1

        echo "[run] r1-onevision on $dataset done $(date +%H:%M)"

        file="rebuttal/${dataset}/R1-OneVision-7B/cot_maxNew${MAXTOK}.jsonl"

        if [ -f "$file" ]; then
            $PY scripts/score_parallel.py --input "$file" --mode cot \
                --output "$SCORES/${dataset}/R1-OneVision-7B/cot.json" \
                --workers 12 >> "$LOGS/r1_score.log" 2>&1
        fi
    done
}

echo "===== all-datasets run started $(date), cap=${CAP:-none} ====="
echo "datasets: $DATASETS"

# ------------------------------------------------------------
# Phase 1: the two Qwen models, one GPU each
# ------------------------------------------------------------

# Sequential on purpose: one model at a time, so a single card is busy and
# the logs stay readable.
run_qwen qwen25vl-3b Qwen2.5-VL-3B-Instruct 0 ""

run_qwen qwen25vl-7b Qwen2.5-VL-7B-Instruct 0 "--vision_attn sdpa"

# ------------------------------------------------------------
# Phase 2: R1-OneVision, alone, GQA style single pass
# ------------------------------------------------------------

run_r1

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

echo "===== tables ====="
$PY scripts/build_table.py --scores "$SCORES" --datasets "$DATASETS" \
    --models Qwen2.5-VL-3B-Instruct,Qwen2.5-VL-7B-Instruct,R1-OneVision-7B \
    --modes "$MODES,cot" \
    --markdown rebuttal/baseline_tables.md
