#!/bin/bash
# Score a folder of generations and print their tables.
#
#   DATASET=POPE TAG=maxNew256 bash runs/score_modes.sh
#   DATASET=POPE TAG=maxNew256 \
#       MODES="direct cot cgr-cg-dino anchor vrg" bash runs/score_modes.sh
#   DATASET=mmstar TAG=boost50_maxNew2000 \
#       MODES="anchor_cot vrg" bash runs/score_modes.sh
#
# The answer field is not always the mode name: the anchoring modes all write
# `anchor_response` and the cg pipelines write `cgr_response`, so the field is
# mapped explicitly instead of being derived.
set -u
cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python

DATASET=${DATASET:-POPE}
MODEL=${MODEL:-Qwen2.5-VL-3B-Instruct}
TAG=${TAG:-maxNew256}
MODES=${MODES:-direct cot cgr-cg-dino anchor anchor_cot vrg}
TOKENS=${TOKENS:-256}

GEN_DIR="rebuttal/${DATASET}/${MODEL}"
SCORE_DIR="rebuttal/scores/${DATASET}/${MODEL}"
LOGS=rebuttal/logs

mkdir -p "$SCORE_DIR" "$LOGS"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "DEEPSEEK_API_KEY is not set: scoring will fail." >&2
fi

for mode in $MODES; do

    file="${GEN_DIR}/${mode}_${TAG}.jsonl"

    if [ ! -f "$file" ]; then
        echo "[score] skip $mode: no $file"
        continue
    fi

    # Which field this mode's answer lives in.
    field="${mode}_response"
    case "$mode" in
        anchor|anchor_cot|vrg|anchor_*|vrg_*) field="anchor_response" ;;
        cgr*) field="cgr_response" ;;
    esac

    # The score file is named after the generation file's mode tag, so a
    # non-default strength does not overwrite the run the paper reports.
    if [[ "$TAG" == maxNew* ]]; then
        name="$mode"
    else
        name="${mode}_${TAG%%_maxNew*}"
    fi

    echo "[score] $name from $file ($field)"

    $PY scripts/score_parallel.py \
        --input "$file" \
        --mode "$mode" \
        --response_field "$field" \
        --output "${SCORE_DIR}/${name}.json" \
        --workers 12 \
        >> "$LOGS/score_anchor.log" 2>&1

done

echo
echo "===== ${DATASET} / ${MODEL} / ${TAG} ====="

$PY tools/report_metrics.py \
    --scores "$SCORE_DIR" \
    --data_name "$DATASET" \
    --generations "$GEN_DIR" \
    --intersect
