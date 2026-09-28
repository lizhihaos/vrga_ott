#!/bin/bash
# Fill the gaps on haloquest: 3B is missing ccot/icot, 7B is missing all four.
#
#   screen -dmS hq_fill bash fill_haloquest.sh      # detached
#   screen -r hq_fill                               # attach
#   tail -f rebuttal/logs/fill_3b_gen.log           # or just watch the logs
#
# Default is parallel: 3B on GPU 0, 7B on GPU 1, which takes about 5 hours
# instead of 7.5. Set PARALLEL=0 to run them one after another on GPU 0.
#
# Everything is resumable. The generator skips samples already in its file and
# the scorer skips ids already judged, so re-running this script after an
# interruption continues instead of starting over.

set -u

cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
DATA=haloquest
MAXTOK=2000
MODELS="Qwen2.5-VL-3B-Instruct,Qwen2.5-VL-7B-Instruct,R1-OneVision-7B"
MODES="direct,cot,ccot,icot"
LOGS=rebuttal/logs
PARALLEL=${PARALLEL:-1}

mkdir -p "$LOGS"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "warning: DEEPSEEK_API_KEY not set, scoring will fail" >&2
fi

score_mode() {
    # model_folder mode
    local folder=$1 mode=$2

    $PY score_parallel.py \
        --input "rebuttal/${DATA}/${folder}/${mode}_maxNew${MAXTOK}.jsonl" \
        --mode "$mode" \
        --output "rebuttal/scores/${DATA}/${folder}/${mode}.json" \
        --workers 12 >> "$LOGS/fill_score.log" 2>&1

    echo "  [score] $folder/$mode done"
}

fill_3b() {

    local device=$1

    echo "[3B] generation start $(date +%H:%M)"

    $PY run_qwen_eval2.py \
        --data_name "$DATA" \
        --model_name Qwen2.5-VL-3B-Instruct \
        --device "$device" \
        --prompt_modes ccot,icot \
        --max_new_tokens "$MAXTOK" \
        > "$LOGS/fill_3b_gen.log" 2>&1

    echo "[3B] generation done $(date +%H:%M)"

    score_mode Qwen2.5-VL-3B-Instruct ccot
    score_mode Qwen2.5-VL-3B-Instruct icot
}

fill_7b() {

    local device=$1

    echo "[7B] generation start $(date +%H:%M)"

    $PY run_qwen_eval2.py \
        --data_name "$DATA" \
        --model_name Qwen2.5-VL-7B-Instruct \
        --device "$device" \
        --vision_attn sdpa \
        --prompt_modes "$MODES" \
        --max_new_tokens "$MAXTOK" \
        > "$LOGS/fill_7b_gen.log" 2>&1

    echo "[7B] generation done $(date +%H:%M)"

    for mode in ${MODES//,/ }; do
        score_mode Qwen2.5-VL-7B-Instruct "$mode"
    done
}

echo "===== fill started $(date), parallel=$PARALLEL ====="

if [ "$PARALLEL" = "1" ]; then

    fill_3b 0 &
    PID_3B=$!

    fill_7b 1 &
    PID_7B=$!

    wait $PID_3B $PID_7B

else

    fill_3b 0
    fill_7b 0

fi

# ------------------------------------------------------------
# Table
# ------------------------------------------------------------

echo "===== table $(date +%H:%M) ====="

$PY build_table.py \
    --scores "rebuttal/scores" \
    --datasets "$DATA" \
    --models "$MODELS" \
    --modes "$MODES" \
    --markdown rebuttal/baseline_tables.md

echo "===== fill finished $(date) ====="
