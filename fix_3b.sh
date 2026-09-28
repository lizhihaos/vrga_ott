#!/bin/bash
# Re-run Qwen2.5-VL-3B direct and cot on haloquest.
#
# The two existing files came from the first pass, which still used the
# checkpoint's generation_config (do_sample=True, temperature=1e-6), while
# ccot, icot, both 7B rows and R1 all use greedy decoding now. Re-running
# these two modes puts the whole table on one decoding setting.
#
# Waits for the 7B fix to release the GPUs, parks the superseded files under
# rebuttal/_superseded/<timestamp>/ instead of deleting them, then runs,
# scores and rebuilds the table.

set -u

cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
DATA=haloquest
MODEL=Qwen2.5-VL-3B-Instruct
MODES=direct,cot
MAXTOK=2000
LOGS=rebuttal/logs

mkdir -p "$LOGS"

echo "===== 3B re-run queued $(date) ====="
echo "waiting for other evaluations to finish..."

while pgrep -f "run_qwen_eval2.py" > /dev/null; do
    sleep 60
done

echo "GPUs free, starting $(date +%H:%M)"

# ------------------------------------------------------------
# Park the superseded generation files and scores
# ------------------------------------------------------------

STAMP=$(date +%Y%m%d-%H%M)
PARK="rebuttal/_superseded/${STAMP}"
mkdir -p "$PARK/data" "$PARK/scores"

for mode in ${MODES//,/ }; do

    src="rebuttal/${DATA}/${MODEL}/${mode}_maxNew${MAXTOK}.jsonl"
    if [ -f "$src" ]; then
        mv "$src" "$PARK/data/"
        echo "  parked $(basename "$src") ($(grep -c . "$PARK/data/$(basename "$src")") records)"
    fi

    src="rebuttal/scores/${DATA}/${MODEL}/${mode}.json"
    if [ -f "$src" ]; then
        mv "$src" "$PARK/scores/"
    fi
done

echo "  superseded files parked under $PARK"

# ------------------------------------------------------------
# Generate, score, table
# ------------------------------------------------------------

$PY run_qwen_eval2.py \
    --data_name "$DATA" \
    --model_name "$MODEL" \
    --device 0 \
    --prompt_modes "$MODES" \
    --max_new_tokens "$MAXTOK" \
    > "$LOGS/fix_3b_gen.log" 2>&1

echo "===== 3B generation done $(date +%H:%M) ====="

for mode in ${MODES//,/ }; do

    $PY score_parallel.py \
        --input "rebuttal/${DATA}/${MODEL}/${mode}_maxNew${MAXTOK}.jsonl" \
        --mode "$mode" \
        --output "rebuttal/scores/${DATA}/${MODEL}/${mode}.json" \
        --workers 12 >> "$LOGS/fix_3b_score.log" 2>&1

    tail -1 "$LOGS/fix_3b_score.log"
done

$PY build_table.py \
    --scores rebuttal/scores \
    --datasets "$DATA" \
    --models "$MODEL,Qwen2.5-VL-7B-Instruct,R1-OneVision-7B" \
    --modes direct,cot,ccot,icot \
    --markdown rebuttal/baseline_tables.md

echo "===== 3B re-run finished $(date) ====="
