#!/bin/bash
# Full CG-VR run on haloquest, then score and rebuild the table.
#
#   screen -dmS cgr bash run_cgr_full.sh
#   tail -f rebuttal/logs/cgr_full.log
#
# Resumable: records already in the output file are skipped.
#
# Budget note: this method calls the model about a dozen times per question,
# so every record stores cgr_generated_tokens and cgr_tool_calls. Any claim
# that it beats CoT has to be reported next to those numbers, not next to
# the answer cap alone.

set -u

cd /home/lizhihao/phd/VRGA

PY=/home/lizhihao/miniconda3/envs/tifa/bin/python
DATA=haloquest
MODEL=Qwen2.5-VL-3B-Instruct
LOGS=rebuttal/logs
OUT="rebuttal/${DATA}/${MODEL}/cgr-vlm_maxNew2000.jsonl"

mkdir -p "$LOGS"

echo "===== CG-VR full run started $(date) ====="

$PY run_cgr.py \
    --data_name "$DATA" \
    --model_name "$MODEL" \
    --device 1 \
    --pipeline claims \
    --answer_tokens 2000 \
    > "$LOGS/cgr_full.log" 2>&1

echo "===== generation done $(date +%H:%M), records: $(grep -c . "$OUT") ====="

$PY score_parallel.py \
    --input "$OUT" \
    --mode cgr \
    --output "rebuttal/scores/${DATA}/${MODEL}/cgr.json" \
    --workers 12 >> "$LOGS/cgr_full_score.log" 2>&1

tail -1 "$LOGS/cgr_full_score.log"

$PY build_table.py \
    --scores rebuttal/scores \
    --datasets "$DATA" \
    --models "$MODEL,Qwen2.5-VL-7B-Instruct,R1-OneVision-7B" \
    --modes direct,cot,ccot,icot,cgr \
    --markdown rebuttal/baseline_tables.md

echo "===== CG-VR full run finished $(date) ====="
