#!/usr/bin/env python3
"""Score result files with DeepSeek, in parallel.

Same judge prompt as evaluate_deepseek.py, but with a thread pool and
resume, because a 600 sample file per (model, mode) pair is too slow to
judge sequentially: twelve runs is 7200 API calls.

    python score_parallel.py --input rebuttal/x.jsonl --mode icot \
        --output rebuttal/scores/model_icot.json --workers 12

Requires DEEPSEEK_API_KEY in the environment. Output shape matches
evaluate_deepseek.py, so the existing summaries keep working.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The repository root holds evaluate_deepseek.py and the packages.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from evaluate_deepseek import (
    JUDGE_SYSTEM_PROMPT,
    MODEL,
    build_prompt,
    client,
)

WRITE_LOCK = threading.Lock()


def parse_args():

    parser = argparse.ArgumentParser(description="Parallel DeepSeek scoring")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        help="Response field to read, as '{mode}_response'")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=5)

    return parser.parse_args()


def ask(question, ground_truth, response, retries):

    prompt = build_prompt(question, ground_truth, response)

    for attempt in range(retries):

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            verdict = completion.choices[0].message.content.strip().upper()

            if verdict.startswith("CORRECT"):
                return 1, "CORRECT"

            if verdict.startswith("WRONG"):
                return 0, "WRONG"

            return 0, verdict

        except Exception as error:

            if attempt == retries - 1:
                return 0, f"ERROR: {type(error).__name__}"

            time.sleep(2 ** attempt)

    return 0, "ERROR"


def score_one(item, mode, index, retries):

    field = f"{mode}_response"
    response = item.get(field, "")

    question = item.get("question", "")
    ground_truth = str(item.get("answer", ""))

    if not response:
        score, verdict = 0, "MISSING"

    else:
        score, verdict = ask(question, ground_truth, response, retries)

    return {
        "id": item.get("id", index),
        "question": question,
        "ground_truth": ground_truth,
        "model_response": response,
        "judge": verdict,
        "score": score,
    }


def main():

    args = parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Resume: skip ids judged before
    # ------------------------------------------------------------

    done = {}

    if output.exists():

        previous = json.loads(output.read_text(encoding="utf-8"))

        for result in previous.get("results", []):
            # MISSING means the generation was empty at the time; a later
            # run may have fixed it, so judge it again.
            if result.get("judge") not in (None, "ERROR", "MISSING"):
                done[result["id"]] = result

        print(f"Resuming: {len(done)} already judged")

    pending = [
        (index, item)
        for index, item in enumerate(records)
        if item.get("id", index) not in done
    ]

    print(f"Scoring {len(pending)} of {len(records)} samples from {args.input}")

    results = list(done.values())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        futures = {
            pool.submit(score_one, item, args.mode, index, args.retries): index
            for index, item in pending
        }

        for counter, future in enumerate(futures, 1):

            results.append(future.result())

            if counter % 25 == 0 or counter == len(futures):
                print(f"  {counter}/{len(futures)}", flush=True)

                with WRITE_LOCK:
                    write(output, args, records, results)

    with WRITE_LOCK:
        write(output, args, records, results)

    scored = [r for r in results if r["judge"] not in ("ERROR", "MISSING", None)]
    correct = sum(r["score"] for r in scored)
    accuracy = correct / len(scored) if scored else 0.0

    print(
        f"{args.mode:8s} accuracy={accuracy:.4f} "
        f"({correct}/{len(scored)}, errors={len(results) - len(scored)})"
    )

    return 0


def write(output, args, records, results):

    """Rewrite the summary file; partial results survive an interrupt."""

    ordered = sorted(results, key=lambda r: str(r["id"]))

    valid = [r for r in ordered if r["judge"] not in ("ERROR", "MISSING", None)]
    correct = sum(r["score"] for r in valid)

    output.write_text(
        json.dumps(
            {
                "input_file": args.input,
                "mode": args.mode,
                "model": MODEL,
                "total_samples": len(records),
                "valid_samples": len(valid),
                "correct": correct,
                "errors": len(ordered) - len(valid),
                "accuracy": correct / len(valid) if valid else 0.0,
                "results": ordered,
            },
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
