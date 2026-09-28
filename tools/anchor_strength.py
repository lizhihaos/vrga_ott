#!/usr/bin/env python3
"""Does the anchor do anything, and how much does its strength matter?

An anchoring result is only readable if the intervention is known to move the
output. This runs the same samples with the boost switched off and at several
strengths, and reports how often the answer changes against the off setting,
plus accuracy on yes/no ground truths where that needs no judge.

The off setting is not a copy of `direct`: it is the anchored run with
`multiply=1.0`, so the grounding call, the prompt and the region all still
happen and only the weight boost is removed. That isolates the boost from the
extra call.

    python tools/anchor_strength.py --dataset POPE --limit 20 --multiply 1.0,1.5,3.0
"""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import torch

from qwen_eval2.anchoring import (
    ANCHOR_MODES,
    ANCHOR_REGION_SOURCE,
    generate_anchored,
    install,
    uninstall,
)
from qwen_eval2.datasets import EvalDataset
from qwen_eval2.modeling import load_model
from tools.report_metrics import is_negative


def parse_args():

    parser = argparse.ArgumentParser(description="Anchor strength sweep")

    parser.add_argument("--dataset", type=str, default="POPE")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--types", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, default="direct")
    parser.add_argument("--region_source", type=str, default="native",
                        choices=["native", "attention"])
    parser.add_argument("--multiply", type=str, default="1.0,1.5,3.0")
    parser.add_argument("--max_new_tokens", type=int, default=64)

    return parser.parse_args()


def answer_of(text):
    """Yes/no/absence from a response, whatever wording it uses.

    POPE answers come back as "Yes, there is a car", "No, there is no car" and
    "There is no car present", so a bare startswith("yes") would drop a third
    of them and report accuracy over the easy subset only.
    """

    text = str(text).strip().lower()

    if text.startswith("yes"):
        return "yes"

    if text.startswith("no"):
        return "no"

    if "there is no" in text or "there are no" in text:
        return "no"

    if text.startswith("there is") or text.startswith("there are"):
        return "yes"

    return None


def main():

    args = parse_args()

    settings = [float(value) for value in args.multiply.split(",")]

    data = EvalDataset(args.dataset).get_data()

    if args.types:
        wanted = [t.strip().lower() for t in args.types.split(",")]
        data = [
            sample for sample in data
            if any(w in str(sample.get("type", "")).lower() for w in wanted)
        ]

    data = data[: args.limit]

    model, processor = load_model(args.model_name, f"cuda:{args.device}")

    print("patched:", install(model))

    # answers[setting][id] = response, so a row can be compared across settings
    answers = {setting: {} for setting in settings}

    for sample in data:

        for setting in settings:

            result = generate_anchored(
                processor,
                model,
                sample,
                prompt_mode=args.prompt_mode,
                region_source=args.region_source,
                max_new_tokens=args.max_new_tokens,
                multiply=setting,
            )

            answers[setting][str(sample["id"])] = result["anchor_response"]

            torch.cuda.empty_cache()

        print(
            f"[{sample['id']}] "
            + " | ".join(
                f"x{setting}: {answers[setting][str(sample['id'])][:28]!r}"
                for setting in settings
            )
        )

    uninstall()

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    off = settings[0]

    print()
    print("=" * 78)
    print(f"{args.dataset}, {len(data)} samples, region source "
          f"{args.region_source}, prompt {args.prompt_mode}")
    print(f"{'boost':>7s} {'changed vs off':>15s} {'yes/no parsed':>14s} "
          f"{'accuracy':>9s}")
    print("-" * 50)

    for setting in settings:

        changed = 0
        parsed = 0
        correct = 0

        for sample in data:

            key = str(sample["id"])

            response = answers[setting][key]

            if setting != off:
                other = answers[off][key]
                changed += answer_of(response) != answer_of(other) or (
                    answer_of(response) is None
                    and response.strip() != other.strip()
                )

            said = answer_of(response)

            if said is None:
                continue

            parsed += 1

            truth = is_negative(sample["answer"])

            correct += (said == "no") == truth

        share = changed / len(data) if setting != off else 0.0

        print(
            f"x{setting:<6.2f} {share * 100:14.0f}% {parsed:14d} "
            f"{correct / parsed * 100 if parsed else float('nan'):8.1f}%"
        )

    print()
    print("'changed vs off' counts samples whose yes/no answer differs from the")
    print("x1.0 run, so it is the boost alone moving the output rather than the")
    print("extra grounding call or the prompt. Read it before any accuracy")
    print("column: a boost that changes nothing cannot explain a difference.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
