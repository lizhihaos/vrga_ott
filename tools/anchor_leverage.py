#!/usr/bin/env python3
"""How much leverage does the anchor have on the model's next token?

An anchoring method can be wired up correctly, fire on every decode step, and
still be unable to change an answer, because the answer's tokens were already
near certain. This measures that instead of assuming it:

    for each sample, decode the same prefix twice, with the boost off and on,
    and compare the next-token distributions step by step until they disagree.

Reported per run: the logit change the boost causes, the total variation
between the two distributions, and how many samples diverge in the first K
tokens. POPE is the sharp case: after "Yes, there is a", the answer token has
probability 0.9998, so a perturbation of 0.25 in logit space cannot move it and
a whole table of POPE rows can be flat for that reason alone.

    python tools/anchor_leverage.py --dataset POPE --limit 10 --steps 32
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
    _CALLS,
    build_anchor,
    install,
    locate_regions,
    set_anchor,
    uninstall,
)
from qwen_eval2.datasets import EvalDataset
from qwen_eval2.images import load_image
from qwen_eval2.inputs import MAX_PIXELS, inputs_from_content
from qwen_eval2.modeling import load_model
from qwen_eval2.prompts import render_prompt


def parse_args():

    parser = argparse.ArgumentParser(description="Anchor leverage on the output")

    parser.add_argument("--dataset", type=str, default="POPE")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--prompt_mode", type=str, default="direct")
    parser.add_argument("--off", type=float, default=1.0)
    parser.add_argument("--on", type=float, default=50.0,
                        help="Boost applied for the treatment run; the default "
                             "is deliberately far above the paper's 1.5 so the "
                             "ceiling is visible")

    return parser.parse_args()


def decode_with_scores(model, inputs, steps):

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=steps,
            do_sample=False,
            output_attentions=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    scores = [step[0].float().cpu() for step in output.scores]
    tokens = output.sequences[0][inputs["input_ids"].shape[1]:].cpu().tolist()

    del output

    return scores, tokens


def main():

    args = parse_args()

    data = EvalDataset(args.dataset).get_data()[: args.limit]

    model, processor = load_model(args.model_name, f"cuda:{args.device}")

    print("patched:", install(model))

    rows = []

    for sample in data:

        prompt, _ = render_prompt(args.prompt_mode, sample["question"])

        inputs = inputs_from_content(
            processor,
            model,
            [
                {
                    "type": "image",
                    "image": load_image(sample["image"]),
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        )

        boxes, _, _ = locate_regions(
            processor, model, sample["image"], sample["question"]
        )

        run = {}

        for name, multiply in (("off", args.off), ("on", args.on)):

            anchor, _ = build_anchor(
                processor, model, inputs, boxes, multiply=multiply
            )

            before = dict(_CALLS)

            set_anchor(anchor)

            scores, tokens = decode_with_scores(model, inputs, args.steps)

            set_anchor(None)

            run[name] = {
                "scores": scores,
                "tokens": tokens,
                "eligible": _CALLS["eligible"] - before["eligible"],
                "modified": _CALLS["modified"] - before["modified"],
                "anchor": anchor,
            }

        # The first score comes from the prefill, where the anchor deliberately
        # does not apply; the comparison starts at the first decode step.
        off, on = run["off"], run["on"]

        divergence = None
        logit_shifts = []
        variations = []

        for step in range(1, min(len(off["scores"]), len(on["scores"]))):

            a, b = off["scores"][step], on["scores"][step]

            if a.argmax().item() != b.argmax().item():
                divergence = step
                break

            logit_shifts.append((a - b).abs().max().item())
            variations.append(
                0.5 * (a.softmax(-1) - b.softmax(-1)).abs().sum().item()
            )

        rows.append({
            "id": str(sample["id"]),
            "anchor_cells": len(off["anchor"].keep),
            "eligible": on["eligible"],
            "modified": on["modified"],
            "divergence": divergence,
            "logit_shift": (
                sum(logit_shifts) / len(logit_shifts) if logit_shifts else None
            ),
            "variation": (
                sum(variations) / len(variations) if variations else None
            ),
        })

        print(
            f"[{sample['id']}] cells {len(off['anchor'].keep):3d} | "
            f"eligible {on['eligible']:4d} | modified {on['modified']:4d} | "
            f"argmax flips at "
            f"{divergence if divergence is not None else f'>{args.steps}':>4} | "
            f"mean |dlogit| "
            f"{(rows[-1]['logit_shift'] or 0):.3f} | mean TV "
            f"{(rows[-1]['variation'] or 0):.4f}"
        )

        del inputs

    uninstall()

    flipped = [row for row in rows if row["divergence"] is not None]

    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else float("nan")

    print()
    print("=" * 78)
    print(f"{args.dataset}, {len(rows)} samples, boost x{args.on} against x{args.off}")
    print(f"anchor cells per sample   : {mean('anchor_cells'):.1f}")
    print(f"decode steps eligible     : {mean('eligible'):.1f}")
    print(f"steps that applied a boost: {mean('modified'):.1f}")
    print(f"mean max |logit change|   : {mean('logit_shift'):.3f}")
    print(f"mean total variation      : {mean('variation'):.4f}")
    print(
        f"argmax flipped within {args.steps:2d} steps: "
        f"{len(flipped)} of {len(rows)} samples"
        + (f", at step {sorted(r['divergence'] for r in flipped)[0]} first"
           if flipped else "")
    )
    print()
    print("A run where the boost changes almost nothing here cannot produce an")
    print("accuracy difference, whatever the region source is; the effect is")
    print("only observable where the answer distribution is flat.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
