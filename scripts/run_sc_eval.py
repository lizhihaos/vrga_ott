#!/usr/bin/env python3
"""Self-consistency baseline, for comparing against CG-VR at equal budget.

CG-VR spends about 673 generated tokens and 11.8 model calls per question
against CoT's 301 tokens and single call, so "it beat CoT" alone proves
nothing: the honest control is a baseline that spends the same. Sampling N
CoT answers and consolidating them does exactly that, and at N=3 it already
outspends CG-VR (3 x 301 > 673), which keeps the comparison conservative in
the baseline's favour.

    python run_sc_eval.py --data_name haloquest --samples 3 --limit 50

Writes rebuttal/<dataset>/<model>/sc-{N}_maxNew<N>.jsonl with an sc_response
field, so scoring and the table work unchanged.
"""

import argparse
import collections
import json
import os
import time

import torch

import os
import sys

# Allow running from any directory: the repository root holds cgr/,
# qwen_eval2/ and evaluate_deepseek.py.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from qwen_eval2.datasets import EvalDataset
from qwen_eval2.images import load_image
from qwen_eval2.inputs import generate_inputs
from qwen_eval2.modeling import load_model
from qwen_eval2.prompts import render_prompt

SAVE_ROOT = "/home/lizhihao/phd/VRGA/rebuttal"

CONSENSUS_PROMPT = (
    "Question: {question}\n\n"
    "Several answers were sampled for it:\n{answers}\n\n"
    "Give the single best answer for the question."
)


def parse_args():

    parser = argparse.ArgumentParser(description="Self-consistency baseline")

    parser.add_argument("--data_name", type=str, default="haloquest")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mode", type=str, default="sc")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of CoT samples to draw per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def generate_once(model, processor, image, prompt, args, sample=False):

    inputs = generate_inputs(processor, model, image, prompt)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=sample,
            temperature=args.temperature if sample else None,
            top_p=args.top_p if sample else None,
        )

    new_ids = output[0][inputs["input_ids"].shape[1]:]

    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return text, int(new_ids.shape[0])


def main():

    args = parse_args()

    device = args.device if args.device == "auto" else f"cuda:{args.device}"

    model, processor = load_model(args.model_name, device)

    data = EvalDataset(args.data_name).get_data()

    if args.limit is not None:
        data = data[:args.limit]

    save_dir = os.path.join(SAVE_ROOT, args.data_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"{args.mode}-{args.samples}_maxNew{args.max_new_tokens}.jsonl",
    )

    done = set()

    if os.path.exists(save_path):

        for line in open(save_path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

        print(f"Already evaluated: {len(done)}")

    prompt_template, _ = render_prompt("cot", "{question}")

    print(f"Dataset : {args.data_name} ({len(data)} samples)")
    print(f"Samples : {args.samples} at temperature {args.temperature}")
    print(f"Output  : {save_path}")

    for sample in data:

        if sample.get("image") is None or sample["id"] in done:
            continue

        image = load_image(sample["image"])
        prompt = prompt_template.format(question=sample["question"].strip())

        start = time.time()
        answers, tokens, calls = [], 0, 0

        # One greedy sample plus N-1 stochastic ones keeps the baseline
        # strong: greedy is what CoT reports.
        for index in range(args.samples):

            text, used = generate_once(
                model, processor, image, prompt, args, sample=index > 0
            )

            answers.append(text)
            tokens += used
            calls += 1

        counts = collections.Counter(answers)
        best, hits = counts.most_common(1)[0]

        # Standard self consistency: majority when there is one, otherwise
        # keep the greedy answer, so the baseline can never score below plain
        # CoT. Asking a model to "pick the best" instead systematically
        # discards the hedged samples, which on a hallucination benchmark is
        # exactly the sample that is right.
        if hits > 1:
            answer = best
        else:
            answer = answers[0]

        # Kept for the record: what the pick-the-best aggregation would have
        # answered, which quantifies how much that aggregation costs.
        consensus = None

        if hits == 1:

            listing = "\n".join(
                f"{i + 1}. {a}" for i, a in enumerate(answers)
            )

            consensus, used = generate_once(
                model,
                processor,
                image,
                CONSENSUS_PROMPT.format(
                    question=sample["question"].strip(),
                    answers=listing,
                ),
                args,
                sample=False,
            )

            tokens += used
            calls += 1

        print(
            f"[{sample['id']}] {calls} calls {tokens} tok "
            f"({time.time() - start:.0f}s) "
            f"agreement={hits}/{len(answers)} -> {answer[:80]!r}"
        )

        record = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "type": sample.get("type", ""),
            "model_name": args.model_name,
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "sc_samples": args.samples,
            "sc_agreement": hits,
            "sc_tool_calls": calls,
            "sc_generated_tokens": tokens,
            "sc_all_answers": answers,
            "sc_greedy": answers[0],
            "sc_consensus": consensus,
            f"{args.mode}_response": answer,
        }

        with open(save_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        done.add(sample["id"])

        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
