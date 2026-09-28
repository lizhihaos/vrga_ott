#!/usr/bin/env python3
"""Run Constraint-Guided Visual Re-grounding over the EvalDataset benchmarks.

    # smoke test, VLM boxes instead of a grounding model
    python run_cgr.py --data_name mmstar --limit 5 --locate_backend vlm

    # with GroundingDINO and the relation/reasoning subsets only
    python run_cgr.py --data_name mmstar --types relation,reasoning

Writes rebuttal/<dataset>/<model>/cgr_maxNew<N>.jsonl with a cgr_response
field, so the scoring and table scripts work unchanged:

    python score_parallel.py --input <that file> --mode cgr \
        --output rebuttal/scores/<dataset>/<model>/cgr.json
"""

import argparse
import json
import os
import time

import torch

from cgr.controller import Controller
from cgr.reground import ReGrounding
from cgr.select import Selector
from cgr.tools import Tools
from qwen_eval2.datasets import EvalDataset
from qwen_eval2.images import load_image
from qwen_eval2.modeling import load_model

MODELS_ROOT = "/home/lizhihao/.cache/huggingface/models"
SAVE_ROOT = "/home/lizhihao/phd/VRGA/rebuttal"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Constraint-Guided Visual Re-grounding"
    )

    parser.add_argument("--data_name", type=str, default="mmstar")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--types", type=str, default=None,
                        help="Only samples whose type contains one of these, comma separated")
    parser.add_argument("--pipeline", type=str, default="claims",
                        choices=["select", "claims", "constraint"],
                        help="select = sample N answers, keep the one the image "
                             "supports most (recommended); claims = draft, verify, "
                             "repair; constraint = question-driven graph")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--max_claims", type=int, default=6)
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of answers the select pipeline samples")
    parser.add_argument("--locate_backend", type=str, default="dino",
                        choices=["dino", "vlm"])
    parser.add_argument("--dino_name", type=str, default="grounding-dino-tiny")
    parser.add_argument("--max_candidates", type=int, default=6)
    parser.add_argument("--answer_tokens", type=int, default=2000,
                        help="The final answer cap, keep it equal to the CoT baseline's "
                             "max_new_tokens so the comparison is budget matched")
    parser.add_argument("--max_calls", type=int, default=48)
    parser.add_argument("--no_verify_relations", action="store_true")

    return parser.parse_args()


def load_dino(name, device):

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    path = os.path.join(MODELS_ROOT, name)

    # The directory existing is not enough: a download in progress leaves
    # .incomplete files and nothing loadable behind.
    weights = [
        os.path.join(path, file)
        for file in ("model.safetensors", "pytorch_model.bin")
        if os.path.exists(os.path.join(path, file))
    ]

    if not os.path.isdir(path) or not weights:
        print(f"GroundingDINO not ready at {path}, falling back to VLM boxes")
        return None, None

    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(path).eval().to(device)

    print(f"Grounding model : {name}")

    return model, processor


def main():

    args = parse_args()

    device = args.device if args.device == "auto" else f"cuda:{args.device}"

    model, processor = load_model(args.model_name, device)

    dino, dino_processor = (None, None)

    if args.locate_backend == "dino":
        dino, dino_processor = load_dino(args.dino_name, model.device)

    backend = "dino" if dino is not None else "vlm"

    tools = Tools(
        model,
        processor,
        dino=dino,
        dino_processor=dino_processor,
        locate_backend=backend,
        max_candidates=args.max_candidates,
    )

    if args.pipeline == "select":

        solver = Selector(
            tools,
            samples=args.samples,
            max_claims=args.max_claims,
            answer_tokens=args.answer_tokens,
        )

    elif args.pipeline == "claims":

        solver = ReGrounding(
            tools,
            max_claims=args.max_claims,
            draft_tokens=args.answer_tokens,
        )

    else:

        solver = Controller(
            tools,
            beam_size=args.beam,
            max_calls=args.max_calls,
            verify_relations=not args.no_verify_relations,
        )

    data = EvalDataset(args.data_name).get_data()

    if args.types:

        wanted = [t.strip().lower() for t in args.types.split(",") if t.strip()]
        data = [
            sample for sample in data
            if any(w in str(sample.get("type", "")).lower() for w in wanted)
        ]
        print(f"Filtered to types {wanted}: {len(data)} samples")

    if args.limit is not None:
        data = data[:args.limit]

    save_dir = os.path.join(SAVE_ROOT, args.data_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # The locate backend changes the numbers, so it belongs in the file name:
    # otherwise a DINO run would resume on the VLM run's records and add
    # nothing.
    save_path = os.path.join(
        save_dir,
        f"cgr-{backend}_maxNew{args.answer_tokens}.jsonl",
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

    print(f"Dataset : {args.data_name} ({len(data)} samples)")
    print(f"Pipeline: {args.pipeline}")
    print(f"Locate  : {backend}")
    print(f"Output  : {save_path}")

    answered = 0
    fell_back = 0

    for sample in data:

        if sample.get("image") is None or sample["id"] in done:
            continue

        tools.calls = 0
        tools.generated_tokens = 0
        start = time.time()

        try:
            result = solver.solve(
                load_image(sample["image"]),
                sample["question"],
            )

        except torch.cuda.OutOfMemoryError:

            print(f"[{sample['id']}] OOM, skipping")
            torch.cuda.empty_cache()
            continue

        except Exception as error:

            print(f"[{sample['id']}] failed: {type(error).__name__}: {error}")

            result = {
                "answer": "",
                "trace": [],
                "fallback": True,
                "tool_calls": tools.calls,
            }

        answered += 1
        fell_back += bool(result.get("fallback"))

        print(
            f"[{sample['id']}] {result['tool_calls']} calls "
            f"{tools.generated_tokens} tok "
            f"({time.time() - start:.0f}s) "
            f"{'FALLBACK ' if result.get('fallback') else ''}"
            f"chains={result.get('surviving_chains', [])} "
            f"-> {str(result['answer'])[:90]!r}"
        )

        record = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "type": sample.get("type", ""),
            "model_name": args.model_name,
            "mode": "cgr",
            "max_new_tokens": args.answer_tokens,
            "cgr_response": result["answer"],
            "cgr_fallback": result.get("fallback", False),
            "cgr_tool_calls": result.get("tool_calls", 0),
            "cgr_generated_tokens": tools.generated_tokens,
            "cgr_draft": result.get("draft"),
            "cgr_claims": result.get("claims"),
            "cgr_verdicts": result.get("verdicts"),
            "cgr_retracted": result.get("retracted"),
            "cgr_repaired": result.get("repaired"),
            "cgr_candidates": result.get("candidates"),
            "cgr_chosen": result.get("chosen"),
            "cgr_graph": result.get("graph"),
            "cgr_chains": result.get("surviving_chains"),
            "cgr_trace": result.get("trace", []),
        }

        with open(save_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        done.add(sample["id"])

        torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print(f"Answered      : {answered}")
    print(f"Fell back     : {fell_back} (parse or grounding failure)")
    print(f"Output        : {save_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
