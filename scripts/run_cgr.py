#!/usr/bin/env python3
"""Run Constraint-Guided Visual Re-grounding over the EvalDataset benchmarks.

    # smoke test, VLM boxes instead of a grounding model
    python run_cgr.py --data_name mmstar --limit 5 --locate_backend vlm

    # with GroundingDINO and the relation/reasoning subsets only
    python run_cgr.py --data_name mmstar --types relation,reasoning

Writes rebuttal/<dataset>/<model>/cgr-<pipeline>-<backend>_maxNew<N>.jsonl
with a cgr_response.
field, so the scoring and table scripts work unchanged:

    python score_parallel.py --input <that file> --mode cgr \
        --output rebuttal/scores/<dataset>/<model>/cgr.json
"""

import argparse
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

from cgr.controller import Controller
from cgr.reground import ReGrounding
from cgr.grounded_cg import GroundedCCoT
from cgr.strict_cg import StrictGroundedCCoT
from cgr.react import React
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
                        choices=["cg", "cgv2", "react", "select", "claims", "constraint"],
                        help="cg = grounded CCoT scene graph (recommended); "
                             "react = reason-act loop; select = sample N answers "
                             "and keep the best supported; claims = draft, verify, "
                             "repair; constraint = question-driven graph search")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--max_claims", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=6,
                        help="ReAct loop budget")
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
    parser.add_argument("--fallback", type=str, default="refuse",
                        choices=["refuse", "direct"],
                        help="cgv2 在 plan 解析失败或 plan 为空时怎么办。refuse "
                             "回答固定的'证据不足'，direct 退化成原方法那样的自由"
                             "生成——那种情况下答案没有任何证据可对照，实测会产出"
                             "很自信的幻觉，所以默认 refuse。")
    parser.add_argument("--vision_attn", type=str, default=None,
                        choices=["eager", "sdpa"],
                        help="视觉塔的注意力实现。7B 用 eager 时峰值 21GB 以上，"
                             "haloquest 这种大图会直接 OOM 并退化成 fallback 答案；"
                             "表里 7B 那一列本来就是 sdpa 跑的，所以这里要一致。")
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

    model, processor = load_model(args.model_name, device, vision_attn=args.vision_attn)

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

    if args.pipeline == "cg":

        solver = GroundedCCoT(tools, max_elements=args.max_claims)

    elif args.pipeline == "cgv2":

        solver = StrictGroundedCCoT(
            tools,
            max_elements=args.max_claims,
            answer_tokens=args.answer_tokens,
            fallback=args.fallback,
        )

    elif args.pipeline == "react":

        solver = React(tools, max_steps=args.max_steps)

    elif args.pipeline == "select":

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
        f"cgr-{args.pipeline}-{backend}_maxNew{args.answer_tokens}.jsonl",
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
            "vision_attn": args.vision_attn or "eager",
            "max_new_tokens": args.answer_tokens,
            "cgr_response": result["answer"],
            "cgr_fallback": result.get("fallback", False),
            "cgr_tool_calls": result.get("tool_calls", 0),
            "cgr_generated_tokens": tools.generated_tokens,
            "cgr_draft": result.get("draft"),
            "cgr_claims": result.get("claims"),
            "cgr_verdicts": result.get("verdicts"),
            "cgr_retracted": result.get("retracted"),
            "cgr_rejected": result.get("rejected"),
            "cgr_repaired": result.get("repaired"),
            "cgr_outcome": result.get("outcome"),
            "cgr_verdict_counts": result.get("verdict_counts"),
            "cgr_uses": result.get("uses"),
            "cgr_violations": result.get("violations"),
            "cgr_repaired_uses": result.get("repaired_uses"),
            "cgr_repaired_violations": result.get("repaired_violations"),
            "cgr_fallback_reason": result.get("fallback_reason"),
            "cgr_fallback_mode": args.fallback if args.pipeline == "cgv2" else None,
            "cgr_steps": result.get("steps"),
            "cgr_plan": result.get("plan"),
            "cgr_grounded": result.get("grounded"),
            "cgr_boxed": result.get("boxes"),
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
