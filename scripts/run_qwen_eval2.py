#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Allow running from any directory: the repository root holds cgr/,
# qwen_eval2/ and evaluate_deepseek.py.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from qwen_eval2.args import parse_args
from qwen_eval2.datasets import EvalDataset
from qwen_eval2.evaluator import evaluate


def main():
    args = parse_args()
    device = args.device if args.device == "auto" else f"cuda:{args.device}"

    print(f"Device   : {device}")
    print(f"Model    : {args.model_name}")
    print(f"Dataset  : {args.data_name}")
    # print(f"Modify   : {args.modify}")
    # print(f"With CoT : {args.with_cot}")
    print(f"Prompts  : {args.prompt_modes or 'default'}")

    if args.vision_attn:
        print(f"VisionAttn: {args.vision_attn}")

    if args.load_4bit:
        print("Load      : 4-bit nf4")

    if "ccot" in (args.prompt_modes or ""):
        print(f"CCoT     : scene graph <= {args.ccot_sg_max_new_tokens} tokens, "
              f"question style = {args.ccot_question_style}")

    if "icot" in (args.prompt_modes or ""):
        print(f"ICoT     : {args.icot_num_patches} patches x "
              f"{args.icot_max_insertions} insertions, "
              f"every {args.icot_insert_every} newlines, "
              f"selector = {args.icot_patch_selector}, "
              f"markers = {args.icot_vision_markers}, "
              f"demo = {args.icot_demo or 'zero-shot'}")

    if "anchor" in (args.prompt_modes or ""):
        print(f"Anchor   : x{args.anchor_multiply} on "
              f"{args.anchor_neighborhood} neighbours, "
              f"head ratio = {args.anchor_head_ratio}, "
              f"cap = {args.anchor_max_token_fraction:.0%} of image tokens, "
              f"grounding <= {args.anchor_grounding_tokens} tokens")

    eval_dataset = EvalDataset(args.data_name)
    data = eval_dataset.get_data()

    if args.types:

        wanted = [t.strip().lower() for t in args.types.split(",") if t.strip()]
        data = [
            sample for sample in data
            if any(w in str(sample.get("type", "")).lower() for w in wanted)
        ]
        print(f"Filtered to types {wanted}: {len(data)} samples")

    if args.limit is not None:
        data = data[:args.limit]
        print(f"Limited to first {len(data)} samples (--limit)")

    print(f"Loaded dataset samples: {len(data)}")

    evaluate(
        data=data,
        data_name=args.data_name,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        prompt_modes=args.prompt_modes,
        device=device,
        ccot_scene_graph_max_new_tokens=args.ccot_sg_max_new_tokens,
        ccot_question_style=args.ccot_question_style,
        icot_num_patches=args.icot_num_patches,
        icot_max_insertions=args.icot_max_insertions,
        icot_insert_every=args.icot_insert_every,
        icot_patch_selector=args.icot_patch_selector,
        icot_vision_markers=args.icot_vision_markers,
        icot_demo=args.icot_demo,
        vision_attn=args.vision_attn,
        load_4bit=args.load_4bit,
        anchor_multiply=args.anchor_multiply,
        anchor_neighborhood=args.anchor_neighborhood,
        anchor_head_ratio=args.anchor_head_ratio,
        anchor_max_token_fraction=args.anchor_max_token_fraction,
        anchor_grounding_tokens=args.anchor_grounding_tokens,
        anchor_grounding_abstain=args.anchor_grounding_abstain,
    )


if __name__ == "__main__":
    main()
