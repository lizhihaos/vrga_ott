#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from qwen_eval.args import parse_args
from qwen_eval.datasets import EvalDataset
from qwen_eval.evaluator import evaluate


def main():
    args = parse_args()
    device = f"cuda:{args.device}"

    print(f"Device   : {device}")
    print(f"Model    : {args.model_name}")
    print(f"Dataset  : {args.data_name}")
    print(f"Modify   : {args.modify}")
    print(f"With CoT : {args.with_cot}")
    print(f"Prompts  : {args.prompt_modes or 'default'}")

    eval_dataset = EvalDataset(args.data_name)
    data = eval_dataset.get_data()
    print(f"Loaded dataset samples: {len(data)}")

    evaluate(
        data=data,
        data_name=args.data_name,
        model_name=args.model_name,
        modify=args.modify,
        max_new_tokens=args.max_new_tokens,
        with_cot=args.with_cot,
        prompt_modes=args.prompt_modes,
        device=device,
    )


if __name__ == "__main__":
    main()
