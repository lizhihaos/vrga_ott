#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# RUN A FULL VQA DATASET WITH QWEN2.5-VL AND EXPORT JSON
# ============================================================
#
# 用法（注意用 tifa 环境）：
#
#   /home/lizhihao/miniconda3/envs/tifa/bin/python \
#       origin/run_vqa_json.py \
#       --dataset HallusionBench \
#       --max-new-tokens 2000
#
# 只跑前 5 条做测试：
#
#   /home/lizhihao/miniconda3/envs/tifa/bin/python \
#       origin/run_vqa_json.py --limit 5 --max-new-tokens 256
#
# 输出（origin/outputs/）：
#
#   HallusionBench_Qwen2.5-VL-3B-Instruct_ori_maxNew2000.jsonl
#   HallusionBench_Qwen2.5-VL-3B-Instruct_ori_maxNew2000.json
#
# 中断后重新执行同一条命令即可续跑（自动跳过 JSONL 里已有的 id）。
# ============================================================


import argparse
import sys
from pathlib import Path


# ============================================================
# 1. Project root
# ============================================================

HERE = Path(__file__).resolve().parent

ROOT = HERE.parent

for path in (str(ROOT), str(HERE)):

    if path not in sys.path:

        sys.path.insert(0, path)


import torch

from vqa_datasets.evalDataset import EvalDataset

from origin.load_qwn_model import load_qwen_model

from origin.vqa_runner import run_dataset


# ============================================================
# 2. Prompts
# ============================================================

PROMPTS = {

    "ori": (
        "Answer the question based on the image.\n"
        "Think step by step.\n"
        "The final answer MUST BE in <answer> </answer> tags."
    ),
}


# ============================================================
# 3. Generation
# ============================================================

def build_generate_fn(model, processor, prompt, max_new_tokens):
    """
    返回 generate_fn(image, question, prompt, max_new_tokens) -> str。
    """

    def generate_fn(image, question, _prompt, _max_new_tokens):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n\n"
                            f"Question: {question}\n\n"
                            "Answer:"
                        ),
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(model.device)
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # ----------------------------------------------------
        # 这一轮的输入张量、生成张量已经没用了，显式释放，
        # 只把文本结果带回给 run_dataset
        # ----------------------------------------------------

        del inputs
        del generated_ids
        del generated_ids_trimmed

        return response

    return generate_fn


# ============================================================
# 4. Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run a VQA dataset with Qwen2.5-VL "
            "and export JSON results."
        )
    )

    parser.add_argument(
        "--dataset",
        default="HallusionBench",
        help="Dataset name, e.g. HallusionBench / mmstar / POPE",
    )

    parser.add_argument(
        "--prompt-tag",
        default="ori",
        choices=list(PROMPTS.keys()),
        help="Which prompt to use.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N samples (debug).",
    )

    parser.add_argument(
        "--output-dir",
        default=str(HERE / "outputs"),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip samples already in the JSONL.",
    )

    parser.add_argument(
        "--cleanup-every",
        type=int,
        default=1,
        help=(
            "Run gc + torch.cuda.empty_cache() every N samples "
            "(0 = never). Default 1 = after every question."
        ),
    )

    parser.add_argument(
        "--free-image",
        action="store_true",
        help=(
            "Set data[i]['image'] = None after each sample to "
            "release the raw image bytes. After that the same "
            "data list cannot be reused for another prompt."
        ),
    )

    parser.add_argument(
        "--model-path",
        default=(
            "/home/lizhihao/.cache/huggingface/"
            "models/Qwen2.5-VL-3B-Instruct"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    prompt = PROMPTS[args.prompt_tag]

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data = EvalDataset(
        data_name=args.dataset,
    ).get_data()

    print(
        f"Loaded {len(data)} samples from {args.dataset}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, processor = load_qwen_model(
        model_path=args.model_path
    )

    generate_fn = build_generate_fn(
        model,
        processor,
        prompt,
        args.max_new_tokens,
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    run_dataset(
        model=model,
        processor=processor,
        data=data,
        prompt=prompt,
        dataset_name=args.dataset,
        generate_fn=generate_fn,
        max_new_tokens=args.max_new_tokens,
        prompt_tag=args.prompt_tag,
        limit=args.limit,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        cleanup_every=args.cleanup_every,
        free_image_after_sample=args.free_image,
    )


if __name__ == "__main__":

    main()
