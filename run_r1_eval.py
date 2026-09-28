#!/usr/bin/env python3
"""Standalone evaluation for reasoning VLMs (R1-OneVision and friends).

Follows the model card's recipe: modelscope imports, trust_remote_code, and a
prompt that asks for the final answer at the end, because these models spend
thousands of tokens thinking before they answer. Datasets come from the same
loader run_qwen_eval2.py uses, so the two are directly comparable.

    # smoke test
    python run_r1_eval.py --data_name haloquest --limit 5

    # full run, reasoning budget raised
    python run_r1_eval.py --data_name haloquest --max_new_tokens 8192

Writes rebuttal/{data}_{model}_{mode}_maxNew{N}.jsonl with a
"{mode}_response" field, so the rest of the pipeline works unchanged:

    python score_parallel.py --input <that file> --mode cot \
        --output rebuttal/scores/r1-onevision-7b_cot.json
    python build_table.py --scores rebuttal/scores
"""

import argparse
import json
import os
import time

import torch
from modelscope import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from qwen_eval2.datasets import EvalDataset
from qwen_eval2.images import load_image

MAX_PIXELS = 1024 * 28 * 28 * 2
MODELS_ROOT = "/home/lizhihao/.cache/huggingface/models"
SAVE_ROOT = "/home/lizhihao/phd/VRGA/rebuttal"

PROMPTS = {
    # From the R1-OneVision model card: it asks for the reasoning to end in
    # the final answer, which is how these models are meant to be prompted.
    "r1": (
        "Hint: Please answer the question and provide the final answer at "
        "the end. Question: {question}"
    ),
    # Bare question, for comparing the model without the format hint.
    "plain": "{question}",
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate R1-OneVision style VLMs on EvalDataset benchmarks"
    )

    parser.add_argument("--data_name", type=str, default="haloquest")
    parser.add_argument("--model_name", type=str, default="R1-OneVision-7B")
    parser.add_argument("--mode", type=str, default="cot",
                        help="Response field becomes '{mode}_response'")
    parser.add_argument("--prompt_style", type=str, default="r1",
                        choices=sorted(PROMPTS))
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N samples")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA index, or 'auto' to shard across GPUs")
    parser.add_argument("--load_4bit", action="store_true")

    return parser.parse_args()


def build_inputs(processor, model, image, question, prompt_style):

    content = [
        {
            "type": "image",
            "image": load_image(image),
            "max_pixels": MAX_PIXELS,
        },
        {
            "type": "text",
            "text": PROMPTS[prompt_style].format(question=question.strip()),
        },
    ]

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs.to(model.device)


def load_processor(model_id):
    """Load the processor, tolerating stale image processor class names.

    R1-OneVision-7B ships "image_processor_type": "Qwen2_5_VLImageProcessor",
    a class early transformers releases had and 5.x dropped, so AutoProcessor
    refuses the checkpoint. Loading the image processor by class skips that
    dispatch, and the config keys themselves are compatible.
    """

    try:

        return AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    except ValueError as error:

        if "image processor" not in str(error).lower():
            raise

        from transformers import AutoTokenizer
        from transformers import Qwen2_5_VLProcessor
        from transformers import Qwen2VLImageProcessor
        from transformers import Qwen2VLVideoProcessor

        print("Falling back to Qwen2VLImageProcessor for this checkpoint.")

        return Qwen2_5_VLProcessor(
            image_processor=Qwen2VLImageProcessor.from_pretrained(model_id),
            tokenizer=AutoTokenizer.from_pretrained(model_id),
            video_processor=Qwen2VLVideoProcessor.from_pretrained(model_id),
        )


def main():

    args = parse_args()

    model_id = os.path.join(MODELS_ROOT, args.model_name)

    # Same layout as run_qwen_eval2.py: dataset / model / mode file.
    save_dir = os.path.join(SAVE_ROOT, args.data_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"{args.mode}_maxNew{args.max_new_tokens}.jsonl",
    )

    print(f"Model    : {model_id}")
    print(f"Dataset  : {args.data_name}")
    print(f"Prompt   : {args.prompt_style}")
    print(f"Output   : {save_path}")

    data = EvalDataset(args.data_name).get_data()

    if args.limit is not None:
        data = data[:args.limit]

    # ------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------

    done = set()

    if os.path.exists(save_path):

        with open(save_path, "r", encoding="utf-8") as handle:

            for line in handle:

                line = line.strip()

                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        print(f"Already evaluated: {len(done)}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    print("Loading model...")

    quantization_config = None

    if args.load_4bit:

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if (args.device == "auto" or args.load_4bit) else None,
        quantization_config=quantization_config,
    ).eval()

    if args.device != "auto" and not args.load_4bit:
        model = model.to(f"cuda:{args.device}")

    processor = load_processor(model_id)

    print("Model loaded.")

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    field = f"{args.mode}_response"
    lengths = []
    truncated = 0

    with open(save_path, "a", encoding="utf-8") as out:

        for sample in data:

            if sample.get("image") is None or sample["id"] in done:
                continue

            inputs = build_inputs(
                processor,
                model,
                sample["image"],
                sample["question"],
                args.prompt_style,
            )

            prompt_length = inputs["input_ids"].shape[1]

            start = time.time()

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            new_ids = generated[0][prompt_length:]
            response = processor.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            hit_cap = len(new_ids) >= args.max_new_tokens
            truncated += hit_cap
            lengths.append(len(new_ids))

            print(
                f"[{sample['id']}] {len(new_ids)} tokens "
                f"({time.time() - start:.0f}s)"
                f"{' TRUNCATED' if hit_cap else ''}: "
                f"{response[-90:].strip()!r}"
            )

            record = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample.get("type", ""),
                "model_name": args.model_name,
                "prompt_style": args.prompt_style,
                "max_new_tokens": args.max_new_tokens,
                "generated_tokens": len(new_ids),
                "hit_token_cap": hit_cap,
                field: response,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            done.add(sample["id"])

            # Reasoning models hold a large KV cache per sample.
            del inputs, generated, new_ids
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    if lengths:

        print()
        print("=" * 60)
        print(f"Samples generated : {len(lengths)}")
        print(f"Mean tokens       : {sum(lengths) / len(lengths):.0f}")
        print(f"Max tokens        : {max(lengths)}")
        print(f"Hit token cap     : {truncated}/{len(lengths)}")

        if truncated:
            print(
                "A truncated sample has no final answer; raise "
                "--max_new_tokens before scoring."
            )

        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
