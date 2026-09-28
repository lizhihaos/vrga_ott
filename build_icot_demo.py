#!/usr/bin/env python3
"""Build a one-shot interleaved demonstration for ICoT.

The paper demonstrates the interleaved format with a worked example whose
crops are the regions the method attends to. The released repository leaves
those crops as ./path/to/icot_image_*.png placeholders, so this script builds
the demonstration mechanically instead:

  1. run the method on a held out sample, collecting the rationale it writes
     and the tiles it injects,
  2. split that rationale at the injection points,
  3. cut the injected tiles out of the image,
  4. write the image, the crops and the segment texts next to demo.json.

The rounds are taken from the dataset's non-eval split, so the demonstration
never shows the model an evaluation sample.

    python build_icot_demo.py --data_name haloquest \
        --out qwen_eval2/demos/haloquest --index 0

Writes:
    <out>/demo.json    segmentation, crop boxes and the settings used
    <out>/image.png    the demonstration image, resized as the model saw it
    <out>/crop_<i>.png the injected regions
    <out>/preview.png  the image with the crop boxes drawn, for inspection
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd
from PIL import Image, ImageDraw

from qwen_eval2.datasets import EvalDataset
from qwen_eval2.icot import generate_icot
from qwen_eval2.images import load_image
from qwen_eval2.inputs import generate_inputs
from qwen_eval2.modeling import load_model
from qwen_eval2.prompts import render_prompt

PATCH_PIXELS = 14
MERGE = 2


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build a one-shot interleaved demonstration for ICoT"
    )

    parser.add_argument("--data_name", type=str, default="haloquest")
    parser.add_argument("--index", type=int, default=0,
                        help="Which held out sample to demonstrate with")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: qwen_eval2/demos/<data_name>)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--max_insertions", type=int, default=3)
    parser.add_argument("--insert_every", type=int, default=2)

    return parser.parse_args()


def load_held_out(data_name, index):
    """Load one non-eval sample, so the demo cannot leak an eval image."""

    root = EvalDataset(data_name).root
    folder = os.path.join(root, data_name)

    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame = frame[frame["split"].astype(str).str.strip().str.lower() != "eval"]
        if len(frame):
            frames.append(frame)

    if not frames:
        raise ValueError(
            f"{data_name} has no non-eval split, so no sample can be held "
            f"out for the demonstration."
        )

    table = pd.concat(frames, ignore_index=True)

    if index >= len(table):
        raise IndexError(f"index {index} out of range ({len(table)} held out samples)")

    row = table.iloc[index].to_dict()

    return {
        "id": row.get("id", index),
        "image": load_image(row["image"]),
        "question": row["question"],
        "answer": row.get("groundtruth_responses", ""),
    }


def split_at_tokens(tokens, cut_points, tokenizer):
    """Decode the rationale in the segments the injections delimit.

    The cuts are token indices recorded by the generator, which is exact;
    newline counts in the decoded text drift because newline characters
    arrive inside merged tokens.
    """

    segments = []
    start = 0

    for cut in cut_points:

        segments.append(
            tokenizer.decode(tokens[start:cut], skip_special_tokens=True)
        )
        start = cut

    segments.append(
        tokenizer.decode(tokens[start:], skip_special_tokens=True)
    )

    return segments


def crop_box(patches, patch_rows, patch_cols):
    """Pixel box of an injected tile, in the resized image space."""

    cols = patch_cols // MERGE
    row0, col0 = divmod(int(patches[0]), cols)
    side = int(len(patches) ** 0.5)

    return (
        col0 * MERGE * PATCH_PIXELS,
        row0 * MERGE * PATCH_PIXELS,
        (col0 + side) * MERGE * PATCH_PIXELS,
        (row0 + side) * MERGE * PATCH_PIXELS,
    )


def main():

    args = parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "qwen_eval2", "demos", args.data_name,
    )
    os.makedirs(out, exist_ok=True)

    sample = load_held_out(args.data_name, args.index)
    print(f"Demo sample : id={sample['id']}")
    print(f"Question    : {sample['question']}")
    print(f"Reference   : {str(sample['answer'])[:100]}")

    model, processor = load_model(args.model_name, f"cuda:{args.device}")

    result = generate_icot(
        processor,
        model,
        sample,
        max_new_tokens=args.max_new_tokens,
        num_patches=args.num_patches,
        max_insertions=args.max_insertions,
        insert_every=args.insert_every,
        return_tokens=True,
    )

    insertions = result["icot_insertions"]
    text = result["icot_response"]

    print(f"\nInsertions  : {len(insertions)}")

    if not insertions:
        raise SystemExit(
            "The demonstration sample produced no insertion, so it cannot "
            "show the interleaved format. Try another --index."
        )

    if len(insertions) < args.max_insertions:
        print(
            f"Warning: only {len(insertions)} of {args.max_insertions} "
            f"insertions fired; the demo will show {len(insertions)} crops."
        )

    # ------------------------------------------------------------
    # The image as the model processed it, so crop boxes line up
    # ------------------------------------------------------------

    prompt, _ = render_prompt("cot", sample["question"])

    prepared = generate_inputs(processor, model, sample["image"], prompt)
    grid = prepared["image_grid_thw"][0]
    patch_rows, patch_cols = int(grid[1]), int(grid[2])

    image = sample["image"].resize(
        (patch_cols * PATCH_PIXELS, patch_rows * PATCH_PIXELS),
        Image.BICUBIC,
    )

    # ------------------------------------------------------------
    # Rationale segments and crops
    # ------------------------------------------------------------

    segments = split_at_tokens(
        result["icot_response_tokens"],
        [insertion["at_token"] for insertion in insertions],
        processor.tokenizer,
    )

    os.makedirs(out, exist_ok=True)
    image.save(os.path.join(out, "image.png"))

    crops = []

    for number, insertion in enumerate(insertions):

        box = crop_box(insertion["patches"], patch_rows, patch_cols)
        name = f"crop_{number}.png"
        image.crop(box).save(os.path.join(out, name))

        crops.append({
            "file": name,
            "box": list(box),
            "patches": insertion["patches"],
            "after_newlines": insertion["after_newlines"],
            "max_attention": insertion["max_attention"],
        })

        print(f"  crop {number}: box={box}")

    spec = {
        "data_name": args.data_name,
        "source_id": sample["id"],
        "question": sample["question"],
        "reference_answer": str(sample["answer"]),
        "prompt": prompt,
        "segments": segments,
        "tail": segments[-1],
        "image": "image.png",
        "crops": crops,
        "settings": {
            "num_patches": args.num_patches,
            "max_insertions": args.max_insertions,
            "insert_every": args.insert_every,
            "max_new_tokens": args.max_new_tokens,
            "model_name": args.model_name,
        },
        "model_answer": text,
    }

    with open(os.path.join(out, "demo.json"), "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------
    # Preview with the crop boxes drawn
    # ------------------------------------------------------------

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for number, crop in enumerate(crops):
        draw.rectangle(crop["box"], outline="red", width=3)
        draw.text((crop["box"][0] + 4, crop["box"][1] + 4), str(number + 1), fill="red")

    preview.save(os.path.join(out, "preview.png"))

    print(f"\nQuestion    : {sample['question']}")
    for number, segment in enumerate(segments):
        print(f"  segment {number}: {segment[:100]!r}")

    print(f"\nWritten to  : {out}")
    print(f"Preview     : {os.path.join(out, 'preview.png')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
