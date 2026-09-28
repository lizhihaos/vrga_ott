#!/usr/bin/env python3
"""Draw the anchored tokens on the image, so the region source can be judged.

A box is not the thing that is anchored: the box is converted to image token
cells, and an off-by-one in that conversion, or the wrong coordinate
convention, leaves a run that looks fine and anchors the wrong corner. This
writes a contact sheet per sample -- the image with the grounding box and every
anchored 28px token cell outlined, plus the grounding text -- and prints the
counts.

    python tools/viz_anchor.py --dataset pope --limit 4 --out rebuttal/figs

Then look at the PNG. Numbers in the JSONL cannot tell you that the anchor is
on the object; the picture can.
"""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from PIL import Image, ImageDraw

import torch

from qwen_eval2.anchoring import (
    AttentionSourceAnchor,
    build_anchor,
    install,
    locate_regions,
    question_end_position,
    sample_geometry,
    set_anchor,
    uninstall,
)
from qwen_eval2.datasets import EvalDataset
from qwen_eval2.images import load_image
from qwen_eval2.inputs import MAX_PIXELS, inputs_from_content
from qwen_eval2.modeling import load_model
from qwen_eval2.prompts import render_prompt


def parse_args():

    parser = argparse.ArgumentParser(description="Visualise the anchored tokens")

    parser.add_argument("--dataset", type=str, default="POPE")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--region_source", type=str, default="native",
                        choices=["native", "attention"],
                        help="Which region source to draw: the grounding tool, "
                             "or the question attention VRGA uses")
    parser.add_argument("--out", type=str, default="rebuttal/figs/anchor")
    parser.add_argument("--max-width", type=int, default=760)

    return parser.parse_args()


def draw(image, anchor, kept_boxes, max_width):

    base = image.convert("RGB").copy()

    # The token grid is defined on the resized image, so the overlay has to be
    # drawn at the same scale: one cell is 28px of the 28 * grid side resize.
    resized_w = anchor.grid_w * 28
    resized_h = anchor.grid_h * 28

    scale = max_width / base.width
    canvas = base.resize((max_width, max(1, int(base.height * scale))))

    draw = ImageDraw.Draw(canvas)

    # `keep` is the dilated set that is actually boosted, `tokens` the
    # undilated cells a box covered. Both are drawn: if the blue cells are
    # wrong the grounding or the box-to-token conversion is wrong, and if only
    # the red ones reach past the object the dilation is too wide.
    located = set(anchor.tokens)
    kept = set(anchor.keep)

    for row in range(anchor.grid_h):
        for col in range(anchor.grid_w):

            index = row * anchor.grid_w + col

            if index not in located and index not in kept:
                continue

            colour = (0, 0, 255) if index in located else (255, 0, 0)

            x0 = col * resized_w / anchor.grid_w * scale
            y0 = row * resized_h / anchor.grid_h * scale
            x1 = (col + 1) * resized_w / anchor.grid_w * scale
            y1 = (row + 1) * resized_h / anchor.grid_h * scale

            draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)

    for item in kept_boxes:

        x1, y1, x2, y2 = item["box"]

        draw.rectangle(
            [
                x1 / resized_w * base.width * scale,
                y1 / resized_h * base.height * scale,
                x2 / resized_w * base.width * scale,
                y2 / resized_h * base.height * scale,
            ],
            outline=(0, 128, 255),
            width=3,
        )

    return canvas


def main():

    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = f"cuda:{args.device}"

    model, processor = load_model(args.model_name, device)

    print("patched:", install(model))

    data = EvalDataset(args.dataset).get_data()[: args.limit]

    for position, sample in enumerate(data):

        prompt, _ = render_prompt("direct", sample["question"])

        content = [
            {
                "type": "image",
                "image": load_image(sample["image"]),
                "max_pixels": MAX_PIXELS,
            },
            {"type": "text", "text": prompt},
        ]

        inputs = inputs_from_content(processor, model, content)

        raw = ""
        boxes = []

        if args.region_source == "native":

            boxes, raw, _ = locate_regions(
                processor, model, sample["image"], sample["question"]
            )

            anchor, kept = build_anchor(processor, model, inputs, boxes)

        else:

            # The attention source only has a region after it has seen the
            # prefill, so this runs the prefill and stops there.
            anchor = AttentionSourceAnchor(
                question_end=question_end_position(processor, inputs),
                **sample_geometry(processor, inputs),
            )

            kept = []

            set_anchor(anchor)

            with torch.inference_mode():
                model(**inputs, use_cache=False)

            anchor.finalise()

        set_anchor(anchor)

        canvas = draw(load_image(sample["image"]), anchor, kept, args.max_width)

        set_anchor(None)

        path = os.path.join(args.out, f"{position:03d}_{args.region_source}.png")
        canvas.save(path)

        print("=" * 70)
        print(f"[{position}] {path}")
        print(f"Q       : {sample['question']}")
        print(f"grounding: {raw[:200]!r}")
        print(f"boxes   : {[item['box'] for item in kept]}")
        print(
            f"grid    : {anchor.grid_h}x{anchor.grid_w} "
            f"| tokens {len(anchor.keep)} of "
            f"{anchor.vision_end - anchor.vision_start} "
            f"| dropped boxes {len(boxes) - len(kept)}"
        )

        if anchor.diagnostics:
            print(f"selection: {anchor.diagnostics}")

        del inputs

    uninstall()


if __name__ == "__main__":
    main()
