#!/usr/bin/env python3
"""Measure where each region source actually points, against the tool's box.

Two anchors of the same size can differ completely in whether they touch the
thing the question is about, and the literature reports the downstream accuracy
rather than that. This measures the region itself, which needs no judge:

    coverage   share of the located object's cells that the anchor covers
    precision  share of the anchor's cells that fall on the object
    border     share of the anchor's cells sitting in the first or last row or
               column of the image grid, which is where the marker tokens that
               bracket the image dump their attention

The tool's box is used as the reference for "the object", so the tool source is
not being scored by its own output but by its box, and the attention source is
scored by the same box. The box is itself a model output and can be wrong; a
sample where it is wrong lowers both sources' numbers, and the picture this
tool writes for each sample is the check on that.

    python tools/anchor_region_quality.py --dataset POPE --limit 30 \
        --out rebuttal/figs/anchor_quality
"""

import argparse
import json
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import torch

from qwen_eval2.anchoring import (
    AttentionSourceAnchor,
    box_to_token_indices,
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
from qwen_eval2.prompts import render_prompt


def parse_args():

    parser = argparse.ArgumentParser(description="Region source quality")

    parser.add_argument("--dataset", type=str, default="POPE")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--prompt_mode", type=str, default="direct")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Where to write the per sample rows")
    parser.add_argument("--out", type=str, default=None,
                        help="Directory for the contact sheets, one per sample")

    return parser.parse_args()


def render(image, name, grid_h, grid_w, tool_cells, attention_cells, out_dir):

    from PIL import ImageDraw

    base = image.convert("RGB")
    max_width = 760
    scale = max_width / base.width
    canvas = base.resize((max_width, max(1, int(base.height * scale))))
    draw = ImageDraw.Draw(canvas)

    for row in range(grid_h):
        for col in range(grid_w):

            index = row * grid_w + col

            if index in tool_cells:
                colour = (0, 128, 255)
            elif index in attention_cells:
                colour = (255, 0, 0)
            else:
                continue

            x0 = col / grid_w * max_width
            y0 = row / grid_h * canvas.height
            x1 = (col + 1) / grid_w * max_width
            y1 = (row + 1) / grid_h * canvas.height

            draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)

    canvas.save(os.path.join(out_dir, f"{name}.png"))


def main():

    args = parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    model, processor = load_model_for(args)

    print("patched:", install(model))

    data = EvalDataset(args.dataset).get_data()[: args.limit]

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

        geometry = sample_geometry(processor, inputs)

        boxes, raw, _ = locate_regions(
            processor, model, sample["image"], sample["question"]
        )

        tool_cells = set()

        for item in boxes:
            tool_cells.update(
                box_to_token_indices(
                    item["box"],
                    geometry["grid_h"],
                    geometry["grid_w"],
                    convention=item.get("convention", "pixel"),
                )
            )

        anchor = AttentionSourceAnchor(
            question_end=question_end_position(processor, inputs),
            neighborhood=0,
            **geometry,
        )

        set_anchor(anchor)

        with torch.inference_mode():
            model(**inputs, use_cache=False)

        anchor.finalise()

        set_anchor(None)

        attention_cells = set(anchor.tokens)

        grid_h, grid_w = geometry["grid_h"], geometry["grid_w"]

        def on_border(cell):
            row, col = divmod(cell, grid_w)
            return row in (0, grid_h - 1) or col in (0, grid_w - 1)

        overlap = tool_cells & attention_cells

        row = {
            "id": str(sample["id"]),
            "type": str(sample.get("type", "")),
            "question": sample["question"],
            "answer": str(sample["answer"]),
            "grounding": raw,
            "grid": [grid_h, grid_w],
            "tool_cells": len(tool_cells),
            "attention_cells": len(attention_cells),
            "overlap": len(overlap),
            "coverage": (
                len(overlap) / len(tool_cells) if tool_cells else None
            ),
            "precision": (
                len(overlap) / len(attention_cells) if attention_cells else None
            ),
            "attention_on_border": (
                sum(1 for cell in attention_cells if on_border(cell))
                / len(attention_cells) if attention_cells else None
            ),
        }

        rows.append(row)

        if args.out:
            render(
                load_image(sample["image"]),
                f"{str(sample['id']):>10s}_{len(rows) - 1:03d}".replace(" ", "0"),
                grid_h,
                grid_w,
                tool_cells,
                attention_cells,
                args.out,
            )

        coverage = (
            "n/a" if row["coverage"] is None else f"{row['coverage']:.2f}"
        )
        border = (
            "n/a"
            if row["attention_on_border"] is None
            else f"{row['attention_on_border']:.2f}"
        )

        print(
            f"[{row['id']}] tool {row['tool_cells']:3d} | "
            f"attention {row['attention_cells']:3d} | "
            f"overlap {row['overlap']:2d} | "
            f"coverage {coverage} | on border {border}"
        )

        del inputs

    uninstall()

    with_boxes = [r for r in rows if r["tool_cells"]]
    with_attention = [r for r in rows if r["attention_cells"]]

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else float("nan")

    summary = {
        "n": len(rows),
        "n_with_boxes": len(with_boxes),
        "n_with_attention": len(with_attention),
        "mean_coverage": mean(r["coverage"] for r in with_boxes),
        "mean_precision": mean(r["precision"] for r in with_attention),
        "mean_attention_on_border": mean(
            r["attention_on_border"] for r in with_attention
        ),
        "samples_touched": sum(1 for r in with_boxes if r["overlap"]),
    }

    print()
    print("=" * 70)
    print(f"samples                 : {summary['n']}")
    print(f"with a located object   : {summary['n_with_boxes']}")
    print(f"with an attention region: {summary['n_with_attention']}")
    print(
        f"attention region touches the object in "
        f"{summary['samples_touched']} of {summary['n_with_boxes']} samples"
    )
    print(f"mean coverage of object : {summary['mean_coverage']:.3f}")
    print(f"mean precision          : {summary['mean_precision']:.3f}")
    print(f"mean cells on the border: {summary['mean_attention_on_border']:.3f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "rows": rows}, handle, indent=2,
                      ensure_ascii=False)
        print(f"wrote {args.json_out}")

    return 0


def load_model_for(args):
    from qwen_eval2.modeling import load_model

    return load_model(args.model_name, f"cuda:{args.device}")


if __name__ == "__main__":
    raise SystemExit(main())
