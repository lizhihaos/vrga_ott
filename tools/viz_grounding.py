#!/usr/bin/env python3
"""Draw what the grounding actually produced, so the boxes can be judged.

For one record of a cgr run it writes a contact sheet:

    panel 1   the image with every located box, labelled with its entity
    panel 2+  for each verified element, the crop the verifier was shown,
              once plain and once with the two boxes drawn on it

The second row is the point: whether a relation verdict is trustworthy is
mostly a question of whether the verifier could see what it was judging.

    python viz_grounding.py --record rebuttal/mmstar/.../cgr-cg-dino_maxNew2000.jsonl \
        --index 0 --out rebuttal/figs
"""

import argparse
import json
import os

from PIL import Image, ImageDraw

import os
import sys

# Allow running from any directory: the repository root holds cgr/,
# qwen_eval2/ and evaluate_deepseek.py.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from qwen_eval2.datasets import EvalDataset


def parse_args():

    parser = argparse.ArgumentParser(description="Visualise grounding boxes")

    parser.add_argument("--record", type=str, required=True,
                        help="A cgr result JSONL (any pipeline that records boxes)")
    parser.add_argument("--index", type=int, default=0,
                        help="Which record of the file")
    parser.add_argument("--dataset", type=str, default="mmstar",
                        help="Dataset to pull the image from, by record id")
    parser.add_argument("--out", type=str, default="rebuttal/figs")
    parser.add_argument("--pad", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=760)

    return parser.parse_args()


def fit(image, max_width):

    if image.width <= max_width:
        return image

    scale = max_width / image.width

    return image.resize(
        (max_width, int(image.height * scale)),
        Image.BICUBIC,
    )


def label(draw, xy, text, fill):

    x, y = xy

    try:
        box = draw.textbbox((x, y), text)
    except AttributeError:
        return

    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill)


def main():

    args = parse_args()

    records = [
        json.loads(line)
        for line in open(args.record, encoding="utf-8")
        if line.strip()
    ]

    if args.index >= len(records):
        raise SystemExit(f"only {len(records)} records in {args.record}")

    record = records[args.index]

    image = None

    for sample in EvalDataset(args.dataset).get_data():

        if sample["id"] == record["id"]:

            from qwen_eval2.images import load_image

            image = load_image(sample["image"]).convert("RGB")
            break

    if image is None:
        raise SystemExit(f"image for id {record['id']} not found")

    panels = []

    # --------------------------------------------------------
    # Panel 1: every located box
    # --------------------------------------------------------

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    boxed = record.get("cgr_boxed") or {}
    elements = record.get("cgr_grounded") or record.get("cgr_verdicts") or []

    plotted = 0

    for name, box in boxed.items():

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        draw.rectangle(box, outline="red", width=4)
        label(draw, (box[0] + 4, box[1] + 4), name, "red")
        plotted += 1

    panels.append(("image + located boxes", fit(annotated, args.max_width)))

    # --------------------------------------------------------
    # Panels 2+: the crops the verifier saw
    # --------------------------------------------------------

    for element in elements:

        subject = element.get("subject") or element.get("name")

        if subject not in boxed:
            continue

        box = boxed[subject]

        pad = 0.25
        width, height = box[2] - box[0], box[3] - box[1]

        crop_box = (
            max(0, box[0] - pad * width),
            max(0, box[1] - pad * height),
            min(image.width, box[2] + pad * width),
            min(image.height, box[3] + pad * height),
        )

        crop = image.crop(crop_box)

        # The same crop with the subject box drawn on it, which is the honest
        # thing to show a verifier that has to judge an attribute.
        marked = crop.copy()
        marked_draw = ImageDraw.Draw(marked)

        marked_draw.rectangle(
            (
                box[0] - crop_box[0],
                box[1] - crop_box[1],
                box[2] - crop_box[0],
                box[3] - crop_box[1],
            ),
            outline="red",
            width=3,
        )

        text = (
            f"{element['kind']}: {subject} "
            f"{element.get('attribute') or element.get('relation') or ''} "
            f"{str(element.get('value', ''))[:18]} -> {element['verdict']}"
        )

        for tag, panel in (("plain", crop), ("boxed", marked)):

            panel = fit(panel, args.max_width // 2)
            canvas = Image.new("RGB", (panel.width, panel.height + 26), "white")
            canvas.paste(panel, (0, 26))
            draw2 = ImageDraw.Draw(canvas)
            draw2.text((4, 6), f"{tag} | {text}"[:110], fill="black")
            panels.append((text, canvas))

    # --------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------

    gap = args.pad
    sheet_width = max(panel.width for _, panel in panels) + 2 * gap
    sheet_height = sum(panel.height + 26 + gap for _, panel in panels) + gap

    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    y = gap

    labels = []

    for text, panel in panels:

        sheet.paste(panel, (gap, y + 26))
        ImageDraw.Draw(sheet).text((gap + 4, y + 6), text[:110], fill="black")
        y += panel.height + 26 + gap

        labels.append(text)

    os.makedirs(args.out, exist_ok=True)

    path = os.path.join(args.out, f"{args.dataset}_{record['id']}_grounding.png")
    sheet.save(path)

    print(f"Question : {record['question'][:100]}")
    print(f"GT       : {str(record['answer'])[:80]}")
    print(f"Answer   : {str(record.get('cgr_response'))[:100]}")
    print(f"Boxes    : {plotted}")
    print()

    for element in elements:

        print(
            f"  [{element['verdict']:14s}] {element['kind']:9s} "
            f"{str(element.get('subject') or element.get('name'))[:18]:18s} "
            f"{str(element.get('attribute') or element.get('relation') or ''):14s} "
            f"-> {element['evidence'][:48]}"
        )

    print()
    print(f"Written  : {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
