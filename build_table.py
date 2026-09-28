#!/usr/bin/env python3
"""Turn scored result files into accuracy tables.

Score files are expected to be named "<model>_<dataset>_<mode>.json", which
is what run_all_datasets.sh produces. One table per dataset is printed, with
a row per model and a column per mode; a missing file shows as "-" so the
tables can be inspected while runs are still going.

    python build_table.py --scores rebuttal/scores \
        --datasets haloquest,POPE --models qwen25vl-3b,qwen25vl-7b \
        --modes direct,cot,ccot,icot --markdown rebuttal/baseline_tables.md
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser(description="Build accuracy tables")

    parser.add_argument("--scores", type=str, default="rebuttal/scores")
    parser.add_argument("--datasets", type=str, default="haloquest")
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen2.5-VL-3B-Instruct,Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--modes", type=str, default="direct,cot,ccot,icot")
    parser.add_argument("--markdown", type=str, default=None)

    return parser.parse_args()


def read_accuracy(path):

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if not data.get("valid_samples"):
        return None

    return data


def cell(data):

    if data is None:
        return "-"

    text = f"{data['accuracy'] * 100:.1f}"

    if data.get("errors"):
        text += f" *({data['valid_samples']}n,{data['errors']}err)*"

    return text


def main():

    args = parse_args()

    root = Path(args.scores)
    datasets = [d.strip() for d in args.datasets.replace(",", " ").split() if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode and mode not in modes:
            modes.append(mode)

    blocks = []
    filled = 0

    for dataset in datasets:

        header = f"| Model | " + " | ".join(modes) + " |"
        divider = "| --- | " + " | ".join("---" for _ in modes) + " |"
        rows = [header, divider]
        any_data = False

        for model in models:

            cells = [model]

            for mode in modes:

                data = read_accuracy(
                    root / dataset / model / f"{mode}.json"
                )

                if data:
                    any_data = True
                    filled += 1

                cells.append(cell(data))

            rows.append("| " + " | ".join(cells) + " |")

        if any_data:
            blocks.append(f"### {dataset}\n\n" + "\n".join(rows))

    if not blocks:
        raise SystemExit(f"No scored files found in {root}")

    report = "\n\n".join(blocks)

    print()
    print("HaloQuest-style accuracy (%), one table per dataset.")
    print()
    print(report)
    print()
    print(f"Filled cells: {filled} of {len(datasets) * len(models) * len(modes)}")

    if args.markdown:

        Path(args.markdown).write_text(report + "\n", encoding="utf-8")
        print(f"Written to {args.markdown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
