#!/usr/bin/env python3
"""Check generated responses for degenerate repetition.

A response counts as degenerate at a given n when its longest repeated word
n-gram repeats at least `threshold` times. Shorter n catches short loops
("fairy tale book" three times), larger n only catches long ones, so the
table reports several n at once.

Pass a result JSONL and optionally the response fields to compare; without
--fields every `*_response` field present in the file is reported, so a
`cot,icot` run is compared automatically.

    python check_degeneration.py rebuttal/haloquest_..._cot-icot_maxNew200.jsonl
"""

import argparse
import collections
import json
import os
import re
import sys


def parse_args():

    parser = argparse.ArgumentParser(
        description="Report repetition degeneration per response field"
    )

    parser.add_argument(
        "input",
        type=str,
        help="Result JSONL produced by run_qwen_eval2.py",
    )

    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help=(
            "Comma-separated response fields to compare, "
            "default: every *_response field found"
        ),
    )

    parser.add_argument(
        "--ngrams",
        type=str,
        default="3,4,6,8",
        help=(
            "Comma-separated n-gram lengths to report (default: 3,4,6,8). "
            "A phrase of k words repeated three times shows up at n=k, so "
            "small n catches short loops and large n only long ones."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Repetitions that count as degenerate (default: 3)",
    )

    return parser.parse_args()


def load(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []

    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {number}: {exc}")

    return records


def max_repeat(text, n):
    """How often the most repeated word n-gram occurs."""
    words = re.findall(r"\S+", text.lower())
    if len(words) < n:
        return 0
    counts = collections.Counter(
        tuple(words[i:i + n]) for i in range(len(words) - n + 1)
    )
    return max(counts.values())


def is_empty(text, min_words=5):
    """A response that collapsed into newlines or a few tokens.

    Word n-grams cannot see this failure, because a flood of newlines has no
    repeated words at all.
    """
    return len(re.findall(r"[A-Za-z0-9]+", text)) < min_words


def words_after_last_insertion(record):
    """Words written after the last injected patch block.

    ICoT can end the turn right after an injection, which leaves a preamble
    and a fragment but no answer, so the counts here are not degeneracy in
    the repetition sense yet still make the sample unusable.
    """

    insertions = record.get("icot_insertions") or []

    if not insertions:
        return None

    text = record.get("icot_response", "")
    target = insertions[-1]["after_newlines"]

    position = 0
    seen = 0

    while seen < target and position < len(text):
        if text[position] == "\n":
            seen += 1
        position += 1

    return len(re.findall(r"[A-Za-z0-9]+", text[position:]))


def report_icot_tails(records, min_words):

    tails = [
        tail
        for tail in (words_after_last_insertion(record) for record in records)
        if tail is not None
    ]

    if not tails:
        return

    stubs = sum(1 for tail in tails if tail < min_words)

    print()
    print("ICoT: answers cut short after the last injection")
    print(
        f"  {stubs}/{len(tails)} = {100 * stubs / len(tails):.1f}% of samples "
        f"write fewer than {min_words} words after the last patch block"
    )


def main():

    args = parse_args()
    records = load(args.input)
    ngrams = [int(n) for n in args.ngrams.split(",") if n.strip()]

    if args.fields:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    else:
        fields = sorted({
            key
            for record in records
            for key in record
            if key.endswith("_response")
        })

    if not fields:
        raise SystemExit(
            "No response fields found. Pass --fields explicitly."
        )

    print(f"Input    : {args.input}")
    print(f"Records  : {len(records)}")
    print(f"Criterion: longest repeated n-gram repeats >= {args.threshold} times")
    print()

    header = f"{'ngram':>6}"
    for field in fields:
        header += f" {field:>22}"
    print(header)
    print("-" * len(header))

    for n in ngrams:

        row = f"{n:>6}"

        for field in fields:

            values = [
                max_repeat(record[field], n)
                for record in records
                if record.get(field)
            ]

            if not values:
                row += f" {'(no data)':>22}"
                continue

            degenerate = sum(1 for value in values if value >= args.threshold)

            row += (
                f" {degenerate:>6}/{len(values):<5}"
                f"{100 * degenerate / len(values):>6.1f}%     "
            )

        print(row)

    # --------------------------------------------------------
    # Responses that collapsed into newlines: no repeated words
    # to count, so the table above cannot see them
    # --------------------------------------------------------

    print()
    print(f"{'field':<22} {'near-empty':>14}")
    print("-" * 38)

    for field in fields:

        texts = [record[field] for record in records if record.get(field)]

        if not texts:
            continue

        empty = sum(1 for text in texts if is_empty(text))

        print(
            f"{field:<22} "
            f"{empty:>6}/{len(texts):<5} "
            f"{100 * empty / len(texts):>6.1f}%"
        )

    report_icot_tails(records, min_words=15)

    return 0


if __name__ == "__main__":
    sys.exit(main())
