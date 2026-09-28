#!/usr/bin/env python3
"""Put two grounded runs side by side, sample by sample.

The point of cgv2 is a change in behaviour, not a number, so the useful view is
per sample: what the old pipeline answered, what the new one answered, and which
verdict and outcome produced the difference.

    python tools/compare_cg.py \
        --a rebuttal/haloquest/Qwen2.5-VL-3B-Instruct/cgr-cg-dino_maxNew2000.jsonl \
        --b rebuttal/haloquest/Qwen2.5-VL-3B-Instruct/cgr-cgv2-dino_maxNew2000.jsonl \
        --ids 8,15,17,18

Without --ids it prints every shared sample whose answer changed, then the
counts: how many answers changed, how many new answers are refusals, and how
the outcomes (clean, repaired, refused, fallback) are distributed.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():

    parser = argparse.ArgumentParser(description="Compare two grounded runs")

    parser.add_argument("--a", type=str, required=True,
                        help="First run, e.g. the cg pipeline")
    parser.add_argument("--b", type=str, required=True,
                        help="Second run, e.g. cgv2")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma separated ids to show in full")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N changed samples")
    parser.add_argument("--response_field", type=str, default="cgr_response")

    return parser.parse_args()


def load(path):

    rows = {}

    with open(path, encoding="utf-8") as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            rows[str(item.get("id"))] = item

    return rows


def verdicts(row):

    counts = row.get("cgr_verdict_counts")

    if counts:
        return "/".join(
            f"{counts.get(name, 0)}{name[0]}" for name in
            ("SUPPORTED", "REFUTED", "UNKNOWN")
        )

    rejected = row.get("cgr_rejected")

    return f"{row.get('cgr_grounded') and '?'} rej={rejected}"


def show(label, row, field):

    print(f"  {label}")
    print(f"    answer : {str(row.get(field, ''))[:150]!r}")
    print(f"    verdict: {verdicts(row)} | rejected={row.get('cgr_rejected')} "
          f"| fallback={row.get('cgr_fallback')} "
          f"| outcome={row.get('cgr_outcome')}")
    violations = row.get("cgr_violations")

    if violations:
        print(f"    violated: {str(violations)[:170]}")

    used = row.get("cgr_uses")

    if used:
        print(f"    asserts : {used}")


def main():

    args = parse_args()

    a, b = load(args.a), load(args.b)

    shared = sorted(set(a) & set(b), key=lambda value: int(value) if value.isdigit() else value)

    print(f"A: {args.a}\n   {len(a)} rows")
    print(f"B: {args.b}\n   {len(b)} rows")
    print(f"shared ids: {len(shared)}")

    changed = []

    for key in shared:

        old = str(a[key].get(args.response_field, ""))
        new = str(b[key].get(args.response_field, ""))

        if old.strip() != new.strip():
            changed.append(key)

    wanted = None

    if args.ids:
        wanted = [value.strip() for value in args.ids.split(",")]

    if wanted:

        for key in wanted:

            if key not in a or key not in b:
                print(f"\n[{key}] missing from one of the runs")
                continue

            print("=" * 78)
            print(f"[{key}] {str(a[key].get('question', ''))[:110]}")
            print(f"  truth : {str(a[key].get('answer', ''))[:110]}")
            show("A", a[key], args.response_field)
            show("B", b[key], args.response_field)

    shown = 0

    for key in changed:

        if args.limit is not None and shown >= args.limit:
            break

        if wanted and key in wanted:
            continue

        print("=" * 78)
        print(f"[{key}] {str(a[key].get('question', ''))[:110]}")
        print(f"  truth : {str(a[key].get('answer', ''))[:110]}")
        show("A", a[key], args.response_field)
        show("B", b[key], args.response_field)

        shown += 1

    # ------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------

    def outcome_counts(run):

        counts = {}

        for key in shared:

            outcome = run[key].get("cgr_outcome") or (
                "fallback" if run[key].get("cgr_fallback") else "unrecorded"
            )

            counts[outcome] = counts.get(outcome, 0) + 1

        return counts

    refusals = sum(
        1 for key in shared
        if str(b[key].get(args.response_field, "")).strip().lower().startswith(
            ("the image does not show", "the image does not provide")
        )
    )

    print()
    print("=" * 78)
    print(f"answers changed        : {len(changed)} of {len(shared)}")
    print(f"B refusals             : {refusals}")
    print(f"A outcomes             : {outcome_counts(a)}")
    print(f"B outcomes             : {outcome_counts(b)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
