#!/usr/bin/env python3
"""Read a scored run the way a saturated benchmark requires.

HaloQuest rewards refusing: 58 per cent of its ground truths are negations,
and a constant "the image does not show it" is judged correct on 43.6 per
cent of them. An overall accuracy number therefore says very little on its
own, and a method built to reject unsupported claims can look strong for the
wrong reason.

So this prints the split that matters:

    - accuracy over the whole set
    - accuracy where the ground truth is negative, i.e. where refusing is right
    - accuracy where the ground truth is positive, i.e. where refusing is wrong
    - the refusal rate, and the refusal rate on the positive subset
    - the accuracy of the constant "does not show" answer as a floor

    python tools/report_metrics.py --scores rebuttal/scores/haloquest/Qwen2.5-VL-3B-Instruct
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_eval2.datasets import EvalDataset

NEGATIVE = re.compile(
    r"\b(not visible|cannot|can not|does not|do not|not possible|not shown|"
    r"no \w+|do not know|don't know|we cannot tell|unclear|not clearly|"
    r"not present|absent|none|unknown)\b",
    re.IGNORECASE,
)

REFUSAL = re.compile(
    r"\b(not visible|cannot be determined|not possible to determine|"
    r"does not show|no \w+ (is|are) (visible|present)|not in the image|"
    r"cannot see|not shown|unable to determine|no such)\b",
    re.IGNORECASE,
)

# POPE and the yes/no half of the hallucination benchmarks answer with a bare
# yes or no, which the NEGATIVE pattern does not match: it looks for phrases
# like "no such object", not a plain "no".
YES_NO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def is_negative(answer):
    """Whether a ground truth says the asked-about thing is absent."""

    text = str(answer)

    match = YES_NO.match(text)

    if match:
        return match.group(1).lower() == "no"

    return bool(NEGATIVE.search(text))


def anchor_cost(generations_dir, mode):
    """Mean cost and coverage of an anchoring run, read from its own records.

    A method that spends an extra call per sample has to be compared against
    what that call would have bought in sampling, and that comparison needs the
    call's size. `anchor_grounding_tokens` is what the grounding pass generated
    and `anchor_token_fraction` is the share of the image the anchor covered,
    both of which say whether the intervention was aimed at anything at all.
    """

    if not generations_dir or not os.path.isdir(generations_dir):
        return None

    candidates = [
        name for name in os.listdir(generations_dir)
        if name.startswith(f"{mode}_maxNew") and name.endswith(".jsonl")
    ]

    if not candidates:
        return None

    records = []

    with open(
        os.path.join(generations_dir, sorted(candidates)[0]),
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "anchor_grounding_tokens" in item:
                records.append(item)

    if not records:
        return None

    def mean(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return sum(values) / len(values) if values else float("nan")

    return {
        "n": len(records),
        "grounding_tokens": mean("anchor_grounding_tokens"),
        "fraction": mean("anchor_token_fraction"),
        "empty": (
            sum(1 for r in records if not r.get("anchor_token_count"))
            / len(records)
        ),
    }


def parse_args():

    parser = argparse.ArgumentParser(description="Per-subset accuracy report")

    parser.add_argument("--scores", type=str, required=True,
                        help="Directory of scored JSON files")
    parser.add_argument("--data_name", type=str, default="haloquest")
    parser.add_argument("--response_field", type=str, default=None,
                        help="Field in the generation file holding the answer text")
    parser.add_argument("--generations", type=str, default=None,
                        help="Directory of the generation JSONL files, to report "
                             "what an anchoring run spent and covered")
    parser.add_argument("--generations_suffix", type=str, default="maxNew2000",
                        help="Token tag of the generation files to read")
    parser.add_argument("--intersect", action="store_true",
                        help="Report every mode on the ids they all share. Needed "
                             "when a mode was run on a subset of the samples the "
                             "others cover, e.g. an anchor arm on the first 200 "
                             "of a file that now holds 1500")

    return parser.parse_args()


def load_ground_truth(data_name):

    """id -> (ground truth text, type), plus the generation text when present."""

    truth = {}

    for sample in EvalDataset(data_name).get_data():
        truth[sample["id"]] = {
            "answer": str(sample["answer"]),
            "type": str(sample.get("type", "")),
        }

    return truth


def summarise(results, truth):
    """Accuracy, its split by ground-truth polarity, and refusal rates.

    Ground-truth polarity decides whether refusing is the right move, which is
    why it is a column rather than a footnote: a method that declines scores
    well on negative items for the wrong reason.
    """

    def negative(record):
        return is_negative(truth.get(record["id"], {}).get("answer", ""))

    def refused(record):
        return bool(REFUSAL.search(str(record.get("model_response", ""))))

    def acc(group):
        return (
            sum(r["score"] for r in group) / len(group)
            if group else float("nan")
        )

    def refusal_rate(group):
        return (
            sum(1 for r in group if refused(r)) / len(group)
            if group else float("nan")
        )

    neg = [r for r in results if negative(r)]
    pos = [r for r in results if not negative(r)]

    return {
        "n": len(results),
        "acc": acc(results),
        "acc_neg": acc(neg),
        "acc_pos": acc(pos),
        "refusal": refusal_rate(results),
        "refusal_pos": refusal_rate(pos),
        "types": {
            t: (
                acc([r for r in results
                     if truth.get(r["id"], {}).get("type") == t]),
                len([r for r in results
                     if truth.get(r["id"], {}).get("type") == t]),
            )
            for t in sorted(
                {truth.get(r["id"], {}).get("type", "") for r in results}
            )
        },
    }


def main():

    args = parse_args()

    truth = load_ground_truth(args.data_name)

    rows = []

    for name in sorted(os.listdir(args.scores)):

        if not name.endswith(".json"):
            continue

        path = os.path.join(args.scores, name)

        try:
            scored = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        results = [
            r for r in scored.get("results", [])
            if r.get("judge") not in ("ERROR", "MISSING", None)
        ]

        if not results:
            continue

        rows.append({
            "mode": name[:-5],
            "results": results,
            "cost": anchor_cost(args.generations, name[:-5]),
            **summarise(results, truth),
        })

    if not rows:
        raise SystemExit(f"No scored files in {args.scores}")

    # ------------------------------------------------------------
    # Optionally restrict every mode to the ids they all share
    # ------------------------------------------------------------

    if args.intersect:

        common = set.intersection(
            *[{r["id"] for r in row["results"]} for row in rows]
        )

        for row in rows:

            row.update(
                summarise(
                    [r for r in row["results"] if r["id"] in common],
                    truth,
                )
            )

        print()
        print(f"Restricted to the {len(common)} ids present in every mode.")

    neg_share = sum(
        1 for v in truth.values() if is_negative(v["answer"])
    ) / max(1, len(truth))

    print()
    print(f"{args.data_name}: {len(truth)} samples, "
          f"{neg_share * 100:.0f}% of ground truths are negative "
          f"(refusing is right on those)")
    print()
    print(f"{'mode':10s} {'n':>4s} {'all':>7s} {'neg-GT':>8s} {'pos-GT':>8s} "
          f"{'refuse':>8s} {'refuse|pos':>11s}")
    print("-" * 62)

    for row in rows:

        print(
            f"{row['mode']:10s} {row['n']:4d} "
            f"{row['acc'] * 100:6.1f}% {row['acc_neg'] * 100:7.1f}% "
            f"{row['acc_pos'] * 100:7.1f}% {row['refusal'] * 100:7.1f}% "
            f"{row['refusal_pos'] * 100:10.1f}%"
        )

    print()
    print("By hallucination type:")
    print(f"{'mode':10s} " + " ".join(f"{t[:16]:>17s}" for t in rows[0]["types"]))
    print("-" * (11 + 18 * len(rows[0]["types"])))

    for row in rows:

        cells = []

        for t, (value, count) in row["types"].items():
            cells.append(f"{value * 100:6.1f}% (n={count:3d})")

        print(f"{row['mode']:10s} " + " ".join(f"{c:>17s}" for c in cells))

    print()
    print("Read the pos-GT column first: that is where refusing is wrong, so")
    print("it separates a method that sees the image from one that declines.")

    if any(row["cost"] for row in rows):

        print()
        print("What the anchoring runs spent and covered:")
        print(f"{'mode':10s} {'n':>4s} {'ground tok':>11s} {'img cover':>10s} "
              f"{'empty':>7s}")
        print("-" * 46)

        for row in rows:

            if not row["cost"]:
                continue

            cost = row["cost"]

            print(
                f"{row['mode']:10s} {cost['n']:4d} "
                f"{cost['grounding_tokens']:11.1f} "
                f"{cost['fraction'] * 100:9.1f}% "
                f"{cost['empty'] * 100:6.1f}%"
            )

        print()
        print("An anchor that covers a large share of the image is not a region,")
        print("and one that is empty means the tool returned nothing usable, so")
        print("both are reported next to the accuracy rather than hidden in it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
