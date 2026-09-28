#!/usr/bin/env python3
"""Verify an uploaded model directory before spending GPU hours on it.

Checks the HuggingFace layout the evaluation code expects, and in particular
that every weight shard named in the index file is present and non-empty: a
half finished upload otherwise only shows up as a load error, or worse, as a
partially initialised model.

    python check_model_dir.py Qwen2.5-VL-32B-Instruct
    python check_model_dir.py --all
"""

import argparse
import glob
import json
import os
import sys

ROOT = "/home/lizhihao/.cache/huggingface/models"

REQUIRED = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
]

TOKENIZER_FILES = [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "generation_config.json",
]

MIN_SHARD_BYTES = 1024 * 1024


def check(model_name):

    path = os.path.join(ROOT, model_name)

    print(f"\n{model_name}")
    print(f"  path: {path}")

    if not os.path.isdir(path):
        print("  ✗ 目录不存在")
        return False

    problems = []

    for name in REQUIRED:
        if not os.path.exists(os.path.join(path, name)):
            problems.append(f"缺少 {name}")

    missing_tokenizer = [
        name for name in TOKENIZER_FILES
        if not os.path.exists(os.path.join(path, name))
    ]

    if len(missing_tokenizer) == len(TOKENIZER_FILES):
        problems.append("分词器相关文件一个都没有")

    # ------------------------------------------------------------
    # Weight shards
    # ------------------------------------------------------------

    index_path = os.path.join(path, "model.safetensors.index.json")
    single = os.path.join(path, "model.safetensors")

    if os.path.exists(index_path):

        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)

        shards = sorted(set(index["weight_map"].values()))

        present, empty, missing = 0, [], []

        for shard in shards:

            shard_path = os.path.join(path, shard)

            if not os.path.exists(shard_path):
                missing.append(shard)

            elif os.path.getsize(shard_path) < MIN_SHARD_BYTES:
                empty.append(shard)

            else:
                present += 1

        print(f"  分片: {present}/{len(shards)} 完整")

        if missing:
            problems.append(f"缺失分片 {len(missing)} 个: {missing[:3]}")

        if empty:
            problems.append(f"空/未传完分片 {len(empty)} 个: {empty[:3]}")

    elif os.path.exists(single):

        if os.path.getsize(single) < MIN_SHARD_BYTES:
            problems.append("model.safetensors 存在但太小")

        print("  权重: 单文件 model.safetensors")

    else:
        problems.append("找不到 model.safetensors.index.json 或 model.safetensors")

    # ------------------------------------------------------------
    # Leftovers from an interrupted download
    # ------------------------------------------------------------

    incomplete = glob.glob(os.path.join(path, "*.incomplete"))

    if incomplete:
        problems.append(f"有 {len(incomplete)} 个 .incomplete 残留文件")

    size = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(path)
        for name in names
    )

    print(f"  体积: {size / 2**30:.1f} GB")

    if problems:
        print("  ✗ 有问题:")
        for problem in problems:
            print(f"      - {problem}")
        return False

    print("  ✓ 可用")
    return True


def main():

    parser = argparse.ArgumentParser(description="Verify model directories")
    parser.add_argument("model", nargs="?", help="Model directory name")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:

        names = sorted(
            name for name in os.listdir(ROOT)
            if os.path.isdir(os.path.join(ROOT, name))
        )

    elif args.model:

        names = [args.model]

    else:

        parser.error("pass a model name or --all")

    results = [check(name) for name in names]
    ok = all(results)

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
