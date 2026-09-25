import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ============================================================
# 3. Dataset
# ============================================================

class EvalDataset:
    """
    统一的数据集加载类。
    """

    def __init__(
        self,
        data_name,
        root=(
            "/home/lizhihao/.cache/"
            "huggingface/datasets/vqa"
        ),
    ):

        self.data_name = data_name
        self.root = root

    # --------------------------------------------------------
    # Parquet
    # --------------------------------------------------------

    def _load_parquet(
        self,
        path,
    ):

        df = pd.read_parquet(path)

        return df.to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    def _load_json(
        self,
        path,
    ):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    # ========================================================
    # MMHal
    # ========================================================

    def _load_mmhal(self):

        path = os.path.join(
            self.root,
            "MMHal-Bench",
            "data",
            "train-00000-of-00001.parquet",
        )

        records = self._load_parquet(
            path
        )

        data = []

        for i, e in enumerate(records):

            e["id"] = i

            e["type"] = (
                e["question_type"]
            )

            e["answer"] = (
                e["gt_answer"]
            )

            data.append(e)

        return data

    # ========================================================
    # HallusionBench
    # ========================================================

    def _load_hallusionbench(self):

        path = os.path.join(
            self.root,
            "HallusionBench",
            "data",
            "image-00000-of-00001.parquet",
        )

        records = self._load_parquet(
            path
        )

        data = []

        for i, e in enumerate(records):

            e["id"] = i

            e["answer"] = e.pop(
                "gt_answer_details"
            )

            e["type"] = (
                f"{e['category']}_"
                f"{e['subcategory']}"
            )

            data.append(e)

        return data

    # ========================================================
    # RH-Bench Halu
    # ========================================================

    def _load_rh_halu(self):

        image_root = os.path.join(
            self.root,
            "RH-Bench",
        )

        ds = self._load_json(
            os.path.join(
                image_root,
                "halu_data.json",
            )
        )

        data = []

        for e in ds:

            e["type"] = e.pop(
                "question_type"
            )

            e["image"] = os.path.join(
                image_root,
                e["image"],
            )

            data.append(e)

        return data

    # ========================================================
    # RH-Bench Reason
    # ========================================================

    def _load_rh_reason(self):

        image_root = os.path.join(
            self.root,
            "RH-Bench",
        )

        ds = self._load_json(
            os.path.join(
                image_root,
                "reason_data.json",
            )
        )

        data = []

        for e in ds:

            e["type"] = e.pop(
                "question_type"
            )

            e["image"] = os.path.join(
                image_root,
                e["image"],
            )

            data.append(e)

        return data

    # ========================================================
    # MMBench
    # ========================================================

    def _load_mmbench(self):

        path = os.path.join(
            self.root,
            "MMBench_EN",
            "data",
            "dev-00000-of-00001-75b6649fb044d38b.parquet",
        )

        records = self._load_parquet(
            path
        )

        data = []

        for e in records:

            e["id"] = e.pop(
                "index"
            )

            hint = e["hint"]

            query = e["question"]

            e["question"] = (
                f"Hint:{hint}"
                f"Question{query}\n"
                f"Options:A.{e['A']}\n"
                f"B.{e['B']}\n"
                f"C.{e['C']}\n"
                f"D.{e['D']}"
            )

            e["type"] = e.pop(
                "category"
            )

            data.append(e)

        return data

    # ========================================================
    # HaloQuest
    # ========================================================

    def _load_haloquest(self):

        folder_path = os.path.join(
            self.root,
            "haloquest",
        )

        parquet_files = sorted(
            [
                os.path.join(
                    folder_path,
                    f,
                )
                for f in os.listdir(
                    folder_path
                )
                if f.endswith(".parquet")
            ]
        )

        if not parquet_files:

            raise FileNotFoundError(
                f"No parquet files found "
                f"in {folder_path}"
            )

        data = []

        for path in parquet_files:

            filename = os.path.basename(
                path
            )

            print(
                f"Loading {filename}..."
            )

            df = pd.read_parquet(
                path
            )

            eval_df = df[
                df["split"]
                .astype(str)
                .str.strip()
                .str.lower()
                == "eval"
            ]

            print(
                f"  Total: {len(df)}, "
                f"Eval: {len(eval_df)}"
            )

            for _, row in (
                eval_df.iterrows()
            ):

                e = row.to_dict()

                gt = e[
                    "groundtruth_responses"
                ]

                if isinstance(
                    gt,
                    str,
                ):

                    try:

                        gt = json.loads(
                            gt
                        )

                    except json.JSONDecodeError:

                        gt = [gt]

                e["answer"] = gt

                e["type"] = (
                    e["hallucination_type"]
                )

                data.append(e)

        print("=" * 60)

        print(
            f"HaloQuest Eval total: "
            f"{len(data)}"
        )

        print("=" * 60)

        return data

    # ========================================================
    # MMStar
    # ========================================================

    def _load_mmstar(self):

        path = os.path.join(
            self.root,
            "MMStar",
            "mmstar.parquet",
        )

        records = self._load_parquet(
            path
        )

        data = []

        for e in records:

            e["id"] = e.pop(
                "index"
            )

            e["type"] = (
                f"{e['category']}_"
                f"{e['l2_category']}"
            )

            data.append(e)

        return data

    # ========================================================
    # MathVista Mini
    # ========================================================

    def _load_mathvista_mini(self):

        path = os.path.join(
            self.root,
            "MathVista",
            "data",
            "testmini-00000-of-00001-725687bf7a18d64b.parquet",
        )

        records = self._load_parquet(
            path
        )

        data = []

        for e in records:

            e["id"] = e.pop(
                "pid"
            )

            e["image"] = (
                e["decoded_image"]
            )

            e["question"] = (
                e["query"]
            )

            e["type"] = (
                e["metadata"]["task"]
            )

            data.append(e)

        return data

    # ========================================================
    # POPE
    # ========================================================

    def _load_pope(self):

        folder_path = os.path.join(
            self.root,
            "POPE",
            "data",
        )

        parquet_files = [
            os.path.join(
                folder_path,
                f,
            )
            for f in os.listdir(
                folder_path
            )
            if f.endswith(".parquet")
        ]

        df_list = [
            pd.read_parquet(f)
            for f in parquet_files
        ]

        combined_df = pd.concat(
            df_list,
            ignore_index=True,
        )

        records = combined_df.to_dict(
            orient="records"
        )

        data = []

        for i, e in enumerate(records):

            # 某些 POPE 文件可能没有 id
            if "id" not in e:

                e["id"] = i

            e["type"] = e[
                "category"
            ]

            data.append(e)

        return data

    # ========================================================
    # AMBER
    # ========================================================

    def _load_amber(self):

        folder_path = os.path.join(
            self.root,
            "amber",
            "discriminative",
        )

        parquet_files = [
            os.path.join(
                folder_path,
                f,
            )
            for f in os.listdir(
                folder_path
            )
            if f.endswith(".parquet")
        ]

        df_list = [
            pd.read_parquet(f)
            for f in parquet_files
        ]

        combined_df = pd.concat(
            df_list,
            ignore_index=True,
        )

        records = combined_df.to_dict(
            orient="records"
        )

        data = []

        for i, e in enumerate(records):

            if "id" not in e:

                e["id"] = i

            e["question"] = (
                e["query"]
            )

            e["answer"] = (
                e["truth"]
            )

            data.append(e)

        return data

    # ========================================================
    # get_data
    # ========================================================

    def get_data(self):

        mapping = {

            "HallusionBench":
                self._load_hallusionbench,

            "RH_halu":
                self._load_rh_halu,

            "RH_reason":
                self._load_rh_reason,

            "mmbench":
                self._load_mmbench,

            "haloquest":
                self._load_haloquest,

            "mmstar":
                self._load_mmstar,

            "MathVistaMini":
                self._load_mathvista_mini,

            "POPE":
                self._load_pope,

            "amber":
                self._load_amber,

            "mmhal":
                self._load_mmhal,
        }

        if self.data_name not in mapping:

            raise ValueError(
                f"Unknown dataset: "
                f"{self.data_name}"
            )

        return mapping[
            self.data_name
        ]()