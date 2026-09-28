import json
import os

import pandas as pd


class EvalDataset:
    """Unified dataset loader."""

    def __init__(self, data_name, root="/home/lizhihao/.cache/huggingface/datasets/vqa"):
        self.data_name = data_name
        self.root = root

    def _load_parquet(self, path):
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_mmhal(self):
        path = os.path.join(
            self.root,
            "MMHal-Bench",
            "data",
            "train-00000-of-00001.parquet",
        )
        records = self._load_parquet(path)
        data = []
        for i, item in enumerate(records):
            item["id"] = i
            item["type"] = item["question_type"]
            item["answer"] = item["gt_answer"]
            data.append(item)
        return data

    def _load_hallusionbench(self):
        path = os.path.join(
            self.root,
            "HallusionBench",
            "data",
            "image-00000-of-00001.parquet",
        )
        records = self._load_parquet(path)
        data = []
        for i, item in enumerate(records):
            item["id"] = i
            item["answer"] = item.pop("gt_answer_details")
            item["type"] = f"{item['category']}_{item['subcategory']}"
            data.append(item)
        return data

    def _load_rh_halu(self):
        image_root = os.path.join(self.root, "RH-Bench")
        records = self._load_json(os.path.join(image_root, "halu_data.json"))
        data = []
        for item in records:
            item["type"] = item.pop("question_type")
            item["image"] = os.path.join(image_root, item["image"])
            data.append(item)
        return data

    def _load_rh_reason(self):
        image_root = os.path.join(self.root, "RH-Bench")
        records = self._load_json(os.path.join(image_root, "reason_data.json"))
        data = []
        for item in records:
            item["type"] = item.pop("question_type")
            item["image"] = os.path.join(image_root, item["image"])
            data.append(item)
        return data

    def _load_mmbench(self):
        path = os.path.join(
            self.root,
            "MMBench_EN",
            "data",
            "dev-00000-of-00001-75b6649fb044d38b.parquet",
        )
        records = self._load_parquet(path)
        data = []
        for item in records:
            item["id"] = item.pop("index")
            hint = item["hint"]
            query = item["question"]
            item["question"] = (
                f"Hint:{hint}"
                f"Question{query}\n"
                f"Options:A.{item['A']}\n"
                f"B.{item['B']}\n"
                f"C.{item['C']}\n"
                f"D.{item['D']}"
            )
            item["type"] = item.pop("category")
            data.append(item)
        return data

    def _load_haloquest(self):
        folder_path = os.path.join(self.root, "haloquest")
        parquet_files = sorted(
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.endswith(".parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {folder_path}")

        data = []
        for path in parquet_files:
            filename = os.path.basename(path)
            print(f"Loading {filename}...")
            df = pd.read_parquet(path)
            eval_df = df[
                df["split"].astype(str).str.strip().str.lower() == "eval"
            ]
            print(f"  Total: {len(df)}, Eval: {len(eval_df)}")
            for _, row in eval_df.iterrows():
                item = row.to_dict()
                gt = item["groundtruth_responses"]
                if isinstance(gt, str):
                    try:
                        gt = json.loads(gt)
                    except json.JSONDecodeError:
                        gt = [gt]
                item["answer"] = gt
                item["type"] = item["hallucination_type"]
                data.append(item)

        print("=" * 60)
        print(f"HaloQuest Eval total: {len(data)}")
        print("=" * 60)
        return data

    def _load_mmstar(self):
        path = os.path.join(self.root, "MMStar", "mmstar.parquet")
        records = self._load_parquet(path)
        data = []
        for item in records:
            item["id"] = item.pop("index")
            item["type"] = f"{item['category']}_{item['l2_category']}"
            data.append(item)
        return data

    def _load_mathvista_mini(self):
        path = os.path.join(
            self.root,
            "MathVista",
            "data",
            "testmini-00000-of-00001-725687bf7a18d64b.parquet",
        )
        records = self._load_parquet(path)
        data = []
        for item in records:
            item["id"] = item.pop("pid")
            item["image"] = item["decoded_image"]
            item["question"] = item["query"]
            item["type"] = item["metadata"]["task"]
            data.append(item)
        return data

    def _load_pope(self):
        folder_path = os.path.join(self.root, "POPE", "data")
        parquet_files = [
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.endswith(".parquet")
        ]
        df_list = [pd.read_parquet(path) for path in parquet_files]
        records = pd.concat(df_list, ignore_index=True).to_dict(orient="records")
        data = []
        for i, item in enumerate(records):
            if "id" not in item:
                item["id"] = i
            item["type"] = item["category"]
            data.append(item)
        return data

    def _load_amber(self):
        folder_path = os.path.join(self.root, "amber", "discriminative")
        parquet_files = [
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.endswith(".parquet")
        ]
        df_list = [pd.read_parquet(path) for path in parquet_files]
        records = pd.concat(df_list, ignore_index=True).to_dict(orient="records")
        data = []
        for i, item in enumerate(records):
            if "id" not in item:
                item["id"] = i
            item["question"] = item["query"]
            item["answer"] = item["truth"]
            data.append(item)
        return data

    def get_data(self):
        mapping = {
            "HallusionBench": self._load_hallusionbench,
            "RH_halu": self._load_rh_halu,
            "RH_reason": self._load_rh_reason,
            "mmbench": self._load_mmbench,
            "haloquest": self._load_haloquest,
            "mmstar": self._load_mmstar,
            "MathVistaMini": self._load_mathvista_mini,
            "POPE": self._load_pope,
            "amber": self._load_amber,
            "mmhal": self._load_mmhal,
        }
        if self.data_name not in mapping:
            raise ValueError(f"Unknown dataset: {self.data_name}")
        return mapping[self.data_name]()

