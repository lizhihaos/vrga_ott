import os
import json
import argparse
import gc

import numpy as np
import pandas as pd
import torch

from PIL import Image
from io import BytesIO
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# 1. 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL VQA Evaluation: ORI vs CoT"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-VL-3B-Instruct",
        help=(
            "Model name, e.g. "
            "Qwen2.5-VL-3B-Instruct"
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index",
    )

    parser.add_argument(
        "--modify",
        type=str,
        default="modify_att",
        help=(
            "Modification type: "
            "modify_att or base"
        ),
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="POPE",
        help=(
            "Dataset name: "
            "HallusionBench, POPE, "
            "mmstar, haloquest, etc."
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2000,
        help="Maximum number of generated tokens",
    )

    # --------------------------------------------------------
    # 关键：增加 CoT 开关
    # --------------------------------------------------------

    parser.add_argument(
        "--with_cot",
        action="store_true",
        help="Generate both ORI and CoT responses",
    )

    return parser.parse_args()


# ============================================================
# 2. 加载模型
# ============================================================

def load_model(
    model_name,
    device,
):
    """
    加载 Qwen2.5-VL 模型和 processor。
    """

    os.environ[
        "HF_ENDPOINT"
    ] = (
        "http://mirrors.tools.huawei.com/huggingface"
    )

    model_id = (
        f"/home/lizhihao/.cache/huggingface/models/"
        f"{model_name}"
    )

    print("=" * 70)
    print("Loading model")
    print("=" * 70)

    print(
        f"Model path: {model_id}"
    )

    # --------------------------------------------------------
    # Qwen3-VL
    # --------------------------------------------------------

    if "Qwen3-VL" in model_name:

        from transformers import (
            Qwen3VLMoeForConditionalGeneration
        )

        model = (
            Qwen3VLMoeForConditionalGeneration
            .from_pretrained(
                "Qwen/Qwen3-VL-30B-A3B-Instruct",
                dtype="auto",
                device_map="auto",
            )
        )

    # --------------------------------------------------------
    # Qwen2.5-VL
    # --------------------------------------------------------

    else:

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .eval()
            .to(device)
        )

    processor = (
        AutoProcessor.from_pretrained(
            model_id
        )
    )

    print("Model loaded.")

    return model, processor


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


# ============================================================
# 4. 图片加载
# ============================================================

def load_image(
    image_input
):
    """
    统一加载图片为 PIL.Image。
    """

    if (
        isinstance(
            image_input,
            dict
        )
        and
        "bytes" in image_input
    ):

        return Image.open(
            BytesIO(
                image_input["bytes"]
            )
        ).convert("RGB")

    elif isinstance(
        image_input,
        bytes
    ):

        return Image.open(
            BytesIO(
                image_input
            )
        ).convert("RGB")

    elif isinstance(
        image_input,
        str
    ):

        return Image.open(
            image_input
        ).convert("RGB")

    else:

        return image_input


# ============================================================
# 5. 构造 ORI / CoT 输入
# ============================================================

def generate_inputs(
    processor,
    model,
    image_path,
    question,
    mode="ori",
    modify="",
):
    """
    构造模型输入。

    ORI:
        直接回答，并使用 <answer>...</answer>

    CoT:
        先逐步推理，最后必须使用
        <answer>...</answer>
    """

    # --------------------------------------------------------
    # 关键修改
    # --------------------------------------------------------

    instruct_map = {

        "ori": (
            "Answer the question directly. "
            "The final answer MUST be in "
            "<answer> </answer> tags."
        ),

        "think": (
            "Let's think step by step. "
            "Analyze the image and question carefully. "
            "Provide your reasoning first, then give "
            "the final answer. "
            "The final answer MUST be in "
            "<answer> </answer> tags."
        ),
    }

    instruct = instruct_map.get(
        mode,
        instruct_map["ori"]
    )

    # --------------------------------------------------------
    # Content
    # --------------------------------------------------------

    content = []

    if image_path is not None:

        raw_image = load_image(
            image_path
        )

        content.append(
            {
                "type":
                    "image",

                "image":
                    raw_image,

                "max_pixels":
                    1024 * 28 * 28 * 2,
            }
        )

    content.append(
        {
            "type":
                "text",

            "text":
                f"{question}\n{instruct}",
        }
    )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages_query = [
        {
            "role":
                "user",

            "content":
                content,
        }
    ]

    # --------------------------------------------------------
    # Vision
    # --------------------------------------------------------

    if image_path is not None:

        image_inputs, _ = (
            process_vision_info(
                messages_query
            )
        )

    else:

        image_inputs = None

    # --------------------------------------------------------
    # Chat template
    # --------------------------------------------------------

    text_query = (
        processor.apply_chat_template(
            messages_query,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(
        text=[text_query],

        images=(
            image_inputs
            if image_path is not None
            else None
        ),

        padding=True,

        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Attention modification
    # --------------------------------------------------------

    if image_path is not None:

        input_ids = (
            inputs["input_ids"][0]
        )

        vision_start_id = (
            processor.tokenizer
            .convert_tokens_to_ids(
                "<|vision_start|>"
            )
        )

        vision_end_id = (
            processor.tokenizer
            .convert_tokens_to_ids(
                "<|vision_end|>"
            )
        )

        vision_start_pos = (
            input_ids
            ==
            vision_start_id
        ).nonzero(
            as_tuple=True
        )[0].item()

        vision_end_pos = (
            input_ids
            ==
            vision_end_id
        ).nonzero(
            as_tuple=True
        )[0].item()

        # ----------------------------------------------------
        # 注意：
        # 这里沿用你原来的逻辑
        # ----------------------------------------------------

        q_ids = (
            processor.tokenizer(
                question,
                add_special_tokens=False,
            ).input_ids
        )

        q_end_pos = (
            vision_end_pos
            +
            len(q_ids)
        )

        if (
            modify
            and
            modify != "base"
        ):

            inputs[modify] = True

            inputs["q_end_pos"] = (
                q_end_pos
            )

            inputs["vision_start"] = (
                vision_start_pos
            )

            inputs["vision_end"] = (
                vision_end_pos
            )

            inputs["grid_w"] = (
                inputs[
                    "image_grid_thw"
                ][0][2].item()
                //
                2
            )

    # --------------------------------------------------------
    # Move to GPU
    # --------------------------------------------------------

    inputs = inputs.to(
        model.device
    )

    return inputs


# ============================================================
# 6. JSON 序列化
# ============================================================

def make_jsonable(
    obj
):

    if isinstance(
        obj,
        np.ndarray
    ):

        return obj.tolist()

    if isinstance(
        obj,
        torch.Tensor
    ):

        return (
            obj.detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        obj,
        (list, tuple)
    ):

        return [
            make_jsonable(x)
            for x in obj
        ]

    if isinstance(
        obj,
        dict
    ):

        return {
            k:
                make_jsonable(v)
            for k, v in obj.items()
        }

    return obj


# ============================================================
# 7. 解码模型输出
# ============================================================

def get_output_text(
    processor,
    output,
    inputs,
):
    """
    只解码 generated tokens，
    不包含原始 prompt。
    """

    generated_ids = (
        output["sequences"]
    )

    generated_ids_trimmed = [

        out_ids[
            len(in_ids):
        ]

        for in_ids, out_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    return processor.batch_decode(
        generated_ids_trimmed,

        skip_special_tokens=True,

        clean_up_tokenization_spaces=False,
    )


# ============================================================
# 8. GPU 清理
# ============================================================

def cleanup_gpu(
    *objects
):

    for obj in objects:

        try:

            del obj

        except Exception:

            pass

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:

            torch.cuda.ipc_collect()

        except Exception:

            pass


# ============================================================
# 9. 单次生成
# ============================================================

def generate_one(
    processor,
    model,
    sample,
    mode,
    modify,
    max_new_tokens,
):
    """
    对一个 sample 做一次生成。

    mode:
        ori
        think
    """

    inputs = None

    output = None

    try:

        inputs = generate_inputs(

            processor,

            model,

            sample["image"],

            sample["question"],

            mode=mode,

            modify=modify,
        )

        with torch.inference_mode():

            output = model.generate(

                **inputs,

                max_new_tokens=max_new_tokens,

                # --------------------------------------------
                # 保留 attention
                # --------------------------------------------

                output_attentions=True,

                output_hidden_states=False,

                return_dict_in_generate=True,
            )

        output_text = get_output_text(

            processor,

            output,

            inputs,
        )

        return output_text

    finally:

        cleanup_gpu(
            output,
            inputs,
        )


# ============================================================
# 10. 主评估
# ============================================================

def evaluate(
    data,
    data_name,
    model_name,
    modify="modify_att",
    max_new_tokens=2000,
    with_cot=False,
    device="cuda:0",
):
    """
    生成 VQA 测试结果。

    输出：
        ori_response

    如果 with_cot=True：
        think_response

    注意：
        本函数只负责生成。
        A / I / S 由后续性能评估脚本计算。
    """

    # ========================================================
    # 保存路径
    # ========================================================

    tag = (
        f"{data_name}_"
        f"{model_name}_"
        f"{modify}_"
        f"{with_cot}_"
        f"maxNew{max_new_tokens}"
    )

    print()
    print("=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)

    print(
        f"Dataset       : {data_name}"
    )

    print(
        f"Model         : {model_name}"
    )

    print(
        f"Modify        : {modify}"
    )

    print(
        f"With CoT      : {with_cot}"
    )

    print(
        f"Max New Token : {max_new_tokens}"
    )

    print(
        f"Device        : {device}"
    )

    print(
        f"Tag           : {tag}"
    )

    print("=" * 80)

    save_root = (
        "/home/lizhihao/phd/VRGA/rebuttal"
    )

    os.makedirs(
        save_root,
        exist_ok=True
    )

    save_path = os.path.join(
        save_root,
        f"{tag}.jsonl",
    )

    # ========================================================
    # 读取已经完成的样本
    # ========================================================

    have_judged = set()

    if os.path.exists(
        save_path
    ):

        print(
            f"Found existing result: "
            f"{save_path}"
        )

        with open(
            save_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    e = json.loads(
                        line
                    )

                    if "id" in e:

                        have_judged.add(
                            e["id"]
                        )

                except json.JSONDecodeError:

                    print(
                        "Warning: "
                        "skip invalid JSON line."
                    )

        print(
            f"Already evaluated: "
            f"{len(have_judged)}"
        )

    else:

        print(
            "No existing result. "
            "Starting from scratch."
        )

    total = len(data)

    # ========================================================
    # 加载模型
    # ========================================================

    model = None

    processor = None

    print()

    model, processor = load_model(
        model_name,
        device,
    )

    model.eval()

    # ========================================================
    # 开始
    # ========================================================

    try:

        with tqdm(
            total=total,
            desc=f"Evaluating {data_name}",
        ) as pbar:

            # ------------------------------------------------
            # 这里不能简单用 enumerate，
            # 因为已经存在的样本也要更新进度
            # ------------------------------------------------

            for sample in data:

                pbar.update(1)

                # --------------------------------------------
                # 图片检查
                # --------------------------------------------

                if sample.get(
                    "image"
                ) is None:

                    print(
                        f"\n[SKIP] "
                        f"id={sample.get('id')} "
                        f"has no image."
                    )

                    continue

                dataset_id = (
                    sample["id"]
                )

                # --------------------------------------------
                # 已经完成
                # --------------------------------------------

                if dataset_id in have_judged:

                    continue

                question = (
                    sample["question"]
                )

                gt_answer = (
                    sample["answer"]
                )

                # --------------------------------------------
                # 初始化
                # --------------------------------------------

                ori_output_text = None

                think_output_text = None

                # =================================================
                # ORI
                # =================================================

                print(
                    f"\n[{dataset_id}] "
                    f"Generating ORI..."
                )

                try:

                    ori_output_text = (
                        generate_one(

                            processor,

                            model,

                            sample,

                            mode="ori",

                            modify=modify,

                            max_new_tokens=(
                                max_new_tokens
                            ),
                        )
                    )

                    print(
                        f"[{dataset_id}] ORI:"
                    )

                    print(
                        ori_output_text
                    )

                except Exception as e:

                    print(
                        f"\n[ERROR] "
                        f"ORI failed "
                        f"for id={dataset_id}"
                    )

                    print(
                        repr(e)
                    )

                    cleanup_gpu()

                    continue

                # =================================================
                # CoT
                # =================================================

                if with_cot:

                    print(
                        f"\n[{dataset_id}] "
                        f"Generating CoT..."
                    )

                    try:

                        think_output_text = (
                            generate_one(

                                processor,

                                model,

                                sample,

                                mode="think",

                                modify=modify,

                                max_new_tokens=(
                                    max_new_tokens
                                ),
                            )
                        )

                        print(
                            f"[{dataset_id}] CoT:"
                        )

                        print(
                            think_output_text
                        )

                    except Exception as e:

                        print(
                            f"\n[ERROR] "
                            f"CoT failed "
                            f"for id={dataset_id}"
                        )

                        print(
                            repr(e)
                        )

                        think_output_text = None

                        cleanup_gpu()

                # =================================================
                # 构造结果
                # =================================================

                result = {

                    "id":
                        dataset_id,

                    "question":
                        question,

                    "answer":
                        gt_answer,

                    "type":
                        sample.get(
                            "type",
                            ""
                        ),

                    "ori_response":
                        ori_output_text,
                }

                # -----------------------------------------------
                # CoT
                # -----------------------------------------------

                if (
                    with_cot
                    and
                    think_output_text
                    is not None
                ):

                    result[
                        "think_response"
                    ] = (
                        think_output_text
                    )

                # =================================================
                # JSON
                # =================================================

                result_safe = (
                    make_jsonable(
                        result
                    )
                )

                # =================================================
                # 保存
                # =================================================

                with open(
                    save_path,
                    "a",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        result_safe,
                        f,
                        ensure_ascii=False,
                    )

                    f.write("\n")

                    f.flush()

                    # --------------------------------------------
                    # 确保数据落盘
                    # --------------------------------------------

                    os.fsync(
                        f.fileno()
                    )

                have_judged.add(
                    dataset_id
                )

                # =================================================
                # 打印
                # =================================================

                print(
                    f"[{dataset_id}] "
                    f"Saved."
                )

                # =================================================
                # 清理
                # =================================================

                cleanup_gpu()

    finally:

        # ========================================================
        # 释放模型
        # ========================================================

        print()
        print(
            "Cleaning model from GPU..."
        )

        cleanup_gpu(
            model,
            processor,
        )

        print(
            "GPU memory cleaned."
        )

    print()
    print("=" * 80)
    print("Evaluation Finished")
    print("=" * 80)

    print(
        f"Output: {save_path}"
    )

    print(
        f"Total samples: {total}"
    )

    print(
        f"Completed samples: "
        f"{len(have_judged)}"
    )

    print("=" * 80)


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    device = (
        f"cuda:{args.device}"
    )

    print(
        f"Device   : {device}"
    )

    print(
        f"Model    : "
        f"{args.model_name}"
    )

    print(
        f"Dataset  : "
        f"{args.data_name}"
    )

    print(
        f"Modify   : "
        f"{args.modify}"
    )

    print(
        f"With CoT : "
        f"{args.with_cot}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    eval_dataset = EvalDataset(
        args.data_name
    )

    data = (
        eval_dataset.get_data()
    )

    print(
        f"Loaded dataset samples: "
        f"{len(data)}"
    )

    # ========================================================
    # Evaluate
    # ========================================================

    evaluate(

        data=data,

        data_name=args.data_name,

        model_name=args.model_name,

        modify=args.modify,

        max_new_tokens=(
            args.max_new_tokens
        ),

        with_cot=args.with_cot,

        device=device,
    )