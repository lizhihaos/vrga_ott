# ============================================================
# RUN A FULL VQA DATASET AND EXPORT JSON / JSONL
# ============================================================
#
# 每条样本的结果格式（与 eval_vl.py / evaluate_dataset.py 一致）：
#
#     {
#         "id": dataset_id,
#         "question": question,
#         "answer": gt_answer,
#         "type": sample.get("type", ""),
#         "ori_response": response,
#         "final_answer": extract_final_answer(response),
#     }
#
# 输出：
#
#     <output_dir>/<tag>.jsonl   逐条追加，支持断点续跑
#     <output_dir>/<tag>.json    全部结果的 JSON 数组
#
# tag = {dataset_name}_{model_tag}_{prompt_tag}_maxNew{max_new_tokens}
# ============================================================


import gc
import json
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# 1. Constants
# ============================================================

MODEL_TAG = "Qwen2.5-VL-3B-Instruct"

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
)

ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# 2. JSON helpers
# ============================================================

def make_jsonable(obj):
    """
    递归地把 numpy / torch 类型转成纯 Python 类型，
    保证 json.dump 不会失败。
    """

    if isinstance(obj, np.ndarray):

        return obj.tolist()

    if isinstance(obj, np.generic):

        return obj.item()

    # torch 只做 duck typing，避免强依赖
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):

        return obj.detach().cpu().tolist()

    if isinstance(obj, (list, tuple)):

        return [
            make_jsonable(x)
            for x in obj
        ]

    if isinstance(obj, dict):

        return {
            k: make_jsonable(v)
            for k, v in obj.items()
        }

    return obj


def extract_final_answer(response):
    """
    从模型输出里抽取 <answer> ... </answer> 中的最终答案。

    抽取不到就返回 None。
    """

    if not response:

        return None

    match = ANSWER_PATTERN.search(
        response
    )

    if match is None:

        return None

    return match.group(1).strip()


# ============================================================
# 3. Image loading
# ============================================================

def load_image(image):
    """
    PIL Image / 路径 / bytes / HF dict -> PIL.Image.Image
    """

    if isinstance(image, Image.Image):

        return image.convert("RGB")

    if isinstance(image, str):

        return Image.open(
            image
        ).convert("RGB")

    if isinstance(image, bytes):

        return Image.open(
            BytesIO(image)
        ).convert("RGB")

    if isinstance(image, dict):

        if image.get("bytes") is not None:

            return Image.open(
                BytesIO(image["bytes"])
            ).convert("RGB")

        if image.get("path") is not None:

            return Image.open(
                image["path"]
            ).convert("RGB")

        raise ValueError(
            f"Unknown image dictionary: {image.keys()}"
        )

    raise TypeError(
        f"Unsupported image type: {type(image)}"
    )


# ============================================================
# 4. Memory helpers
# ============================================================

def free_memory():
    """
    每问完一个问题后调用：回收 Python 垃圾对象 + 归还
    PyTorch 显存缓存，只保留还在被引用的 model / processor。

    显存碎片是长任务慢慢变慢甚至 OOM 的主要原因
    （图片尺寸每次都不同），所以这里主动 empty_cache()。
    """

    gc.collect()

    try:

        import torch

    except ImportError:

        return

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def gpu_mem_gb():
    """
    返回 (allocated, reserved)，单位 GB；没有 CUDA 返回 None。
    """

    try:

        import torch

    except ImportError:

        return None

    if not torch.cuda.is_available():

        return None

    allocated = 0.0

    reserved = 0.0

    for i in range(torch.cuda.device_count()):

        allocated += torch.cuda.memory_allocated(i)

        reserved += torch.cuda.memory_reserved(i)

    return (
        allocated / 1024 ** 3,
        reserved / 1024 ** 3,
    )


def is_cuda_oom(exc):
    """
    判断异常是不是 CUDA OOM。
    """

    try:

        import torch

    except ImportError:

        return False

    oom_error = getattr(
        torch.cuda,
        "OutOfMemoryError",
        None,
    )

    if oom_error is not None and isinstance(exc, oom_error):

        return True

    return "out of memory" in str(exc).lower()


# ============================================================
# 5. Resume helpers
# ============================================================

def load_done_ids(jsonl_path):
    """
    读取已有 JSONL，返回已经跑完的 id 集合，用于断点续跑。
    """

    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():

        return set()

    done = set()

    with open(
        jsonl_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError:

                # 上次写盘被打断的最后一行，忽略
                continue

            done.add(
                row.get("id")
            )

    return done


def jsonl_to_json(jsonl_path, json_path):
    """
    JSONL -> JSON 数组。
    """

    rows = []

    with open(
        jsonl_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            rows.append(
                json.loads(line)
            )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            rows,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return rows


# ============================================================
# 6. Run the whole dataset
# ============================================================

def run_dataset(
    model,
    processor,
    data,
    prompt,
    dataset_name,
    generate_fn,
    max_new_tokens=2000,
    prompt_tag="ori",
    limit=None,
    output_dir=OUTPUT_DIR,
    resume=True,
    save_every=1,
    cleanup_every=1,
    free_image_after_sample=False,
):
    """
    在整份数据集上跑推理，并把结果写成 JSONL + JSON。

    每问完一个问题就丢掉这一轮的临时变量（PIL 图像、模型输入、
    生成张量），只保留 model 与 processor，所以显存不会随样本
    数增长。

    Args:
        model, processor: 已加载的 Qwen 模型与 processor。
        data: EvalDataset(...).get_data() 得到的样本列表。
        prompt: 系统提示词，例如 ORI_PROMPT。
        dataset_name: 数据集名，例如 "HallusionBench"。
        generate_fn: (image, question, prompt, max_new_tokens) -> str。
        max_new_tokens: 单条最大生成 token 数。
        prompt_tag: 提示词标记，用于文件名，例如 "ori" / "cot"。
        limit: 只跑前 limit 条（调试用），None 表示全量。
        output_dir: 输出目录。
        resume: True 表示跳过 JSONL 里已经跑完的 id。
        save_every: 每多少条重写一次 JSON 数组。
        cleanup_every: 每多少条做一次 gc + torch.cuda.empty_cache()。
            1 表示每条都清；0 表示不清理。
        free_image_after_sample: True 表示跑完一条就把 data[i]["image"]
            置为 None，进一步释放原始图像占用的内存。注意：置空后
            这份 data 不能再用来跑别的 prompt，需要重新 get_data()。

    Returns:
        results: 结果列表。
        jsonl_path: JSONL 路径。
        json_path: JSON 路径。
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{dataset_name}_{MODEL_TAG}"
        f"_{prompt_tag}_maxNew{max_new_tokens}"
    )

    jsonl_path = output_dir / f"{tag}.jsonl"

    json_path = output_dir / f"{tag}.json"

    done_ids = (
        load_done_ids(jsonl_path)
        if resume
        else set()
    )

    if resume and done_ids:

        print(
            f"Resume: {len(done_ids)} samples already "
            f"in {jsonl_path}"
        )

    # --------------------------------------------------------
    # Which samples to run
    # --------------------------------------------------------

    total = len(data)

    if limit is not None:

        total = min(total, limit)

    todo = [
        i
        for i in range(total)
        if data[i].get("id", i) not in done_ids
    ]

    print("=" * 70)
    print(f"Dataset:        {dataset_name}")
    print(f"Samples:        {total}")
    print(f"To run:         {len(todo)}")
    print(f"Output JSONL:   {jsonl_path}")
    print(f"Output JSON:    {json_path}")
    print("=" * 70)

    results = {

        row["id"]: row

        for row in (
            jsonl_to_json(jsonl_path, json_path)
            if done_ids
            else []
        )
    }

    # --------------------------------------------------------
    # Single-sample inference
    #
    # 放在函数里，这样图像 / 输入张量 / 生成结果都是局部变量，
    # 函数返回后引用立刻消失，不会累积到下一轮。
    # --------------------------------------------------------

    def infer(question, sample):

        image = load_image(
            sample["image"]
        )

        try:

            try:

                return generate_fn(
                    image,
                    question,
                    prompt,
                    max_new_tokens,
                )

            except Exception as e:

                # 共享 GPU 上别的进程抢显存 / 显存碎片导致的 OOM：
                # 清理一次再试一遍，仍然失败才交给外层报错。
                if not is_cuda_oom(e):

                    raise

                print(
                    f"[oom] cleanup + retry: {e}"
                )

                free_memory()

                return generate_fn(
                    image,
                    question,
                    prompt,
                    max_new_tokens,
                )

        finally:

            del image

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    start_time = time.time()

    n_new = 0

    with open(
        jsonl_path,
        "a",
        encoding="utf-8",
    ) as f_out, tqdm(
        todo,
        desc=f"Evaluating {dataset_name}",
    ) as pbar:

        for i in pbar:

            sample = data[i]

            dataset_id = sample.get("id", i)

            # ------------------------------------------------
            # Skip empty image
            # ------------------------------------------------

            if sample.get("image") is None:

                print(f"[skip] id={dataset_id} empty image")

                continue

            question = sample["question"]

            gt_answer = sample.get("answer", None)

            sample_type = sample.get("type", "")

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            try:

                response = infer(
                    question,
                    sample,
                )

            except Exception as e:

                print(
                    f"[error] id={dataset_id} "
                    f"{type(e).__name__}: {e}"
                )

                response = None

            # ------------------------------------------------
            # Cleanup after each question
            #
            # 图像与输入张量已经在 infer() 里释放；这里把这一轮的
            # 本地引用也断开，并归还显存缓存。
            # ------------------------------------------------

            if free_image_after_sample:

                sample["image"] = None

            del sample

            if cleanup_every and (n_new + 1) % cleanup_every == 0:

                free_memory()

            if response is None:

                continue

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            result = {
                "id": dataset_id,
                "question": question,
                "answer": gt_answer,
                "type": sample_type,
                "ori_response": response,
                "final_answer": extract_final_answer(
                    response
                ),
            }

            result = make_jsonable(result)

            # ------------------------------------------------
            # Append to JSONL（先落盘，避免长任务丢结果）
            # ------------------------------------------------

            f_out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f_out.flush()

            results[dataset_id] = result

            n_new += 1

            # ------------------------------------------------
            # 这一轮的文本结果已经存进 results / JSONL，
            # 原始 response 可以丢掉了
            # ------------------------------------------------

            del response, result

            # ------------------------------------------------
            # Periodic JSON export
            # ------------------------------------------------

            if save_every and n_new % save_every == 0:

                with open(
                    json_path,
                    "w",
                    encoding="utf-8",
                ) as f_json:

                    json.dump(
                        list(results.values()),
                        f_json,
                        ensure_ascii=False,
                        indent=2,
                    )

            # ------------------------------------------------
            # Progress / memory
            # ------------------------------------------------

            elapsed = time.time() - start_time

            if n_new > 0:

                postfix = {
                    "sec_per_sample": f"{elapsed / n_new:.1f}",
                }

                mem = gpu_mem_gb()

                if mem is not None:

                    allocated, reserved = mem

                    postfix["gpu_alloc/reserved"] = (
                        f"{allocated:.1f}/{reserved:.1f}G"
                    )

                pbar.set_postfix(**postfix)

    # --------------------------------------------------------
    # Final JSON export
    # --------------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f_json:

        json.dump(
            list(results.values()),
            f_json,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print(f"Done. {n_new} new samples in {time.time() - start_time:.1f}s")
    print(f"JSONL: {jsonl_path}")
    print(f"JSON:  {json_path}")
    print("=" * 70)

    return list(results.values()), jsonl_path, json_path
