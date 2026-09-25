import json
import os

from tqdm import tqdm

from .generation import cleanup_gpu, generate_one, make_jsonable
from .modeling import load_model


MODE_LABELS = {
    "direct": "Direct",
    "cot": "CoT",
    "ccot": "CCoT",
    "icot": "ICoT",
}

MODE_OUTPUT_FIELDS = {
    "direct": "ori_response",
    "cot": "think_response",
    "ccot": "ccot_response",
    "icot": "icot_response",
}


def resolve_prompt_modes(with_cot=False, prompt_modes=None):
    if prompt_modes:
        modes = [mode.strip().lower() for mode in prompt_modes.split(",") if mode.strip()]
    elif with_cot:
        modes = ["direct", "cot"]
    else:
        modes = ["direct"]

    aliases = {
        "ori": "direct",
        "baseline": "direct",
        "direct_answer": "direct",
        "think": "cot",
        "reason": "cot",
    }
    resolved = []
    for mode in modes:
        mode = aliases.get(mode, mode)
        if mode not in MODE_LABELS:
            raise ValueError(f"Unknown prompt mode: {mode}")
        if mode not in resolved:
            resolved.append(mode)
    return resolved


def evaluate(
    data,
    data_name,
    model_name,
    modify="modify_att",
    max_new_tokens=2000,
    with_cot=False,
    prompt_modes=None,
    device="cuda:0",
):
    """Generate VQA results. A/I/S should be computed by a judge script later."""

    modes = resolve_prompt_modes(with_cot=with_cot, prompt_modes=prompt_modes)
    tag = (
        f"{data_name}_{model_name}_{modify}_{with_cot}_"
        f"{'-'.join(modes)}_maxNew{max_new_tokens}"
    )

    print()
    print("=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)
    print(f"Dataset       : {data_name}")
    print(f"Model         : {model_name}")
    print(f"Modify        : {modify}")
    print(f"With CoT      : {with_cot}")
    print(f"Prompt modes  : {', '.join(MODE_LABELS[mode] for mode in modes)}")
    print(f"Max New Token : {max_new_tokens}")
    print(f"Device        : {device}")
    print(f"Tag           : {tag}")
    print("=" * 80)

    save_root = "/home/lizhihao/phd/VRGA/rebuttal"
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, f"{tag}.jsonl")

    have_judged = set()
    if os.path.exists(save_path):
        print(f"Found existing result: {save_path}")
        with open(save_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "id" in item:
                        have_judged.add(item["id"])
                except json.JSONDecodeError:
                    print("Warning: skip invalid JSON line.")
        print(f"Already evaluated: {len(have_judged)}")
    else:
        print("No existing result. Starting from scratch.")

    total = len(data)
    model = None
    processor = None
    print()
    model, processor = load_model(model_name, device)
    model.eval()

    try:
        with tqdm(total=total, desc=f"Evaluating {data_name}") as pbar:
            for sample in data:
                pbar.update(1)

                if sample.get("image") is None:
                    print(f"\n[SKIP] id={sample.get('id')} has no image.")
                    continue

                dataset_id = sample["id"]
                if dataset_id in have_judged:
                    continue

                question = sample["question"]
                gt_answer = sample["answer"]
                generated = {}

                for mode in modes:
                    label = MODE_LABELS[mode]
                    print(f"\n[{dataset_id}] Generating {label}...")
                    try:
                        output_text = generate_one(
                            processor,
                            model,
                            sample,
                            mode=mode,
                            modify=modify,
                            max_new_tokens=max_new_tokens,
                        )
                        generated[mode] = output_text
                        print(f"[{dataset_id}] {label}:")
                        print(output_text)
                    except Exception as exc:
                        print(f"\n[ERROR] {label} failed for id={dataset_id}")
                        print(repr(exc))
                        cleanup_gpu()
                        if mode == "direct":
                            break

                if "direct" not in generated:
                    continue

                result = {
                    "id": dataset_id,
                    "question": question,
                    "answer": gt_answer,
                    "type": sample.get("type", ""),
                }
                for mode, output_text in generated.items():
                    result[MODE_OUTPUT_FIELDS[mode]] = output_text

                with open(save_path, "a", encoding="utf-8") as handle:
                    json.dump(make_jsonable(result), handle, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())

                have_judged.add(dataset_id)
                print(f"[{dataset_id}] Saved.")
                cleanup_gpu()
    finally:
        print()
        print("Cleaning model from GPU...")
        cleanup_gpu(model, processor)
        print("GPU memory cleaned.")

    print()
    print("=" * 80)
    print("Evaluation Finished")
    print("=" * 80)
    print(f"Output: {save_path}")
    print(f"Total samples: {total}")
    print(f"Completed samples: {len(have_judged)}")
    print("=" * 80)
