import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_model(model_name, device):
    """Load Qwen VL model and processor."""

    os.environ["HF_ENDPOINT"] = "http://mirrors.tools.huawei.com/huggingface"
    model_id = f"/home/lizhihao/.cache/huggingface/models/{model_name}"

    print("=" * 70)
    print("Loading model")
    print("=" * 70)
    print(f"Model path: {model_id}")

    if "Qwen3-VL" in model_name:
        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            dtype="auto",
            device_map="auto",
        )
    else:
        model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .eval()
            .to(device)
        )

    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded.")
    return model, processor

