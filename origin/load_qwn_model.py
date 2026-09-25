# ============================================================
# CELL 4
# QWEN2.5-VL-3B MODEL
# ============================================================

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


# ============================================================
# Load Qwen2.5-VL-3B
# ============================================================

def load_qwen_model(
    model_path="/home/lizhihao/.cache/huggingface/models/Qwen2.5-VL-3B-Instruct",
):
    """
    Load Qwen2.5-VL-3B-Instruct model and processor.

    Args:
        model_path: Local path of Qwen2.5-VL-3B-Instruct.

    Returns:
        model: Qwen2.5-VL-3B model
        processor: Qwen processor
    """

    print("=" * 70)
    print("Loading Qwen2.5-VL-3B-Instruct...")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        model_path
    )

    print("Model loaded.")
    print("=" * 70)

    return model, processor