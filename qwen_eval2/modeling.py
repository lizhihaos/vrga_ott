import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_model(model_name, device, vision_attn=None, load_4bit=False):
    """Load Qwen VL model and processor.

    vision_attn overrides the attention implementation of the vision tower
    only. Its eager attention materialises a [heads, tokens, tokens] fp32
    matrix, which dominates memory on large images: on a 1508 token image,
    Qwen2.5-VL-7B peaks at 21.1GB with eager and 16.4GB with sdpa, i.e. the
    difference between 0.7GB and 6.4GB of headroom on a 24GB card. The text
    model stays eager because VRGA reads its attentions.

    load_4bit quantises the weights with bitsandbytes, which is the only way
    a 30B class checkpoint fits on this machine: 30B in bf16 is about 60GB
    and the two cards together hold 48GB.
    """

    os.environ["HF_ENDPOINT"] = "http://mirrors.tools.huawei.com/huggingface"
    model_id = f"/home/lizhihao/.cache/huggingface/models/{model_name}"

    print("=" * 70)
    print("Loading model")
    print("=" * 70)
    print(f"Model path: {model_id}")

    quantization_config = None

    if load_4bit:

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Quantisation : 4-bit nf4")

    if "Qwen3-VL" in model_name:

        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
        ).eval()

    elif str(device) == "auto" or load_4bit:

        # Shard across every visible GPU. Needed once the weights alone no
        # longer fit on one card, e.g. Qwen2.5-VL-32B; slower than a single
        # card because activations cross PCIe between layers.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto" if not load_4bit else {"": device if str(device) != "auto" else 0},
            quantization_config=quantization_config,
        ).eval()

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

    if vision_attn is not None:

        model.config.vision_config._attn_implementation = vision_attn
        print(f"Vision attention : {vision_attn}")

    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded.")
    return model, processor

