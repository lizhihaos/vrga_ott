import gc

import numpy as np
import torch

from .inputs import generate_inputs


def make_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: make_jsonable(value) for key, value in obj.items()}
    return obj


def get_output_text(processor, output, inputs):
    """Decode only generated tokens, excluding the prompt tokens."""

    generated_ids = output["sequences"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def cleanup_gpu(*objects):
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


def generate_one(processor, model, sample, mode, modify, max_new_tokens):
    """Generate one ORI or CoT response for one sample."""

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
                output_attentions=True,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
        return get_output_text(processor, output, inputs)
    finally:
        cleanup_gpu(output, inputs)

