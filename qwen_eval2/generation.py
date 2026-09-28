import gc

import numpy as np
import torch

from .inputs import generate_inputs
from .prompts import render_prompt


# ============================================================
# Convert objects to JSON serializable objects
# ============================================================

def make_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    if isinstance(obj, (list, tuple)):
        return [
            make_jsonable(item)
            for item in obj
        ]

    if isinstance(obj, dict):
        return {
            key: make_jsonable(value)
            for key, value in obj.items()
        }

    return obj


# ============================================================
# Decode generated text
# ============================================================

def get_output_text(processor, output, inputs):
    """
    Decode only generated tokens, excluding prompt tokens.

    Important:
        output["sequences"] is moved to CPU before decoding,
        so the GPU generation output can be released earlier.
    """

    generated_ids = output["sequences"]

    # ------------------------------------------------------------
    # Move generated token IDs to CPU.
    #
    # This prevents the following decoding operation from
    # continuing to hold GPU memory through the output object.
    # ------------------------------------------------------------

    generated_ids = generated_ids.detach().cpu()

    input_ids = inputs.input_ids.detach().cpu()

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            input_ids,
            generated_ids
        )
    ]

    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # ------------------------------------------------------------
    # batch_decode returns list[str].
    #
    # Since generate_one() processes one sample at a time,
    # return the first string directly.
    # ------------------------------------------------------------

    if len(decoded) == 1:
        return decoded[0]

    return decoded


# ============================================================
# GPU cleanup
# ============================================================

def cleanup_gpu(*objects):
    """
    Release temporary objects and clear CUDA cache.

    This function does NOT delete model/processor unless they
    are explicitly passed to it.
    """

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
# Generate one response from one prompt
# ============================================================

def generate_from_prompt(
    processor,
    model,
    image,
    prompt,
    max_new_tokens,
):
    """
    Run a single generation pass and return the decoded text.

    GPU lifecycle:

        model / processor
              │
              │ remain resident
              ▼
        generate_inputs()
              │
              ├── inputs: temporary GPU tensors
              │
              ▼
        model.generate()
              │
              ├── output: temporary GPU tensors
              │
              ▼
        decode generated tokens
              │
              ▼
        return Python string
              │
              ▼
        finally:
            delete output
            delete inputs
            empty CUDA cache

    Important:
        model and processor are NOT deleted here.
    """

    inputs = None
    output = None

    try:

        # ========================================================
        # Prepare inputs
        # ========================================================

        inputs = generate_inputs(
            processor,
            model,
            image,
            prompt,
        )

        # ========================================================
        # Generation
        # ========================================================

        with torch.inference_mode():

            output = model.generate(
                **inputs,

                max_new_tokens=max_new_tokens,

                # The checkpoint's generation_config sets do_sample=True with
                # temperature=1e-6, which resolves near-tie logits randomly:
                # the same sample then gives different answers on every run.
                # Greedy keeps runs reproducible and matches the ICoT loop.
                do_sample=False,

                # Keep True if VRGA needs generation attention.
                output_attentions=True,

                output_hidden_states=False,

                return_dict_in_generate=True,
            )

        # ========================================================
        # Decode
        # ========================================================

        output_text = get_output_text(
            processor,
            output,
            inputs,
        )

        # --------------------------------------------------------
        # At this point output_text should be CPU-side Python text.
        # Do not keep output / inputs in the returned object.
        # --------------------------------------------------------

        return output_text

    finally:

        # ========================================================
        # IMPORTANT:
        #
        # Always release the temporary tensors, including when
        # model.generate() throws CUDA OutOfMemoryError.
        # ========================================================

        if output is not None:
            del output

        if inputs is not None:
            del inputs

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


# ============================================================
# Generate one response for one sample, single-stage modes
# ============================================================

def generate_one(
    processor,
    model,
    sample,
    mode,
    max_new_tokens,
):
    """
    Generate one response for one sample with a single generation pass.

    Handles the modes whose whole prompt is derived from the question
    alone: direct, cot, region_guided. Multi-pass methods such as CCoT
    live in their own module, see qwen_eval2/ccot.py.
    """

    prompt, _ = render_prompt(
        mode,
        sample["question"],
    )

    return generate_from_prompt(
        processor,
        model,
        sample["image"],
        prompt,
        max_new_tokens,
    )