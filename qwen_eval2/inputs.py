from qwen_vl_utils import process_vision_info

from .images import load_image
from .prompts import render_prompt

MAX_PIXELS = 1024 * 28 * 28 * 2


def generate_inputs(processor, model, image, prompt):
    """Build model inputs for one image and one already-rendered prompt.

    Args:
        processor: Qwen VL processor.
        model: loaded model, used only for its device.
        image: image path, raw bytes, {"bytes": ...} dict, or PIL image.
            Pass None for text-only inputs.
        prompt: prompt string, already rendered by the caller.
    """

    content = []
    if image is not None:
        content.append({
            "type": "image",
            "image": load_image(image),
            "max_pixels": MAX_PIXELS,
        })
    content.append({"type": "text", "text": prompt})

    return inputs_from_content(processor, model, content)


def inputs_from_content(processor, model, content):
    """Turn a list of chat content items into model inputs.

    The items become one user message. ICoT uses inputs_from_messages
    instead, because its demonstration needs a second turn.
    """

    return inputs_from_messages(
        processor,
        model,
        [{"role": "user", "content": content}],
    )


def inputs_from_messages(processor, model, messages):
    """Turn a chat message list into model inputs.

    ICoT builds the messages itself: its one-shot demonstration puts the
    interleaved rationale in an assistant turn, so the model sees the format
    as its own previous answer rather than as user supplied text.
    """

    messages_query = messages
    has_image = any(
        item["type"] == "image"
        for message in messages
        for item in message["content"]
        if isinstance(item, dict)
    )

    image_inputs = None
    if has_image:
        image_inputs, _ = process_vision_info(messages_query)

    text_query = processor.apply_chat_template(
        messages_query,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_query],
        images=image_inputs if has_image else None,
        padding=True,
        return_tensors="pt",
    )

    if has_image:
        input_ids = inputs["input_ids"][0]
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0][-1].item()
        vision_end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0][-1].item()

        # inputs["q_end_pos"] = q_end_pos
        # inputs["vision_start"] = vision_start_pos
        # inputs["vision_end"] = vision_end_pos
        # inputs["grid_w"] = inputs["image_grid_thw"][0][2].item() // 2

    return inputs.to(model.device)


def generate_inputs_for_mode(processor, model, image, question, mode="direct"):
    """Build inputs for one of the single-stage prompt modes."""

    prompt, _ = render_prompt(mode, question)

    return generate_inputs(
        processor,
        model,
        image,
        prompt,
    )
