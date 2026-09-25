from qwen_vl_utils import process_vision_info

from .images import load_image
from .prompts import render_prompt


def generate_inputs(processor, model, image_path, question, mode="ori", modify=""):
    """Build model inputs for ORI or CoT generation."""

    prompt, _ = render_prompt(mode, question)

    content = []
    if image_path is not None:
        content.append({
            "type": "image",
            "image": load_image(image_path),
            "max_pixels": 1024 * 28 * 28 * 2,
        })
    content.append({"type": "text", "text": prompt})

    messages_query = [{"role": "user", "content": content}]
    image_inputs = None
    if image_path is not None:
        image_inputs, _ = process_vision_info(messages_query)

    text_query = processor.apply_chat_template(
        messages_query,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_query],
        images=image_inputs if image_path is not None else None,
        padding=True,
        return_tensors="pt",
    )

    if image_path is not None:
        input_ids = inputs["input_ids"][0]
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0].item()
        vision_end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0].item()
        q_ids = processor.tokenizer(question, add_special_tokens=False).input_ids
        q_end_pos = vision_end_pos + len(q_ids)

        if modify and modify != "base":
            inputs[modify] = True
            inputs["q_end_pos"] = q_end_pos
            inputs["vision_start"] = vision_start_pos
            inputs["vision_end"] = vision_end_pos
            inputs["grid_w"] = inputs["image_grid_thw"][0][2].item() // 2

    return inputs.to(model.device)
