"""ICoT: Interleaved-Modal Chain-of-Thought prompting.

Gao et al., "Interleaved-Modal Chain-of-Thought", CVPR 2025.
Reference implementation: https://github.com/jungao1106/ICoT

The rationale is generated one token at a time. Every couple of newlines,
the image patches the model currently attends to most are re-injected into
the sequence as soft visual tokens, so the remaining reasoning steps are
conditioned on the regions the model is thinking about:

    text step ... text step [patches] text step ... text step [patches] ...

The reference implementation patches model.forward() and relies on the
generate() internals of transformers 4.4x. This version drives the decode
loop itself, which keeps the same maths while running on the installed
transformers.

Two choices differ from the reference script, both on purpose:

1. The prompt is the CoT prompt, so `cot` and `icot` differ only by the
   patch injection rather than by prompt wording.

2. The patches are not wrapped in <|vision_start|>/<|vision_end|>. With the
   markers, Qwen2.5-VL reads the injected block as the start of a new chat
   turn and the output collapses into "<|im_start|>" repetitions. Pass
   vision_markers=True to reproduce that behaviour.

The patches are picked as the most attended side x side region rather than
as the side*side individually most attended patches, see _select_patches for
why the literal version is unusable here.

The decode loop is greedy, applies the checkpoint's repetition penalty, and
is verified to reproduce model.generate() token for token when the
injection is disabled.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from .inputs import MAX_PIXELS, generate_inputs, inputs_from_messages
from .images import load_image
from .prompts import render_prompt

VISION_START_ID = 151652
VISION_END_ID = 151653


def load_demo(path):
    """Load a one-shot interleaved demonstration.

    The directory is written by build_icot_demo.py and holds the
    demonstration image, its prompt, the rationale split into segments, and
    the crop injected between each pair of segments.
    """

    path = Path(path)

    if path.is_dir():
        path = path / "demo.json"

    if not path.exists():
        raise FileNotFoundError(f"ICoT demo not found: {path}")

    spec = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    segments = spec["segments"]
    crops = spec["crops"]

    if len(segments) != len(crops) + 1:

        raise ValueError(
            f"Demo has {len(segments)} segments but {len(crops)} crops; "
            f"a demo with k crops needs k + 1 segments."
        )

    return {
        "prompt": spec["prompt"],
        "question": spec["question"],
        "segments": segments,
        "image": Image.open(root / spec["image"]).convert("RGB"),
        "crops": [
            Image.open(root / crop["file"]).convert("RGB")
            for crop in crops
        ],
    }


def demo_messages(demo, image, prompt):
    """Messages for [user asks demo][assistant answers interleaved][user asks query].

    The interleaved rationale sits in an assistant turn, so the model reads it
    as its own previous answer. Putting it all in the user turn instead makes
    the model treat the demonstration as user supplied text, and it then ends
    its own turn right after an injection.
    """

    assistant = [{"type": "text", "text": demo["segments"][0]}]

    for crop, segment in zip(demo["crops"], demo["segments"][1:]):

        assistant.append({
            "type": "image",
            "image": crop,
            "max_pixels": MAX_PIXELS,
        })

        if segment:
            assistant.append({"type": "text", "text": segment})

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": demo["image"],
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": demo["prompt"]},
            ],
        },
        {"role": "assistant", "content": assistant},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": load_image(image),
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _build_patch_block(patch_embeds, base, side):
    """Position ids for the injected patch grid.

    The patches are laid out as a fresh side x side grid, the same layout the
    reference implementation gives its 4x4 block: the temporal row stays
    constant while the height and width rows index the grid.
    """

    k = patch_embeds.shape[1]
    pos = torch.zeros((3, 1, k), dtype=torch.long, device=patch_embeds.device)

    for i in range(k):
        row, col = divmod(i, side)
        pos[0, :, i] = base + 1
        pos[1, :, i] = base + 1 + row
        pos[2, :, i] = base + 1 + col

    return pos


def _select_patches(image_attention, image_grid_thw, side, selector, used):
    """Choose the patches to inject from the attention over image tokens.

    Patches that earlier insertions already used are excluded, the same way
    the reference implementation clears them from its query image mask. Every
    insertion therefore points at a fresh region.

    "tile" takes the side x side region around the most attended patch, which
    keeps the selection spatially coherent and matches the rectangular
    regions the paper draws.

    "topk" is the literal behaviour of the reference code: the side*side
    individually most attended patches. Those come out scattered over the
    first image rows, so the injected block claims adjacency between patches
    that are far apart, and on the larger benchmark images Qwen2.5-VL then
    collapses into repeated digits. Kept for reproduction only.

    image_grid_thw holds one row per image in the prompt; the query image is
    the last one.

    Returns the selected indices and the updated used-patch mask.
    """

    rows = int(image_grid_thw[-1][1]) // 2
    cols = int(image_grid_thw[-1][2]) // 2

    scores = image_attention.masked_fill(used, float("-inf"))

    if selector == "topk" or rows < side or cols < side:

        indices = scores.topk(side * side)[1].sort()[0]
        used = used.clone()
        used[indices] = True

        return indices, used

    heat = scores.reshape(rows, cols)
    best = int(heat.argmax())

    row0 = min(max(best // cols - side // 2, 0), rows - side)
    col0 = min(max(best % cols - side // 2, 0), cols - side)

    indices = [
        (row0 + row) * cols + (col0 + col)
        for row in range(side)
        for col in range(side)
    ]

    indices = torch.tensor(
        indices,
        dtype=torch.long,
        device=image_attention.device,
    )

    used = used.clone()
    used[indices] = True

    return indices, used


def generate_icot(
    processor,
    model,
    sample,
    max_new_tokens,
    num_patches=16,
    max_insertions=3,
    insert_every=2,
    patch_selector="tile",
    vision_markers=False,
    demo=None,
    return_tokens=False,
):
    """Generate one ICoT response, returning text plus the injection log.

    `demo` is a demonstration loaded by load_demo(); with it the model sees
    the interleaved format before answering, which is the setting the paper
    evaluates.

    The whole body runs under inference_mode: without it the vision tower
    keeps an autograd graph alive for every forward, which costs tens of
    gigabytes on the large benchmark images.
    """

    with torch.inference_mode():
        return _generate_icot(
            processor,
            model,
            sample,
            max_new_tokens=max_new_tokens,
            num_patches=num_patches,
            max_insertions=max_insertions,
            insert_every=insert_every,
            patch_selector=patch_selector,
            vision_markers=vision_markers,
            demo=demo,
            return_tokens=return_tokens,
        )


def _generate_icot(
    processor,
    model,
    sample,
    max_new_tokens,
    num_patches,
    max_insertions,
    insert_every,
    patch_selector,
    vision_markers,
    demo,
    return_tokens,
):
    """Decode loop behind generate_icot."""

    side = int(round(num_patches ** 0.5))

    if side * side != num_patches:
        raise ValueError(
            f"num_patches must be a perfect square, got {num_patches}"
        )

    prompt, _ = render_prompt("cot", sample["question"])

    if demo is None:

        inputs = generate_inputs(
            processor,
            model,
            sample["image"],
            prompt,
        )

    else:

        inputs = inputs_from_messages(
            processor,
            model,
            demo_messages(demo, sample["image"], prompt),
        )

    tokenizer = processor.tokenizer
    device = model.device
    embed = model.model.get_input_embeddings()

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_thw = inputs["image_grid_thw"]

    # ------------------------------------------------------------
    # Positions of the query image's tokens, and its features.
    #
    # The last vision span in the prompt is the query image: with a one-shot
    # demonstration in front, the earlier spans belong to the demo image and
    # its crops.
    # ------------------------------------------------------------

    image_start = (input_ids[0] == VISION_START_ID).nonzero().max().item() + 1
    image_end = (input_ids[0] == VISION_END_ID).nonzero().max().item()
    query_tokens = image_end - image_start

    # The vision tower is the memory hot spot of this pipeline, so the images
    # are encoded once here and the features serve both the prompt embeddings
    # and the injections below. Letting the prefill forward encode them as
    # well would run the tower twice and exhaust the GPU.
    image_embeds = torch.cat(
        model.model.get_image_features(
            inputs["pixel_values"],
            image_grid_thw,
        ).pooler_output,
        dim=0,
    )

    inputs_embeds = embed(input_ids)
    image_embeds = image_embeds.to(
        inputs_embeds.device,
        inputs_embeds.dtype,
    )

    image_mask, _ = model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_embeds,
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Injections only ever use the query image's patches. Vision features
    # arrive in prompt order, so the query image owns the last block.
    query_embeds = image_embeds[-query_tokens:]

    if query_embeds.shape[0] != query_tokens:

        raise ValueError(
            f"Query image features ({query_embeds.shape[0]}) do not match "
            f"the query image tokens in the prompt ({query_tokens})."
        )

    # ------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------

    position_ids = model.model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        inputs_embeds=None,
        attention_mask=attention_mask,
        past_key_values=None,
        second_per_grid_ts=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )

    with torch.inference_mode():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            logits_to_keep=1,
        )

    del inputs_embeds

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    # Positions of generated tokens continue from the prompt's last one.
    next_position = int(position_ids.max().item()) + 1

    # ------------------------------------------------------------
    # Decoding setup: greedy, with the checkpoint's repetition penalty so
    # this matches what the other prompt modes get from model.generate().
    # ------------------------------------------------------------

    penalty = getattr(model.generation_config, "repetition_penalty", None) or 1.0
    penalty_processor = (
        RepetitionPenaltyLogitsProcessor(penalty)
        if penalty != 1.0
        else None
    )

    eos_token_ids = model.generation_config.eos_token_id
    if not isinstance(eos_token_ids, (list, tuple)):
        eos_token_ids = [eos_token_ids]

    marker_embeds = None
    if vision_markers:
        marker_embeds = (
            embed(torch.tensor([[VISION_START_ID]], device=device)),
            embed(torch.tensor([[VISION_END_ID]], device=device)),
        )

    generated = []
    prefix = input_ids.clone()
    newlines = 0
    next_insertion = insert_every
    insertions = []
    used_patches = torch.zeros(
        query_tokens,
        dtype=torch.bool,
        device=device,
    )

    # ============================================================
    # Decode loop
    # ============================================================

    for _ in range(max_new_tokens):

        # --------------------------------------------------------
        # Sample
        # --------------------------------------------------------

        step_logits = logits
        if penalty_processor is not None:
            step_logits = penalty_processor(prefix, step_logits.clone())

        next_token = step_logits.argmax(-1, keepdim=True)
        token_id = int(next_token[0, 0])
        generated.append(token_id)
        prefix = torch.cat([prefix, next_token], dim=1)

        # --------------------------------------------------------
        # Feed the sampled token back in
        #
        # Attentions are only needed on steps that may still inject.
        # --------------------------------------------------------

        wants_attention = num_patches > 0 and len(insertions) < max_insertions

        position = torch.full(
            (3, 1, 1),
            next_position,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )

        with torch.inference_mode():
            outputs = model(
                inputs_embeds=embed(next_token),
                attention_mask=attention_mask,
                position_ids=position,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=wants_attention,
                logits_to_keep=1,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_position += 1

        # Newline characters can arrive inside merged tokens, e.g. ".\n\n",
        # so count characters rather than matching the "\n" token id.
        newlines += tokenizer.decode([token_id]).count("\n")

        # --------------------------------------------------------
        # Inject the most attended image patches
        # --------------------------------------------------------

        if newlines >= next_insertion and len(insertions) < max_insertions:

            attention = torch.cat(outputs.attentions, dim=1).mean(dim=1)[0, -1, :]
            image_attention = attention[image_start:image_end]
            patches, used_patches = _select_patches(
                image_attention,
                image_grid_thw,
                side,
                patch_selector,
                used_patches,
            )
            patch_embeds = query_embeds[patches].unsqueeze(0)

            if vision_markers:
                start_embed, end_embed = marker_embeds
                block = torch.cat([start_embed, patch_embeds, end_embed], dim=1)
                block_positions = _build_patch_block(block, next_position, side)
                block_positions[:, :, 0] = next_position
                block_positions[:, :, -1] = next_position + side + 1
                block_span = side + 2
            else:
                block = patch_embeds
                block_positions = _build_patch_block(block, next_position, side)
                block_span = side + 1

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, block.shape[1]),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )

            with torch.inference_mode():
                outputs = model(
                    inputs_embeds=block,
                    attention_mask=attention_mask,
                    position_ids=block_positions,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    logits_to_keep=1,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_position += block_span

            insertions.append({
                "after_newlines": newlines,
                "at_token": len(generated),
                "patches": patches.tolist(),
                "max_attention": float(image_attention.max()),
            })

            next_insertion = (
                newlines // insert_every + 1
            ) * insert_every

        if token_id in eos_token_ids:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)

    result = {
        "icot_response": text,
        "icot_insertions": insertions,
    }

    if return_tokens:
        result["icot_response_tokens"] = generated

    return result
