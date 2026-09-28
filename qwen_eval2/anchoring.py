"""Tool-anchored attention: VRGA's intervention with a non-drifting region source.

VRGA reads the regions it boosts out of the question token's cross-attention
over the image tokens, computed during the prefill. The paper's own framing is
that reasoning degrades perception, and the same mechanism works against that
region source: the attention map is produced by the reasoning prefix that the
paper says drifts, so the anchor can drift with it.

This module keeps the intervention and replaces the source. Regions come from a
tool call made before decoding starts -- Qwen2.5-VL's own grounding mode, or
GroundingDINO -- so the anchor is fixed by the question alone and is identical
at decode step 1 and step 2000. It is training free and needs no box
annotations, which is the gap left by Faithful-MR1 style box-guided anchoring.

The intervention is the one VRGA already validated, so an anchored run differs
from a VRGA run in exactly one variable, the region source:

    at every decode step (query length 1), for the heads whose mean attention
    on the image tokens is above `head_ratio` times their mean attention over
    all keys, multiply the post-softmax weights of the anchored tokens by
    `multiply` and renormalise.

Injection point: `Qwen2_5_VLAttention.forward` resolves its default attention
implementation from the module globals of `modeling_qwen2_5_vl` at call time,
and transformers registers no "eager" entry in `ALL_ATTENTION_FUNCTIONS` (it is
only reachable as that default). Replacing the module global therefore
intercepts every text attention call, and nothing under site-packages is
modified. `install` counts its invocations and raises if the hook never fired,
so a silently inert patch cannot produce a table of null results.
"""

import math
import re

import torch

from .generation import get_output_text
from .images import load_image
from .inputs import MAX_PIXELS, inputs_from_content
from .prompts import render_prompt


# The anchoring modes share one intervention and one prompt each; only the
# region source and the prompt differ, so they are one method with three rows.
#   anchor      tool located regions, question asked directly
#   anchor_cot  tool located regions, question asked with a reasoning prompt
#   vrg         attention derived regions, which is the paper's own source
ANCHOR_MODES = ("anchor", "anchor_cot", "vrg")

ANCHOR_PROMPT_MODE = {
    "anchor": "direct",
    "anchor_cot": "cot",
    "vrg": "cot",
}

ANCHOR_REGION_SOURCE = {
    "anchor": "native",
    "anchor_cot": "native",
    "vrg": "attention",
}

# The strength the paper uses. A run at any other strength writes to its own
# file, because the boost is not in the mode name and two strengths of one mode
# would otherwise share a file and a resume state.
ANCHOR_DEFAULT_MULTIPLY = 1.5


# ============================================================
# Grounding tool: question -> boxes
# ============================================================

GROUNDING_PROMPT = (
    "Locate the objects that the question asks about, and only those.\n"
    "Question: {question}\n"
    "Answer with a JSON list and nothing else: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "name"}}] '
    "with the coordinates scaled to 0-1000."
)

# The same call with permission to find nothing. Measured on 120 POPE samples,
# the plain prompt returns a box for every single one, including all 60 where
# the question's object is absent by construction, and anchoring on that box
# anchors a hallucination. This asks the tool to abstain instead, which is the
# only way its precision is observable at all.
GROUNDING_PROMPT_ABSTAIN = (
    "Locate the objects that the question asks about, and only those.\n"
    "If an object the question asks about is not visible in the image, leave it "
    "out. It is correct to answer with an empty list.\n"
    "Question: {question}\n"
    "Answer with a JSON list and nothing else: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "name"}}] '
    "with the coordinates scaled to 0-1000."
)

# Four numbers, optionally in Qwen's native <|box_start|>(x,y),(x,y) form.
_BOX_JSON = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_BOX_TUPLE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,?\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)
_LABEL = re.compile(r'"label"\s*:\s*"([^"]*)"')


def parse_boxes(text, convention="pixel"):
    """Pull bounding boxes out of a grounding answer.

    One sample's `convention` is set by `locate_regions` and recorded with the
    result, see `box_to_token_indices` for why it cannot be assumed.
    """

    matches = [m.groups() for m in _BOX_JSON.finditer(text)]

    if not matches:
        matches = [m.groups() for m in _BOX_TUPLE.finditer(text)]

    labels = _LABEL.findall(text)

    boxes = []

    for index, values in enumerate(matches):

        box = [float(value) for value in values]

        if box[2] < box[0]:
            box[0], box[2] = box[2], box[0]

        if box[3] < box[1]:
            box[1], box[3] = box[3], box[1]

        boxes.append({
            "box": box,
            "label": labels[index] if index < len(labels) else "",
            "convention": convention,
        })

    return boxes


def locate_regions(
    processor,
    model,
    image,
    question,
    max_new_tokens=96,
    convention="pixel",
    abstain=False,
):
    """One short generation that turns the question into boxes.

    Returns (boxes, raw_text, generated_tokens). The token count is returned
    because an extra call is the cost this method has to justify: the
    comparison that the literature review found missing is exactly
    "tool call" against "the same call spent on sampling".
    """

    template = GROUNDING_PROMPT_ABSTAIN if abstain else GROUNDING_PROMPT

    prompt = template.replace("{question}", question.strip())

    content = [
        {"type": "image", "image": load_image(image), "max_pixels": MAX_PIXELS},
        {"type": "text", "text": prompt},
    ]

    inputs = inputs_from_content(processor, model, content)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=False,
            return_dict_in_generate=True,
        )

    raw = get_output_text(processor, output, inputs)
    generated = int(output.sequences.shape[1] - inputs.input_ids.shape[1])

    del output, inputs

    return parse_boxes(raw, convention=convention), raw, generated


# ============================================================
# Boxes -> image token indices
# ============================================================

def box_to_token_indices(box, grid_h, grid_w, convention="pixel"):
    """Map one box to the flat indices of the image tokens it covers.

    The image tokens arrive as a row major grid: grid_h = image_grid_thw[1] // 2
    and grid_w = image_grid_thw[2] // 2, because the patch merger folds each
    2x2 group of 14px patches into one token, so one token cell is 28px.

    `convention` is what the grounding answer's coordinates are in. It is a
    measured property of the checkpoint, not a documented one: Qwen2.5-VL-3B
    under transformers 5.15 answers in the pixel space of the resized image
    (asked for a red square occupying x in [300, 600] of a 600x400 image, it
    replied [435, 109, 672, 388], where 392 is the height smart_resize picks),
    while the published format says 0-1000. Passing the wrong one puts the
    anchor in the wrong corner, which is why the raw grounding text is stored
    with every result.
    """

    x1, y1, x2, y2 = box

    if convention == "norm1000":

        scale_x, scale_y = grid_w / 1000.0, grid_h / 1000.0

    else:

        scale_x = scale_y = 1.0 / 28.0

    col_start = math.floor(x1 * scale_x)
    col_end = math.ceil(x2 * scale_x) - 1
    row_start = math.floor(y1 * scale_y)
    row_end = math.ceil(y2 * scale_y) - 1

    col_start = max(0, min(col_start, grid_w - 1))
    col_end = max(col_start, min(col_end, grid_w - 1))
    row_start = max(0, min(row_start, grid_h - 1))
    row_end = max(row_start, min(row_end, grid_h - 1))

    indices = []

    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            indices.append(row * grid_w + col)

    return indices


def boxes_to_token_indices(boxes, grid_h, grid_w, max_tokens=None):
    """Union of the per box token sets, capped at `max_tokens`.

    The cap bounds the cells the boxes cover, not the final anchored set: each
    surviving cell is later dilated into a square, so the anchored count is
    larger and is recorded as a fraction of the image tokens instead of being
    predicted from here.

    A degenerate answer such as one box covering the whole image would anchor
    everything and change nothing except the normalisation, which is worse than
    useless: it looks like an experiment. Boxes are therefore added smallest
    first and any box that would push the union past the cap is dropped, so a
    whole image box leaves the anchor empty rather than diffuse. The caller
    records how many boxes survived, which makes the tool's precision readable
    from the results instead of hidden in them.
    """

    ordered = sorted(
        boxes,
        key=lambda item: (
            (item["box"][2] - item["box"][0])
            * (item["box"][3] - item["box"][1])
        ),
    )

    kept_boxes = []
    tokens = set()

    for item in ordered:

        candidate = set(
            box_to_token_indices(
                item["box"],
                grid_h,
                grid_w,
                convention=item.get("convention", "norm1000"),
            )
        )

        union = tokens | candidate

        if max_tokens is not None and len(union) > max_tokens:
            continue

        tokens = union
        kept_boxes.append(item)

    return sorted(tokens), kept_boxes


# ============================================================
# The anchor
# ============================================================

class RegionAnchor:
    """Token indices to boost during decoding, and the boost itself.

    `tokens` are grid indices relative to the image window, so adding
    `vision_start` gives the absolute position the key dimension uses; keeping
    them relative means they stay valid at every decode step, because the cache
    keeps the prompt's positions fixed.
    """

    def __init__(
        self,
        tokens,
        vision_start,
        vision_end,
        grid_h,
        grid_w,
        multiply=1.5,
        neighborhood=1,
        head_ratio=0.6,
        layers=None,
    ):
        self.vision_start = vision_start
        self.vision_end = vision_end
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.multiply = multiply
        self.neighborhood = neighborhood
        self.head_ratio = head_ratio
        self.layers = None if layers is None else set(layers)

        self.tokens = list(tokens)
        self._gain_cache = {}
        self.steps = 0
        self.diagnostics = {}

        self.rebuild()

    @property
    def active(self):
        return len(self.keep) > 0 and self.multiply != 1.0

    def prepare_decode(self, module, attn_weights, query_length):
        """Hook for sources that have to read the prefill before decoding.

        A fixed region source has nothing to do here. The attention source
        overrides it, see `AttentionSourceAnchor`.
        """

        return None

    def rebuild(self):
        """Recompute the dilated set and drop the cached gain vector."""

        self.keep = self._neighborhood_indices()
        self._gain_cache = {}

    def _neighborhood_indices(self):
        """Focus tokens widened into a square of `neighborhood` cell radius.

        The dilation is done on the 2D grid rather than on the flat index as
        VRGA does it: neighbouring flat indices wrap across the row boundary, so
        a flat window grows into a strip that spans two rows at an arbitrary
        column, which is not a region a reader would recognise from the figure.

        The window guards mirror VRGA's: the first and last few image tokens are
        skipped, because their attention is dominated by the marker tokens that
        bracket the image rather than by image content.
        """

        width = self.vision_end - self.vision_start
        keep = set()

        for index in self.tokens:

            if not 3 <= index < width:
                continue

            row, col = divmod(index, self.grid_w)

            for row_offset in range(-self.neighborhood, self.neighborhood + 1):
                for col_offset in range(-self.neighborhood, self.neighborhood + 1):

                    neighbour_row = row + row_offset
                    neighbour_col = col + col_offset

                    if not 0 <= neighbour_row < self.grid_h:
                        continue

                    if not 0 <= neighbour_col < self.grid_w:
                        continue

                    neighbour = neighbour_row * self.grid_w + neighbour_col

                    if 3 <= neighbour < width:
                        keep.add(neighbour)

        return sorted(keep)

    def gain(self, width, device, dtype):
        """Multiplier per image token position, with the cache keyed per call."""

        key = (width, device, dtype)

        if key not in self._gain_cache:

            gain = torch.ones(width, device=device, dtype=dtype)

            if self.keep:
                gain[torch.tensor(self.keep, device=device)] = self.multiply

            self._gain_cache[key] = gain.view(1, 1, 1, -1)

        return self._gain_cache[key]


class AttentionSourceAnchor(RegionAnchor):
    """VRGA's own region source: the question token's attention over the image.

    This is the source the paper uses and the one this module replaces, ported
    so that the two can be compared inside one codebase, one prompt and one
    intervention. The selection follows the patched modelling file:

        per layer and head, take the row of the last question token, then
        EFR = image attention entropy / image attention ratio; heads below the
        5th percentile of EFR and above the 95th of the ratio are kept; the
        heads that barely look at the image are subtracted from their mean; the
        first row and column of the grid are excluded, because the marker
        tokens that bracket the image collect attention there and would
        otherwise be anchored every time; the tokens whose share exceeds
        mean + 0.6 sd are the region.

    Two deliberate differences from the original: the rows are read from the
    eager attention the patch already intercepts, so no attention matrices are
    materialised and memory does not grow with the prompt; and the region is
    the last question token before the assistant turn, rather than
    vision_end + len(tokenize(question)), which tokenises the text separately
    and lands one token short.
    """

    def __init__(
        self,
        vision_start,
        vision_end,
        grid_h,
        grid_w,
        question_end,
        percentile_efr=5.0,
        percentile_ratio=95.0,
        top_ratio=0.6,
        min_tokens=10,
        border_fraction=0.2,
        subtract_low_heads=True,
        **kwargs,
    ):
        self.question_end = question_end
        self.percentile_efr = percentile_efr
        self.percentile_ratio = percentile_ratio
        self.top_ratio = top_ratio
        self.min_tokens = min_tokens
        self.border_fraction = border_fraction
        self.subtract_low_heads = subtract_low_heads

        self.rows = []
        self.finalised = False
        self.diagnostics = {}

        super().__init__(
            tokens=[],
            vision_start=vision_start,
            vision_end=vision_end,
            grid_h=grid_h,
            grid_w=grid_w,
            **kwargs,
        )

    def prepare_decode(self, module, attn_weights, query_length):

        if query_length > 1:

            # One prefill call per layer; keep every head's row for the
            # question token.
            if self.question_end < attn_weights.shape[-1]:

                self.rows.append(
                    attn_weights[0, :, self.question_end, :].detach().float()
                )

            return None

        if not self.finalised:
            self.finalise()

        return None

    def _border_indices(self, width):

        rows = torch.arange(self.grid_h).view(-1, 1)
        cols = torch.arange(self.grid_w).view(1, -1)
        flat = rows * self.grid_w + cols

        candidates = torch.cat(
            [
                flat[:2, :2].flatten(),
                flat[0, 2:].flatten(),
                flat[2:, 0].flatten(),
            ]
        )

        limit = max(0, int(width * self.border_fraction))

        return candidates[:limit] if limit else candidates[:0]

    def finalise(self):

        self.finalised = True

        if not self.rows:
            return

        width = self.vision_end - self.vision_start

        stacked = torch.stack(self.rows)  # (layers, heads, keys)

        if width <= 0 or stacked.shape[-1] < self.vision_end:
            return

        image = stacked[:, :, self.vision_start:self.vision_end]

        image_sum = image.sum(-1, keepdim=True).clamp(min=1e-12)
        normalised = image / image_sum

        entropy = -(normalised * (normalised + 1e-12).log()).sum(-1)
        ratio = image.mean(-1) / stacked.mean(-1).clamp(min=1e-12)

        efr = entropy / (ratio + 1e-12)

        flat_efr = efr.reshape(-1)
        flat_ratio = ratio.reshape(-1)

        threshold_efr = torch.quantile(flat_efr, self.percentile_efr / 100.0)
        threshold_ratio = torch.quantile(flat_ratio, self.percentile_ratio / 100.0)

        selected = (flat_efr <= threshold_efr) & (flat_ratio >= threshold_ratio)

        if not bool(selected.any()):
            selected = flat_ratio >= threshold_ratio

        if not bool(selected.any()):
            return

        needs = image.reshape(-1, width)[selected].mean(0)

        low = (
            (flat_ratio < 0.01)
            & (
                torch.arange(flat_ratio.numel(), device=flat_ratio.device)
                < 64
            )
        )

        if self.subtract_low_heads and bool(low.any()):

            others = image.reshape(-1, width)[low].mean(0)

            needs = (
                needs / needs.sum().clamp(min=1e-12)
                - others / others.sum().clamp(min=1e-12)
            ).clamp(min=0)

        total = needs.sum().clamp(min=1e-12)
        needs = needs / total

        border = self._border_indices(width).to(device=needs.device)

        scores = needs.clone()
        valid = torch.ones(width, dtype=torch.bool, device=needs.device)
        valid[border] = False

        if not bool(valid.any()):
            return

        mean = scores[valid].mean()
        scores = scores / (mean + 1e-12)
        scores[border] = -1e9

        threshold = (
            scores[valid].mean() + scores[valid].std() * self.top_ratio
        )

        chosen = torch.where(scores > threshold)[0]

        if chosen.numel() < self.min_tokens:
            chosen = torch.topk(scores, k=min(self.min_tokens, width)).indices

        self.tokens = sorted(int(index) for index in chosen)
        self.rebuild()

        self.diagnostics = {
            "heads": int(flat_ratio.numel()),
            "selected_heads": int(selected.sum()),
            "low_heads": int(low.sum()),
            "border_excluded": int(border.numel()),
            "tokens": len(self.tokens),
        }


# ============================================================
# Attention patch
# ============================================================

# Candidate modules that hold the eager attention implementation Qwen VL uses
# as its default. Qwen3-VL is listed because the 30B checkpoint in the tables is
# a Qwen3 model; installing on a module that is not imported is a no-op.
_MODULE_PATHS = [
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
]

_ACTIVE = None
_ORIGINALS = {}
_CALLS = {"attention": 0, "eligible": 0, "modified": 0}


def _apply_anchor(module, attn_weights, query_length):

    anchor = _ACTIVE

    if anchor is None:
        return attn_weights

    # The vision tower uses the same eager implementation; it has no layer_idx.
    if not hasattr(module, "layer_idx"):
        return attn_weights

    # Sources that read the prefill collect their row here, before the early
    # return for non-decode steps.
    anchor.prepare_decode(module, attn_weights, query_length)

    if not anchor.active:
        return attn_weights

    # Only decode steps. In the prefill the query is the whole prompt and the
    # question's own tokens must stay free to look at the image.
    if query_length != 1:
        return attn_weights

    if anchor.layers is not None and module.layer_idx not in anchor.layers:
        return attn_weights

    vision_start, vision_end = anchor.vision_start, anchor.vision_end

    if vision_end > attn_weights.shape[-1] or vision_end <= vision_start:
        return attn_weights

    _CALLS["eligible"] += 1

    window = attn_weights[:, :, :, vision_start:vision_end]

    ratio = window.mean(dim=-1) / attn_weights.mean(dim=-1)

    selected = (ratio > anchor.head_ratio).unsqueeze(-1)

    if not bool(selected.any()):
        return attn_weights

    gain = anchor.gain(window.shape[-1], attn_weights.device, attn_weights.dtype)

    window = window * torch.where(selected, gain, torch.ones_like(gain))

    attn_weights = torch.cat(
        [
            attn_weights[:, :, :, :vision_start],
            window,
            attn_weights[:, :, :, vision_end:],
        ],
        dim=-1,
    )

    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

    anchor.steps += 1
    _CALLS["modified"] += 1

    return attn_weights


def _make_patched(original, repeat_kv):
    """The upstream eager attention, with the anchor applied before the values."""

    def patched(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):

        _CALLS["attention"] += 1

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        attn_weights = _apply_anchor(module, attn_weights, query.shape[2])

        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    patched.__name__ = getattr(original, "__name__", "patched_attention")

    return patched


def install(model=None):
    """Replace the eager attention default in every module that defines one.

    The intervention needs the softmaxed weights, which only the eager path
    exposes; with sdpa there is nothing to intercept and the patch would sit
    there doing nothing. The loaders set eager for Qwen2.5-VL, but the Qwen3-VL
    loader leaves the default, so the text config is checked and corrected here
    rather than left to produce a silent no-op on the 30B rows.
    """

    import importlib

    if model is not None:

        configs = [getattr(model, "config", None)]

        text_config = getattr(getattr(model, "config", None), "text_config", None)

        if text_config is not None:
            configs.append(text_config)

        for config in configs:

            if config is None:
                continue

            if getattr(config, "_attn_implementation", None) != "eager":

                print(
                    f"Anchoring needs eager attention; switching "
                    f"{type(config).__name__} from "
                    f"{getattr(config, '_attn_implementation', None)} to eager."
                )

                config._attn_implementation = "eager"

    installed = []

    for path in _MODULE_PATHS:

        try:
            module = importlib.import_module(path)
        except ImportError:
            continue

        original = getattr(module, "eager_attention_forward", None)
        repeat_kv = getattr(module, "repeat_kv", None)

        if original is None or repeat_kv is None:
            continue

        if path in _ORIGINALS:
            installed.append(path)
            continue

        _ORIGINALS[path] = module.eager_attention_forward

        module.eager_attention_forward = _make_patched(original, repeat_kv)

        installed.append(path)

    if not installed:
        raise RuntimeError(
            "No Qwen VL attention module was found to patch, so anchoring "
            "would silently do nothing."
        )

    return installed


def uninstall():
    """Put the original functions back."""

    import importlib

    for path, original in list(_ORIGINALS.items()):

        module = importlib.import_module(path)

        module.eager_attention_forward = original

        del _ORIGINALS[path]

    set_anchor(None)


def set_anchor(anchor):

    global _ACTIVE

    _ACTIVE = anchor


def claim_hook_fired(minimum=1):
    """Fail loudly if the patch was never in a position to apply the anchor.

    The counter that matters is `eligible`: it counts decode steps where the
    patch saw a non-empty anchor and the image window inside the key range, so
    it proves the positions are right. How many of those steps then pass the
    head ratio test is a property of the sample -- a one token answer has few
    steps and can legitimately modify nothing -- so it is recorded, not
    asserted.
    """

    if _CALLS["eligible"] < minimum:

        raise RuntimeError(
            f"The attention patch ran {_CALLS['attention']} times but was "
            f"never eligible to apply the anchor ({_CALLS['eligible']}). The "
            "region probably maps to no image token, or the intervention is "
            "aimed at the wrong attention implementation."
        )

    return _CALLS["modified"]


# ============================================================
# Generation
# ============================================================

def build_anchor(processor, model, inputs, boxes, max_token_fraction=0.25,
                 multiply=1.5, neighborhood=1, head_ratio=0.6, layers=None):
    """Turn boxes plus one sample's inputs into a RegionAnchor."""

    geometry = sample_geometry(processor, inputs)

    max_tokens = max(
        1, int(max_token_fraction * geometry["grid_h"] * geometry["grid_w"])
    )

    tokens, kept_boxes = boxes_to_token_indices(
        boxes, geometry["grid_h"], geometry["grid_w"], max_tokens=max_tokens
    )

    anchor = RegionAnchor(
        tokens=tokens,
        multiply=multiply,
        neighborhood=neighborhood,
        head_ratio=head_ratio,
        layers=layers,
        **geometry,
    )

    return anchor, kept_boxes


def sample_geometry(processor, inputs):
    """Where the image lives in the sequence, and the token grid it forms."""

    input_ids = inputs["input_ids"][0]

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    start_marker = (input_ids == vision_start_id).nonzero(as_tuple=True)[0][-1]
    end_marker = (input_ids == vision_end_id).nonzero(as_tuple=True)[0][-1]

    grid = inputs["image_grid_thw"][0]

    return {
        # Image tokens sit strictly between the two markers.
        "vision_start": int(start_marker) + 1,
        "vision_end": int(end_marker),
        "grid_h": int(grid[1]) // 2,
        "grid_w": int(grid[2]) // 2,
    }


def question_end_position(processor, inputs):
    """Index of the last question token.

    The chat template puts the question straight before the <|im_end|> that
    closes the user turn, so the last question token is the one before that
    marker. VRGA computes the same position as
    vision_end + len(tokenize(question)), which re-tokenises the question on
    its own and can disagree with the templated text by a token.

    The marker itself is deliberately not used: <|im_end|> and the assistant
    opener are the tokens whose job is to close the turn, and their attention
    row answers a different question than the question's own last token does.
    """

    input_ids = inputs["input_ids"][0]

    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    positions = (input_ids == end_id).nonzero(as_tuple=True)[0]

    if positions.numel() == 0:
        return int(input_ids.shape[0]) - 3

    return max(0, int(positions[-1]) - 1)


def generate_anchored(
    processor,
    model,
    sample,
    prompt_mode="direct",
    max_new_tokens=2000,
    region_source="native",
    max_token_fraction=0.25,
    multiply=1.5,
    neighborhood=1,
    head_ratio=0.6,
    layers=None,
    grounding_max_new_tokens=96,
    grounding_abstain=False,
):
    """Answer with the intervention active, anchored on located regions.

    The tool source spends a short grounding call first and fixes the regions
    before decoding; the attention source reads the prefill instead and adds no
    calls. Either way the patch is armed only for the answering pass.
    """

    question = sample["question"]
    image = sample["image"]

    boxes = []
    raw_boxes = ""
    grounding_tokens = 0

    if region_source == "native":

        boxes, raw_boxes, grounding_tokens = locate_regions(
            processor,
            model,
            image,
            question,
            max_new_tokens=grounding_max_new_tokens,
            abstain=grounding_abstain,
        )

    elif region_source != "attention":

        raise ValueError(f"Unsupported region source: {region_source!r}")

    prompt, _ = render_prompt(prompt_mode, question)

    content = [
        {"type": "image", "image": load_image(image), "max_pixels": MAX_PIXELS},
        {"type": "text", "text": prompt},
    ]

    inputs = inputs_from_content(processor, model, content)

    geometry = sample_geometry(processor, inputs)

    if region_source == "native":

        anchor, kept_boxes = build_anchor(
            processor,
            model,
            inputs,
            boxes,
            max_token_fraction=max_token_fraction,
            multiply=multiply,
            neighborhood=neighborhood,
            head_ratio=head_ratio,
            layers=layers,
        )

    else:

        kept_boxes = []
        anchor = AttentionSourceAnchor(
            question_end=question_end_position(processor, inputs),
            multiply=multiply,
            neighborhood=neighborhood,
            head_ratio=head_ratio,
            layers=layers,
            **geometry,
        )

    result = {
        "anchor_response": "",
        "anchor_region_source": region_source,
        "anchor_grounding_abstain": grounding_abstain,
        "anchor_boxes": kept_boxes,
        "anchor_grounding": raw_boxes,
        "anchor_grid": [geometry["grid_h"], geometry["grid_w"]],
        "anchor_token_count": 0,
        "anchor_token_fraction": 0.0,
        "anchor_dropped_boxes": len(boxes) - len(kept_boxes),
        "anchor_grounding_tokens": grounding_tokens,
    }

    output = None

    before = dict(_CALLS)

    try:

        set_anchor(anchor)

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # The anchor acts inside the attention function, so the run does
                # not need attention outputs; leaving them off is what keeps
                # this cheaper in memory than the attention-derived source.
                output_attentions=False,
                return_dict_in_generate=True,
            )

        result["anchor_response"] = get_output_text(processor, output, inputs)

        # The attention source only knows its region once the prefill has run,
        # so the counts are read after generation for both sources.
        width = max(1, geometry["vision_end"] - geometry["vision_start"])

        result["anchor_token_count"] = len(anchor.keep)
        result["anchor_token_fraction"] = round(len(anchor.keep) / width, 4)

        if anchor.diagnostics:
            result["anchor_selection"] = anchor.diagnostics

        # An empty anchor means every box was degenerate; that is a result to
        # report, not an error, so only a non-empty anchor has to prove it
        # fired. Both counters are deltas for this sample rather than running
        # totals, so one record describes one sample.
        result["anchor_modified_steps"] = (
            claim_hook_fired() - before["modified"] if anchor.active else 0
        )
        result["anchor_eligible_steps"] = _CALLS["eligible"] - before["eligible"]

    finally:

        set_anchor(None)

        if output is not None:
            del output

        del inputs

        torch.cuda.empty_cache()

    return result
