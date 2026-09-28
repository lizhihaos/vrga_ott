# qwen_eval2

Generation pipeline for Qwen2.5-VL / Qwen3-VL on the `EvalDataset` benchmarks.
Entry point: `scripts/run_qwen_eval2.py`.

## Prompt Modes

| Mode            | Passes | Prompt                                              | Output field        |
| --------------- | ------ | --------------------------------------------------- | ------------------- |
| `direct`        | 1      | question only, this is the direct answer baseline   | `direct_response`   |
| `cot`           | 1      | question + "Think step by step."                    | `cot_response`      |
| `region_guided` | 1      | question + focus on the question-related region     | `region_guided`     |
| `ccot`          | 2      | scene graph generation, then answer extraction      | `ccot_response`     |
| `icot`          | 1      | CoT with image patches injected mid-generation      | `icot_response`     |
| `anchor`        | 1      | direct prompt, attention anchored on tool boxes     | `anchor_response`   |
| `anchor_cot`    | 1      | CoT prompt, attention anchored on tool boxes        | `anchor_response`   |
| `vrg`           | 1      | CoT prompt, attention anchored on question attention | `anchor_response`  |

The output field is always `{mode}_response`, so the name to read a
generation is the same as the name used to produce it.

Modes can be combined, for example `--prompt_modes direct,cot,ccot`, in which
case every sample gets all requested generations in one run.

## CCoT

CCoT is Compositional Chain-of-Thought prompting
(Mitra et al., CVPR 2024, <https://github.com/chancharikmitra/CCoT>).
Instead of answering in one pass it answers in two:

```text
stage 1   image + question               -> scene graph (JSON)
stage 2   image + scene graph + question -> answer
```

The scene graph is a compact linguistic description of the objects,
attributes and relations that matter for the question, so stage 2 reasons
over it rather than re-deriving the visual evidence.

Implementation:

| File           | Role                                                            |
| -------------- | --------------------------------------------------------------- |
| `ccot.py`      | the two-stage pipeline, `generate_ccot()`                        |
| `prompts.py`   | scene graph and answer extraction prompts, copied from the paper |
| `generation.py`| `generate_from_prompt()`, one generation pass                    |

Both CCoT passes run back to back inside one call, so the evaluator's OOM
retry covers the whole pipeline: an OOM in either pass restarts both with a
smaller token budget.

### Scene graph token budget

Stage 1 is capped separately by `--ccot_sg_max_new_tokens` (default `256`,
the value used in the paper). Stage 2 uses the normal `--max_new_tokens`.

### Question style

`--ccot_question_style` controls what stage 1 sees:

| Style  | Behaviour                                                                                  |
| ------ | ------------------------------------------------------------------------------------------ |
| `full` | question exactly as the dataset stores it, options and answer-format hints included (default) |
| `stem` | only the text before the first `?`, the setup used by the paper's SEED-Bench script         |

## ICoT

ICoT is Interleaved-Modal Chain-of-Thought
(Gao et al., CVPR 2025, <https://github.com/jungao1106/ICoT>).
The rationale is decoded one token at a time; every `insert_every` newline
characters, the image region the model attends to most is re-injected as
soft visual tokens, so the remaining reasoning is conditioned on it:

```text
text step ... text step [patch tile] text step ... text step [patch tile]
```

The prompt is the plain CoT prompt, so `cot` and `icot` differ only by the
injection. Implementation: `icot.py`.

### Parameters

| Flag                       | Default | Meaning                                          |
| -------------------------- | ------- | ------------------------------------------------ |
| `--icot_num_patches`       | `16`    | patches per insertion, must be a perfect square  |
| `--icot_max_insertions`    | `3`     | maximum number of insertions per sample          |
| `--icot_insert_every`      | `2`     | newline characters between insertions            |
| `--icot_patch_selector`    | `tile`  | `tile` = most attended region, `topk` = reference |
| `--icot_vision_markers`    | off     | wrap patches in `<|vision_start|>` / `<|vision_end|>` |

### Differences from the reference implementation

The reference script for Qwen models is unfinished (its one-shot demo images
are `./path/to/icot_image_*.png` placeholders) and targets transformers 4.4x
with Qwen2-VL, so this is a port of its mechanism, not of its code. Three
points differ, each for a measured reason:

1. **No `<|vision_start|>` / `<|vision_end|>` around the patches.** With the
   markers, Qwen2.5-VL reads the injected block as the start of a new chat
   turn and collapses into `<|im_start|>` repetitions. Reproduce with
   `--icot_vision_markers`.
2. **`tile` selection instead of `topk`.** The reference takes the
   `num_patches` individually most attended patches. Those come out scattered
   over the first image rows, so the injected block claims adjacency between
   patches that are far apart; on the larger benchmark images the model then
   collapses into repeated digits ("5555555"). `tile` takes the contiguous
   region around the most attended patch, which is what the paper figures as
   well. Reproduce the old behaviour with `--icot_patch_selector topk`.
3. **Already used patches are excluded.** The reference clears the selected
   patches from its query image mask so every insertion points at a fresh
   region; the released Qwen script dropped that step, and without it all
   three insertions injected the same top-left tile in 122 of 125 samples.
4. **Greedy decoding with the checkpoint's repetition penalty**, like every
   other mode. The loop is verified to reproduce `model.generate()` token for
   token when the injection is disabled.

### Stability caveat

Injection is fragile on Qwen2.5-VL. Attention over the image is nearly flat
on these benchmarks (the peak patch gets ~0.002, a few times uniform), and
its argmax sits on the top-left patch, the usual attention sink, so the
selection carries little content signal. After an injection the model often
switches from reasoning to describing the injected region
("…-style, anime-style, 3d-rendering, high-quality, detailed…") and
sometimes degenerates.

Measured on a 137 sample HaloQuest pilot, counting responses whose longest
8-gram repeats three or more times:

| Run                            | Degenerate |
| ------------------------------ | ---------- |
| `cot` (control)                | 0.7 %      |
| `icot`, repeated-patch bug     | 22.5 %     |
| `icot`, after patch exclusion  | 8.3 % (12 samples) |

The paper's own setting demonstrates the interleaved format with a one-shot
example, whose images are exactly the crops injected during the rationale;
that demonstration is missing from the released Qwen script and is not
implemented here, which is the most likely reason the zero-shot port stays
unstable.

Before using ICoT numbers, run a pilot and check coherence:

```bash
/home/lizhihao/miniconda3/envs/tifa/bin/python scripts/run_qwen_eval2.py \
    --data_name haloquest --prompt_modes cot,icot --max_new_tokens 200
```

Each record carries `icot_insertions` (newline count, patch indices and
attention peak per insertion), which also shows whether successive
insertions picked fresh regions.

### One-shot demonstration

The paper demonstrates the interleaved format with a worked example before
asking the query, and its crops are the regions the method would inject. The
released repository leaves those crops as `./path/to/icot_image_*.png`
placeholders, so the demonstration is built mechanically here:

```bash
cd /home/lizhihao/phd/VRGA
/home/lizhihao/miniconda3/envs/tifa/bin/python tools/build_icot_demo.py \
    --data_name haloquest --out qwen_eval2/demos/haloquest --index 8 --device 0
```

The builder takes a sample from the dataset's **non-eval** split (haloquest
has 7609 held out samples, so the demonstration never shows an evaluation
image), runs the method on it, splits the rationale at the injection points
recorded as token indices, and cuts the injected tiles out of the image:

```text
<out>/demo.json     segments, crop boxes and the settings used
<out>/image.png     the demonstration image, resized as the model saw it
<out>/crop_<i>.png  the injected regions
<out>/preview.png   the image with the crop boxes drawn, for inspection
```

`demo.json` carries `segments_cleaned` and a note: the segment that follows
an injection can lose its sentence opening to the cut, so its leading
prompt-style fragment was replaced with the sentence opener. Everything else
is the model's own text.

The demonstration is placed in an **assistant turn**:

```text
user       demo image + demo question
assistant  interleaved rationale, crops inline
user       query image + query question
```

Putting the whole demonstration in the user turn instead makes the model read
it as user supplied text, and it then ends its own turn right after an
injection. See the measurements below.

Run with the demonstration:

```bash
/home/lizhihao/miniconda3/envs/tifa/bin/python scripts/run_qwen_eval2.py \
    --data_name haloquest --prompt_modes icot \
    --icot_demo qwen_eval2/demos/haloquest --max_new_tokens 200
```

Results go to a separate file, `..._icot_demo-haloquest_maxNew200.jsonl`, next
to the zero-shot `..._icot_maxNew200.jsonl`, so both settings can be scored
and compared without collisions.

### What the demonstration actually does on this model

Measured on HaloQuest, 200 tokens. The single turn placement was measured on
30 samples, the assistant turn and zero-shot on the same 12 samples, so read
the two blocks separately.

| Placement              | Samples | Median words | Stub (<25 words) | Under 15 words after last injection |
| ---------------------- | ------- | ------------ | ---------------- | ---------------------------------- |
| zero-shot              | 12      | 105          | 1/12             | 1/12                               |
| assistant turn         | 12      | 122          | 2/12             | 4/12                               |
| single user turn       | 30      | 46           | 6/30             | 19/30                              |

Single user turn, 30 samples, repeated n-grams: 3-gram 33.3 % -> 16.7 %,
8-gram 3.3 % -> 0.0 % against zero-shot, so it does suppress repetition, but
at the cost of 63 % of samples ending without an answer.

**On Qwen2.5-VL-3B the demonstration trades one failure mode for another and
does not clearly beat zero-shot.** The assistant turn placement is the
correct chat formulation and removes most of the early stopping, but its
repetition is then no better than zero-shot (3 vs 1 at 8-gram over 12
samples, i.e. within noise at that size). Score both settings with
evaluate_deepseek.py before choosing; the zero-shot file is the safer
default.

## Attention anchoring

VRGA boosts the attention weights of a set of image tokens during decoding.
This section covers where that set of tokens comes from, which is the whole
difference between the three anchoring modes:

| Mode         | Region source                    | Prompt   | Output field      |
| ------------ | -------------------------------- | -------- | ----------------- |
| `anchor`     | tool located boxes               | direct   | `anchor_response` |
| `anchor_cot` | tool located boxes               | CoT      | `anchor_response` |
| `vrg`        | question attention, i.e. VRGA's  | CoT      | `anchor_response` |

`direct` and `cot` are the controls for these rows: they use the same prompts
with the intervention off, so a table of `direct / anchor` isolates the anchor
and `cot / vrg` reproduces the paper's own source inside this codebase.

The intervention is identical in all three modes, and it is the one VRGA
validated: at every decode step (query length 1), for the heads whose mean
attention on the image tokens is above `--anchor_head_ratio` (0.6) times their
mean attention over all keys, the post-softmax weights of the anchored tokens
are multiplied by `--anchor_multiply` (1.5) and the row is renormalised. Anchors
are dilated by `--anchor_neighborhood` cells (1) and capped at
`--anchor_max_token_fraction` of the image tokens (0.25).

### The two region sources

**Tool (`anchor`, `anchor_cot`).** One short grounding call before decoding:
the question is asked in Qwen2.5-VL's grounding mode, the boxes it returns are
mapped to image token cells, and the anchor is fixed from then on. It costs
`--anchor_grounding_tokens` (96) tokens per sample at most, which is recorded
per sample as `anchor_grounding_tokens` because an extra call has to be
compared against spending the same call on sampling.

**Attention (`vrg`).** The source the paper uses, ported so the two can be
compared under one intervention: per layer and head, the last question token's
row over the image tokens, with heads selected by entropy over image attention
ratio (EFR) at the 5th percentile and image ratio at the 95th, the heads that
barely look at the image subtracted, the first row and column excluded because
the marker tokens that bracket the image collect attention there, and the cells
above mean + 0.6 sd anchored. Reading `<|im_end|>` or the assistant opener
instead of the question's own last token changes the region completely, so the
position is taken as the token before the `<|im_end|>` that closes the user
turn.

### Where the patch goes

`Qwen2_5_VLAttention.forward` resolves its default attention implementation
from the globals of `modeling_qwen2_5_vl` at call time, and transformers
registers no `eager` entry in `ALL_ATTENTION_FUNCTIONS`, so replacing that
global intercepts every text attention call. `anchoring.install(model)` does
that for Qwen2.5-VL and both Qwen3-VL modules, and nothing under
site-packages is modified. It also switches the text config to eager if a
loader left it at sdpa, because sdpa exposes no weights to intercept and the
patch would otherwise be a silent no-op. `claim_hook_fired()` raises if the
patch was never in a position to apply an anchor, so an inert patch cannot
produce a table of null results.

`output_attentions` stays off for anchored runs: the patch sees the weights
inside the attention call, so no attention matrices are materialised and memory
does not grow with the prompt. The intervention needs no training and no box
annotations.

### Region quality

`tools/anchor_region_quality.py` measures where each source points without
scoring anything, using the tool's box as the reference for the object. On 120
POPE samples with Qwen2.5-VL-3B:

| Source    | Touches the object | Mean coverage of it | Mean precision |
| --------- | ------------------ | ------------------- | -------------- |
| attention | 83 of 120          | 22 %                | 23 %           |
| tool      | by construction    | 100 %               | 100 %          |

So the source the paper replaces lands on the question's referent in about
seven cases in ten and misses it entirely in the rest; the tool source is on
the referent by construction, and its failure mode is different -- it will
return a box for an object that is not in the image, which is what POPE's
negative items measure. `tools/viz_anchor.py` draws both sources on the image
for a handful of samples, which is how the numbers above were checked.

## Usage

```bash
cd /home/lizhihao/phd/VRGA

/home/lizhihao/miniconda3/envs/tifa/bin/python scripts/run_qwen_eval2.py \
    --model_name Qwen2.5-VL-3B-Instruct \
    --data_name haloquest \
    --prompt_modes ccot \
    --max_new_tokens 2000 \
    --device 0
```

Run CCoT next to the baselines on the same samples:

```bash
/home/lizhihao/miniconda3/envs/tifa/bin/python scripts/run_qwen_eval2.py \
    --data_name haloquest \
    --prompt_modes direct,cot,ccot
```

The `tifa` environment is the one that loads these checkpoints
(transformers 5.15.0); with transformers 5.14.1 model loading fails with
`KeyError: 'default'` on the mrope config.

## Running bigger models

Memory is dominated by two things: the weights, and the vision tower, whose
eager attention materialises a `[heads, tokens, tokens]` fp32 matrix. A
1536x816 image becomes 1508 vision tokens, and that matrix is what pushes a
7B model over the edge of a 24GB card.

Measured on one RTX 4090, Qwen2.5-VL-7B, that image:

| Vision attention | Peak   | Free   | Result                          |
| ---------------- | ------ | ------ | ------------------------------- |
| `eager`          | 21.1GB | 0.7GB  | fits only barely, larger images OOM |
| `sdpa`           | 16.4GB | 6.4GB  | comfortable                     |

So the first lever is the vision tower's attention implementation, not a
second GPU:

```bash
python scripts/run_qwen_eval2.py --model_name Qwen2.5-VL-7B-Instruct \
    --data_name haloquest --prompt_modes cot,icot --vision_attn sdpa
```

The text model stays eager, because VRGA reads its attentions. `sdpa` changes
numerics slightly, so every mode you compare must use the same setting.

When the weights alone no longer fit, shard across every visible GPU:

```bash
python scripts/run_qwen_eval2.py --model_name Qwen2.5-VL-32B-Instruct \
    --data_name haloquest --prompt_modes cot,icot --device auto --vision_attn sdpa
```

Verified with 7B split over two cards (7.1GB + 8.4GB) for the single pass
modes and for the ICoT decode loop. Sharding is slower than a single card,
because activations cross PCIe between layers, so prefer `--vision_attn sdpa`
first. Lowering `max_pixels` also shrinks the vision activations, but it
changes the input resolution and therefore the comparison against the other
prompt modes.

## Output

Each mode writes its own file, so scoring, re-running one mode and the
resume state stay independent:

```text
rebuttal/{data_name}_{model_name}_{mode}_maxNew{max_new_tokens}.jsonl
```

A run with `--prompt_modes direct,cot,ccot,icot` produces four files. Modes
are ordered canonically in the name, so `direct,cot,icot,ccot` on the command
line writes to the same `..._cot_...` file rather than a second one.

One JSON record per sample. CCoT records carry two extra fields, the
intermediate scene graph and the settings used to produce it:

```json
{
    "id": 0,
    "question": "What type of shoes is the girl wearing?",
    "answer": "The shoes are not visible in the image",
    "type": "insufficient context",
    "ccot_scene_graph": "```json\n{\"objects\": [\"shoes\"], ...}\n```",
    "ccot_scene_graph_max_new_tokens": 256,
    "ccot_question_style": "full",
    "ccot_response": "The girl in the image is wearing white shoes.",
    "actual_max_new_tokens": {"ccot": 2000},
    "retry_count": {"ccot": 0}
}
```

ICoT records carry the injection log, one entry per insertion:

```json
{
    "id": 0,
    "icot_response": "To determine the type of shoes ...",
    "icot_insertions": [
        {
            "after_newlines": 2,
            "patches": [0, 1, 2, 3, 29, 30, 31, 32, 58, 59, 60, 61, 87, 88, 89, 90],
            "max_attention": 0.002166748046875
        }
    ],
    "icot_num_patches": 16,
    "icot_max_insertions": 3,
    "icot_insert_every": 2,
    "icot_patch_selector": "tile",
    "icot_vision_markers": false
}
```

Existing records are skipped on restart, so an interrupted run can be resumed
by re-running the same command.

## Scoring

`evaluate_deepseek.py` reads the response as `{mode}_response`, so pass the
same mode name used for generation:

| `--mode` | Field read        |
| -------- | ----------------- |
| `direct` | `direct_response` |
| `cot`    | `cot_response`    |
| `ccot`   | `ccot_response`   |
| `icot`   | `icot_response`   |

```bash
/home/lizhihao/miniconda3/envs/tifa/bin/python scripts/../evaluate_deepseek.py \
    --input rebuttal/haloquest_Qwen2.5-VL-3B-Instruct_ccot_maxNew2000.jsonl \
    --mode ccot
```
