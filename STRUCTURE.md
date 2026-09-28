# Repository layout

## Evaluation pipeline (`qwen_eval2/`)

Prompt-mode evaluation on the `EvalDataset` benchmarks. One folder per dataset,
one folder per model, one file per mode:

```text
rebuttal/<dataset>/<model>/<mode>_maxNew<N>.jsonl
```

| Mode | What it does |
| ---- | ------------ |
| `direct` | question only |
| `cot` | question + "Think step by step." |
| `ccot` | Compositional CoT: scene graph, then answer from it |
| `icot` | Compositional CoT with image patches injected during decoding |
| `anchor` / `anchor_cot` | VRGA's attention boost, anchored on tool located boxes |
| `vrg` | the same boost, anchored on the question attention VRGA uses |

Runs are resumable, greedy, and record the token budget they used. See
`qwen_eval2/README.md` for the modes, the ICoT deviations, the one-shot
demonstration and the anchoring sources.

## Attention anchoring (`qwen_eval2/anchoring.py`)

VRGA's intervention with the region source replaced. The paper reads the region
out of the question token's attention, which is produced by the same reasoning
the paper says drifts; the anchoring modes fix the region before decoding
starts, from a tool call or from the prefill, and keep it fixed.

```text
locate_regions     one grounding call: question -> boxes
box_to_token_indices  box -> image token cells, on the 2x2 merged grid
RegionAnchor       fixed regions from those boxes
AttentionSourceAnchor  the paper's own source, ported for the comparison
install            replaces the eager attention default, no site-packages edit
```

Needs eager text attention, needs no training and no box annotations, and
stores the raw grounding text with every result so a wrong coordinate
convention is visible rather than silently anchored.

## Methods (`cgr/`)

Constraint-guided visual re-grounding: the model's reasoning is checked
against the image with tools instead of being trusted.

```text
tools.py         LOCATE / VERIFY_ATTRIBUTE / RELATE / INSPECT
                 grounding: GroundingDINO or the VLM's own boxes
                 relations: geometry for spatial, VLM for semantic
grounded_cg.py   cg        plan a scene graph, ground every element, answer
                          from what the image supports   <- current method
react.py         react     thought / action / observation loop, image observations
select.py        select    sample N answers, keep the best supported
reground.py      claims    draft, verify, revise
controller.py    constraint question -> constraint graph -> candidate search
```

`cg`, `react` and `select` share the tool layer, so all three report the same
budget numbers and the same per-element verdicts.

## Entry points (`scripts/`)

| Script | Purpose |
| ------ | ------- |
| `run_qwen_eval2.py` | the four prompt modes |
| `run_cgr.py` | the five method pipelines (`--pipeline cg|react|select|claims|constraint`) |
| `run_r1_eval.py` | R1-style reasoning models, one prompt, their own answer format |
| `run_sc_eval.py` | self-consistency baseline at a matched budget |
| `score_parallel.py` | parallel DeepSeek judging, resumable |
| `build_table.py` | accuracy tables per dataset |

All of them work from any directory; each adds the repository root to
`sys.path`.

## Tools (`tools/`)

| Script | Purpose |
| ------ | ------- |
| `build_icot_demo.py` | builds the one-shot ICoT demonstration from a held out sample |
| `viz_grounding.py` | draws the boxes a record produced, plus the crops the verifier saw |
| `viz_anchor.py` | draws the anchored cells and the box they came from |
| `anchor_region_quality.py` | measures where each region source points, no judge needed |
| `anchor_leverage.py` | how far the boost moves the next-token distribution |
| `anchor_strength.py` | answer changes and accuracy against the boost's size |
| `check_degeneration.py` | repetition, near-empty and post-injection truncation rates |
| `check_model_dir.py` | verifies an uploaded checkpoint before spending GPU hours |
| `report_metrics.py` | per-subset accuracy, refusal rate and anchoring cost |

## Orchestration (`runs/`)

| Script | Purpose |
| ------ | ------- |
| `run_all_datasets.sh` | every dataset, every mode, then scoring and tables |
| `run_mmstar_compare.sh` | the current comparison on the MMStar perception subset |
| `run_mmstar_full.sh` | the four prompt modes on the whole 1500 sample MMStar |
| `run_pope_anchor.sh` | POPE pilot: controls, both region sources, all three anchor rows |
| `run_anchor_boost.sh` | both region sources at 50x, where the boost moves the output |

## Results

Written under `rebuttal/` and ignored by git; regenerate with the pipeline.

## What the measurements say so far

The intervention's size is the binding constraint, before the region source is
even in question. `tools/anchor_leverage.py`, 8 POPE samples, Qwen2.5-VL-3B,
boost at 50x the paper's strength:

| Quantity | Value |
| -------- | ----- |
| decode steps eligible to apply the anchor | 598 per sample |
| steps that applied it | 126 per sample |
| mean max change in a logit | 0.83 |
| mean total variation between the distributions | 0.022 |
| samples whose argmax flipped within 24 steps | 3 of 8 |

At the paper's 1.5x the change is a fraction of this, and POPE's answer tokens
carry probability 0.96 to 0.9998. A boost that shifts a logit by ~0.03 cannot
move a token that certain, so a short-answer benchmark is flat under any region
source; `tools/anchor_strength.py` confirms it, changing no answer at 50x on 20
samples. Whatever the anchoring rows show at 1.5 is the prompt, not the anchor.
`runs/run_anchor_boost.sh` re-runs both sources at 50x, where the intervention
does move the reasoning.

Where each region source points, 120 POPE samples, Qwen2.5-VL-3B, measured
against the tool's own box and checked by eye with `tools/viz_anchor.py`:

| Source | Touches the object | Coverage of it | Precision |
| ------ | ------------------ | -------------- | --------- |
| question attention (`vrg`) | 83 of 120 | 22 % | 23 % |
| tool boxes (`anchor`) | by construction | 100 % | 100 % |

The source the paper replaces misses the question's referent entirely in about
three cases in ten. The tool source is on it by construction, and its own
failure mode is the opposite one: asked plainly it returns a box for every one
of the 120 samples, including all 60 whose object is absent by construction, so
a negative item gets anchored on a hallucinated region. Asked to abstain
(`--anchor_grounding_abstain`) it answers with an empty list on 18 of 20 absent
objects and on only 1 of 20 present ones, which is what makes POPE's negative
half a test rather than a foregone conclusion.

| Grounding prompt | Abstains on absent | Abstains on present |
| ---------------- | ------------------ | ------------------- |
| plain | 0 of 20 | 0 of 20 |
| abstain | 18 of 20 | 1 of 20 |

That is the verifier precision the literature reports least often, and here it
turns a one-line prompt change into a usable gate: an empty anchor is recorded
and leaves the intervention inert for that sample instead of aiming it at a
region the question was about and the image does not contain.

On HaloQuest (600 samples, one decoding setting, no dropped samples):

| Model | direct | cot | ccot | icot |
| ----- | ------ | --- | ---- | ---- |
| Qwen2.5-VL-3B | 55.7 | 37.2 | 43.8 | 34.5 |
| Qwen2.5-VL-7B | 73.8 | 62.2 | 46.2 | 45.7 |
| R1-OneVision-7B | - | 60.0 | - | - |

Every reasoning variant scores below direct, CCoT and ICoT the worst of them.

Two caveats from the same work:

- HaloQuest is saturated by refusal. 58 % of its ground truths are negations,
  and a constant "the image does not show it" scores 43.6 %, the wording "what
  the question asks about" 64.4 %. A method that refuses scores well there for
  the wrong reason, which is why the method comparison moved to MMStar's
  perception subset, where refusing scores zero.
- Grounding quality is the binding constraint for anything built on it. A
  suitcase resting on a brochure was verified as "not on the brochure" while
  the ground truth said it was, which is what the geometric `on`/`under` rules
  in `tools.py` now handle.
