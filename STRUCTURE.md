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

## Results

Written under `rebuttal/` and ignored by git; regenerate with the pipeline.

## What the measurements say so far

Where each region source points, 120 POPE samples, Qwen2.5-VL-3B, measured
against the tool's own box and checked by eye with `tools/viz_anchor.py`:

| Source | Touches the object | Coverage of it | Precision |
| ------ | ------------------ | -------------- | --------- |
| question attention (`vrg`) | 83 of 120 | 22 % | 23 % |
| tool boxes (`anchor`) | by construction | 100 % | 100 % |

The source the paper replaces misses the question's referent entirely in about
three cases in ten. The tool source is on it by construction, and its own
failure mode is the opposite one: it returns a box for an object that is not in
the image, which is what the negative half of POPE measures.

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
