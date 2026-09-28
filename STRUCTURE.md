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

Runs are resumable, greedy, and record the token budget they used. See
`qwen_eval2/README.md` for the modes, the ICoT deviations and the one-shot
demonstration.

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
| `check_degeneration.py` | repetition, near-empty and post-injection truncation rates |
| `check_model_dir.py` | verifies an uploaded checkpoint before spending GPU hours |

## Orchestration (`runs/`)

| Script | Purpose |
| ------ | ------- |
| `run_all_datasets.sh` | every dataset, every mode, then scoring and tables |
| `run_mmstar_compare.sh` | the current comparison on the MMStar perception subset |

## Results

Written under `rebuttal/` and ignored by git; regenerate with the pipeline.

## What the measurements say so far

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
