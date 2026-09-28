import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL VQA Evaluation: ORI vs CoT"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-VL-3B-Instruct",
        help="Model name, e.g. Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help=(
            "CUDA device index, or 'auto' to shard the model across every "
            "visible GPU (needed once the weights alone exceed one card)"
        ),
    )
    # parser.add_argument(
    #     "--modify",
    #     type=str,
    #     default="modify_att",
    #     help="Modification type: modify_att or base",
    # )
    parser.add_argument(
        "--data_name",
        type=str,
        default="POPE",
        help="Dataset name: HallusionBench, POPE, mmstar, haloquest, etc.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2000,
        help="Maximum number of generated tokens",
    )
    # parser.add_argument(
    #     "--with_cot",
    #     action="store_true",
    #     help="Generate both ORI and CoT responses",
    # )
    parser.add_argument(
        "--vision_attn",
        type=str,
        default=None,
        choices=["eager", "sdpa"],
        help=(
            "Attention implementation for the vision tower only. 'sdpa' "
            "avoids materialising the token x token attention matrix and "
            "saves about 4.7GB on Qwen2.5-VL-7B, which is what makes that "
            "model fit on a 24GB card. It changes numerics slightly, so use "
            "one setting across every mode you compare."
        ),
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help=(
            "Load the weights in 4-bit nf4 with bitsandbytes. Required for "
            "30B+ checkpoints, whose bf16 weights exceed this machine's two "
            "cards; it changes numerics, so do not mix it across the rows of "
            "one table without saying so."
        ),
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help=(
            "Only samples whose type contains one of these, comma "
            "separated. Use the same filter across modes so the "
            "comparison stays on one sample set."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process the first N samples, for pilots. Saved samples "
            "are skipped as usual, so re-running without --limit continues "
            "where the pilot stopped."
        ),
    )
    parser.add_argument(
        "--prompt_modes",
        type=str,
        default=None,
        help=(
            "Comma-separated prompt modes to generate. "
            "Choices: direct,cot,region_guided,ccot,anchor,anchor_cot. "
            "Example: --prompt_modes direct,cot,ccot"
        ),
    )
    parser.add_argument(
        "--anchor_multiply",
        type=float,
        default=1.5,
        help=(
            "Anchoring: factor applied to the attention weights of the tool "
            "located patches. 1.0 turns the intervention off, which makes the "
            "run a prompt-only control."
        ),
    )
    parser.add_argument(
        "--anchor_neighborhood",
        type=int,
        default=1,
        help=(
            "Anchoring: patches dilated around each located patch, matching "
            "the VRGA window so the two region sources are comparable"
        ),
    )
    parser.add_argument(
        "--anchor_head_ratio",
        type=float,
        default=0.6,
        help=(
            "Anchoring: a head is boosted only if its mean attention on the "
            "image tokens exceeds this multiple of its mean attention over all "
            "keys. VRGA uses 0.6."
        ),
    )
    parser.add_argument(
        "--anchor_max_token_fraction",
        type=float,
        default=0.25,
        help=(
            "Anchoring: cap on the fraction of image tokens one anchor may "
            "cover. Boxes are dropped largest first until the union fits, so a "
            "degenerate whole-image box cannot anchor everything."
        ),
    )
    parser.add_argument(
        "--anchor_grounding_tokens",
        type=int,
        default=96,
        help=(
            "Anchoring: token budget for the grounding call that produces the "
            "boxes. Recorded per sample, because that call is the cost the "
            "method has to justify against sampling."
        ),
    )
    parser.add_argument(
        "--ccot_question_style",
        type=str,
        default="full",
        choices=["full", "stem"],
        help=(
            "Question passed to the CCoT scene graph stage. "
            "'full' keeps answer-format instructions and choices, "
            "'stem' keeps only the text before the first '?' "
            "(the reference SEED-Bench setup)."
        ),
    )
    parser.add_argument(
        "--ccot_sg_max_new_tokens",
        type=int,
        default=256,
        help=(
            "Maximum number of tokens for the CCoT scene graph stage"
        ),
    )
    parser.add_argument(
        "--icot_num_patches",
        type=int,
        default=16,
        help=(
            "Number of image patches injected per ICoT insertion, "
            "must be a perfect square (16 gives the 4x4 grid used by the "
            "reference implementation)"
        ),
    )
    parser.add_argument(
        "--icot_max_insertions",
        type=int,
        default=3,
        help="Maximum number of ICoT patch insertions per sample",
    )
    parser.add_argument(
        "--icot_insert_every",
        type=int,
        default=2,
        help=(
            "Insert patches every N newline characters of the rationale"
        ),
    )
    parser.add_argument(
        "--icot_patch_selector",
        type=str,
        default="tile",
        choices=["tile", "topk"],
        help=(
            "How ICoT picks patches to inject. 'tile' takes the most "
            "attended side x side region, 'topk' takes the individually most "
            "attended patches as in the reference code, which degenerates on "
            "large images."
        ),
    )
    parser.add_argument(
        "--icot_demo",
        type=str,
        default=None,
        help=(
            "Directory of a one-shot interleaved demonstration built by "
            "build_icot_demo.py, e.g. qwen_eval2/demos/haloquest. Without it "
            "ICoT runs zero-shot, which is not the setting the paper "
            "evaluates."
        ),
    )
    parser.add_argument(
        "--icot_vision_markers",
        action="store_true",
        help=(
            "Wrap injected patches in <|vision_start|>/<|vision_end|> as the "
            "reference implementation does. Qwen2.5-VL reads that block as a "
            "new chat turn and the output degenerates, so this is off by "
            "default and only kept for reproduction."
        ),
    )
    return parser.parse_args()
