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
        type=int,
        default=0,
        help="CUDA device index",
    )
    parser.add_argument(
        "--modify",
        type=str,
        default="modify_att",
        help="Modification type: modify_att or base",
    )
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
    parser.add_argument(
        "--with_cot",
        action="store_true",
        help="Generate both ORI and CoT responses",
    )
    parser.add_argument(
        "--prompt_modes",
        type=str,
        default=None,
        help=(
            "Comma-separated prompt modes to generate. "
            "Choices: direct,cot,ccot,icot. "
            "Example: --prompt_modes direct,cot,ccot,icot"
        ),
    )
    return parser.parse_args()
