"""CCoT: Compositional Chain-of-Thought prompting.

Mitra et al., "Compositional Chain-of-Thought Prompting for Large
Multimodal Models", CVPR 2024.
Reference implementation: https://github.com/chancharikmitra/CCoT

Each sample is answered with two passes over the same image:

    stage 1   image + question               -> scene graph (JSON)
    stage 2   image + scene graph + question -> answer

The scene graph is a compact linguistic description of the parts of the
image that matter for the question, so stage 2 reasons over it instead
of re-deriving the visual evidence from scratch.

Both passes are plain generations, so OOM handling stays with the
caller: if either pass raises, the evaluator retries the whole pipeline
with a smaller token budget.
"""

from .generation import generate_from_prompt
from .prompts import (
    render_ccot_answer_prompt,
    render_ccot_scene_graph_prompt,
)


def generate_ccot(
    processor,
    model,
    sample,
    max_new_tokens,
    scene_graph_max_new_tokens=256,
    question_style="full",
):
    """Run both CCoT passes on one sample.

    Returns a dict whose keys are already the field names used by the
    evaluator, so its result record can be built with a plain update().
    """

    question = sample["question"]
    image = sample["image"]

    # ------------------------------------------------------------
    # Stage 1: scene graph generation
    # ------------------------------------------------------------

    scene_graph_prompt = render_ccot_scene_graph_prompt(
        question,
        question_style=question_style,
    )

    scene_graph = generate_from_prompt(
        processor,
        model,
        image,
        scene_graph_prompt,
        scene_graph_max_new_tokens,
    )

    print(
        f"[CCoT] Scene graph ready "
        f"({len(scene_graph)} chars)."
    )

    # ------------------------------------------------------------
    # Stage 2: answer extraction
    # ------------------------------------------------------------

    answer_prompt = render_ccot_answer_prompt(
        question,
        scene_graph,
    )

    response = generate_from_prompt(
        processor,
        model,
        image,
        answer_prompt,
        max_new_tokens,
    )

    return {
        "ccot_scene_graph": scene_graph,
        "ccot_response": response,
    }
