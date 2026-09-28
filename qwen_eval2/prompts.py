# PROMPTS = {

#     # ============================================================
#     # Direct / ORI
#     # ============================================================
#     "direct": """
# Question:
# {question}

# Answer:
# """,

#     # ============================================================
#     # Chain-of-Thought / THINK
#     # ============================================================
#     "cot": """
# Think step by step before giving the final answer.

# Question:
# {question}

# Reasoning:
# """,

#     # ============================================================
#     # CCoT
#     # ============================================================
#     "ccot": """
# Analyze the image and answer the question.

# First, identify the relevant visual evidence in the image.
# Then, use this evidence to reason about the answer.

# Your reasoning should be concise and grounded in the image.
# Only mention visual information that is relevant to the question.
# Do not introduce information that cannot be supported by the image.

# Question:
# {question}

# Visual Evidence and Reasoning:
# """,

#     # ============================================================
#     # ICoT
#     # ============================================================
#     "icot": """
# Analyze the image and answer the question using interleaved visual reasoning.

# For each important inference, explicitly connect the reasoning
# to the corresponding visual evidence in the image.

# Follow this structure:

# Observation:
# Describe the relevant visual evidence.

# Reasoning:
# Explain how this observation supports the answer.

# Verification:
# Check whether the reasoning and answer are consistent with
# the visual evidence.

# Final Answer:
# Provide the final answer.

# Question:
# {question}
# """
# }


# # ============================================================
# # Backward Compatibility
# # ============================================================

# MODE_ALIASES = {

#     "ori": "direct",
#     "direct": "direct",
#     "think": "cot",
#     "cot": "cot",
#     "ccot": "ccot",
#     "icot": "icot",
# }


# # ============================================================
# # Prompt Renderer
# # ============================================================

# def render_prompt(mode, question):

#     mode = MODE_ALIASES.get(mode, mode)

#     if mode not in PROMPTS:
#         raise ValueError(
#             f"Unknown prompt mode: {mode}. "
#             f"Available modes: {list(PROMPTS.keys())}"
#         )

#     prompt = PROMPTS[mode].format(
#         question=question
#     )

#     return prompt, mode
PROMPTS = {
    # Direct：直接提问
    "direct": "{question}",

    # Reason：逐步推理
    "cot": "{question} Think step by step.",

    # Region-Guided：关注相关区域并逐步推理
    "region_guided": (
        "{question} Focus on the region related to the question "
        "and Think step by step."
    ),
}

MODE_ALIASES = {
    "direct": "direct",
    "cot": "cot",
    "reason": "cot",
    "region": "region_guided",
    "region-guided": "region_guided",
    "region_guided": "region_guided",
}


def render_prompt(mode: str, question: str) -> tuple[str, str]:
    mode = mode.strip().lower()
    mode = MODE_ALIASES.get(mode, mode)

    if mode not in PROMPTS:
        raise ValueError(
            f"Unknown prompt mode: {mode!r}. "
            f"Available modes: {list(PROMPTS.keys())}"
        )

    prompt = PROMPTS[mode].format(question=question.strip())
    return prompt, mode


# ============================================================
# CCoT (Compositional Chain-of-Thought)
#
# Mitra et al., "Compositional Chain-of-Thought Prompting for
# Large Multimodal Models", CVPR 2024.
# Reference implementation: https://github.com/chancharikmitra/CCoT
#
# CCoT answers in two passes instead of one:
#
#   Stage 1, scene graph generation
#       image + question -> scene graph (JSON)
#
#   Stage 2, answer extraction
#       image + scene graph + question -> answer
#
# Both prompt texts are copied verbatim from the reference
# implementation so that the numbers stay comparable to the paper.
# ============================================================

CCOT_SCENE_GRAPH_PROMPT = """
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Scene Graph:
"""

CCOT_ANSWER_PROMPT = (
    "Use the image and scene graph as context "
    "and answer the following question: "
)

# "full" keeps the question exactly as the dataset stores it, including
# answer-format instructions and options. This matches the reference
# MMBench / Whoops / Winoground scripts.
# "stem" keeps only the question stem, matching the reference SEED-Bench
# script, which feeds `question.split("?")[0]` to the scene graph stage.
CCOT_QUESTION_STYLES = ("full", "stem")


def render_ccot_scene_graph_prompt(
    question: str,
    question_style: str = "full",
) -> str:
    """CCoT stage 1: prompt for a task-relevant scene graph."""

    if question_style not in CCOT_QUESTION_STYLES:
        raise ValueError(
            f"Unknown CCoT question style: {question_style!r}. "
            f"Available styles: {list(CCOT_QUESTION_STYLES)}"
        )

    question = question.strip()

    if question_style == "stem":
        question = question.split("?")[0].strip() + "?"

    return f"{question}\n{CCOT_SCENE_GRAPH_PROMPT}"


def render_ccot_answer_prompt(question: str, scene_graph: str) -> str:
    """CCoT stage 2: answer with the scene graph as additional context."""

    return (
        f"Scene Graph: {scene_graph.strip()}\n\n"
        f"{CCOT_ANSWER_PROMPT}{question.strip()}"
    )