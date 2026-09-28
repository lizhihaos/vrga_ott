
PROMPTS = {


# ============================================================
# Direct
# ============================================================

"direct": """
Answer the question based on the image.

Question:
{question}

Answer:
""",



# ============================================================
# Chain-of-Thought (CoT)
# ============================================================

"cot": """
Think step by step before giving the final answer.
Question:
{question}

Reasoning:
""",



# ============================================================
# Concise Chain-of-Thought (CCoT)
# ============================================================

"ccot": """
Analyze the image and answer the question.

Provide only concise reasoning containing the key visual evidence
needed to support the answer.

Avoid unnecessary descriptions.
Do not introduce information that is not visible in the image.

Question:
{question}

Reasoning:
""",



# ============================================================
# Interleaved-Modal Chain-of-Thought (ICoT)
# ============================================================

"icot": """
Analyze the image and answer the question using interleaved visual reasoning.

During reasoning, connect each inference with the corresponding
visual evidence from the image.

Follow this format:

Observation:
Describe the relevant visual evidence.

Reasoning:
Explain how the observation supports the answer.

Verification:
Check whether the answer is consistent with the image.

Final Answer:
Provide the final answer.

Question:
{question}

"""
}
# ============================================================
# Backward Compatibility
# ============================================================

MODE_ALIASES = {

    # old name -> new name
    "ori": "direct",

    "think": "cot",

}


# ============================================================
# Prompt Renderer
# ============================================================

def render_prompt(
    mode,
    question
):

    # compatible with old code
    mode = MODE_ALIASES.get(
        mode,
        mode
    )


    if mode not in PROMPTS:

        raise ValueError(
            f"Unknown prompt mode: {mode}. "
            f"Available modes: {list(PROMPTS.keys())}"
        )


    prompt = PROMPTS[mode].format(
        question=question
    )


    return prompt, mode