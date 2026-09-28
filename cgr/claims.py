"""Draft answer to atomic visual claims.

The constraint-graph pipeline needs the question to name visual entities,
which most benchmark questions do not: MMBench asks about colonies and
endotherms, HallusionBench about figure semantics. The claim pipeline takes
the entity names from somewhere else, namely from the model's own draft, so
the same verification tools apply to any question type.

A claim is one checkable statement about the image:

    "The cat is wearing a red collar" ->
        subject="cat", kind="attribute", predicate="collar", value="red"
    "There is no snowboard" ->
        subject="snowboard", kind="existence", value="absent"
"""

import json
import re

CLAIM_PROMPT = """You are given an image, a question about it, and a draft answer.
List the statements the draft answer makes about the image that must be true for the answer to be correct.

Return JSON only, a list, no prose:
[{"subject": "<object the statement is about, short lowercase noun>",
  "kind": "existence" | "attribute" | "relation" | "count",
  "predicate": "<attribute or relation name, empty for existence and count>",
  "object": "<the other object, only for relations>",
  "value": "<claimed value, number, or present/absent>",
  "text": "<the sentence it came from>"}]

Rules:
- Check what the ANSWER depends on, not what merely appears in the image.
- When the answer chooses an option, state what that option claims about the image.
- Relations and attributes are the useful claims; plain presence is almost never
  enough to justify an answer, so use "existence" only when the answer is about
  presence or absence. For a statement like "the suitcase is on the book" emit a
  relation claim with subject "suitcase", object "book", predicate "on".
- List statements the answer asserts to be true; do not list the ones it rejects.
- At most {max_claims} entries, the ones the answer most depends on.

Question:
{question}

Draft answer:
{draft}

JSON list:"""


def parse_claims(generate, draft, image=None, max_claims=8, question=""):
    """Ask the model to turn its draft into a list of claims."""

    # The image goes along: without it the model reads "checkable" as
    # "checkable by me right now" and returns an empty list.
    reply = generate(
        CLAIM_PROMPT
        .replace("{max_claims}", str(max_claims))
        .replace("{question}", (question or "(not given)").strip())
        .replace("{draft}", draft.strip()),
        image,
    )

    claims = extract_json_list(reply)

    out = []

    for claim in claims:

        if not isinstance(claim, dict):
            continue

        subject = str(claim.get("subject", "")).strip().lower()

        if not subject:
            continue

        out.append({
            "subject": subject,
            "kind": str(claim.get("kind", "existence")).strip().lower(),
            "predicate": str(claim.get("predicate", "")).strip().lower(),
            "object": str(claim.get("object", "")).strip().lower(),
            "value": str(claim.get("value", "")).strip().lower(),
            "text": str(claim.get("text", "")).strip(),
        })

    return out[:max_claims]


def extract_json_list(text):
    """First JSON array in a model reply, fences and prose tolerated."""

    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)

    if fence:
        text = fence.group(1).strip()

    start = text.find("[")

    if start == -1:
        return []

    depth = 0

    for index in range(start, len(text)):

        if text[index] == "[":
            depth += 1

        elif text[index] == "]":

            depth -= 1

            if depth == 0:

                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return []

    return []
