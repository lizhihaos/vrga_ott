"""Grounded CCoT: a scene graph that gets checked against the image.

Compositional chain-of-thought writes a scene graph and then answers from it.
The graph is only ever described, never verified, so a hallucinated node
becomes the premise of the answer:

    question: What design does the cat's collar have?      (truth: no collar)
    graph:    {"cat": {"collar": {"design": "striped"}}}   <- collar invented
    answer:   "the cat's collar has a striped design"      <- follows the node

ReAct has the opposite problem: it can check anything, but it has to decide
what to check, and on a small model it wanders, repeating the same locate
until the step budget runs out.

This pipeline joins them. The scene graph is the plan, so the loop is
directed; every node and edge of that plan is then grounded with a tool, so a
node the image does not contain is dropped before it can become a premise:

    plan      question -> elements: objects, attributes, relations
    ground    every element -> locate / verify / relate, keeping the boxes
    answer    the supported elements only, and an explicit list of what the
              image does not show

The grounded graph is also the artefact worth showing: every node carries the
box it was found at, and every rejection carries the reason.
"""

import json
import re

from .schema import extract_json

PLAN_PROMPT = """You are given an image and a question about it.
Write the scene graph needed to answer the question: which objects are involved, which of their attributes the question asks about, and how they relate.

Return JSON only:
{"target": {"entity": "<what the question asks about>", "attribute": "<which property>"},
 "elements": [
   {"kind": "object", "name": "<object, short lowercase noun>"},
   {"kind": "attribute", "subject": "<object name>", "attribute": "<property>", "value": "<value>"},
   {"kind": "relation", "subject": "<object name>", "relation": "<relation>", "object": "<object name>"}
 ]}

Rules:
- Only elements the question actually needs, at most six.
- An object element is what has to be found in the image.
- An attribute element is a property the question asks about or states.
- A relation element connects two objects the question relates.
- Use the same object names everywhere, short lowercase nouns without articles.

Question: {question}

JSON:"""

ANSWER_PROMPT = """Question: {question}

The image was checked against the scene graph for this question.

Holds in the image:
{supported}

Does not hold, the image contradicts it or does not contain it:
{rejected}

Answer the question using only what holds.
- Do not mention or imply anything from the rejected list.
- If the answer needs something that does not hold, say plainly that the image does not show it.
- Keep the answer format the question expects.

Answer:"""


def parse_plan(reply):

    data = extract_json(reply)

    elements = []

    for item in data.get("elements") or []:

        if not isinstance(item, dict):
            continue

        kind = str(item.get("kind", "")).strip().lower()

        if kind == "object" and item.get("name"):
            elements.append({"kind": "object", "name": str(item["name"]).lower()})

        elif kind == "attribute" and item.get("subject"):
            elements.append({
                "kind": "attribute",
                "subject": str(item["subject"]).lower(),
                "attribute": str(item.get("attribute", "")).lower(),
                "value": str(item.get("value", "")).lower(),
            })

        elif kind == "relation" and item.get("subject") and item.get("object"):
            elements.append({
                "kind": "relation",
                "subject": str(item["subject"]).lower(),
                "relation": str(item.get("relation", "")).lower(),
                "object": str(item["object"]).lower(),
            })

    target = data.get("target") or {}

    return {
        "target": {
            "entity": str(target.get("entity", "")).lower(),
            "attribute": str(target.get("attribute", "")).lower(),
        },
        "elements": elements[:6],
        "raw": reply[:600],
    }


class GroundedCCoT:
    """Plan, ground every element, answer from what survived."""

    def __init__(self, tools, max_elements=6, answer_tokens=400):

        self.tools = tools
        self.max_elements = max_elements
        self.answer_tokens = answer_tokens

    # ========================================================
    # Grounding
    # ========================================================

    def ground(self, image, plan):

        boxes = {}
        grounded = []

        for element in plan["elements"]:

            kind = element["kind"]

            # ------------------------------------------------
            # object: is it in the image at all
            # ------------------------------------------------

            if kind == "object":

                name = element["name"]
                found = self.tools.locate(image, name)

                if found:
                    boxes[name] = found[0]["box"]

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if found else "NOT_SUPPORTED",
                    "evidence": (
                        f"{len(found)} {name} found" if found
                        else f"no {name} in the image"
                    ),
                    "boxes": len(found),
                })

                continue

            # ------------------------------------------------
            # attribute: does the located object hold it
            # ------------------------------------------------

            if kind == "attribute":

                subject = element["subject"]

                if subject not in boxes:
                    found = self.tools.locate(image, subject)
                    if found:
                        boxes[subject] = found[0]["box"]

                if subject not in boxes:

                    grounded.append({
                        **element,
                        "verdict": "NOT_SUPPORTED",
                        "evidence": f"no {subject} in the image",
                    })

                    continue

                ok = self.tools.verify_attribute(
                    image, boxes[subject], subject,
                    element["attribute"] or "attribute",
                    element["value"],
                )

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if ok else "NOT_SUPPORTED",
                    "evidence": (
                        f"the {subject} is {element['value']}" if ok
                        else f"the {subject} is not {element['value']}"
                    ),
                })

                continue

            # ------------------------------------------------
            # relation: is it true of the two located objects
            # ------------------------------------------------

            if kind == "relation":

                first, second = element["subject"], element["object"]

                for name in (first, second):
                    if name not in boxes:
                        found = self.tools.locate(image, name)
                        if found:
                            boxes[name] = found[0]["box"]

                if first not in boxes or second not in boxes:

                    missing = first if first not in boxes else second

                    grounded.append({
                        **element,
                        "verdict": "NOT_SUPPORTED",
                        "evidence": f"no {missing} in the image",
                    })

                    continue

                ok = self.tools.relate(
                    image, boxes[first], first,
                    element["relation"], boxes[second], second,
                )

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if ok else "NOT_SUPPORTED",
                    "evidence": (
                        f"the {first} is {element['relation']} the {second}" if ok
                        else f"the {first} is not {element['relation']} the {second}"
                    ),
                })

        return grounded, boxes

    # ========================================================
    # Entry point
    # ========================================================

    def solve(self, image, question):

        record = {"trace": [], "fallback": False}
        trace = record["trace"]

        # ----------------------------------------------------
        # 1. Plan
        # ----------------------------------------------------

        try:

            reply = self.tools.generate(
                image,
                PLAN_PROMPT.replace("{question}", question.strip()),
                max_new_tokens=512,
            )

            plan = parse_plan(reply)

        except (ValueError, KeyError, TypeError) as error:

            trace.append({"tool": "plan", "error": str(error)})

            record["fallback"] = True
            record["answer"] = self.tools.ask_question(image, question)
            record["tool_calls"] = self.tools.calls

            return record

        record["plan"] = plan
        trace.append({"tool": "plan", "elements": len(plan["elements"])})

        # ----------------------------------------------------
        # 2. Ground every element
        #
        # An empty plan means the question needs no visual structure (an
        # abstract or definitional question): answer it directly rather than
        # forcing the method onto it.
        # ----------------------------------------------------

        if not plan["elements"]:

            record["fallback"] = True
            record["answer"] = self.tools.ask_question(image, question)
            record["grounded"] = []
            record["tool_calls"] = self.tools.calls

            trace.append({"tool": "ground", "skipped": "empty plan"})

            return record

        grounded, boxes = self.ground(image, plan)

        record["grounded"] = grounded
        record["boxes"] = boxes
        record["rejected"] = sum(
            1 for item in grounded if item["verdict"] == "NOT_SUPPORTED"
        )

        trace.append({
            "tool": "ground",
            "elements": len(grounded),
            "rejected": record["rejected"],
        })

        # ----------------------------------------------------
        # 3. Answer from the grounded graph
        # ----------------------------------------------------

        supported = [
            f"- {item['evidence']} ({item['kind']})"
            for item in grounded
            if item["verdict"] == "SUPPORTED"
        ]

        rejected = [
            f"- {item['evidence']} ({item['kind']})"
            for item in grounded
            if item["verdict"] == "NOT_SUPPORTED"
        ]

        answer = self.tools.generate(
            image,
            (ANSWER_PROMPT
             .replace("{question}", question.strip())
             .replace("{supported}", "\n".join(supported) or "- (none)")
             .replace("{rejected}", "\n".join(rejected) or "- (none)")),
            max_new_tokens=self.answer_tokens,
        )

        record["answer"] = answer
        record["tool_calls"] = self.tools.calls

        trace.append({
            "tool": "answer",
            "supported": len(supported),
            "rejected": len(rejected),
            "answer": answer,
        })

        return record
