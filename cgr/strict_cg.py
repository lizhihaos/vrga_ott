"""Strict grounded CCoT: verdicts that constrain the answer, not just describe it.

The first version of this pipeline grounds every element of the plan and then
hands the answer model two lists, one of what holds and one of what does not.
Measured on the first 19 HaloQuest samples, that is not enough, and the failures
are all of one kind: the verdict is computed, printed, and then ignored.

    id=8   plan writes book.title = samantha       (the answer, guessed)
           grounding refutes it                    (rejected = 1)
           answer still says "the author is Samantha"

    id=15  hat colour = black is refuted, answer says "the hat is black"
    id=17  answer asserts a banana that no element ever grounded
    id=18  orange is refuted, answer moves to "blue", which nothing supports

Four things follow from those, and this module is those four:

1. Three verdicts instead of two. NOT_SUPPORTED conflates "the image shows
   something else" with "the image cannot settle this", and the answer step
   reads the second as licence to guess. REFUTED and UNKNOWN are separated.

2. An observability gate at the plan. A property no image can settle (a book's
   author, a newspaper's title) is never sent to a tool, because a tool that
   cannot see it will still answer, and the answer then looks verified. It is
   UNKNOWN by construction. This is where the laundering in id=8 starts.

3. The answer must declare what it asserts, and every declared item has to
   trace to a SUPPORTED element. A claim that traces to REFUTED is a
   contradiction, one that traces to nothing is unsupported; either is a
   violation, repaired once, then answered conservatively. This is what stops
   id=15, id=17 and id=18.

4. A fallback that does not free-generate. When the plan cannot be parsed the
   first version falls back to a plain answer, and on id=7 that produced purple
   alien trees for a question whose truth is "there is no tree". The fallback
   here refuses instead, and the choice is recorded so the two can be told
   apart in the results.
"""

import json
import re

from .schema import extract_json

VERDICTS = ("SUPPORTED", "REFUTED", "UNKNOWN")

# Properties an image cannot settle. Sending them to a tool invites a guess:
# the tool is asked "is the author Samantha" and answers, and the guess comes
# back wearing a verdict. They are UNKNOWN without a call.
NON_VISUAL = {
    "author", "brand", "date", "identity", "language", "meaning", "name",
    "nationality", "occupation", "owner", "price", "purpose", "reason", "time",
    "title", "year", "age", "address", "phone", "model", "maker", "origin",
    "cost", "value", "intention", "relationship", "history",
}

REFUSAL = (
    "The image does not provide enough evidence to answer this question."
)

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
- Write an attribute value only if it is a property of the image: a colour,
  shape, material, count, position or clothing. Never write a value you would
  have to read, look up or guess, such as a name, title, author, brand or date;
  for those write "value": "unknown".
- A relation element connects two objects the question relates.
- Use the same object names everywhere, short lowercase nouns without articles.

Question: {question}

JSON:"""

ANSWER_PROMPT = """Question: {question}

The image was checked against the scene graph for this question.

Holds in the image:
{supported}

Contradicted by the image, the image shows something else here:
{refuted}

Not settled by the image, either not found or not a property an image can show:
{unknown}

Answer the question from what holds.
- Say plainly that the image does not show it when the answer would need
  anything from the other two lists.
- Do not mention a contradicted or unsettled value, not even to deny it.

Then list what your answer asserts, so it can be checked:
- "uses" holds every object, attribute value and relation the answer states.
- An answer that only declines to answer uses an empty list.

Return JSON only:
{{"answer": "<the answer the question expects>", "uses": ["<what it asserts>"]}}

Question: {question}

JSON:"""

REPAIR_PROMPT = """Question: {question}

An answer was written and then checked against what the image holds. These parts
of it are not supported by the image:

{violations}

What the image holds:
{supported}

Rewrite the answer so that everything it asserts is in that list, and say the
image does not show it for the rest. Keep the format the question expects.

Return JSON only:
{{"answer": "<the answer>", "uses": ["<what it asserts>"]}}

Question: {question}

JSON:"""


def normalise(text):

    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def is_visual(attribute):

    """Whether an image could settle this property at all."""

    attribute = normalise(attribute)

    if not attribute:
        return True

    return not any(word in NON_VISUAL for word in attribute.split())


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


class StrictGroundedCCoT:
    """Plan, ground every element into three verdicts, answer from evidence."""

    def __init__(self, tools, max_elements=6, answer_tokens=400,
                 fallback="refuse"):

        self.tools = tools
        self.max_elements = max_elements
        self.answer_tokens = answer_tokens
        self.fallback = fallback

    # ========================================================
    # Grounding
    # ========================================================

    def _locate(self, image, name, boxes):

        if name in boxes:
            return boxes[name]

        found = self.tools.locate(image, name)

        if found:
            boxes[name] = found[0]["box"]

        return boxes.get(name)

    def ground(self, image, plan):

        boxes = {}
        grounded = []

        for element in plan["elements"]:

            kind = element["kind"]

            # ------------------------------------------------
            # object: is it in the image at all
            #
            # A tool that finds nothing is not proof of absence: the detector
            # misses small and unusual things, so "no dog found" is UNKNOWN
            # rather than "the image contradicts a dog". REFUTED is kept for a
            # check that positively saw something else.
            # ------------------------------------------------

            if kind == "object":

                name = element["name"]
                found = self.tools.locate(image, name)

                if found:
                    boxes[name] = found[0]["box"]

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if found else "UNKNOWN",
                    "evidence": (
                        f"{len(found)} {name} found" if found
                        else f"no {name} found in the image"
                    ),
                    "boxes": len(found),
                })

                continue

            # ------------------------------------------------
            # attribute: does the located object hold it
            # ------------------------------------------------

            if kind == "attribute":

                subject = element["subject"]
                attribute = element["attribute"]
                value = element["value"]

                # The question asks for something no image can show. Asking a
                # tool anyway is how a guess acquires a verdict.
                if not is_visual(attribute) or not is_visual(value):

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": (
                            f"the {attribute} of the {subject} is not a "
                            f"property the image can show"
                        ),
                    })

                    continue

                if self._locate(image, subject, boxes) is None:

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": f"no {subject} located to check",
                    })

                    continue

                ok = self.tools.verify_attribute(
                    image, boxes[subject], subject,
                    attribute or "attribute",
                    value,
                )

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if ok else "REFUTED",
                    "evidence": (
                        f"the {subject} is {value}" if ok
                        else f"the {subject} is not {value}"
                    ),
                })

                continue

            # ------------------------------------------------
            # relation: is it true of the two located objects
            # ------------------------------------------------

            if kind == "relation":

                first, second = element["subject"], element["object"]

                if self._locate(image, first, boxes) is None or \
                        self._locate(image, second, boxes) is None:

                    missing = (
                        first if first not in boxes else second
                    )

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": f"no {missing} located to check",
                    })

                    continue

                ok = self.tools.relate(
                    image, boxes[first], first,
                    element["relation"], boxes[second], second,
                )

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if ok else "REFUTED",
                    "evidence": (
                        f"the {first} is {element['relation']} the {second}"
                        if ok else
                        f"the {first} is not {element['relation']} the {second}"
                    ),
                })

        return grounded, boxes

    # ========================================================
    # Answer constraint
    # ========================================================

    def _supported_terms(self, grounded):

        terms = []

        for item in grounded:

            if item["verdict"] != "SUPPORTED":
                continue

            terms.append(normalise(item.get("name", "")))
            terms.append(normalise(item.get("subject", "")))
            terms.append(normalise(item.get("object", "")))
            terms.append(normalise(item.get("value", "")))
            terms.append(normalise(item.get("attribute", "")))
            terms.append(normalise(item.get("relation", "")))
            terms.append(normalise(item.get("evidence", "")))

        return [term for term in terms if term]

    def _refuted_terms(self, grounded):

        terms = []

        for item in grounded:

            if item["verdict"] != "REFUTED":
                continue

            terms.append(normalise(item.get("value", "")))
            terms.append(normalise(item.get("evidence", "")))
            terms.append(normalise(item.get("relation", "")))

        return [term for term in terms if term]

    def check(self, uses, grounded):

        """Which of the things the answer asserts the image does not hold.

        A declared item is checked against the text of the SUPPORTED elements:
        evidence lines already carry the object, the attribute and the value, so
        a substring match in either direction is enough to trace it. Matching
        against REFUTED evidence is worse than not matching at all, because
        those lines read "the bottom stripe is not orange" and would otherwise
        license the value they deny.
        """

        supported = self._supported_terms(grounded)
        refuted = self._refuted_terms(grounded)

        violations = []

        for use in uses:

            term = normalise(use)

            if not term:
                continue

            if any(
                term in other or other in term
                for other in supported
            ):
                continue

            kind = "contradicted" if any(
                term in other or other in term
                for other in refuted
            ) else "unsupported"

            violations.append({"item": use, "kind": kind})

        return violations

    def _parse_answer(self, reply):

        try:
            data = extract_json(reply)
        except (ValueError, KeyError, TypeError):
            return None

        answer = str(data.get("answer", "")).strip()

        if not answer:
            return None

        uses = data.get("uses") or []

        if isinstance(uses, str):
            uses = [uses]

        return {
            "answer": answer,
            "uses": [str(use) for use in uses][:12],
            "raw": reply[:600],
        }

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

            return self._fallback(image, question, record, str(error))

        record["plan"] = plan
        trace.append({"tool": "plan", "elements": len(plan["elements"])})

        # ----------------------------------------------------
        # 2. Ground every element
        #
        # An empty plan means the question needs no visual structure. There is
        # nothing to check the answer against, so answering it here is the
        # unconstrained generation this pipeline exists to avoid.
        # ----------------------------------------------------

        if not plan["elements"]:

            return self._fallback(image, question, record, "empty plan")

        grounded, boxes = self.ground(image, plan)

        record["grounded"] = grounded
        record["boxes"] = boxes
        record["rejected"] = sum(
            1 for item in grounded if item["verdict"] != "SUPPORTED"
        )
        record["verdict_counts"] = {
            verdict: sum(1 for item in grounded if item["verdict"] == verdict)
            for verdict in VERDICTS
        }

        trace.append({
            "tool": "ground",
            "elements": len(grounded),
            "verdicts": record["verdict_counts"],
        })

        # ----------------------------------------------------
        # 3. Answer from the grounded graph, and check the answer
        # ----------------------------------------------------

        def lists():

            def lines(verdict):

                return [
                    f"- {item['evidence']} ({item['kind']})"
                    for item in grounded
                    if item["verdict"] == verdict
                ]

            return (
                "\n".join(lines("SUPPORTED")) or "- (none)",
                "\n".join(lines("REFUTED")) or "- (none)",
                "\n".join(lines("UNKNOWN")) or "- (none)",
            )

        supported, refuted, unknown = lists()

        prompt = (
            ANSWER_PROMPT
            .replace("{question}", question.strip())
            .replace("{supported}", supported)
            .replace("{refuted}", refuted)
            .replace("{unknown}", unknown)
        )

        parsed = self._parse_answer(
            self.tools.generate(image, prompt, max_new_tokens=self.answer_tokens)
        )

        record["tool_calls"] = self.tools.calls

        if parsed is None:

            return self._finish(record, REFUSAL, [], [], "unparsable answer")

        violations = self.check(parsed["uses"], grounded)

        record["uses"] = parsed["uses"]
        record["violations"] = violations

        if not violations:

            return self._finish(
                record, parsed["answer"], parsed["uses"], [], "clean"
            )

        # ----------------------------------------------------
        # 4. One repair, then refuse
        #
        # Repairing is a generation with the violations named, so it can fail
        # the same way. When it does, the conservative answer is a refusal, not
        # the unconstrained answer: a claim the evidence does not carry is the
        # failure this pipeline is meant to prevent.
        # ----------------------------------------------------

        violation_lines = "\n".join(
            f"- {item['item']} ({item['kind']})" for item in violations
        )

        repaired = self._parse_answer(
            self.tools.generate(
                image,
                (REPAIR_PROMPT
                 .replace("{question}", question.strip())
                 .replace("{violations}", violation_lines)
                 .replace("{supported}", supported)),
                max_new_tokens=self.answer_tokens,
            )
        )

        if repaired is not None:

            again = self.check(repaired["uses"], grounded)

            record["repaired_uses"] = repaired["uses"]
            record["repaired_violations"] = again

            if not again:

                return self._finish(
                    record, repaired["answer"], repaired["uses"],
                    violations, "repaired",
                )

        return self._finish(record, REFUSAL, [], violations, "refused")

    # ========================================================
    # Finishing
    # ========================================================

    def _finish(self, record, answer, uses, violations, outcome):

        record["answer"] = answer
        record["tool_calls"] = self.tools.calls
        record["outcome"] = outcome

        record["trace"].append({
            "tool": "answer",
            "outcome": outcome,
            "uses": uses,
            "violations": violations,
        })

        return record

    def _fallback(self, image, question, record, reason):

        record["fallback"] = True
        record["fallback_reason"] = reason
        record["grounded"] = []
        record["rejected"] = 0
        record["verdict_counts"] = {verdict: 0 for verdict in VERDICTS}
        record["tool_calls"] = self.tools.calls

        if self.fallback == "direct":

            record["answer"] = self.tools.ask_question(image, question)
            record["outcome"] = "fallback-direct"

        else:

            # An answer generated here has nothing to be checked against, and
            # on the samples where this path was taken with free generation the
            # result was a confident hallucination.
            record["answer"] = REFUSAL
            record["outcome"] = "fallback-refuse"

        record["trace"].append({"tool": "fallback", "reason": reason})

        return record
