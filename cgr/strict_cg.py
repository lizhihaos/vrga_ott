"""Strict grounded CCoT: check what the question asks, then answer from evidence.

The first version grounds the plan and then hands the answer model two lists.
Measured on the first 19 HaloQuest samples, the verdicts are computed, printed,
and then ignored, and one structural reason sits behind most of the failures:
the plan is allowed to contain the answer.

    id=8   plan writes book.title = samantha, grounding refutes it, and the
           answer still says Samantha
    id=15  hat colour = black is refuted, the answer says the hat is black
    id=17  the answer asserts a banana no element ever grounded
    id=18  the plan writes bottom stripe = orange, grounding refutes orange,
           and the answer moves to blue, which nothing supports

A plan that carries a value turns the verifier into a check of the model's own
guess: refuting orange licenses nothing, and the hole that opens is what the
answer model fills. So the value is removed from the plan entirely, and the
grounding call is changed from "is it orange" to "what colour is it". The plan
says what has to be read off the image; the image supplies the value.

Six changes, in the order they matter:

1. No value in the plan. An attribute element names a property to read
   ("bottom stripe color"), never a candidate for it.

2. describe_attribute instead of verify_attribute. "Is the bottom stripe
   orange" can only refute the plan's guess; "what colour is the bottom
   stripe" reads the value from the image, and a refusal is UNKNOWN.

3. Three verdicts. NOT_SUPPORTED conflated "the image shows something else"
   with "the image cannot settle this", and the answer step read the second as
   licence to guess.

4. Multi-instance entities. Three cats is a set, and keeping only the first
   box silently verified every later attribute on one cat. Instances are kept
   as a set, attributes are read across them, and relations are tried over the
   pairs.

5. Identity separate from position. "left man" is not an entity name: it is the
   entity "man" plus an ordering, and querying the locator with "left man"
   returned two boxes for both the left and the right one. The selector is
   stripped before the query and applied to the instances geometrically.

6. An answer that cannot look at the image. Evidence is the only input, so a
   value that is not in the evidence graph has nowhere to come from, and a
   declared assertion that traces to nothing is caught before it is recorded.

The fallback follows the same logic: the plan is retried, then reduced to the
question's own object, and only then refused. Free generation on a failed plan
produced purple alien trees for a question whose truth is that there is no tree.
"""

import json
import re

from .schema import extract_json

VERDICTS = ("SUPPORTED", "REFUTED", "UNKNOWN")

# Properties an image cannot settle. Asking about them invites a guess: the
# tool cannot see an author but will still answer, and the answer comes back
# wearing a verdict. They are UNKNOWN without a call.
NON_VISUAL = {
    "author", "brand", "date", "identity", "language", "meaning", "name",
    "nationality", "occupation", "owner", "price", "purpose", "reason", "time",
    "title", "year", "age", "address", "phone", "model", "maker", "origin",
    "cost", "value", "intention", "relationship", "history",
}

# Ordering words that read as part of an entity name but are really an
# attribute of the set: "left man", "smaller animal". Stripped before the
# locator is queried and applied to the instances afterwards.
SELECTORS = (
    "left", "right", "top", "bottom", "upper", "lower", "middle", "center",
    "smaller", "larger", "bigger", "taller", "shorter", "first", "second",
    "third", "front", "back", "nearest", "closest",
)

POSITION_WORDS = {
    "position", "side", "location", "place", "left", "right", "top", "bottom",
    "middle", "center", "centre", "front", "back", "leftmost", "rightmost",
}

# Words that describe where an attribute sits inside the question rather than
# what it is, and are ignored when deciding whether it is a position.
POSITION_FILLER = {"of", "the", "its", "a", "an", "subject", "object"}

# The minimal plan has no model-written attribute, so the question itself is
# read for one. Only the properties it can name unambiguously are listed: the
# point of the floor is to check something, not to guess.
ATTRIBUTE_HINTS = (
    ("how many", "count"),
    ("what color", "color"),
    ("what colour", "color"),
    ("color of", "color"),
    ("colour of", "color"),
    ("what type", "type"),
    ("what kind", "type"),
    ("what material", "material"),
    ("what shape", "shape"),
    ("what design", "design"),
    ("what is the name", "text"),
    ("who is", "identity"),
    ("where is", "position"),
    ("what is written", "text"),
)

REFUSAL = (
    "The image does not provide enough evidence to answer this question."
)

PLAN_PROMPT = """You are given an image and a question about it.
Write what has to be checked in the image in order to answer the question. Do not answer it.

Return JSON only:
{"target": {"entity": "<what the question asks about>", "attribute": "<which property>"},
 "elements": [
   {"kind": "object", "name": "<object, short lowercase noun>"},
   {"kind": "attribute", "subject": "<object name>", "attribute": "<property to read>"},
   {"kind": "relation", "subject": "<object name>", "relation": "<relation>", "object": "<object name>"}
 ]}

Rules:
- At most six elements, only what the question needs.
- An object element is what has to be found in the image.
- An attribute element names a property to READ OFF the image, such as color,
  material, count, shape or position. Never write the value: the value is the
  thing being asked for, and a value written here would be checked against the
  image instead of read from it.
- Write an attribute of a part as the property of the whole: for "the bottom
  stripe of the rainbow", subject "rainbow", attribute "bottom stripe color".
- One object appearing several times keeps one name for the type; do not put
  left, right, first or smaller into the name.
- Write only properties an image could show. Never ask for a name, title,
  author, brand or date.
- Use the same object names everywhere, short lowercase nouns without articles.

Question: {question}

JSON:"""

PLAN_RETRY_PROMPT = """Your previous reply could not be read as JSON.

Return JSON only, no prose and no code fence:
{"target": {"entity": "<what the question asks about>", "attribute": "<which property>"},
 "elements": [{"kind": "object", "name": "<object>"}]}

Question: {question}

JSON:"""

TARGET_PROMPT = """Question: {question}

Which object does this question ask about? Answer with one short lowercase noun
only, such as "cat" or "book". If the question asks about no object at all,
answer exactly: NONE

Answer:"""

ANSWER_PROMPT = """Question: {question}

The image was checked against the scene graph for this question. This is all
that is known:

Holds in the image:
{supported}

Contradicted by the image, the image shows something else where this was asked:
{refuted}

Not settled by the image, either not found or not a property an image can show:
{unknown}

Answer the question using only the first list.
- You cannot see the image. Anything not written above is unknown to you.
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
of it are not supported:

{violations}

Everything known about the image:
{supported}

Rewrite the answer so that everything it asserts is in that list, and say the
image does not show it for the rest. You cannot see the image.

Return JSON only:
{{"answer": "<the answer>", "uses": ["<what it asserts>"]}}

Question: {question}

JSON:"""


def normalise(text):

    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def is_positional(attribute):

    """Whether the attribute is a position rather than a readable property.

    A word match is not enough: "bottom" appears in "bottom stripe color",
    which is a colour to read off the image, and routing it here skipped the
    read entirely. Only an attribute made of position words is one.
    """

    words = [
        word for word in normalise(attribute).split()
        if word not in POSITION_FILLER
    ]

    return bool(words) and all(word in POSITION_WORDS for word in words)


def attribute_from_question(question):

    """The property a question asks for, when it names one outright."""

    lowered = normalise(question)

    for phrase, attribute in ATTRIBUTE_HINTS:

        if phrase in lowered:
            return attribute

    return ""


def is_visual(attribute):

    """Whether an image could settle this property at all."""

    attribute = normalise(attribute)

    if not attribute:
        return True

    return not any(word in NON_VISUAL for word in attribute.split())


def split_selector(name):

    """("man", "left") out of "left man"; (name, None) when there is no order.

    The order is a property of the set, not part of the type: querying a
    locator for "left man" asks it to find the phrase, and it answers with both
    men, which is how id=12 ended up with two boxes for "left man" and two for
    "right man".
    """

    words = normalise(name).split()

    found = [word for word in words if word in SELECTORS]

    remaining = [word for word in words if word not in SELECTORS]

    if not found or not remaining:
        return name, None

    return " ".join(remaining), found[0]


def select_instance(instances, selector):

    """The instance an ordering word picks out, geometrically.

    `instances` are the locator's own records, each with a "box" key, because
    that is what every caller holds.
    """

    if not instances:
        return []

    if selector is None:
        return instances

    def centre(item):
        box = item["box"]
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def area(item):
        box = item["box"]
        return abs((box[2] - box[0]) * (box[3] - box[1]))

    choosers = {
        "left": lambda: min(instances, key=lambda item: centre(item)[0]),
        "right": lambda: max(instances, key=lambda item: centre(item)[0]),
        "top": lambda: min(instances, key=lambda item: centre(item)[1]),
        "upper": lambda: min(instances, key=lambda item: centre(item)[1]),
        "bottom": lambda: max(instances, key=lambda item: centre(item)[1]),
        "lower": lambda: max(instances, key=lambda item: centre(item)[1]),
        "smaller": lambda: min(instances, key=area),
        "shorter": lambda: min(instances, key=area),
        "larger": lambda: max(instances, key=area),
        "bigger": lambda: max(instances, key=area),
        "taller": lambda: max(instances, key=area),
        "first": lambda: instances[0],
        "second": lambda: instances[1] if len(instances) > 1 else instances[0],
        "third": lambda: instances[2] if len(instances) > 2 else instances[-1],
        "middle": lambda: instances[len(instances) // 2],
        "center": lambda: instances[len(instances) // 2],
        "centre": lambda: instances[len(instances) // 2],
    }

    for word in (selector, "middle", "center"):
        chooser = choosers.get(word)
        if chooser is not None:
            return [chooser()]

    return instances


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
    """Plan what to read, ground it, and answer from the evidence alone."""

    def __init__(self, tools, max_elements=6, answer_tokens=400,
                 fallback="refuse"):

        self.tools = tools
        self.max_elements = max_elements
        self.answer_tokens = answer_tokens
        self.fallback = fallback

    # ========================================================
    # Instances
    # ========================================================

    def instances(self, image, name, cache):

        """Every box for an entity, and never one box for a set of them."""

        if name in cache:
            return cache[name]

        query, selector = split_selector(name)

        found = self.tools.locate(image, query)

        cache[name] = select_instance(found, selector)

        return cache[name]

    # ========================================================
    # Grounding
    # ========================================================

    def ground(self, image, plan):

        cache = {}
        grounded = []

        for element in plan["elements"]:

            kind = element["kind"]

            # ------------------------------------------------
            # object: is it in the image at all
            #
            # A tool that finds nothing is not proof of absence, so that is
            # UNKNOWN rather than "the image contradicts a cat". REFUTED is
            # kept for a check that positively saw something else.
            # ------------------------------------------------

            if kind == "object":

                name = element["name"]
                found = self.instances(image, name, cache)

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
            # attribute: read the property off the image
            # ------------------------------------------------

            if kind == "attribute":

                subject = element["subject"]
                attribute = element["attribute"]

                if not is_visual(attribute):

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": (
                            f"the {attribute} of the {subject} is not a "
                            f"property the image can show"
                        ),
                    })

                    continue

                found = self.instances(image, subject, cache)

                if not found:

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": f"no {subject} found to read the {attribute} from",
                    })

                    continue

                # ------------------------------------------------
                # Position is not read by a model, it is computed from the
                # boxes: asking a VLM which of two similar instances is the
                # left one is how "left man" and "right man" got the same box.
                # ------------------------------------------------

                if is_positional(attribute):

                    read = self._read_position(image, subject, found)

                else:

                    read = self.tools.describe_attribute(
                        image, [box["box"] for box in found], subject, attribute
                    )

                if read.get("value") is None:

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": read.get("evidence", f"no {attribute} read"),
                    })

                    continue

                grounded.append({
                    **element,
                    "value": read["value"],
                    "verdict": "SUPPORTED",
                    "evidence": read["evidence"],
                    "boxes": len(found),
                })

                continue

            # ------------------------------------------------
            # relation: true of the instances
            #
            # Both sides existing does not make the edge true, so the edge is
            # checked, over every pair when a side has several instances: any
            # pair holding is enough to support it, and only all pairs failing
            # refutes it.
            # ------------------------------------------------

            if kind == "relation":

                first, second = element["subject"], element["object"]

                first_boxes = self.instances(image, first, cache)[:3]
                second_boxes = self.instances(image, second, cache)[:3]

                if not first_boxes or not second_boxes:

                    missing = first if not first_boxes else second

                    grounded.append({
                        **element,
                        "verdict": "UNKNOWN",
                        "evidence": f"no {missing} found to check the relation",
                    })

                    continue

                verdicts = []

                for box_a in first_boxes:
                    for box_b in second_boxes:

                        box_a, box_b = box_a["box"], box_b["box"]

                        verdicts.append(
                            self.tools.relate(
                                image, box_a, first,
                                element["relation"], box_b, second,
                            )
                        )

                held = any(verdicts)

                grounded.append({
                    **element,
                    "verdict": "SUPPORTED" if held else "REFUTED",
                    "evidence": (
                        f"the {first} is {element['relation']} the {second}"
                        if held else
                        f"the {first} is not {element['relation']} the {second}"
                    ),
                    "pairs": len(verdicts),
                })

        return grounded, cache

    def _read_position(self, image, subject, found):

        """The ordering of the instances, from their boxes.

        Position is not read by a model: asking a VLM which of two similar
        instances is the left one is how "left man" and "right man" came back
        with the same box. Which instance the question means is decided by the
        selector in its name, see `split_selector`; what is recorded here is
        the fact that the set has an order at all, so the answer can say
        "the left one" about a set the evidence describes.
        """

        if len(found) < 2:

            return {
                "value": None,
                "evidence": (
                    f"only one {subject} in the image, so it has no position "
                    f"relative to another"
                ),
            }

        ordered = sorted(
            found,
            key=lambda item: (item["box"][0] + item["box"][2]) / 2,
        )

        labels = []

        for rank in range(len(ordered)):

            if rank == 0:
                labels.append("leftmost")
            elif rank == len(ordered) - 1:
                labels.append("rightmost")
            else:
                labels.append(f"middle {rank}")

        value = "left to right: " + ", ".join(labels)

        return {
            "value": value,
            "evidence": (
                f"the image holds {len(found)} {subject}, ordered left to "
                f"right as {', '.join(labels)}"
            ),
        }

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

        A declared item is traced against the text of the SUPPORTED elements,
        which already carry the object, the attribute and the value read from
        the image. Matching against REFUTED text is worse than not matching at
        all, because those lines read "the bottom stripe is not orange" and
        would otherwise license the value they deny.
        """

        supported = self._supported_terms(grounded)
        refuted = self._refuted_terms(grounded)

        violations = []

        for use in uses:

            term = normalise(use)

            if not term:
                continue

            if any(term in other or other in term for other in supported):
                continue

            kind = "contradicted" if any(
                term in other or other in term for other in refuted
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

        return {"answer": answer, "uses": [str(use) for use in uses][:12]}

    # ========================================================
    # Plan, with a retry and a floor under it
    # ========================================================

    def _plan(self, image, question, trace):

        for name, prompt in (
            ("plan", PLAN_PROMPT),
            ("plan-retry", PLAN_RETRY_PROMPT),
        ):

            try:

                reply = self.tools.generate(
                    image,
                    prompt.replace("{question}", question.strip()),
                    max_new_tokens=512,
                )

                plan = parse_plan(reply)

            except (ValueError, KeyError, TypeError) as error:

                trace.append({"tool": name, "error": str(error)})
                continue

            trace.append({"tool": name, "elements": len(plan["elements"])})

            if plan["elements"]:
                return plan

        # ----------------------------------------------------
        # The floor: the question's own object, nothing guessed
        # ----------------------------------------------------

        try:

            noun = self.tools.generate(
                image,
                TARGET_PROMPT.replace("{question}", question.strip()),
                max_new_tokens=12,
            ).strip().lower()

        except (ValueError, KeyError, TypeError):

            return None

        noun = re.sub(r"[^a-z ]+", "", noun).strip()

        if not noun or noun in ("none", "no object"):

            return None

        attribute = attribute_from_question(question)

        elements = [{"kind": "object", "name": noun}]

        # Without this the floor can only check that the object exists, and a
        # question about its colour is answered "the image does not provide
        # enough evidence" even though the colour could have been read.
        if attribute and is_visual(attribute):

            elements.append({
                "kind": "attribute",
                "subject": noun,
                "attribute": attribute,
            })

        trace.append({
            "tool": "plan-minimal",
            "entity": noun,
            "attribute": attribute,
        })

        return {
            "target": {"entity": noun, "attribute": attribute},
            "elements": elements,
            "raw": "minimal plan",
        }

    # ========================================================
    # Entry point
    # ========================================================

    def solve(self, image, question):

        record = {"trace": [], "fallback": False}
        trace = record["trace"]

        plan = self._plan(image, question, trace)

        if plan is None:

            return self._fallback(image, question, record, "no plan")

        record["plan"] = plan

        grounded, cache = self.ground(image, plan)

        record["grounded"] = grounded
        record["boxes"] = {
            name: [box["box"] for box in boxes]
            for name, boxes in cache.items()
        }
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
        # Answer from the evidence, without the image
        # ----------------------------------------------------

        def lines(verdict):

            return [
                f"- {item['evidence']} ({item['kind']})"
                for item in grounded
                if item["verdict"] == verdict
            ]

        supported = "\n".join(lines("SUPPORTED")) or "- (none)"
        refuted = "\n".join(lines("REFUTED")) or "- (none)"
        unknown = "\n".join(lines("UNKNOWN")) or "- (none)"

        prompt = (
            ANSWER_PROMPT
            .replace("{question}", question.strip())
            .replace("{supported}", supported)
            .replace("{refuted}", refuted)
            .replace("{unknown}", unknown)
        )

        # No image: an answer that cannot see is an answer that cannot invent.
        parsed = self._parse_answer(
            self.tools.generate(None, prompt, max_new_tokens=self.answer_tokens)
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

        violation_lines = "\n".join(
            f"- {item['item']} ({item['kind']})" for item in violations
        )

        repaired = self._parse_answer(
            self.tools.generate(
                None,
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
        record["boxes"] = {}
        record["rejected"] = 0
        record["verdict_counts"] = {verdict: 0 for verdict in VERDICTS}
        record["tool_calls"] = self.tools.calls

        if self.fallback == "direct":

            record["answer"] = self.tools.ask_question(image, question)
            record["outcome"] = "fallback-direct"

        else:

            # An answer generated here has nothing to be checked against, and
            # on the samples where this path used free generation the result
            # was a confident hallucination.
            record["answer"] = REFUSAL
            record["outcome"] = "fallback-refuse"

        record["trace"].append({"tool": "fallback", "reason": reason})

        return record
