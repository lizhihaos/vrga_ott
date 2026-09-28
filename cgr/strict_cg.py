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

# Words that carry no fact of their own. Everything else in an answer has to be
# traceable to a supported element, which is what stops an answer from
# introducing the value the evidence does not contain.
GENERIC = {
    "the", "and", "are", "but", "can", "cannot", "could", "does", "for",
    "from", "has", "have", "image", "into", "its", "not", "only", "other",
    "show", "shows", "than", "that", "their", "them", "there", "these",
    "they", "this", "those", "visible", "was", "were", "which", "while",
    "with", "without", "would", "yes", "answer", "question", "appears",
    "appear", "seems", "seem", "because", "about", "above", "across",
    "after", "also", "any", "both", "each", "either", "enough", "even",
    "every", "here", "however", "itself", "more", "most", "much", "neither",
    "never", "none", "nor", "nothing", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "some", "such", "then",
    "therefore", "unclear", "unknown", "where", "whether", "yet", "given",
    "provide", "provided", "provides", "determine", "determined", "evidence",
    "insufficient", "enough", "cannot", "unable", "possible", "possibly",
}

REFUSAL = (
    "The image does not provide enough evidence to answer this question."
)

# A refusal that names what it is refusing about scores the way the answers do.
# HaloQuest's ground truth for a false premise is "there is no ball in the
# image", and the generic sentence was judged wrong against it on three of the
# nineteen samples whose old answers scored right.
REFUSAL_TEMPLATE = (
    "The image does not show the {attribute} of the {entity}, so it cannot be "
    "answered from the image."
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

Holds in the image, and the only things you may state:
{supported}

Contradicted by the image, the image shows something else where this was asked:
{refuted}

Not settled by the image, either not found or not a property an image can show:
{unknown}

Answer the question using only the first list.
- You cannot see the image. Anything not written above is unknown to you.
- Answer in one full sentence, the way the question expects.
- When the answer would need anything from the other two lists, answer exactly:
  "The image does not show what the question asks about."
- Do not mention a contradicted or unsettled value, not even to deny it.

Every word of your answer has to come from the first list. Do not introduce an
object, a colour, a number or a relation that is not written there.

Then give the numbers of the facts you used.

Return JSON only:
{{"answer": "<the answer the question expects>", "evidence": [<numbers from the first list>]}}

Question: {question}

JSON:"""

VERIFY_PROMPT = """Question: {question}

What the image holds:
{supported}

What the image contradicts:
{refuted}

What the image cannot settle:
{unknown}

An answer was written from those facts:

{answer}

Is every factual claim in that answer supported by what the image holds?
- An answer that declines to answer, because the facts do not provide it, is
  supported.
- A claim that contradicts the image, or that names something none of the three
  lists mention, is not supported.

Answer only YES or NO."""

REPAIR_PROMPT = """Question: {question}

An answer was written and then checked against what the image holds. These parts
of it are not supported:

{violations}

Everything known about the image:
{supported}

Rewrite the answer so that every word of it appears in that list, and say the
image does not show it for the rest. You cannot see the image.

Return JSON only:
{{"answer": "<the answer>", "evidence": [<numbers from the list>]}}

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
                 fallback="refuse", verify_objects=False):

        self.tools = tools
        self.max_elements = max_elements
        self.answer_tokens = answer_tokens
        self.fallback = fallback

        # Off by default: asking a model "does this crop show a bottom stripe
        # of the rainbow" is unreliable on phrase-like names, and a rejected
        # instance takes the whole element with it, which cost 18 of 19 samples
        # their evidence in the first run that had it on.
        self.verify_objects = verify_objects
        self.dropped_by_verification = 0
        self.last_judge_readable = None

    # ========================================================
    # Instances
    # ========================================================

    def instances(self, image, name, cache):

        """Every box for an entity, and never one box for a set of them."""

        if name in cache:
            return cache[name]

        query, selector = split_selector(name)

        found = self.tools.locate(image, query)

        # A box is not an object: the locator returns one for everything it is
        # asked about, so on an absent object the pipeline checks the place
        # where it would be. Optionally each crop is shown back to the model
        # first, which is the only thing here that can say "there is no tree".
        if self.verify_objects:

            before = len(found)

            found = self.tools.verify_object(image, found, query)

            self.dropped_by_verification += before - len(found)

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

                for item_a in first_boxes:
                    for item_b in second_boxes:

                        verdicts.append(
                            self.tools.relate(
                                image, item_a["box"], first,
                                element["relation"], item_b["box"], second,
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

    def check(self, answer, cited, grounded, question, facts):

        """Whether the answer stays inside what the image holds.

        Two passes:

        1. Structural: the cited evidence numbers must exist and point at
           SUPPORTED elements. Numbers rather than quoted text, because text
           can be copied verbatim into a claim and then trivially "match" the
           line it copied, which is how a hallucinated sentence passed the
           earlier version.

        2. Semantic: a call that reads the facts and the answer together and
           names any claim the facts do not carry. A word list was tried here
           first and it failed in both directions: it rejected the refusal
           sentence the prompt itself asks for ("what the question asks about"
           is not a word any evidence line contains), and it could not tell a
           refusal from an assertion that reused the question's words. Whether
           a sentence claims something is a question about meaning.

        The refuted values keep a lexical guard as well, because naming a value
        the image contradicts is an error no reading of the facts can excuse.
        """

        supported_elements = [
            item for item in grounded if item["verdict"] == "SUPPORTED"
        ]

        refuted = self._refuted_terms(grounded)

        violations = []

        for number in cited:

            if not isinstance(number, int) or not 1 <= number <= len(supported_elements):

                violations.append({
                    "item": f"cited evidence {number}",
                    "kind": "not a fact that holds",
                })

        lowered = normalise(answer)

        for term in refuted:

            words = [word for word in term.split() if len(word) > 3]

            if not words:
                continue

            # Only a whole phrase counts: "the bottom stripe is not orange"
            # contains orange, and the answer naming orange is the mistake.
            if term[:24] and term[:24] in lowered and "not " not in lowered:

                violations.append({"item": term[:40], "kind": "contradicted"})

        verdict = self._judge(question, answer, facts)

        self.last_judge_readable = verdict.get("readable")

        if not verdict["ok"]:

            violations.append({
                "item": (
                    "; ".join(verdict["unsupported"])
                    or "the answer as written"
                ),
                "kind": "not supported by the facts",
            })

        return violations

    def _judge(self, question, answer, facts):

        """Whether the answer stays inside the facts. Yes or no, nothing else.

        A JSON verdict was tried first and the model could not produce it: an
        unreadable reply was treated as a failure, every draft then failed, and
        the run answered the generic refusal to all 19 samples. YES/NO is what
        the rest of this codebase's checks use and what a 3B model answers
        reliably.

        An unreadable verdict does not veto. The structural checks -- the cited
        numbers and the contradicted values -- still stand on their own, and a
        verifier that cannot speak must not be able to turn the pipeline into a
        refusal machine.
        """

        prompt = (
            VERIFY_PROMPT
            .replace("{question}", question.strip())
            .replace("{supported}", facts["supported"])
            .replace("{refuted}", facts["refuted"])
            .replace("{unknown}", facts["unknown"])
            .replace("{answer}", answer)
        )

        try:

            reply = self.tools.generate(None, prompt, max_new_tokens=8)

        except (ValueError, KeyError, TypeError) as error:

            return {"ok": True, "unsupported": [], "readable": False,
                    "error": str(error)}

        match = re.search(r"\b(yes|no)\b", str(reply).strip().lower())

        if match is None:

            return {"ok": True, "unsupported": [], "readable": False,
                    "error": str(reply)[:60]}

        if match.group(1) == "yes":

            return {"ok": True, "unsupported": [], "readable": True}

        return {
            "ok": False,
            "unsupported": [],
            "readable": True,
            "error": "",
        }

    def _parse_answer(self, reply):

        try:
            data = extract_json(reply)
        except (ValueError, KeyError, TypeError):
            return None

        answer = str(data.get("answer", "")).strip()

        if not answer:
            return None

        cited = data.get("evidence") or []

        if isinstance(cited, (int, str)):
            cited = [cited]

        numbers = []

        for item in cited:

            try:
                numbers.append(int(item))
            except (TypeError, ValueError):
                continue

        return {"answer": answer, "cited": numbers[:12]}

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

    def complete_plan(self, plan, question):

        """Add the property the question asks for when the plan left it out.

        A plan of one object element grounds existence and nothing else, and the
        answer step is then asked a question the evidence says nothing about:
        id=16 got "the dog is sitting" out of "1 dog found". The target carries
        the question's own property, so it is the one element worth trusting to
        be about the question, and it is added when the plan forgets it.
        """

        target = plan.get("target") or {}

        entity = normalise(target.get("entity", ""))
        attribute = normalise(target.get("attribute", ""))

        if not entity or not attribute or not is_visual(attribute):
            return plan

        for element in plan["elements"]:

            if element["kind"] != "attribute":
                continue

            existing = normalise(element.get("attribute", ""))

            if existing and (
                attribute[:4] in existing or existing[:4] in attribute
            ):
                return plan

        names = [
            element["name"] for element in plan["elements"]
            if element["kind"] == "object"
        ]

        # The attribute belongs to the entity the question is about, so when the
        # plan grounded something else the entity is added as its own element:
        # attaching the property to the plan's first object made id=12 check
        # "the light color of the motorcycle" for a question about a man.
        subject = next(
            (name for name in names if entity[:4] in name or name[:4] in entity),
            None,
        )

        if subject is None:

            subject = entity

            plan["elements"] = plan["elements"] + [{
                "kind": "object",
                "name": entity,
            }]

        plan["elements"] = plan["elements"] + [{
            "kind": "attribute",
            "subject": subject,
            "attribute": attribute,
        }]

        plan["completed"] = attribute

        return plan

    # ========================================================
    # Entry point
    # ========================================================

    def solve(self, image, question):

        record = {"trace": [], "fallback": False}
        trace = record["trace"]

        plan = self._plan(image, question, trace)

        if plan is None:

            return self._fallback(image, question, record, "no plan")

        plan = self.complete_plan(plan, question)

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
        record["dropped_by_verification"] = self.dropped_by_verification
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

        # Numbered, because the answer cites the numbers rather than repeating
        # the text: free text can be copied verbatim into a claim and then
        # "match" the evidence it copied, which is how a hallucinated sentence
        # passed the first version of this check.
        supported = "\n".join(
            f"{index}. {item['evidence']} ({item['kind']})"
            for index, item in enumerate(
                [item for item in grounded if item["verdict"] == "SUPPORTED"],
                start=1,
            )
        ) or "- (none)"
        refuted = "\n".join(lines("REFUTED")) or "- (none)"
        unknown = "\n".join(lines("UNKNOWN")) or "- (none)"

        facts = {
            "supported": supported,
            "refuted": refuted,
            "unknown": unknown,
        }

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

            return self._finish(record, self.refusal(plan), [], [], "unparsable answer")

        violations = self.check(
            parsed["answer"], parsed["cited"], grounded, question, facts
        )

        record["cited_evidence"] = parsed["cited"]
        record["judge_readable"] = self.last_judge_readable
        record["violations"] = violations

        if not violations:

            return self._finish(
                record, parsed["answer"], parsed["cited"], [], "clean"
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

            again = self.check(
                repaired["answer"], repaired["cited"], grounded, question, facts
            )

            record["repaired_uses"] = repaired["answer"]
            record["repaired_violations"] = again

            if not again:

                return self._finish(
                    record, repaired["answer"], repaired["cited"],
                    violations, "repaired",
                )

        return self._finish(record, self.refusal(plan), [], violations, "refused")

    # ========================================================
    # Finishing
    # ========================================================

    def refusal(self, plan):

        """The sentence to answer with when the facts do not provide one."""

        target = (plan or {}).get("target") or {}

        entity = str(target.get("entity", "")).strip()
        attribute = str(target.get("attribute", "")).strip()

        if entity and attribute and is_visual(attribute):

            return REFUSAL_TEMPLATE.format(entity=entity, attribute=attribute)

        if entity:

            return (
                f"The image does not show the {entity}, so it cannot be "
                f"answered from the image."
            )

        return REFUSAL

    def _finish(self, record, answer, cited, violations, outcome):

        record["answer"] = answer
        record["tool_calls"] = self.tools.calls
        record["outcome"] = outcome

        record["trace"].append({
            "tool": "answer",
            "outcome": outcome,
            "evidence": cited,
            "violations": violations,
        })

        return record

    def _fallback(self, image, question, record, reason):

        record["fallback"] = True
        record["fallback_reason"] = reason
        record["grounded"] = []
        record["boxes"] = {}
        record["rejected"] = 0
        record["dropped_by_verification"] = self.dropped_by_verification
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
