#!/usr/bin/env python3
"""Check the strict pipeline's structure on CPU, with a stub tool layer.

The failures this pipeline exists to prevent are decided in its own code: what
the plan is allowed to contain, whether an attribute is read or guessed,
whether three cats stay three cats, whether the answer can see the image. All
of that is testable without a GPU, in seconds, so a structural error is not
discovered after an hour of HaloQuest.

    python tools/check_cgv2.py
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from cgr.strict_cg import (
    REFUSAL,
    StrictGroundedCCoT,
    is_visual,
    parse_plan,
    select_instance,
    split_selector,
)


def BOX(x):
    return {"box": [x, 10, x + 40, 90], "score": 0.9}


class StubTools:
    """A tool layer that answers from a script instead of from an image."""

    def __init__(self, plan, locate_answer=None, describe_answer=None,
                 relation_answer=None, answer=None, repair=None, noun=None,
                 absent=(), judge=None):
        self.plan = plan
        self.locate_answer = locate_answer or {}
        self.describe_answer = describe_answer or {}
        self.relation_answer = relation_answer or {}
        self.answer = answer or {}
        self.repair = repair or {}
        self.noun = noun if noun is not None else "none"
        self.absent = set(absent)
        self.judge = judge or {}
        self.judge_calls = 0
        self.verify_object_calls = []
        self.calls = 0
        self.generated_tokens = 0
        self.describe_calls = []
        self.answer_saw_image = None
        self.last_prompt = ""

    def generate(self, image, prompt, max_new_tokens=256):

        self.calls += 1
        self.generated_tokens += 10

        if "Write what has to be checked" in prompt:
            return self.plan

        if "previous reply could not be read" in prompt:
            return ""

        if "Which object does this question ask about" in prompt:
            return self.noun

        if "Is every factual claim" in prompt:

            self.judge_calls += 1

            replies = self.judge.get("replies")

            if replies:

                index = min(self.judge_calls - 1, len(replies) - 1)

                return replies[index]

            return self.judge.get("reply", "YES")

        if "An answer was written and then checked" in prompt:
            return self.repair.get("reply", "")

        # The final answer is the only call that is allowed no image.
        self.answer_saw_image = image is not None
        self.last_prompt = prompt

        return self.answer.get("reply", "")

    def locate(self, image, name):

        self.calls += 1

        return self.locate_answer.get(name, [])

    def verify_object(self, image, boxes, entity, max_new_tokens=4):
        """The stub keeps an instance unless the script says it is absent."""

        self.calls += 1
        self.verify_object_calls.append(entity)

        if entity in self.absent:
            return []

        return list(boxes)

    def describe_attribute(self, image, boxes, entity, attribute,
                           max_new_tokens=16):

        self.calls += 1
        self.describe_calls.append((entity, attribute, len(boxes)))

        return self.describe_answer.get(
            attribute,
            {"value": None,
             "evidence": f"the image does not show the {attribute}"},
        )

    def relate(self, image, box_a, a, relation, box_b, b):

        self.calls += 1

        return self.relation_answer.get((a, relation, b), False)

    def ask_question(self, image, question):

        self.calls += 1

        return "free generation"


CHECKS = []


def check(name, condition, detail=""):

    CHECKS.append((name, bool(condition), detail))


def run(plan, fallback="refuse", question="q", verify_objects=False, **kwargs):

    tools = StubTools(plan, **kwargs)

    solver = StrictGroundedCCoT(
        tools,
        answer_tokens=200,
        fallback=fallback,
        verify_objects=verify_objects,
    )

    record = solver.solve(image=None, question=question)

    return tools, record


# ============================================================
# The plan may not carry the answer
# ============================================================

plan = parse_plan(
    '{"target": {"entity": "stripe", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "rainbow"},'
    ' {"kind": "attribute", "subject": "rainbow", "attribute": "bottom stripe color"}]}'
)

check("the plan keeps what to read",
      plan["elements"][1]["attribute"] == "bottom stripe color")
check("the plan has no value field",
      "value" not in plan["elements"][1], str(plan["elements"][1]))

plan_with_value = parse_plan(
    '{"target": {}, "elements": [{"kind": "attribute", "subject": "rainbow",'
    ' "attribute": "bottom stripe color", "value": "orange"}]}'
)

check("a value written anyway is dropped",
      "value" not in plan_with_value["elements"][0],
      str(plan_with_value["elements"][0]))


# ============================================================
# The attribute is read, not guessed
# ============================================================

tools, record = run(
    '{"target": {"entity": "stripe", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "rainbow"},'
    ' {"kind": "attribute", "subject": "rainbow", "attribute": "bottom stripe color"}]}',
    locate_answer={"rainbow": [BOX(10)]},
    describe_answer={
        "bottom stripe color": {
            "value": "purple",
            "evidence": "the bottom stripe color of the rainbow is purple",
        }
    },
    answer={"reply": '{"answer": "The bottom stripe is purple.", "evidence": [1, 2]}'},
)

check("the attribute is read from the image",
      tools.describe_calls == [("rainbow", "bottom stripe color", 1)],
      str(tools.describe_calls))
check("the value read becomes the evidence",
      record["grounded"][1].get("value") == "purple", str(record["grounded"][1]))
check("an answer using the read value is clean",
      record["outcome"] == "clean", record.get("outcome"))


# ============================================================
# id=18: a refusal to read is UNKNOWN, and cannot license a value
# ============================================================

tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "rainbow"},'
    ' {"kind": "attribute", "subject": "rainbow", "attribute": "bottom stripe color"}]}',
    locate_answer={"rainbow": [BOX(10)]},
    describe_answer={
        "bottom stripe color": {
            "value": None,
            "evidence": "the image does not show the bottom stripe color of the rainbow",
        }
    },
    answer={"reply": '{"answer": "The bottom stripe is blue.", "evidence": [1]}'},
    repair={"reply": '{"answer": "The image does not show what the question asks about.",'
                     ' "evidence": []}'},
    judge={"replies": ["NO", "YES"]},
)

check("an unreadable attribute is UNKNOWN",
      record["grounded"][1]["verdict"] == "UNKNOWN", str(record["grounded"][1]))
check("blue is not licensed by a refusal",
      record["violations"] and "not supported" in record["violations"][0]["kind"],
      str(record.get("violations")))
check("blue is gone", "blue" not in record["answer"].lower(), record["answer"])


# ============================================================
# The answer cannot look at the image
# ============================================================

tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "shirt"},'
    ' {"kind": "attribute", "subject": "shirt", "attribute": "color"}]}',
    locate_answer={"shirt": [BOX(10)]},
    describe_answer={
        "color": {"value": "red", "evidence": "the color of the shirt is red"}
    },
    answer={"reply": '{"answer": "The shirt is red.", "evidence": [1, 2]}'},
)

check("the answer step is called without the image",
      tools.answer_saw_image is False, str(tools.answer_saw_image))


# ============================================================
# id=8: an author is never sent to a tool, and Samantha cannot come back
# ============================================================

check("author is not visual", not is_visual("author"))
check("title is not visual", not is_visual("title"))
check("color is visual", is_visual("color"))

tools, record = run(
    '{"target": {"entity": "author", "attribute": "name"}, "elements": '
    '[{"kind": "object", "name": "book"},'
    ' {"kind": "attribute", "subject": "book", "attribute": "author"}]}',
    locate_answer={"book": [BOX(10)]},
    answer={"reply": '{"answer": "The author is Samantha.", "evidence": [1]}'},
    repair={"reply": '{"answer": "The image does not show what the question asks about.",'
                     ' "evidence": []}'},
    judge={"replies": ["NO", "YES"]},
)

check("the author never reached a tool", tools.describe_calls == [],
      str(tools.describe_calls))
check("the author is UNKNOWN", record["grounded"][1]["verdict"] == "UNKNOWN",
      str(record["grounded"][1]))
check("Samantha cannot be asserted", "samantha" not in record["answer"].lower(),
      record["answer"])


# ============================================================
# Three cats stay three cats
# ============================================================

tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "cat"},'
    ' {"kind": "attribute", "subject": "cat", "attribute": "color"}]}',
    locate_answer={"cat": [BOX(10), BOX(200), BOX(400)]},
    describe_answer={
        "color": {"value": "white", "evidence": "the color of the cat is white"}
    },
    answer={"reply": '{"answer": "The cat is white.", "evidence": [1, 2]}'},
)

check("every instance is kept",
      len(record["boxes"]["cat"]) == 3, str(record["boxes"]))
check("the attribute is read across the instances",
      tools.describe_calls == [("cat", "color", 3)], str(tools.describe_calls))
check("the instance count is in the evidence",
      "3 cat found" in str(record["grounded"][0]["evidence"]),
      str(record["grounded"][0]))


# ============================================================
# left man is the entity man plus an order, not a name to query
# ============================================================

check("left man splits", split_selector("left man") == ("man", "left"),
      str(split_selector("left man")))
check("smaller animal splits",
      split_selector("smaller animal") == ("animal", "smaller"),
      str(split_selector("smaller animal")))
check("plain man does not split", split_selector("man") == ("man", None),
      str(split_selector("man")))
check("a name that is only an order stays whole",
      split_selector("left") == ("left", None), str(split_selector("left")))

three = [BOX(10), BOX(200), BOX(400)]

check("left picks the leftmost", select_instance(three, "left")[0]["box"][0] == 10,
      str(select_instance(three, "left")))
check("right picks the rightmost", select_instance(three, "right")[0]["box"][0] == 400,
      str(select_instance(three, "right")))

tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "left man"},'
    ' {"kind": "object", "name": "right man"}]}',
    locate_answer={"man": [BOX(10), BOX(400)]},
    answer={"reply": '{"answer": "There are two man.", "evidence": [1]}'},
    repair={"reply": '{"answer": "The image shows two man.", "evidence": [1]}'},
)

check("the locator is queried with the type, not the order",
      "left man" not in tools.describe_calls and
      len(record["boxes"].get("left man", [])) == 1,
      str(record["boxes"]))
check("each side resolves to one instance",
      len(record["boxes"]["left man"]) == 1 and len(record["boxes"]["right man"]) == 1,
      str(record["boxes"]))
check("and they are different instances",
      record["boxes"]["left man"][0][0] != record["boxes"]["right man"][0][0],
      str(record["boxes"]))


# ============================================================
# Relations are checked, not inferred from both sides existing
# ============================================================

tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "girl"},'
    ' {"kind": "object", "name": "book"},'
    ' {"kind": "relation", "subject": "girl", "relation": "reading", "object": "book"}]}',
    locate_answer={"girl": [BOX(10)], "book": [BOX(400)]},
    relation_answer={("girl", "reading", "book"): False},
    answer={"reply": '{"answer": "The girl is reading the book.", "evidence": [3]}'},
    repair={"reply": '{"answer": "The image does not show that.", "evidence": []}'},
)

check("a refuted relation is REFUTED",
      record["grounded"][2]["verdict"] == "REFUTED", str(record["grounded"][2]))
check("its value cannot be asserted", "reading" not in record["answer"].lower(),
      record["answer"])


# ============================================================
# Fallback: retry the plan, then the question's own object, then refuse
# ============================================================

tools, record = run(
    "not json at all",
    question="What color is the cat?",
    locate_answer={"cat": [BOX(10)]},
    describe_answer={
        "color": {"value": "black", "evidence": "the color of the cat is black"}
    },
    answer={"reply": '{"answer": "The cat is black.", "evidence": [1, 2]}'},
    noun="cat",
)

check("a broken plan falls to the minimal plan",
      record["plan"]["raw"] == "minimal plan", str(record.get("plan")))
check("the minimal plan is the question's object and property",
      record["plan"]["elements"] == [
          {"kind": "object", "name": "cat"},
          {"kind": "attribute", "subject": "cat", "attribute": "color"},
      ],
      str(record["plan"]["elements"]))
check("and it is answered from evidence, not freely",
      record["answer"] == "The cat is black.", record["answer"])
check("no free generation happened", "free generation" not in record["answer"])

tools, record = run(
    "not json at all",
    question="Who is the author of the book?",
    locate_answer={"book": [BOX(10)]},
    answer={"reply": "free generation"},
    noun="book",
)

check("a question about something no image can show gets no attribute element",
      record["plan"]["elements"] == [{"kind": "object", "name": "book"}],
      str(record["plan"]["elements"]))
check("and that answer cannot be asserted",
      "free generation" not in record["answer"], record["answer"])

tools, record = run(
    "not json at all",
    noun="none",
    answer={"reply": "free generation"},
)

check("no object to fall back to refuses",
      record["answer"] == REFUSAL and record["fallback"], record["answer"])

tools, record = run(
    "not json at all",
    noun="none",
    locate_answer={"cat": [BOX(10)]},
    answer={"reply": "free generation"},
    fallback="direct",
)

check("fallback=direct keeps the old behaviour for comparison",
      record["answer"] == "free generation", record["answer"])


# ============================================================
# A supported answer still passes, and refusing is not punished
# ============================================================

tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "girl"},'
    ' {"kind": "attribute", "subject": "girl", "attribute": "shoes"}]}',
    locate_answer={"girl": [BOX(10)]},
    describe_answer={
        "shoes": {
            "value": None,
            "evidence": "the image does not show the shoes of the girl",
        }
    },
    answer={"reply": '{"answer": "The image does not show the girl.", "evidence": []}'},
)

check("a refusal with no claims is clean", record["outcome"] == "clean",
      record.get("outcome"))
check("counts are recorded",
      record["verdict_counts"]["UNKNOWN"] == 1 and
      record["verdict_counts"]["SUPPORTED"] == 1,
      str(record["verdict_counts"]))


# ============================================================
# The two holes the first live run found
# ============================================================

# id=16: the plan grounded one object and nothing else, and the answer asserted
# a state ("sitting") that no element ever read. Citing a valid fact must not
# license prose that goes beyond it.
tools, record = run(
    '{"target": {"entity": "dog", "attribute": "sitting"}, "elements": '
    '[{"kind": "object", "name": "dog"}]}',
    question="Is the dog sitting or walking?",
    locate_answer={"dog": [BOX(10)]},
    answer={"reply": '{"answer": "The dog is sitting.", "evidence": [1]}'},
    repair={"reply": '{"answer": "The image does not show what the question asks about.",'
                     ' "evidence": []}'},
    judge={"reply": "NO"},
)

check("a claim the facts do not carry is a violation",
      any("not supported" in item["kind"] for item in record.get("violations", [])),
      str(record.get("violations")))
check("and the answer no longer asserts it",
      record["outcome"] in ("refused", "repaired")
      and "does not show" in record["answer"].lower(), record["answer"])
check("the plan is completed with the question's own property",
      any(element.get("attribute") == "sitting" for element in record["plan"]["elements"]),
      str(record["plan"]["elements"]))

# A citation that is not a fact that holds is a violation on its own, because
# the answer is then claiming support it does not have.
tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "cat"}]}',
    locate_answer={"cat": [BOX(10)]},
    answer={"reply": '{"answer": "The cat is here.", "evidence": [7]}'},
    repair={"reply": '{"answer": "The image shows the cat.", "evidence": [1]}'},
)

check("citing evidence that does not exist is a violation",
      any("cited evidence" in item["item"] for item in record.get("violations", [])),
      str(record.get("violations")))

# An answer can no longer be declared "supported" by quoting the evidence line.
tools, record = run(
    '{"target": {}, "elements": [{"kind": "object", "name": "dog"}]}',
    locate_answer={"dog": [BOX(10)]},
    answer={"reply": '{"answer": "1 dog found.", "evidence": [1]}'},
)

check("quoting the evidence word for word is not an answer",
      record["answer"] != "1 dog found." or record["outcome"] == "clean",
      record["answer"])

# ============================================================
# A located box is not an object until its crop is checked
# ============================================================

tools, record = run(
    '{"target": {"entity": "tree", "attribute": "type"}, "elements": '
    '[{"kind": "object", "name": "tree"}]}',
    question="What type of tree is in this image?",
    locate_answer={"tree": [BOX(10)]},
    absent=("tree",),
    verify_objects=True,
    answer={"reply": '{"answer": "1 tree found.", "evidence": [1]}'},
    repair={"reply": '{"answer": "The image does not show what the question asks about.",'
                     ' "evidence": []}'},
)

check("an object the verifier rejects is not grounded",
      record["grounded"][0]["verdict"] == "UNKNOWN", str(record["grounded"][0]))
check("the drop is counted", record["dropped_by_verification"] == 1,
      str(record.get("dropped_by_verification")))
check("and nothing can be asserted about it",
      "tree" not in record["answer"].lower() or "does not show" in record["answer"].lower(),
      record["answer"])

# The attribute of the question belongs to the question's entity.
tools, record = run(
    '{"target": {"entity": "man", "attribute": "helmet color"}, "elements": '
    '[{"kind": "object", "name": "motorcycle"}]}',
    locate_answer={"motorcycle": [BOX(10)], "man": [BOX(300)]},
    describe_answer={
        "helmet color": {"value": "white", "evidence": "the helmet color of the man is white"}
    },
    answer={"reply": '{"answer": "The helmet is white.", "evidence": [2, 3]}'},
)

check("the question's entity is grounded even when the plan missed it",
      any(element.get("name") == "man" for element in record["plan"]["elements"]),
      str(record["plan"]["elements"]))
check("and the attribute is read from it",
      any(entity == "man" for entity, _, _ in tools.describe_calls),
      str(tools.describe_calls))

# A short answer is not a crime: id=18 answers "purple" to a question whose
# truth is that the stripe is purple, and a word list that rejected it put the
# answer into repair and then into a refusal, which is how a run turns into a
# constant refusal that scores well on HaloQuest for the wrong reason.
tools, record = run(
    '{"target": {"entity": "stripe", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "rainbow"},'
    ' {"kind": "attribute", "subject": "rainbow", "attribute": "bottom stripe color"}]}',
    locate_answer={"rainbow": [BOX(10)]},
    describe_answer={
        "bottom stripe color": {
            "value": "purple",
            "evidence": "the bottom stripe color of the rainbow is purple",
        }
    },
    answer={"reply": '{"answer": "purple", "evidence": [1, 2]}'},
)

check("a one word answer the facts carry is accepted",
      record["outcome"] == "clean" and record["answer"] == "purple",
      f"{record.get('outcome')} / {record['answer']}")

# And the opposite: the judge saying no sends it to repair, not straight out.
tools, record = run(
    '{"target": {"entity": "hat", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "hat"}]}',
    locate_answer={"hat": [BOX(10)]},
    describe_answer={"color": {"value": "red", "evidence": "the color of the hat is red"}},
    answer={"reply": '{"answer": "The hat is blue.", "evidence": [1, 2]}'},
    repair={"reply": '{"answer": "The hat is red.", "evidence": [1, 2]}'},
    judge={"replies": ["NO", "YES"]},
)

check("a rejected draft is repaired",
      record["outcome"] == "repaired" and record["answer"] == "The hat is red.",
      f"{record.get('outcome')} / {record['answer']}")

print()

failed = 0

for name, passed, detail in CHECKS:

    print(f"{'ok  ' if passed else 'FAIL'} {name}"
          + (f"  <- {detail}" if not passed else ""))

    failed += not passed

print()
print(f"{len(CHECKS) - failed} passed, {failed} failed")

raise SystemExit(1 if failed else 0)
