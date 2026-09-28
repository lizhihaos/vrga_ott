#!/usr/bin/env python3
"""Check the strict pipeline's verdict and answer-constraint logic on CPU.

The failures this pipeline is meant to stop are decided in its own code, not in
the model: whether a contradicted value is refused, whether an ungrounded value
in the answer is caught, whether an unobservable property ever reaches a tool.
Those can be tested with a stub tool layer, which runs in seconds and needs no
GPU, so a logic error is not discovered after an hour of HaloQuest.

The cases are the ones from the first 19 HaloQuest samples:

    id=8   book.title = samantha is refuted, answer says Samantha anyway
    id=15  hat colour = black is refuted, answer says black
    id=17  the answer asserts a banana that no element grounded
    id=18  orange is refuted, the answer moves to blue
    id=0   shoes are not visible, the answer refuses, which is correct

    python tools/check_cgv2.py
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from cgr.strict_cg import (
    NON_VISUAL,
    REFUSAL,
    StrictGroundedCCoT,
    is_visual,
    parse_plan,
)


class StubTools:
    """A tool layer that answers from a script instead of from an image."""

    def __init__(self, plan, locate_answer, verify_answer, relation=None,
                 answer=None, repair=None):
        self.plan = plan
        self.locate_answer = locate_answer
        self.verify_answer = verify_answer
        self.relation_answer = relation or {}
        self.answer = answer or {}
        self.repair = repair or {}
        self.calls = 0
        self.generated_tokens = 0
        self.verify_calls = []

    def generate(self, image, prompt, max_new_tokens=256):

        self.calls += 1
        self.generated_tokens += 10

        # The answer prompt also mentions the scene graph, so route on the
        # instruction that only the plan prompt carries.
        if "Write the scene graph needed" in prompt:
            return self.plan

        if "An answer was written and then checked" in prompt:
            return self.repair.get("reply", "")

        return self.answer.get("reply", "")

    def locate(self, image, name):

        self.calls += 1

        return self.locate_answer.get(name, [])

    def verify_attribute(self, image, box, subject, attribute, value):

        self.calls += 1
        self.verify_calls.append((subject, attribute, value))

        return self.verify_answer.get(value, False)

    def relate(self, image, box_a, a, relation, box_b, b):

        self.calls += 1

        return self.relation_answer.get((a, relation, b), False)

    def ask_question(self, image, question):

        self.calls += 1

        return "free generation"


BOX = [{"box": [1, 1, 2, 2], "score": 0.9}]

CHECKS = []


def check(name, condition, detail=""):

    CHECKS.append((name, bool(condition), detail))


def run(plan, locate_answer, verify_answer, answer, repair=None,
        fallback="refuse"):

    tools = StubTools(
        plan,
        locate_answer,
        verify_answer,
        answer=answer,
        repair=repair,
    )

    solver = StrictGroundedCCoT(tools, answer_tokens=200, fallback=fallback)

    return solver, solver.solve(image=None, question="q")


# ============================================================
# The observability gate: an author is never sent to a tool
# ============================================================

check("author is not a visual property", not is_visual("author"))
check("title is not a visual property", not is_visual("title"))
check("colour is a visual property", is_visual("color"))
check("material is a visual property", is_visual("material"))

tools = StubTools(
    plan='{"target": {"entity": "author", "attribute": "name"}, "elements": '
         '[{"kind": "object", "name": "book"},'
         ' {"kind": "attribute", "subject": "book", "attribute": "title",'
         '  "value": "samantha"}]}',
    locate_answer={"book": BOX},
    verify_answer={},
    answer={"reply": '{"answer": "The author is Samantha.", "uses": ["samantha"]}'},
)

solver = StrictGroundedCCoT(tools, answer_tokens=200)

record = solver.solve(image=None, question="Who is the author of the book?")

check("title never reached a tool", tools.verify_calls == [], str(tools.verify_calls))
check(
    "title is UNKNOWN",
    all(item["verdict"] == "UNKNOWN" for item in record["grounded"]
        if item["kind"] == "attribute"),
    str(record["grounded"]),
)
check(
    "Samantha is refused",
    "Samantha" not in record["answer"],
    record["answer"],
)
check("outcome recorded", record["outcome"] in ("refused", "repaired"),
      record.get("outcome"))


# ============================================================
# id=15: a refuted attribute may not appear in the answer
# ============================================================

plan_hat = (
    '{"target": {"entity": "hat", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "hat"},'
    ' {"kind": "attribute", "subject": "hat", "attribute": "color",'
    '  "value": "black"}]}'
)

solver, record = run(
    plan_hat,
    locate_answer={"hat": BOX},
    verify_answer={"black": False},
    answer={"reply": '{"answer": "The hat is black.", "uses": ["black"]}'},
    repair={"reply": '{"answer": "The image does not show the hat\'s colour.",'
                     ' "uses": []}'},
)

check("refuted value is a violation", record["violations"] and
      record["violations"][0]["kind"] == "contradicted", str(record.get("violations")))
check("repair removed it", "black" not in record["answer"].lower(), record["answer"])
check("outcome is repaired", record["outcome"] == "repaired", record.get("outcome"))


# ============================================================
# id=15 without a usable repair: refuse rather than assert it
# ============================================================

solver, record = run(
    plan_hat,
    locate_answer={"hat": BOX},
    verify_answer={"black": False},
    answer={"reply": '{"answer": "The hat is black.", "uses": ["black"]}'},
    repair={"reply": '{"answer": "The hat is black.", "uses": ["black"]}'},
)

check("a failed repair refuses", record["answer"] == REFUSAL, record["answer"])
check("violation kept for the record", bool(record["violations"]))


# ============================================================
# id=17: an entity the answer asserts is in no grounded element
# ============================================================

solver, record = run(
    '{"target": {"entity": "fruit", "attribute": "kind"}, "elements": '
    '[{"kind": "object", "name": "smaller animal"},'
    ' {"kind": "attribute", "subject": "smaller animal", "attribute": "tail",'
    '  "value": "holding"}]}',
    locate_answer={"smaller animal": BOX},
    verify_answer={"holding": True},
    answer={"reply": '{"answer": "It is holding a banana.", "uses": ["banana"]}'},
    repair={"reply": '{"answer": "It is holding a piece of fruit.", "uses": []}'},
)

check("ungrounded value is a violation", record["violations"] and
      record["violations"][0]["kind"] == "unsupported", str(record.get("violations")))
check("banana is gone", "banana" not in record["answer"].lower(), record["answer"])


# ============================================================
# id=18: refuting orange does not license blue
# ============================================================

solver, record = run(
    '{"target": {"entity": "bottom stripe", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "rainbow"},'
    ' {"kind": "attribute", "subject": "rainbow", "attribute": "bottom stripe",'
    '  "value": "orange"}]}',
    locate_answer={"rainbow": BOX},
    verify_answer={"orange": False},
    answer={"reply": '{"answer": "The bottom stripe is blue.", "uses": ["blue"]}'},
    repair={"reply": '{"answer": "The image does not show the stripe colour.",'
                     ' "uses": []}'},
)

check("blue is unsupported, not licensed", record["violations"] and
      record["violations"][0]["kind"] == "unsupported", str(record.get("violations")))
check("blue is gone", "blue" not in record["answer"].lower(), record["answer"])


# ============================================================
# id=0: refusing is the right answer and must pass clean
# ============================================================

solver, record = run(
    '{"target": {"entity": "shoes", "attribute": "type"}, "elements": '
    '[{"kind": "object", "name": "girl"},'
    ' {"kind": "attribute", "subject": "girl", "attribute": "shoes",'
    '  "value": "sandals"}]}',
    locate_answer={"girl": BOX},
    verify_answer={"sandals": False},
    answer={"reply": '{"answer": "The image does not show the girl\'s shoes.",'
                     ' "uses": []}'},
)

check("a refusal with no claims is clean", record["outcome"] == "clean",
      record.get("outcome"))
check("refusal kept", record["answer"].startswith("The image does not show"),
      record["answer"])


# ============================================================
# A supported answer passes and keeps its evidence
# ============================================================

solver, record = run(
    '{"target": {"entity": "shirt", "attribute": "color"}, "elements": '
    '[{"kind": "object", "name": "shirt"},'
    ' {"kind": "attribute", "subject": "shirt", "attribute": "color",'
    '  "value": "red"}]}',
    locate_answer={"shirt": BOX},
    verify_answer={"red": True},
    answer={"reply": '{"answer": "The shirt is red.", "uses": ["red"]}'},
)

check("supported answer is clean", record["outcome"] == "clean", record.get("outcome"))
check("supported answer kept", record["answer"] == "The shirt is red.", record["answer"])
check("verdict counts recorded",
      record["verdict_counts"]["SUPPORTED"] == 2, str(record["verdict_counts"]))


# ============================================================
# Fallback: refuse, and only refuse, unless asked otherwise
# ============================================================

solver, record = run(
    "not json at all",
    locate_answer={},
    verify_answer={},
    answer={"reply": "ignored"},
)

check("broken plan falls back", record["fallback"])
check("fallback refuses", record["answer"] == REFUSAL, record["answer"])
check("fallback answer did not call the model",
      "free generation" not in record["answer"])

solver, record = run(
    "not json at all",
    locate_answer={},
    verify_answer={},
    answer={"reply": "ignored"},
    fallback="direct",
)

check("fallback=direct uses the plain answer",
      record["answer"] == "free generation", record["answer"])


# ============================================================
# Empty plan is not grounded, so it cannot be checked
# ============================================================

solver, record = run(
    '{"target": {}, "elements": []}',
    locate_answer={},
    verify_answer={},
    answer={"reply": "ignored"},
)

check("empty plan falls back", record["fallback"])
check("empty plan refuses", record["answer"] == REFUSAL, record["answer"])


print()

failed = 0

for name, passed, detail in CHECKS:

    print(f"{'ok  ' if passed else 'FAIL'} {name}" + (f"  <- {detail}" if not passed else ""))

    failed += not passed

print()
print(f"{len(CHECKS) - failed} passed, {failed} failed")

raise SystemExit(1 if failed else 0)
