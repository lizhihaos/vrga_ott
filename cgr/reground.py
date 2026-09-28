"""Answer, check every visual claim against the image, then rewrite.

This is the dataset-agnostic form of the method. The constraint graph came
from the question, which only works when the question names visual entities;
here the claims come from the model's own draft, so the same re-grounding,
candidate-set and verification machinery applies to any benchmark:

    draft      plain CoT answer
    decompose  draft -> atomic visual claims
    verify     each claim -> LOCATE / VERIFY_ATTRIBUTE / RELATE on the image
    repair     drop what the image does not support and answer again
    report     the per-claim verdicts, which are the mechanism evidence

The verdicts are also the metric that needs no ground truth: the share of
claims the image supports, and the share of questions where at least one
claim had to be retracted.
"""

import re

from .claims import parse_claims

REPAIR_PROMPT = """Question: {question}

Draft answer:
{draft}

Facts checked against the image.

Supported by the image:
{supported}

NOT supported by the image (the image contradicts or does not contain them):
{retracted}

Write the answer to the question subject to these constraints:
- You may keep any part of the draft that is consistent with the supported facts.
- You must not state or imply anything from the NOT supported list.
- If the question can no longer be answered without the unsupported statements,
  say plainly that the image does not show it.

Answer:"""


class ReGrounding:
    """Draft, verify, repair."""

    def __init__(
        self,
        tools,
        max_claims=6,
        draft_tokens=700,
        max_revisions=2,
        count_tolerance=1,
        candidates_per_claim=3,
        invert_negations=True,
    ):

        self.tools = tools
        self.max_claims = max_claims
        self.draft_tokens = draft_tokens
        self.max_revisions = max_revisions
        self.count_tolerance = count_tolerance
        self.candidates_per_claim = candidates_per_claim
        self.invert_negations = invert_negations

    # ========================================================
    # Verification
    # ========================================================

    def _negated(self, text):

        return bool(re.search(r"\b(no|not|never|without|absent)\b", text.lower()))

    def verify(self, image, claim):

        """Check one claim on the image, keeping the candidate set.

        The top candidate decides, but a claim that the top candidate
        contradicts while a lower ranked one supports it is marked ambiguous
        rather than false: that is the multi-entity case the candidate set
        exists for.
        """

        subject = claim["subject"]
        kind = claim["kind"]

        # The model labels kinds unreliably: "the suitcase is on the book"
        # comes back as kind "existence", and an existence check only asks
        # whether a suitcase is somewhere in the image, which is always true
        # and therefore never contradicts anything. Infer the check from the
        # content instead.
        weak_values = ("", "present", "absent", "true", "false", "yes", "no")

        if claim["object"] and claim["predicate"]:

            kind = "relation"

        elif claim["predicate"] and claim["value"] not in weak_values:

            kind = "attribute"

        elif kind not in ("existence", "count"):

            kind = "existence"

        candidates = self.tools.locate(image, subject)

        # A value like "unknown" carries nothing to check; asking the VLM
        # about it produces a confident answer to a meaningless question.
        if claim["value"] in (
            "unknown", "unclear", "unspecified", "not visible",
            "not specified", "n/a", "none", "",
        ) and kind != "existence":

            return {
                "claim": claim,
                "subject": subject,
                "kind": kind,
                "candidates": 0,
                "verdict": "SKIPPED",
                "reason": "nothing checkable in this claim",
            }

        result = {
            "claim": claim,
            "subject": subject,
            "kind": kind,
            "candidates": len(candidates),
            "boxes": [item["box"] for item in candidates],
        }

        # ----------------------------------------------------
        # existence
        # ----------------------------------------------------

        if kind == "existence":

            present = len(candidates) > 0
            expects_absent = claim["value"] in ("absent", "no", "none", "not present")

            if self.invert_negations and self._negated(claim["text"]):
                expects_absent = not expects_absent

            ok = present != expects_absent

            result["verdict"] = "SUPPORTED" if ok else "NOT_SUPPORTED"
            result["reason"] = (
                f"found {len(candidates)} {subject} in the image"
                if present else
                f"found no {subject} in the image"
            )

            return result

        # ----------------------------------------------------
        # everything else needs the subject to be there
        # ----------------------------------------------------

        if not candidates:

            result["verdict"] = "NOT_SUPPORTED"
            result["reason"] = f"found no {subject} in the image"

            return result

        # ----------------------------------------------------
        # attribute
        # ----------------------------------------------------

        if kind == "attribute":

            attribute = claim["predicate"] or "attribute"
            checks = []

            for candidate in candidates[:self.candidates_per_claim]:

                checks.append(self.tools.verify_attribute(
                    image,
                    candidate["box"],
                    subject,
                    attribute,
                    claim["value"],
                ))

            if checks and checks[0]:

                result["verdict"] = "SUPPORTED"
                result["reason"] = f"the {subject} is {claim['value']}"

            elif any(checks):

                result["verdict"] = "AMBIGUOUS"
                result["reason"] = (
                    f"the most confident {subject} is not {claim['value']}, "
                    f"but another one is"
                )

            else:

                result["verdict"] = "NOT_SUPPORTED"
                result["reason"] = f"no {subject} looks {claim['value']}"

            return result

        # ----------------------------------------------------
        # relation
        # ----------------------------------------------------

        if kind == "relation":

            predicate = claim["predicate"] or claim["value"]

            others = self.tools.locate(image, claim["object"])

            result["object_candidates"] = len(others)

            if not others:

                result["verdict"] = "NOT_SUPPORTED"
                result["reason"] = f"found no {claim['object']} in the image"

                return result

            for first in candidates[:2]:
                for second in others[:2]:

                    if self.tools.relate(
                        image,
                        first["box"],
                        subject,
                        predicate,
                        second["box"],
                        claim["object"],
                    ):

                        result["verdict"] = "SUPPORTED"
                        result["reason"] = (
                            f"the {subject} is {predicate} the "
                            f"{claim['object']}"
                        )

                        return result

            result["verdict"] = "NOT_SUPPORTED"
            result["reason"] = (
                f"no {subject} is {predicate} a {claim['object']}"
            )

            return result

        # ----------------------------------------------------
        # count
        # ----------------------------------------------------

        if kind == "count":

            numbers = re.findall(r"\d+", claim["value"])

            if not numbers:

                result["verdict"] = "SKIPPED"
                result["reason"] = "no number to check"

                return result

            claimed = int(numbers[0])
            found = len(candidates)

            ok = abs(found - claimed) <= self.count_tolerance

            result["verdict"] = "SUPPORTED" if ok else "NOT_SUPPORTED"
            result["reason"] = f"claimed {claimed}, grounding found {found}"

            return result

        result["verdict"] = "SKIPPED"
        result["reason"] = f"unknown claim kind {kind}"

        return result

    # ========================================================
    # Entry point
    # ========================================================

    def solve(self, image, question):

        record = {"trace": [], "fallback": False}
        trace = record["trace"]

        # ----------------------------------------------------
        # 1. Draft
        # ----------------------------------------------------

        draft = self.tools.ask_question(
            image, question, max_new_tokens=self.draft_tokens
        )

        record["draft"] = draft
        trace.append({"tool": "draft", "chars": len(draft)})

        # ----------------------------------------------------
        # 2. Decompose
        # ----------------------------------------------------

        claims = parse_claims(
            lambda prompt, img=None: self.tools.generate(
                img, prompt, max_new_tokens=512
            ),
            draft,
            image=image,
            max_claims=self.max_claims,
        )

        record["claims"] = claims
        trace.append({"tool": "decompose", "claims": len(claims)})

        if not claims:

            record["answer"] = draft
            record["verdicts"] = []
            record["retracted"] = 0
            record["tool_calls"] = self.tools.calls

            return record

        # ----------------------------------------------------
        # 3. Verify each claim on the image
        # ----------------------------------------------------

        verdicts = []

        for claim in claims:

            verdict = self.verify(image, claim)

            verdicts.append(verdict)

            trace.append({
                "tool": "verify",
                "subject": claim["subject"],
                "kind": claim["kind"],
                "value": claim["value"],
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
                "candidates": verdict.get("candidates", 0),
            })

        record["verdicts"] = verdicts
        record["retracted"] = sum(
            1 for v in verdicts if v["verdict"] == "NOT_SUPPORTED"
        )

        # ----------------------------------------------------
        # 4. The answer must pass the same check
        #
        # A small model paraphrases a forbidden claim instead of dropping it
        # ("red collar" comes back as "solid red collar"), so revising once is
        # not enough: the revised answer is decomposed and verified again, and
        # when it still fails the answer is built from the tools instead.
        # ----------------------------------------------------

        answer = draft
        attempts = 0

        while attempts < self.max_revisions:

            attempts += 1

            answer_claims = parse_claims(
                lambda prompt, img=None: self.tools.generate(
                    img, prompt, max_new_tokens=512
                ),
                answer,
                image=image,
                max_claims=self.max_claims,
            )

            answer_verdicts = [
                self.verify(image, claim) for claim in answer_claims
            ]

            failed = [
                verdict for verdict in answer_verdicts
                if verdict["verdict"] == "NOT_SUPPORTED"
            ]

            trace.append({
                "tool": "reverify",
                "attempt": attempts,
                "claims": len(answer_claims),
                "failed": len(failed),
            })

            if not failed:
                break

            if attempts >= self.max_revisions:
                answer = self.deterministic_answer(failed, draft)
                trace.append({"tool": "deterministic", "answer": answer})
                break

            answer = self.revise(image, question, draft, answer_verdicts)

            trace.append({"tool": "revise", "attempt": attempts, "answer": answer})

        record["answer"] = answer
        record["repaired"] = attempts > 0 and answer != draft
        record["revision_attempts"] = attempts
        record["tool_calls"] = self.tools.calls

        return record

    # ========================================================
    # Revision and fallback
    # ========================================================

    def revise(self, image, question, draft, verdicts):

        supported = [
            f"- {verdict['claim']['text']} ({verdict['reason']})"
            for verdict in verdicts
            if verdict["verdict"] in ("SUPPORTED", "AMBIGUOUS")
        ]

        retracted = [
            f"- {verdict['claim']['text']} ({verdict['reason']})"
            for verdict in verdicts
            if verdict["verdict"] == "NOT_SUPPORTED"
        ]

        return self.tools.generate(
            image,
            (REPAIR_PROMPT
             .replace("{question}", question.strip())
             .replace("{draft}", draft.strip())
             .replace("{supported}", "\n".join(supported) or "- (none)")
             .replace("{retracted}", "\n".join(retracted) or "- (none)")),
            max_new_tokens=256,
        )

    @staticmethod
    def deterministic_answer(failed, draft):

        """What is left when the model will not drop an unsupported claim.

        For a hallucination benchmark this is usually the right answer
        anyway: the question asks about something the image does not show.
        """

        subjects = []

        for verdict in failed:

            claim = verdict["claim"]

            # For an attribute claim the missing thing is what the attribute
            # is about, not the object that carries it: the girl is visible,
            # her shoes are not.
            subject = verdict["subject"]

            if claim["kind"] == "attribute" and claim["predicate"]:

                head = claim["predicate"].split()[0].strip()

                if head and head not in ("attribute", "type", "color", "design"):
                    subject = head

            if subject not in subjects:
                subjects.append(subject)

        if subjects:

            listing = ", ".join(subjects[:3])

            return f"The image does not show the {listing}."

        return "The image does not show what the question asks about."
