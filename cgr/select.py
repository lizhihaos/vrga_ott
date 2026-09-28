"""Pick among sampled answers by how much of each one the image supports.

This is the selector form of the method. The editor form (reground.py) has
to make a small model drop a claim it believes, and it systematically fails
at that: asked not to say "red collar" it answers "solid red collar". The
selector never asks the model to rewrite anything. It samples several
answers, decomposes and verifies each one against the image, and keeps the
answer whose claims the image actually supports.

The evidence for this shape:

    draft (cot)            29.7 %   one sample, no checking
    editor (reground.py)   41.2 %   checks, then asks the model to revise
    pass@3, oracle         58.5 %   the right answer is often in the samples
    constant refusal       45-64 %  haloquest rewards refusing, so use a
                                    benchmark where refusing scores zero

So the answers exist in the sampling space and the verifier is the thing
that can find them. Selection needs no ground truth, unlike pass@3.
"""

from .claims import parse_claims


class Selector:
    """Sample, verify each sample's claims, keep the best supported one."""

    def __init__(
        self,
        tools,
        samples=3,
        max_claims=4,
        temperature=0.7,
        top_p=0.9,
        answer_tokens=700,
        portfolio=("direct", "cot"),
    ):

        self.tools = tools
        self.samples = max(1, samples)
        self.max_claims = max_claims
        self.temperature = temperature
        self.top_p = top_p
        self.answer_tokens = answer_tokens
        self.portfolio = list(portfolio)

    # ========================================================
    # Sampling
    # ========================================================

    def _sample_answers(self, image, question):

        """Draw the portfolio of answers.

        The first samples are greedy with different prompts, which gives the
        pool differently biased answers; the rest are stochastic draws of the
        chain-of-thought prompt for diversity.
        """

        from qwen_eval2.prompts import PROMPTS

        answers = []

        for index in range(self.samples):

            mode = self.portfolio[index % len(self.portfolio)]

            prompt = PROMPTS[mode].format(question=question.strip())

            # One greedy pass per prompt, then stochastic draws: greedy is
            # what the baseline reports, so the selector should at least be
            # able to return it.
            stochastic = index >= len(self.portfolio)

            text = self.tools.generate(
                image,
                prompt,
                max_new_tokens=self.answer_tokens,
                temperature=self.temperature if stochastic else None,
                top_p=self.top_p if stochastic else None,
            )

            answers.append({"mode": mode, "stochastic": stochastic, "text": text})

        return answers

    # ========================================================
    # Scoring
    # ========================================================

    def _score_answer(self, image, answer, question=""):

        """How much of this answer does the image support.

        One point per supported claim, one against per contradicted claim, so
        an answer with nothing checkable scores zero rather than winning by
        vacuum.
        """

        claims = parse_claims(
            lambda prompt, img=None: self.tools.generate(
                img, prompt, max_new_tokens=512
            ),
            answer,
            image=image,
            max_claims=self.max_claims,
            question=self._question,
        )

        verdicts = [self.verify(image, claim) for claim in claims]

        supported = sum(
            1 for v in verdicts if v["verdict"] in ("SUPPORTED", "AMBIGUOUS")
        )
        contradicted = sum(
            1 for v in verdicts if v["verdict"] == "NOT_SUPPORTED"
        )

        return {
            "claims": claims,
            "verdicts": verdicts,
            "supported": supported,
            "contradicted": contradicted,
            "score": supported - contradicted,
        }

    # Verification is shared with the editor pipeline.
    def verify(self, image, claim):

        from .reground import ReGrounding

        if not hasattr(self, "_verifier"):
            self._verifier = ReGrounding(self.tools, max_claims=self.max_claims)

        return self._verifier.verify(image, claim)

    # ========================================================
    # Entry point
    # ========================================================

    def solve(self, image, question):

        record = {"trace": [], "fallback": False}
        trace = record["trace"]

        # ----------------------------------------------------
        # 1. Sample a portfolio of answers
        # ----------------------------------------------------

        answers = self._sample_answers(image, question)

        trace.append({
            "tool": "sample",
            "n": len(answers),
            "modes": [a["mode"] for a in answers],
        })

        # ----------------------------------------------------
        # 2. Verify each answer's claims on the image
        # ----------------------------------------------------

        scored = []

        for index, answer in enumerate(answers):

            self._question = question
            result = self._score_answer(image, answer["text"], question)
            result["index"] = index
            result["text"] = answer["text"]
            result["mode"] = answer["mode"]

            scored.append(result)

            trace.append({
                "tool": "score",
                "index": index,
                "mode": answer["mode"],
                "stochastic": answer["stochastic"],
                "claims": len(result["claims"]),
                "supported": result["supported"],
                "contradicted": result["contradicted"],
                "score": result["score"],
                "answer": answer["text"],
            })

        # ----------------------------------------------------
        # 3. Keep the best supported one
        #
        # Ties go to the earlier sample, which is the greedy one, so the
        # selector can never do worse than the baseline it samples first.
        # ----------------------------------------------------

        best = max(scored, key=lambda item: (item["score"], -item["index"]))

        record["candidates"] = [
            {
                "index": item["index"],
                "mode": item["mode"],
                "supported": item["supported"],
                "contradicted": item["contradicted"],
                "score": item["score"],
            }
            for item in scored
        ]
        record["chosen"] = best["index"]
        record["answer"] = best["text"]
        record["claims"] = best["claims"]
        record["verdicts"] = best["verdicts"]
        record["retracted"] = best["contradicted"]
        record["tool_calls"] = self.tools.calls

        trace.append({
            "tool": "select",
            "chosen": best["index"],
            "score": best["score"],
            "answer": best["text"],
        })

        return record
