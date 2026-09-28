"""The controller: candidate sets, constraint filtering, re-grounding.

For every question:

    parse            question -> constraint graph
    hop 0            LOCATE the first entity, keep it as a candidate set
    attribute filter VERIFY_ATTRIBUTE on every candidate, drop the ones that
                     do not match
    relation expand  LOCATE the next entity, keep the ones RELATE accepts
    re-ground        repeat on the image, never on the text history
    target           LOCATE the asked-about entity inside the surviving
                     candidate, then INSPECT its attribute

Candidates are carried as a beam rather than collapsed to one winner, so a
wrong grounding at hop 1 does not end the question at hop 2. Every hop stores
what it looked at, which is what makes the drift analysis possible.
"""

from dataclasses import dataclass, field

from .schema import ConstraintGraph
from .parser import parse_question


@dataclass
class State:
    """One surviving chain of grounded entities."""

    chain: list[dict] = field(default_factory=list)
    score: float = 1.0

    @property
    def last(self):
        return self.chain[-1] if self.chain else None


class Controller:
    """Runs constraint-guided candidate search over one image."""

    def __init__(
        self,
        tools,
        beam_size=3,
        max_calls=48,
        verify_relations=True,
    ):

        self.tools = tools
        self.beam_size = beam_size
        self.max_calls = max_calls
        self.verify_relations = verify_relations

    # ========================================================
    # Chain expansion
    # ========================================================

    def _filter_attributes(self, image, entity, box, attributes, trace):

        """Keep a candidate only if every stated attribute holds on its crop."""

        for attribute in attributes:

            if not attribute.value:
                continue

            passed = self.tools.verify_attribute(
                image, box, entity, attribute.name, attribute.value
            )

            trace.append({
                "tool": "verify_attribute",
                "entity": entity,
                "attribute": attribute.name,
                "value": attribute.value,
                "passed": passed,
            })

            if not passed:
                return False

        return True

    def _filter_relations(self, image, entity, box, relations, trace):

        """Check the relations the question states about this entity.

        Each relation is a second grounding: locate the related entity, keep
        it only if the relation holds, and apply its own attribute
        constraints on the related crop.
        """

        for relation in relations:

            if not relation.predicate or not relation.entity:
                continue

            if not self.verify_relations:
                continue

            related = self.tools.locate(image, relation.entity)
            trace.append({
                "tool": "locate",
                "entity": relation.entity,
                "for": f"{entity}.{relation.predicate}",
                "candidates": len(related),
            })

            accepted = False

            for candidate in related:

                if not self.tools.relate(
                    image,
                    box,
                    entity,
                    relation.predicate,
                    candidate["box"],
                    relation.entity,
                ):
                    continue

                if not self._filter_attributes(
                    image,
                    relation.entity,
                    candidate["box"],
                    relation.attributes,
                    trace,
                ):
                    continue

                accepted = True
                break

            trace.append({
                "tool": "verify_relation",
                "predicate": relation.predicate,
                "entity": relation.entity,
                "passed": accepted,
            })

            if not accepted:
                return False

        return True

    def _expand_hop(self, image, states, hop, trace):

        """One hop: locate the entity, then filter and relate on the image."""

        located = self.tools.locate(image, hop.entity)

        trace.append({
            "hop": hop.entity,
            "link": hop.link,
            "tool": "locate",
            "candidates": len(located),
            "boxes": [item["box"] for item in located],
            "scores": [round(item["score"], 3) for item in located],
        })

        expanded = []

        for state in states:

            for candidate in located:

                box = candidate["box"]

                # The link to the previous hop is checked on the image,
                # which is the re-grounding step.
                if hop.link and state.last is not None:

                    linked = self.tools.relate(
                        image,
                        state.last["box"],
                        state.last["entity"],
                        hop.link,
                        box,
                        hop.entity,
                    )

                    trace.append({
                        "tool": "relate",
                        "from": state.last["entity"],
                        "predicate": hop.link,
                        "to": hop.entity,
                        "passed": linked,
                    })

                    if not linked:
                        continue

                if not self._filter_attributes(
                    image, hop.entity, box, hop.attributes, trace
                ):
                    continue

                if not self._filter_relations(
                    image, hop.entity, box, hop.relations, trace
                ):
                    continue

                expanded.append(State(
                    chain=state.chain + [{
                        "entity": hop.entity,
                        "box": box,
                        "score": candidate["score"],
                        "source": candidate["source"],
                    }],
                    score=state.score * max(candidate["score"], 1e-3),
                ))

        expanded.sort(key=lambda state: -state.score)

        return expanded[:self.beam_size]

    # ========================================================
    # Target and answer
    # ========================================================

    def _target_box(self, image, state, graph):

        """Where to look for the asked-about attribute.

        When the target entity is the last hop (what colour is the car), the
        box is already there. Otherwise the target is localized and kept
        only where it sits with the last hop (the windows of that car).
        """

        if state.last is None:
            return None

        if not graph.target.entity:
            return state.last["box"]

        if graph.target.entity == state.last["entity"]:
            return state.last["box"]

        inside = self.tools.locate(image, graph.target.entity)

        for candidate in inside:

            if self.tools.relate(
                image,
                state.last["box"],
                state.last["entity"],
                "overlap",
                candidate["box"],
                graph.target.entity,
            ):
                return candidate["box"]

        # Nothing inside the grounded object: the chain itself is what the
        # question is about, so inspect the last hop.
        return state.last["box"]

    def _final_answer(self, image, question, states, graph, trace):

        answers = []

        for state in states:

            box = self._target_box(image, state, graph)

            if box is None:
                continue

            answer = self.tools.inspect(image, box, question)

            trace.append({
                "tool": "inspect",
                "box": box,
                "chain": [step["entity"] for step in state.chain],
                "answer": answer,
            })

            answers.append(answer.strip())

        if not answers:
            return None

        if len(set(answers)) == 1:
            return answers[0]

        # Disagreement between candidate chains: let the model arbitrate,
        # which keeps the decision at the answer level instead of the
        # grounding level.
        listing = "\n".join(
            f"{index + 1}. {answer}" for index, answer in enumerate(answers)
        )

        reply = self.tools.generate(
            image,
            f"Question: {question.strip()}\n\n"
            f"Possible answers from different visual readings:\n{listing}\n\n"
            f"Give the single best answer for the question.",
            max_new_tokens=128,
        )

        trace.append({"tool": "arbitrate", "answers": answers, "chosen": reply})

        return reply.strip()

    # ========================================================
    # Entry point
    # ========================================================

    def solve(self, image, question):

        """Answer one question, returning the answer and the full trace."""

        record = {
            "graph": None,
            "trace": [],
            "fallback": False,
            "tool_calls": 0,
        }

        trace = record["trace"]

        # ----------------------------------------------------
        # Parse
        # ----------------------------------------------------

        try:

            graph = parse_question(
                lambda prompt: self.tools.generate(None, prompt, max_new_tokens=256),
                question,
            )

        except (ValueError, KeyError, TypeError) as error:

            trace.append({"tool": "parse", "error": str(error)})

            record["fallback"] = True
            record["answer"] = self.tools.ask_question(image, question)
            record["tool_calls"] = self.tools.calls

            return record

        record["graph"] = {
            "target": {
                "entity": graph.target.entity,
                "attribute": graph.target.attribute,
            },
            "hops": [
                {
                    "entity": hop.entity,
                    "attributes": [
                        [attribute.name, attribute.value]
                        for attribute in hop.attributes
                    ],
                    "relations": [
                        [relation.predicate, relation.entity]
                        for relation in hop.relations
                    ],
                    "link": hop.link,
                }
                for hop in graph.hops
            ],
        }

        # ----------------------------------------------------
        # Candidate search over the hops
        # ----------------------------------------------------

        states = [State()]

        for hop in graph.hops:

            if self.tools.calls > self.max_calls:
                trace.append({"tool": "budget", "stopped_at": hop.entity})
                break

            states = self._expand_hop(image, states, hop, trace)

            if not states:
                break

        record["surviving_chains"] = [
            [step["entity"] for step in state.chain] for state in states
        ]

        # ----------------------------------------------------
        # Answer, with a plain CoT fallback when grounding failed
        # ----------------------------------------------------

        answer = self._final_answer(image, question, states, graph, trace) \
            if states else None

        if not answer:

            record["fallback"] = True
            answer = self.tools.ask_question(image, question)

        record["answer"] = answer
        record["tool_calls"] = self.tools.calls

        return record
