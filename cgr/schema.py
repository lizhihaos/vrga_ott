"""Constraint graph structures and the JSON parsing around them.

The graph is deliberately small: an ordered chain of hops, each hop naming
one entity type, the attributes the question pins on it, and the relations
the question pins on it. Multi-hop questions become longer chains.

    "What color are the windows of the black car next to the man wearing a
     red shirt who is holding a blue bag?"

    target = (window, color)
    hops   = [ man(attributes={shirt: red},
                   relations=[holding -> bag(color=blue)]),
               car(attributes={color: black}, link=next_to) ]
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class Attribute:
    """A property the question pins on an entity, e.g. color = black."""

    name: str
    value: str


@dataclass
class Relation:
    """A relation from this hop's entity to another entity.

    The other entity carries its own attribute constraints, which is how
    "the man holding a blue bag" is represented on the man hop.
    """

    predicate: str
    entity: str
    attributes: list[Attribute] = field(default_factory=list)


@dataclass
class Hop:
    """One entity in the chain, with the constraints the question gives it."""

    entity: str
    attributes: list[Attribute] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    # Relation to the previous hop, empty for the first one.
    link: str = ""


@dataclass
class ConstraintGraph:
    """What the question asks for, and the chain that leads to it."""

    target: "Target"
    hops: list[Hop]
    question: str = ""
    raw: str = ""


@dataclass
class Target:
    """The entity whose attribute the question asks about."""

    entity: str
    attribute: str


def _attributes(raw):

    if isinstance(raw, dict):
        return [Attribute(str(k), str(v)) for k, v in raw.items()]

    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                out.append(Attribute(str(item["name"]), str(item.get("value", ""))))
        return out

    return []


def graph_from_dict(data, question="", raw=""):

    target = data.get("target") or {}
    hops = []

    for item in data.get("hops") or []:

        relations = [
            Relation(
                predicate=str(rel.get("predicate", "")),
                entity=str(rel.get("entity", "")),
                attributes=_attributes(rel.get("attributes")),
            )
            for rel in (item.get("relations") or [])
            if isinstance(rel, dict)
        ]

        hops.append(Hop(
            entity=str(item.get("entity", "")),
            attributes=_attributes(item.get("attributes")),
            relations=relations,
            link=str(item.get("link", "")),
        ))

    return ConstraintGraph(
        target=Target(
            entity=str(target.get("entity", "")),
            attribute=str(target.get("attribute", "")),
        ),
        hops=[hop for hop in hops if hop.entity],
        question=question,
        raw=raw,
    )


def extract_json(text):
    """Pull the first JSON object out of a model reply.

    The 3B model wraps its answer in code fences often enough that a plain
    json.loads is not enough.
    """

    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)

    if fence:
        text = fence.group(1).strip()

    start = text.find("{")

    if start == -1:
        raise ValueError("no JSON object in reply")

    depth = 0

    for index in range(start, len(text)):

        if text[index] == "{":
            depth += 1

        elif text[index] == "}":

            depth -= 1

            if depth == 0:
                return json.loads(text[start:index + 1])

    raise ValueError("unbalanced JSON object in reply")
