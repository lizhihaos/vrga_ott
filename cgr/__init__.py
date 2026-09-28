"""Constraint-Guided Visual Re-grounding (CG-VR).

A question becomes a constraint graph, the graph drives a candidate-set
search over grounded entities, and every hop re-grounds on the image instead
of trusting the textual reasoning history:

    question
       -> constraint graph          (parser.py)
       -> candidate sets            (controller.py)
       -> attribute filtering       (tools.py, VERIFY_ATTRIBUTE)
       -> relation expansion        (tools.py, RELATE)
       -> step-wise re-grounding    (controller.py)
       -> answer                    (tools.py, INSPECT)

Nothing here trains anything: the grounding tool answers where, the VLM
answers what and which, and a Python controller keeps the search honest.
"""

from .schema import Attribute, ConstraintGraph, Hop, Relation, Target
from .parser import parse_question
from .claims import parse_claims
from .reground import ReGrounding
from .select import Selector

__all__ = [
    "Attribute",
    "ConstraintGraph",
    "Hop",
    "Relation",
    "Target",
    "parse_question",
    "parse_claims",
    "ReGrounding",
    "Selector",
]
