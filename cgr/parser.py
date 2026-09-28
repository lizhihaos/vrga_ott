"""Question to constraint graph, using the same Qwen model the pipeline uses.

The parse runs text-only, with the example from the method sketch as the
one-shot demonstration. If the reply cannot be parsed the controller falls
back to answering the question directly, so a bad parse costs accuracy on
one sample rather than crashing the run.
"""

from .schema import extract_json, graph_from_dict

PARSE_PROMPT = """You convert a visual question into a constraint graph in JSON.

Schema:
{
  "target": {"entity": "<what the question asks about>", "attribute": "<which property>"},
  "hops": [
    {
      "entity": "<entity type>",
      "attributes": {"<attribute>": "<value>"},
      "relations": [{"predicate": "<relation>", "entity": "<other entity>", "attributes": {"<attribute>": "<value>"}}],
      "link": "<relation to the previous hop, empty for the first>"
    }
  ]
}

Rules:
- Order the hops from the first entity the question mentions to the last.
- "attributes" holds only what the question states about that entity.
- "relations" holds relations the question states from that entity to another.
- Use short lowercase singular nouns, and no articles: car, man, bag, window.
- If the question names only one entity, produce a single hop.

Example
Question: What color are the windows of the black car next to the man wearing a red shirt who is holding a blue bag?
Answer: {"target": {"entity": "window", "attribute": "color"}, "hops": [{"entity": "man", "attributes": {"shirt_color": "red"}, "relations": [{"predicate": "holding", "entity": "bag", "attributes": {"color": "blue"}}], "link": ""}, {"entity": "car", "attributes": {"color": "black"}, "relations": [], "link": "next_to"}]}

Question: {question}
Answer:"""


def parse_question(generate, question):
    """Ask the model for a constraint graph.

    `generate` is a callable(prompt) -> text, so the parser does not need to
    know how the model is wired up.
    """

    # str.format would treat the JSON braces in the prompt as fields, so the
    # single placeholder is substituted by hand.
    reply = generate(
        PARSE_PROMPT.replace("{question}", question.strip())
    )

    data = extract_json(reply)

    return graph_from_dict(data, question=question, raw=reply)
