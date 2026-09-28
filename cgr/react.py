"""Visual ReAct: the reason-act loop with image operations as actions.

ReAct (Yao et al.) interleaves Thought, Action and Observation, with the
model choosing which action to take next and the environment returning what
the action produced. Here the actions are operations on the image instead of
a Wikipedia search:

    locate[cat]                 -> boxes, so the model knows where things are
    crop[2]                     -> the region at box 2, returned as an image
    verify[cat, collar color, red] -> yes or no
    relate[man, holding, bag]   -> yes or no
    count[person]               -> how many there are
    answer[the girl]            -> finish

Two things make this different from the pipelines next door. First, the model
drives: reground.py and select.py decide the step order in Python, while here
the model decides what to look at next. Second, observations can be images:
crop[] returns the pixels the model just located, so the next thought is
formed on a re-perceived region rather than on a remembered one, which is the
visual evidence refresh idea in its most literal form.

The loop depends on the format, so it ships with a worked trajectory in the
prompt. Without it a 3B model stops emitting parsable actions within a couple
of steps; with it the loop is stable, which is also why ReAct itself is a
few-shot method.
"""

import re
import time

DEMO = """Thought: I need to find the object the question is about, then check its property.
Action: locate[hydrant]
Observation: 1 box, the most confident is [120, 300, 480, 900]
Thought: Now I check the property the question asks about on that box.
Action: verify[hydrant, cap color, yellow]
Observation: yes
Thought: The property holds for the located object, so I have the answer.
Action: answer[yellow]
"""

PROMPT = """You answer questions about an image by using tools that look at it.

Tools:
  locate[object]                        -> where every object of that type is, as boxes
  verify[object, attribute, value]      -> yes or no, whether the object has that attribute
  relate[object A, relation, object B]  -> yes or no, whether A is related to B
  count[object]                         -> how many of them there are
  crop[index]                           -> shows you the region at that box index
  answer[text]                          -> finish with the answer

Reply in exactly this format, one Thought and one Action per turn:
Thought: <why you are doing this>
Action: <tool>[<arguments>]

Example
{demo}
Question: {question}

Thought:"""

ACTION = re.compile(
    r"Action:\s*(\w+)\s*\[(.*?)\]",
    re.DOTALL,
)


class React:
    """Thought, action, observation, until the model answers."""

    def __init__(
        self,
        tools,
        max_steps=6,
        answer_tokens=400,
        action_tokens=200,
        max_context_chars=4000,
    ):

        self.tools = tools
        self.max_steps = max_steps
        self.answer_tokens = answer_tokens
        self.action_tokens = action_tokens
        self.max_context_chars = max_context_chars

    # ========================================================
    # Action execution
    # ========================================================

    def _split_args(self, raw):

        return [part.strip() for part in raw.split(",") if part.strip()]

    def execute(self, image, action, state):

        """Run one action, returning the observation text.

        `state` carries the boxes found so far, so a later crop[] or verify[]
        can address them by index, which is what keeps the model from having
        to repeat coordinates back.
        """

        name = action[0].lower()
        args = action[1]

        if name == "locate":

            entity = args[0] if args else ""

            # Locating the same thing twice must not append a second copy of
            # its boxes: the index list grows, the model loses track of which
            # index is which, and it repeats the action until the budget runs
            # out.
            if entity in state["located"]:
                return state["located"][entity]

            boxes = self.tools.locate(image, entity)

            if not boxes:
                state["located"][entity] = f"no {entity} found in the image"
                return state["located"][entity]

            start = len(state["boxes"])
            state["boxes"].extend(
                {"entity": entity, "box": item["box"]} for item in boxes
            )

            listing = ", ".join(
                f"[{index}] {[round(v) for v in item['box']]}"
                for index, item in enumerate(boxes, start)
            )

            state["located"][entity] = f"{len(boxes)} {entity} found: {listing}"

            return state["located"][entity]

        if name == "crop":

            if not args or not args[0].isdigit():
                return "crop needs a box index from a locate action"

            index = int(args[0])

            if index >= len(state["boxes"]):
                return f"box {index} does not exist, locate something first"

            box = state["boxes"][index]["box"]

            state["pending_image"] = self.tools.crop(image, box)

            return (
                f"showing the region at box {index} "
                f"({state['boxes'][index]['entity']}), it is attached as an image"
            )

        if name == "verify":

            if len(args) < 3:
                return "verify needs object, attribute and value"

            entity, attribute, value = args[0], args[1], args[2]

            boxes = [
                item["box"] for item in state["boxes"]
                if item["entity"] == entity
            ] or [item["box"] for item in self.tools.locate(image, entity)]

            if not boxes:
                return f"no {entity} found in the image, so the attribute cannot hold"

            ok = self.tools.verify_attribute(image, boxes[0], entity, attribute, value)

            return "yes" if ok else "no"

        if name == "relate":

            if len(args) < 3:
                return (
                    "relate needs three arguments: object A, relation, object B, "
                    "for example relate[man, holding, bag]"
                )

            if args[2] in ("yes", "no"):
                return (
                    "the third argument must be an object, not yes or no, "
                    "for example relate[man, holding, bag]"
                )

            first, relation, second = args[0], args[1], args[2]

            a_boxes = [i["box"] for i in state["boxes"] if i["entity"] == first] \
                or [i["box"] for i in self.tools.locate(image, first)]
            b_boxes = [i["box"] for i in state["boxes"] if i["entity"] == second] \
                or [i["box"] for i in self.tools.locate(image, second)]

            if not a_boxes or not b_boxes:
                return "at least one of the two objects is not in the image"

            ok = self.tools.relate(
                image, a_boxes[0], first, relation, b_boxes[0], second
            )

            return "yes" if ok else "no"

        if name == "count":

            entity = args[0] if args else ""

            return f"{len(self.tools.locate(image, entity))} {entity}"

        return f"unknown tool {name}"

    # ========================================================
    # Loop
    # ========================================================

    def solve(self, image, question):

        record = {"trace": [], "fallback": False}
        trace = record["trace"]

        context = PROMPT.format(demo=DEMO, question=question.strip())

        if len(context) > self.max_context_chars:
            context = context[-self.max_context_chars:]

        state = {
            "boxes": [],
            "pending_image": None,
            "located": {},
            "last_thought": None,
            "action_counts": {},
        }

        for step in range(self.max_steps):

            # The action follows the thought, and crop[] attaches the region
            # it just showed, so the model reasons on pixels rather than on a
            # remembered description.
            images = [state.pop("pending_image")] if state.get("pending_image") else None

            # A small model imitates the demonstration so literally that it
            # sometimes writes the Observation itself and skips the Action.
            # Generation is cut at the observation boundary, and a reply
            # without an action is re-asked instead of ending the episode:
            # that nudge is what keeps the loop alive.
            match = None

            for attempt in range(3):

                reply = self.tools.generate(
                    image,
                    context,
                    max_new_tokens=self.action_tokens,
                    extra_images=images if attempt == 0 else None,
                    stop_strings=["Observation:"],
                )

                if "Observation:" in reply:
                    reply = reply.split("Observation:")[0].rstrip()

                match = ACTION.search(reply)

                trace.append({
                    "step": step,
                    "attempt": attempt,
                    "reply": reply[:300],
                })

                if match:
                    break

                context += (
                    f"{reply.strip()}\n"
                    f"(Your reply contained no Action. Reply with exactly one "
                    f"Action line now, for example Action: locate[<object>] or "
                    f"Action: answer[<answer>].)\n"
                )

            if not match:

                trace.append({"step": step, "error": "no parsable action"})
                break

            name = match.group(1).lower()
            args = self._split_args(match.group(2))

            if name == "answer":

                answer = match.group(2).strip()

                record["answer"] = answer
                record["steps"] = step + 1
                record["tool_calls"] = self.tools.calls

                trace.append({"step": step, "tool": "answer", "answer": answer})

                return record

            observation = self.execute(image, (name, args), state)

            trace.append({
                "step": step,
                "tool": name,
                "args": args,
                "observation": observation,
            })

            # Feeding a small model its own thought every turn makes it
            # continue that thought instead of acting: the trace collapses
            # into the same sentence repeated until the budget runs out. Only
            # the first occurrence is kept, and a repeated action is answered
            # with a hard instruction rather than a suggestion.
            thought = reply.split("Action:")[0].strip()

            if thought and thought != state.get("last_thought"):
                context += thought + "\n"
                state["last_thought"] = thought

            signature = f"{name}[{','.join(args)}]"

            repeats = state["action_counts"].get(signature, 0) + 1
            state["action_counts"][signature] = repeats

            context += f"\nObservation: {observation}\n"

            if repeats >= 2:

                context += (
                    f"You have already run {signature} {repeats} times and the "
                    f"answer is unchanged. You must not run it again. Reply "
                    f"now with Action: answer[<answer>] using the observations "
                    f"above.\n"
                )

            if len(context) > self.max_context_chars:
                context = context[-self.max_context_chars:]

        # ----------------------------------------------------
        # The loop never finished: fall back to a plain answer so a broken
        # trajectory costs one sample rather than the run.
        # ----------------------------------------------------

        # The observations are already gathered; throwing them away for a
        # fresh chain-of-thought pass is strictly worse than forcing an answer
        # out of what the tools returned.
        record["fallback"] = True

        record["answer"] = self.tools.generate(
            image,
            context
            + "\nThought: I cannot gather more evidence. I answer from the "
              "observations above.\nAction: answer[",
            max_new_tokens=self.answer_tokens,
        )
        record["steps"] = self.max_steps
        record["tool_calls"] = self.tools.calls

        trace.append({"tool": "fallback", "answer": record["answer"]})

        return record
