"""The four tools the controller drives.

    LOCATE(entity)                  -> candidate boxes, from GroundingDINO or
                                       from the VLM's own box output
    VERIFY_ATTRIBUTE(box, attr)     -> yes/no, by showing the VLM the crop
    RELATE(box_a, predicate, box_b) -> yes/no, geometry for spatial
                                       predicates and the VLM for the rest
    INSPECT(box, question)          -> the answer, from the crop

Grounding answers where, the VLM answers what and which, and the two never
guess for each other. Every call is counted so a run can be reported against
a token and call budget.
"""

import json
import re

import torch
from PIL import Image

from qwen_eval2.images import load_image
from qwen_eval2.inputs import MAX_PIXELS, inputs_from_content
from qwen_eval2.prompts import render_prompt

SPATIAL = {
    "left_of", "left", "right_of", "right", "above", "below", "near",
    "next_to", "beside", "overlap", "in_front_of",
}

# Prompts that ask for one word back, so parsing stays trivial.
ATTR_PROMPT = (
    "Look at this cropped region of a larger image. "
    "Does this crop show a {entity} whose {attribute} is {value}? "
    "Answer only YES or NO."
)

RELATION_PROMPT = (
    "Look at this cropped region of a larger image. "
    "It contains a {a_entity} and a {b_entity}. "
    "Is the {a_entity} {predicate} the {b_entity}? "
    "Answer only YES or NO."
)

LOCATE_PROMPT = (
    "Localize every {entity} in the image. "
    "Reply with JSON only, no prose: "
    '[{{"box_2d": [y1, x1, y2, x2]}}] '
    "where every coordinate is an integer between 0 and 1000, "
    "measured on a 1000x1000 grid over the image, and y1 < y2, x1 < x2. "
    "If the image contains no {entity}, reply exactly: none"
)

INSPECT_PROMPT = (
    "This is a cropped region of a larger image, showing the object the "
    "question is about.\n\nQuestion: {question}\n\n"
    "Answer the question using only what is visible in this crop."
)


def _yes(text):
    """First YES/NO token, defaulting to NO when the reply is unclear."""

    match = re.search(r"\b(yes|no)\b", text.strip().lower())
    return bool(match and match.group(1) == "yes")


class Tools:
    """Tool layer over one VLM and one grounding model."""

    def __init__(
        self,
        model,
        processor,
        dino=None,
        dino_processor=None,
        locate_backend="dino",
        dino_threshold=0.25,
        dino_text_threshold=0.20,
        max_candidates=6,
    ):

        self.model = model
        self.processor = processor
        self.dino = dino
        self.dino_processor = dino_processor
        self.locate_backend = locate_backend
        self.dino_threshold = dino_threshold
        self.dino_text_threshold = dino_text_threshold
        self.max_candidates = max_candidates

        self.calls = 0
        self.generated_tokens = 0

    # ========================================================
    # Generation helpers
    # ========================================================

    def generate(self, image, prompt, max_new_tokens=256,
                 temperature=None, top_p=None, extra_images=None,
                 stop_strings=None):
        """One VLM call, image optional.

        temperature None keeps it greedy, which is what the verification
        calls want; the selector passes a temperature to draw diverse
        samples to choose from.
        """

        # Datasets hand back raw bytes, path strings or PIL images; normalise
        # here so qwen_vl_utils never sees a bare bytes object.
        image = load_image(image) if image is not None else None

        content = []

        if image is not None:
            content.append({
                "type": "image",
                "image": image,
                "max_pixels": MAX_PIXELS,
            })

        # Observations can be images: a crop the previous action returned is
        # handed back as a second image, so the next thought is formed on the
        # re-perceived region rather than on a remembered description.
        for extra in (extra_images or []):
            content.append({
                "type": "image",
                "image": load_image(extra),
                "max_pixels": MAX_PIXELS,
            })

        content.append({"type": "text", "text": prompt})

        inputs = inputs_from_content(self.processor, self.model, content)

        self.calls += 1

        sampling = temperature is not None and temperature > 0

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=top_p if sampling else None,
                # A small model will happily write the tool's Observation
                # itself and then stop emitting actions, so generation is cut
                # at the boundary instead.
                stop_strings=stop_strings,
                tokenizer=self.processor.tokenizer if stop_strings else None,
            )

        new_ids = output[0][inputs["input_ids"].shape[1]:]

        # Budget accounting: a method that calls the model eleven times has
        # to report the tokens it actually spent, not just its answer cap.
        self.generated_tokens += int(new_ids.shape[0])

        return self.processor.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()

    def ask_question(self, image, question, max_new_tokens=256):
        """The plain-vanilla path, used by the controller as a fallback."""

        prompt, _ = render_prompt("cot", question)

        return self.generate(image, prompt, max_new_tokens=max_new_tokens)

    # ========================================================
    # Boxes
    # ========================================================

    def crop(self, image, box, margin=0.10):
        """Crop a box, with a little context around it."""

        width, height = image.size
        x1, y1, x2, y2 = box

        pad_x = margin * (x2 - x1)
        pad_y = margin * (y2 - y1)

        return image.crop((
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        ))

    def union_box(self, box_a, box_b, margin=0.15):

        x1 = min(box_a[0], box_b[0])
        y1 = min(box_a[1], box_b[1])
        x2 = max(box_a[2], box_b[2])
        y2 = max(box_a[3], box_b[3])

        width = x2 - x1
        height = y2 - y1

        return (
            x1 - margin * width,
            y1 - margin * height,
            x2 + margin * width,
            y2 + margin * height,
        )

    # ========================================================
    # LOCATE
    # ========================================================

    def locate(self, image, entity):
        """Candidate boxes for an entity type, most confident first."""

        if self.locate_backend == "dino" and self.dino is not None:

            boxes = self._locate_dino(image, entity)

        else:

            boxes = self._locate_vlm(image, entity)

        boxes.sort(key=lambda item: -item["score"])

        return boxes[:self.max_candidates]

    def _locate_dino(self, image, entity):
        """GroundingDINO: the reference backend for where.

        The label needs to be lowercase with a trailing period, which is the
        format the model was trained on.
        """

        inputs = self.dino_processor(
            images=image,
            text=entity.lower().strip() + ".",
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.dino(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.dino_threshold,
            text_threshold=self.dino_text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = []

        for box, score in zip(results["boxes"], results["scores"]):

            boxes.append({
                "box": [float(v) for v in box.tolist()],
                "score": float(score),
                "source": "dino",
            })

        return boxes

    def _locate_vlm(self, image, entity):
        """VLM boxes: the fallback when no grounding model is installed.

        Qwen2.5-VL emits boxes on a 0-1000 grid, which is fine for
        candidate generation and keeps the pipeline runnable everywhere.
        """

        reply = self.generate(image, LOCATE_PROMPT.format(entity=entity))

        text = reply.strip()

        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)

        if fence:
            text = fence.group(1).strip()

        start = text.find("[")

        if start == -1:
            return []

        try:
            items = json.loads(text[start:text.rfind("]") + 1])
        except (json.JSONDecodeError, ValueError):
            return []

        width, height = image.size
        boxes = []
        raw_boxes = []

        for item in items:

            coords = item.get("box_2d") if isinstance(item, dict) else None

            if not coords or len(coords) != 4:
                continue

            raw_boxes.append([float(v) for v in coords])

        if not raw_boxes:
            return []

        # The model mixes conventions: sometimes the 0-1000 grid the prompt
        # asks for, sometimes pixel-like values. Anything above 1000 means the
        # reply is not on that grid, so rescale by the largest value seen.
        largest = max(max(box) for box in raw_boxes)
        divisor = largest if largest > 1000 else 1000.0

        for y1, x1, y2, x2 in raw_boxes:

            box = [
                min(max(x1 / divisor * width, 0), width),
                min(max(y1 / divisor * height, 0), height),
                min(max(x2 / divisor * width, 0), width),
                min(max(y2 / divisor * height, 0), height),
            ]

            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue

            boxes.append({"box": box, "score": 1.0, "source": "vlm"})

        return boxes

    # ========================================================
    # VERIFY_ATTRIBUTE
    # ========================================================

    def verify_attribute(self, image, box, entity, attribute, value):
        """Does this box hold the attribute the question pins on it?

        Only the crop is shown, so the model cannot ground the attribute on
        some other object elsewhere in the image.
        """

        crop = self.crop(image, box)

        reply = self.generate(
            crop,
            ATTR_PROMPT.format(
                entity=entity,
                attribute=attribute.replace("_", " "),
                value=value,
            ),
            max_new_tokens=8,
        )

        return _yes(reply)

    # ========================================================
    # RELATE
    # ========================================================

    def relate(self, image, box_a, a_entity, predicate, box_b, b_entity):
        """Is a_entity <predicate> b_entity?

        Spatial predicates come from geometry, which is exact and free;
        semantic ones (holding, wearing, riding) go to the VLM as a yes/no
        question over the union crop.
        """

        predicate = predicate.lower().replace(" ", "_")

        if predicate in SPATIAL:

            return self._relate_geometry(box_a, predicate, box_b)

        crop = self.crop(image, self.union_box(box_a, box_b))

        reply = self.generate(
            crop,
            RELATION_PROMPT.format(
                a_entity=a_entity,
                b_entity=b_entity,
                predicate=predicate.replace("_", " "),
            ),
            max_new_tokens=8,
        )

        return _yes(reply)

    @staticmethod
    def _relate_geometry(box_a, predicate, box_b):

        ax = (box_a[0] + box_a[2]) / 2
        ay = (box_a[1] + box_a[3]) / 2
        bx = (box_b[0] + box_b[2]) / 2
        by = (box_b[1] + box_b[3]) / 2

        if predicate in ("left_of", "left"):
            return ax < bx

        if predicate in ("right_of", "right"):
            return ax > bx

        if predicate == "above":
            return ay < by

        if predicate == "below":
            return ay > by

        if predicate in ("overlap", "in_front_of"):
            return Tools._iou(box_a, box_b) > 0.1

        # "on" and "under" are about vertical contact plus horizontal
        # overlap, which geometry computes exactly. Sending them to the VLM
        # made it judge a suitcase resting on the ground with a brochure
        # sticking out under its edge as "not on", and the ground truth
        # disagreed.
        if predicate in ("on", "on_top_of", "on_top", "under", "beneath",
                         "underneath", "below"):

            # Support means: the upper object sits higher, they overlap
            # horizontally, and the upper one's bottom reaches the lower one's
            # top rather than floating far above it. Bounding boxes only, so
            # "reaches" has to tolerate the upper box crossing the lower one.
            upper, lower = (box_a, box_b) if predicate.startswith("on") else (box_b, box_a)

            span = min(upper[2], lower[2]) - max(upper[0], lower[0])
            overlap = span / max(1e-6, min(upper[2] - upper[0], lower[2] - lower[0]))

            upper_cy = (upper[1] + upper[3]) / 2
            lower_cy = (lower[1] + lower[3]) / 2

            return (
                upper_cy < lower_cy
                and overlap > 0.25
                and upper[3] > lower[1] - 0.35 * (upper[3] - upper[1])
            )

        if predicate in ("inside", "in", "contains"):
            cx = (box_a[0] + box_a[2]) / 2
            cy = (box_a[1] + box_a[3]) / 2
            return box_b[0] <= cx <= box_b[2] and box_b[1] <= cy <= box_b[3]

        if predicate in ("near", "next_to", "beside"):
            # Edge gap rather than centre distance: two boxes of very
            # different sizes can touch while their centres are far apart.
            gap_x = max(0.0, max(box_a[0], box_b[0]) - min(box_a[2], box_b[2]))
            gap_y = max(0.0, max(box_a[1], box_b[1]) - min(box_a[3], box_b[3]))
            gap = max(gap_x, gap_y)
            scale = min(
                (box_a[2] - box_a[0] + box_a[3] - box_a[1]) / 2,
                (box_b[2] - box_b[0] + box_b[3] - box_b[1]) / 2,
            )
            limit = 0.5 if predicate == "next_to" else 1.5
            return gap < limit * scale

        return False

    @staticmethod
    def _iou(box_a, box_b):

        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter / (area_a + area_b - inter + 1e-6)

    # ========================================================
    # INSPECT
    # ========================================================

    def inspect(self, image, box, question, max_new_tokens=200):
        """Answer the question from a crop of the grounded object."""

        crop = self.crop(image, box, margin=0.05)

        reply = self.generate(
            crop,
            INSPECT_PROMPT.format(question=question.strip()),
            max_new_tokens=max_new_tokens,
        )

        return reply
