#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VQA Semantic Evaluation with ORI / CoT
=======================================

使用 DeepSeek API 对 VQA 模型的 ORI 和 CoT 回答进行评价。

评价指标：

1. Correctness A
----------------
A ∈ {0, 1}

A = 1:
    最终答案正确

A = 0:
    最终答案错误


2. Irrelevance Degree I
-----------------------
I ∈ [0, 1]

I = 0:
    没有明显无关、冗余、偏题内容

I 越大：
    无关、冗余、偏题内容越严重


3. Comprehensive Score S
------------------------
S = A * (1 - alpha * I)

其中：

    alpha ∈ [0, 1]

用于控制无关内容的惩罚强度。


支持：

    ori_response
    think_response

输入示例：

{
    "id": 0,
    "question": "What color is the object?",
    "ori_response": ["<answer>Yellow</answer>"],
    "think_response": [
        "Let's think step by step. "
        "The object appears yellow. "
        "<answer>Yellow</answer>"
    ],
    "answer": "Yellow",
    "type": "attribute"
}


输出：

{
    "id": 0,
    "question": "...",
    "answer": "Yellow",

    "ori_response": "...",
    "ori_A": 1,
    "ori_I": 0.0,
    "ori_S": 1.0,

    "cot_response": "...",
    "cot_A": 1,
    "cot_I": 0.12,
    "cot_S": 0.94
}


本版本：

    - 不使用缓存
    - 不读取历史 judgments.jsonl
    - 不读取 cache
    - 不保存 cache
    - 每条样本重新调用 DeepSeek API
    - 每次运行重新生成结果
"""


import os
import re
import json
import time
import argparse

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ============================================================
# Configuration
# ============================================================

API_URL = "https://api.deepseek.com/chat/completions"

DEFAULT_MODEL = "deepseek-chat"

MAX_RETRIES = 3

REQUEST_TIMEOUT = 120

DEFAULT_ALPHA = 0.5


# ============================================================
# Utils
# ============================================================

def clean_answer(text):
    """
    清理模型回答。

    支持：

        <answer>Yellow</answer>
        -> Yellow

        <answer>Yellow
        -> Yellow

    如果是 CoT：

        Let's think...
        <answer>Yellow</answer>

    则尽量提取 <answer> 中的最终答案。
    """

    if text is None:
        return ""

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    if isinstance(text, list):

        if len(text) == 0:
            return ""

        text = text[-1]

    text = str(text).strip()

    # --------------------------------------------------------
    # 优先提取 <answer>...</answer>
    # --------------------------------------------------------

    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:

        return match.group(1).strip()

    # --------------------------------------------------------
    # <answer>...
    # --------------------------------------------------------

    match = re.search(
        r"<answer>\s*(.*?)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:

        return match.group(1).strip()

    return text


def get_full_response(row, field):
    """
    获取完整模型回答。

    不做 <answer> 截断。

    因为：

        ORI:
            主要用于最终答案

        CoT:
            需要保留完整 reasoning，
            用于评价 Irrelevance Degree。
    """

    if field not in row:

        return ""

    response = row[field]

    if isinstance(response, list):

        if len(response) == 0:
            return ""

        response = response[-1]

    if response is None:
        return ""

    return str(response).strip()


def get_final_answer(response):
    """
    从完整 response 中提取最终答案。

    对 CoT：

        reasoning...
        <answer>Yellow</answer>

    -> Yellow
    """

    if not response:
        return ""

    return clean_answer(response)


def get_reference_answer(row):
    """
    获取 ground-truth answer。
    """

    if "answer" in row:

        answer = row["answer"]

        if isinstance(answer, list):

            if len(answer) == 0:
                return ""

            return " / ".join(
                str(x).strip()
                for x in answer
            )

        return str(answer).strip()

    if "gt" in row:

        return str(
            row["gt"]
        ).strip()

    if "ground_truth" in row:

        return str(
            row["ground_truth"]
        ).strip()

    return ""


def load_jsonl(path):
    """
    读取 JSONL。

    支持：

        ori_response
        think_response
        prediction
        response
        model_response
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    records = []

    seen_ids = set()

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(
            f,
            1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_no}: {e}"
                )

            # ------------------------------------------------
            # ID
            # ------------------------------------------------

            if "id" not in row:

                raise ValueError(
                    f"Missing id at line "
                    f"{line_no}"
                )

            rid = row["id"]

            if rid in seen_ids:

                raise ValueError(
                    f"Duplicate id: {rid}"
                )

            seen_ids.add(rid)

            # ------------------------------------------------
            # Question
            # ------------------------------------------------

            question = str(
                row.get(
                    "question",
                    ""
                )
            ).strip()

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            answer = get_reference_answer(
                row
            )

            # ------------------------------------------------
            # ORI response
            # ------------------------------------------------

            ori_response = ""

            if "ori_response" in row:

                ori_response = get_full_response(
                    row,
                    "ori_response"
                )

            elif "prediction" in row:

                ori_response = get_full_response(
                    row,
                    "prediction"
                )

            elif "response" in row:

                ori_response = get_full_response(
                    row,
                    "response"
                )

            elif "model_response" in row:

                ori_response = get_full_response(
                    row,
                    "model_response"
                )

            # ------------------------------------------------
            # CoT response
            # ------------------------------------------------

            think_response = ""

            if "think_response" in row:

                think_response = get_full_response(
                    row,
                    "think_response"
                )

            # ------------------------------------------------
            # Type
            # ------------------------------------------------

            typ = str(
                row.get(
                    "type",
                    "unknown"
                )
            ).strip()

            record = {

                "id": rid,

                "question": question,

                "answer": answer,

                "ori_response": ori_response,

                "think_response": think_response,

                "type": typ,
            }

            records.append(record)

    return records


# ============================================================
# Prompt
# ============================================================

JUDGE_SYSTEM_PROMPT = r"""
You are a strict evaluator for a Visual Question Answering benchmark.

You will evaluate a model response using three metrics:

1. Correctness A
2. Irrelevance Degree I
3. Comprehensive Score S

The reference answer is the ground-truth semantic target.


============================================================
PART 1: CORRECTNESS A
============================================================

A is binary:

    A = 1
    A = 0

A = 1 if the FINAL ANSWER correctly answers the question.

A = 0 if the FINAL ANSWER is incorrect.

Important:

For a chain-of-thought response, judge correctness primarily from
the final answer, not from whether the reasoning sounds plausible.

Do NOT mark an answer correct merely because the reasoning contains
some correct observations.

The final answer must correspond to the reference answer.


Rules:

1. Semantic equivalence is correct.

Different wording is acceptable if it expresses the same answer.

2. Concise answers are acceptable.

The model does not need to reproduce the reference answer word-for-word.

3. Do not require exact string matching.

Judge meaning rather than surface form.

4. Counting questions:

The number must be correct.

5. Attribute questions:

Relevant attributes such as color, shape, size, identity,
gender, material, etc. must be correct.

6. Spatial questions:

Relationships such as left/right, above/below,
front/behind, inside/outside, near/far, etc. must be correct.

7. Object relationship questions:

The relationship between the relevant objects must be correct.

8. Yes/no questions:

The response must correspond to the correct yes/no answer.

9. Multiple-choice questions:

The selected option must correspond to the reference answer.

If both option letter and textual answer are given,
evaluate whether they correspond to the reference.

10. Adversarial questions:

Carefully check whether the premise and requested object/event
are supported by the reference answer.

11. A direct contradiction to the reference answer is incorrect.

12. Extra details are allowed only when they do not introduce
substantive factual errors relevant to the question.

13. Do not reward an answer merely because part of it is correct.

If a substantive contradiction changes the answer,
mark it incorrect.

14. Descriptive questions:

Minor wording differences or omission of unimportant details
are acceptable.

Significant fabricated or contradictory details are incorrect.

15. Do not use outside knowledge to override the reference answer.

16. The QUESTION is the primary criterion.

The REFERENCE ANSWER defines the semantic target.

17. Do not judge based only on keyword overlap.

18. If the model response is clearly equivalent to the reference answer,
mark it correct.

19. If the response contains uncertainty but ultimately gives the
correct answer, it can still be correct unless the uncertainty
changes the answer.

20. If the response gives multiple conflicting final answers and
there is no clear final answer, mark it incorrect.


============================================================
PART 2: IRRELEVANCE DEGREE I
============================================================

I measures how much of the model response is irrelevant,
unnecessary, off-topic, or redundant relative to the question.

I is a continuous value:

    0 <= I <= 1


I = 0:

The response is concise and contains essentially no irrelevant content.

I close to 0:

The response contains very little unnecessary content.

I around 0.5:

A substantial portion of the response is unnecessary,
but the response still mainly addresses the question.

I close to 1:

The response is dominated by irrelevant, off-topic,
rambling, or unrelated content.


Important:

Do NOT penalize necessary reasoning.

Do NOT treat correct explanations as irrelevant simply because
they are longer.

Do NOT penalize normal short reasoning needed to answer the question.

Only content that is unnecessary, repetitive, unrelated,
or substantially off-topic should increase I.


For CoT responses:

Reasoning itself is NOT automatically irrelevant.

Evaluate whether the reasoning contributes to answering the question.

Examples:

Question:
"What color is the ball?"

Response:
"The ball is red. <answer>Red</answer>"

I should be close to 0.

Response:
"Let's inspect the image. The ball appears red.
Therefore the answer is red. <answer>Red</answer>"

I should still be low.

Response:
"The ball appears red. I will now discuss football history,
different types of sports, famous players, and weather conditions.
<answer>Red</answer>"

I should be substantially higher.

For a direct-answer response, a short answer should generally
have low I.

Do not penalize a concise correct answer for lacking explanation.


============================================================
PART 3: COMPREHENSIVE SCORE
============================================================

The comprehensive score is:

    S = A * (1 - alpha * I)

The value of alpha is provided separately.

If:

    A = 0

then:

    S = 0

regardless of I.

If:

    A = 1

then:

    S = 1 - alpha * I


============================================================
IMPORTANT EVALUATION PRINCIPLE
============================================================

Correctness and irrelevance are DIFFERENT dimensions.

A response can be:

    correct + concise
    correct + verbose
    correct + irrelevant
    incorrect + concise
    incorrect + verbose

Do not let verbosity automatically make an answer incorrect.

Do not let a correct final answer automatically make I = 0.

Evaluate A and I independently.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not return Markdown.

Do not return additional fields.

Format:

{
  "A": 0 or 1,
  "I": 0.0,
  "reason_A": "brief explanation of correctness",
  "reason_I": "brief explanation of irrelevance"
}

I must be a number between 0 and 1.

Do not calculate S yourself.
The program will calculate S.
"""


def build_user_prompt(
    question,
    reference_answer,
    model_response,
    alpha
):

    return f"""
Question:
{question}

Reference answer:
{reference_answer}

Model response:
{model_response}

Alpha:
{alpha}

Evaluate the model response using:

A = correctness of the final answer
I = degree of irrelevant / unnecessary / off-topic content

Return JSON only:

{{
  "A": 0 or 1,
  "I": 0.0,
  "reason_A": "brief explanation",
  "reason_I": "brief explanation"
}}
"""


# ============================================================
# Validation
# ============================================================

def validate_judgment(data):

    if not isinstance(data, dict):

        raise ValueError(
            "Judge result must be a JSON object"
        )

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    A = data.get("A")

    if isinstance(A, bool):

        A = int(A)

    if isinstance(A, str):

        try:
            A = int(A.strip())
        except Exception:

            raise ValueError(
                f"Invalid A: {A!r}"
            )

    if A not in (0, 1):

        raise ValueError(
            f"Invalid A: {A!r}"
        )

    # --------------------------------------------------------
    # I
    # --------------------------------------------------------

    I = data.get("I")

    try:

        I = float(I)

    except Exception:

        raise ValueError(
            f"Invalid I: {I!r}"
        )

    if I < 0:

        I = 0.0

    if I > 1:

        I = 1.0

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reason_A = data.get(
        "reason_A",
        ""
    )

    reason_I = data.get(
        "reason_I",
        ""
    )

    if not isinstance(
        reason_A,
        str
    ):

        reason_A = str(
            reason_A
        )

    if not isinstance(
        reason_I,
        str
    ):

        reason_I = str(
            reason_I
        )

    return {

        "A": A,

        "I": round(
            I,
            6
        ),

        "reason_A":
            reason_A.strip(),

        "reason_I":
            reason_I.strip(),
    }


# ============================================================
# DeepSeek Judge
# ============================================================

class JudgeError(Exception):

    def __init__(
        self,
        message,
        fatal=False
    ):

        super().__init__(message)

        self.fatal = fatal


class DeepSeekJudge:

    def __init__(
        self,
        api_key,
        model=DEFAULT_MODEL,
        retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT,
        alpha=DEFAULT_ALPHA,
    ):

        self.api_key = api_key

        self.model = model

        self.retries = retries

        self.timeout = timeout

        self.alpha = alpha

    def _request(
        self,
        question,
        reference_answer,
        model_response
    ):

        payload = {

            "model": self.model,

            "messages": [

                {
                    "role":
                        "system",

                    "content":
                        JUDGE_SYSTEM_PROMPT
                },

                {
                    "role":
                        "user",

                    "content":
                        build_user_prompt(
                            question,
                            reference_answer,
                            model_response,
                            self.alpha
                        )
                }

            ],

            "temperature": 0,

            "response_format": {
                "type": "json_object"
            },

            "thinking": {
                "type": "disabled"
            }
        }

        data = json.dumps(
            payload,
            ensure_ascii=False
        ).encode(
            "utf-8"
        )

        request = Request(

            API_URL,

            data=data,

            headers={

                "Content-Type":
                    "application/json",

                "Authorization":
                    f"Bearer {self.api_key}",
            },

            method="POST"
        )

        # ----------------------------------------------------
        # Request
        # ----------------------------------------------------

        try:

            with urlopen(
                request,
                timeout=self.timeout
            ) as response:

                raw = (
                    response
                    .read()
                    .decode("utf-8")
                )

        except HTTPError as e:

            if e.code in (
                400,
                401,
                403
            ):

                raise JudgeError(
                    f"HTTP {e.code}: "
                    f"{e.reason}",
                    fatal=True
                )

            raise JudgeError(
                f"HTTP {e.code}: "
                f"{e.reason}"
            )

        except URLError as e:

            raise JudgeError(
                f"Network error: {e}"
            )

        except Exception as e:

            raise JudgeError(
                f"Request failed: {e}"
            )

        # ----------------------------------------------------
        # API JSON
        # ----------------------------------------------------

        try:

            response_json = json.loads(
                raw
            )

        except json.JSONDecodeError:

            raise JudgeError(
                "Invalid API JSON response"
            )

        # ----------------------------------------------------
        # Get content
        # ----------------------------------------------------

        try:

            choice = (
                response_json
                ["choices"]
                [0]
            )

            finish_reason = (
                choice.get(
                    "finish_reason"
                )
            )

            if finish_reason == "length":

                raise JudgeError(
                    "API response truncated"
                )

            content = (
                choice
                ["message"]
                ["content"]
            )

        except Exception as e:

            if isinstance(
                e,
                JudgeError
            ):

                raise

            raise JudgeError(
                "Invalid API response "
                f"structure: {e}"
            )

        # ----------------------------------------------------
        # Parse judge JSON
        # ----------------------------------------------------

        try:

            judgment = json.loads(
                content
            )

        except json.JSONDecodeError:

            content = content.strip()

            # Remove Markdown code block

            if content.startswith(
                "```"
            ):

                content = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    content,
                    flags=re.IGNORECASE
                )

                content = re.sub(
                    r"\s*```$",
                    "",
                    content
                )

            try:

                judgment = json.loads(
                    content
                )

            except json.JSONDecodeError:

                raise JudgeError(
                    "Judge returned "
                    "invalid JSON"
                )

        return validate_judgment(
            judgment
        )

    def judge(
        self,
        question,
        reference_answer,
        model_response
    ):

        """
        完全无缓存。

        每次调用都会重新请求 DeepSeek。
        """

        last_error = None

        for attempt in range(
            self.retries + 1
        ):

            try:

                judgment = self._request(

                    question,

                    reference_answer,

                    model_response
                )

                judgment["cached"] = False

                return judgment

            except JudgeError as e:

                last_error = e

                if e.fatal:

                    raise

                if attempt < self.retries:

                    wait = min(
                        2 ** attempt,
                        10
                    )

                    print(
                        f"    Retry "
                        f"{attempt + 1}/"
                        f"{self.retries} "
                        f"after {wait}s..."
                    )

                    time.sleep(
                        wait
                    )

        raise last_error


# ============================================================
# Score
# ============================================================

def calculate_score(
    A,
    I,
    alpha
):

    """
    S = A * (1 - alpha * I)
    """

    S = A * (
        1.0 -
        alpha * I
    )

    # 数值保护

    S = max(
        0.0,
        min(
            1.0,
            S
        )
    )

    return round(
        S,
        6
    )


# ============================================================
# Aggregate Statistics
# ============================================================

def aggregate_metric(
    judgments,
    prefix
):

    """
    prefix:

        ori
        cot
    """

    A_values = []

    I_values = []

    S_values = []

    for item in judgments:

        A_key = f"{prefix}_A"

        I_key = f"{prefix}_I"

        S_key = f"{prefix}_S"

        if (
            A_key in item
            and
            I_key in item
            and
            S_key in item
        ):

            A_values.append(
                item[A_key]
            )

            I_values.append(
                item[I_key]
            )

            S_values.append(
                item[S_key]
            )

    if len(A_values) == 0:

        return {

            "evaluated":
                0,

            "A":
                None,

            "I":
                None,

            "S":
                None,
        }

    return {

        "evaluated":
            len(A_values),

        "A":
            round(
                sum(A_values)
                /
                len(A_values),
                6
            ),

        "I":
            round(
                sum(I_values)
                /
                len(I_values),
                6
            ),

        "S":
            round(
                sum(S_values)
                /
                len(S_values),
                6
            ),
    }


def build_summary(
    records,
    judged,
    api_error_ids,
    alpha
):

    total = len(records)

    record_map = {
        r["id"]: r
        for r in records
    }

    # ========================================================
    # ORI
    # ========================================================

    ori_stats = aggregate_metric(
        judged,
        "ori"
    )

    # ========================================================
    # CoT
    # ========================================================

    cot_stats = aggregate_metric(
        judged,
        "cot"
    )

    # ========================================================
    # Differences
    # ========================================================

    comparison = {}

    if (
        ori_stats["A"] is not None
        and
        cot_stats["A"] is not None
    ):

        comparison["delta_A"] = round(
            cot_stats["A"]
            -
            ori_stats["A"],
            6
        )

        comparison["delta_I"] = round(
            cot_stats["I"]
            -
            ori_stats["I"],
            6
        )

        comparison["delta_S"] = round(
            cot_stats["S"]
            -
            ori_stats["S"],
            6
        )

    else:

        comparison = {

            "delta_A":
                None,

            "delta_I":
                None,

            "delta_S":
                None,
        }

    # ========================================================
    # CoT-induced errors
    # ========================================================

    ori_correct = 0

    ori_incorrect = 0

    cot_correct = 0

    cot_incorrect = 0

    cot_induced_error = 0

    cot_recovery = 0

    paired_count = 0

    for item in judged:

        ori_A = item.get(
            "ori_A"
        )

        cot_A = item.get(
            "cot_A"
        )

        if (
            ori_A not in (0, 1)
            or
            cot_A not in (0, 1)
        ):

            continue

        paired_count += 1

        if ori_A == 1:

            ori_correct += 1

        else:

            ori_incorrect += 1

        if cot_A == 1:

            cot_correct += 1

        else:

            cot_incorrect += 1

        # ----------------------------------------------------
        # No-CoT correct -> CoT incorrect
        # ----------------------------------------------------

        if (
            ori_A == 1
            and
            cot_A == 0
        ):

            cot_induced_error += 1

        # ----------------------------------------------------
        # No-CoT incorrect -> CoT correct
        # ----------------------------------------------------

        if (
            ori_A == 0
            and
            cot_A == 1
        ):

            cot_recovery += 1

    if ori_correct > 0:

        cot_error_rate = (
            cot_induced_error
            /
            ori_correct
        )

    else:

        cot_error_rate = None

    if ori_incorrect > 0:

        cot_recovery_rate = (
            cot_recovery
            /
            ori_incorrect
        )

    else:

        cot_recovery_rate = None

    # ========================================================
    # By type
    # ========================================================

    by_type = {}

    for record in records:

        typ = record["type"]

        if typ not in by_type:

            by_type[typ] = {

                "total":
                    0,

                "ori_evaluated":
                    0,

                "cot_evaluated":
                    0,

                "ori_A":
                    [],

                "ori_I":
                    [],

                "ori_S":
                    [],

                "cot_A":
                    [],

                "cot_I":
                    [],

                "cot_S":
                    [],
            }

        by_type[typ]["total"] += 1

    for item in judged:

        rid = item["id"]

        if rid not in record_map:

            continue

        typ = record_map[
            rid
        ]["type"]

        stat = by_type[typ]

        # ----------------------------------------------------
        # ORI
        # ----------------------------------------------------

        if (
            "ori_A" in item
            and
            "ori_I" in item
            and
            "ori_S" in item
        ):

            stat[
                "ori_evaluated"
            ] += 1

            stat[
                "ori_A"
            ].append(
                item["ori_A"]
            )

            stat[
                "ori_I"
            ].append(
                item["ori_I"]
            )

            stat[
                "ori_S"
            ].append(
                item["ori_S"]
            )

        # ----------------------------------------------------
        # CoT
        # ----------------------------------------------------

        if (
            "cot_A" in item
            and
            "cot_I" in item
            and
            "cot_S" in item
        ):

            stat[
                "cot_evaluated"
            ] += 1

            stat[
                "cot_A"
            ].append(
                item["cot_A"]
            )

            stat[
                "cot_I"
            ].append(
                item["cot_I"]
            )

            stat[
                "cot_S"
            ].append(
                item["cot_S"]
            )

    # --------------------------------------------------------
    # Convert lists -> means
    # --------------------------------------------------------

    for typ, stat in by_type.items():

        for prefix in (
            "ori",
            "cot"
        ):

            for metric in (
                "A",
                "I",
                "S"
            ):

                key = (
                    f"{prefix}_{metric}"
                )

                values = stat[key]

                if len(values) > 0:

                    stat[key] = round(
                        sum(values)
                        /
                        len(values),
                        6
                    )

                else:

                    stat[key] = None

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        if (
            stat["ori_A"] is not None
            and
            stat["cot_A"] is not None
        ):

            stat["delta_A"] = round(
                stat["cot_A"]
                -
                stat["ori_A"],
                6
            )

            stat["delta_I"] = round(
                stat["cot_I"]
                -
                stat["ori_I"],
                6
            )

            stat["delta_S"] = round(
                stat["cot_S"]
                -
                stat["ori_S"],
                6
            )

        else:

            stat["delta_A"] = None

            stat["delta_I"] = None

            stat["delta_S"] = None

    # ========================================================
    # Incorrect IDs
    # ========================================================

    ori_incorrect_ids = [

        x["id"]

        for x in judged

        if x.get("ori_A") == 0
    ]

    cot_incorrect_ids = [

        x["id"]

        for x in judged

        if x.get("cot_A") == 0
    ]

    # ========================================================
    # CoT-induced error IDs
    # ========================================================

    cot_induced_error_ids = [

        x["id"]

        for x in judged

        if (
            x.get("ori_A") == 1
            and
            x.get("cot_A") == 0
        )
    ]

    # ========================================================
    # Recovery IDs
    # ========================================================

    cot_recovery_ids = [

        x["id"]

        for x in judged

        if (
            x.get("ori_A") == 0
            and
            x.get("cot_A") == 1
        )
    ]

    # ========================================================
    # API errors
    # ========================================================

    api_error_ids = sorted(
        api_error_ids
    )

    not_processed_ids = [

        r["id"]

        for r in records

        if r["id"]
        in api_error_ids
    ]

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "total":
            total,

        "evaluated":
            len(judged),

        "api_errors":
            len(api_error_ids),

        "not_processed":
            len(not_processed_ids),

        "complete":
            (
                len(api_error_ids) == 0
                and
                len(judged) == total
            ),

        "alpha":
            alpha,

        "ori":
            ori_stats,

        "cot":
            cot_stats,

        "comparison":
            comparison,

        "cot_error_analysis": {

            "paired":
                paired_count,

            "ori_correct":
                ori_correct,

            "ori_incorrect":
                ori_incorrect,

            "cot_correct":
                cot_correct,

            "cot_incorrect":
                cot_incorrect,

            "cot_induced_error":
                cot_induced_error,

            "cot_induced_error_rate":
                (
                    round(
                        cot_error_rate,
                        6
                    )
                    if cot_error_rate
                    is not None
                    else None
                ),

            "cot_recovery":
                cot_recovery,

            "cot_recovery_rate":
                (
                    round(
                        cot_recovery_rate,
                        6
                    )
                    if cot_recovery_rate
                    is not None
                    else None
                ),
        },

        "ori_incorrect_ids":
            ori_incorrect_ids,

        "cot_incorrect_ids":
            cot_incorrect_ids,

        "cot_induced_error_ids":
            cot_induced_error_ids,

        "cot_recovery_ids":
            cot_recovery_ids,

        "api_error_ids":
            api_error_ids,

        "not_processed_ids":
            not_processed_ids,

        "by_type":
            dict(
                sorted(
                    by_type.items()
                )
            ),
    }

    return summary


# ============================================================
# Output
# ============================================================

def save_json(
    path,
    data
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def append_jsonl(
    path,
    data
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(

            json.dumps(
                data,
                ensure_ascii=False
            )

            + "\n"
        )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    records,
    judge,
    output_dir,
    run_name="full",
    limit=None
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if limit is not None:

        records = records[
            :limit
        ]

    judgments_path = (
        output_dir
        /
        "judgments.jsonl"
    )

    errors_path = (
        output_dir
        /
        "api_errors.jsonl"
    )

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    # ========================================================
    # 完全重新开始
    # ========================================================

    if judgments_path.exists():

        judgments_path.unlink()

    if errors_path.exists():

        errors_path.unlink()

    if summary_path.exists():

        summary_path.unlink()

    errors_path.write_text(
        "",
        encoding="utf-8"
    )

    judged = []

    api_error_ids = []

    # ========================================================
    # Header
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "VQA ORI / CoT Semantic Evaluation"
    )

    print(
        "=" * 80
    )

    print(
        f"Model       : "
        f"{judge.model}"
    )

    print(
        f"Samples     : "
        f"{len(records)}"
    )

    print(
        f"Alpha       : "
        f"{judge.alpha}"
    )

    print(
        f"Run         : "
        f"{run_name}"
    )

    print(
        "Cache       : DISABLED"
    )

    print(
        "Evaluation   : ORI + CoT"
    )

    print(
        "=" * 80
    )

    print()

    # ========================================================
    # Evaluation loop
    # ========================================================

    for index, record in enumerate(
        records,
        1
    ):

        rid = record["id"]

        print(
            f"[{index}/{len(records)}] "
            f"id={rid}"
        )

        # ----------------------------------------------------
        # ORI
        # ----------------------------------------------------

        ori_response = record[
            "ori_response"
        ]

        if not ori_response:

            print(
                "    ORI -> EMPTY"
            )

            ori_result = None

        else:

            try:

                ori_result = judge.judge(

                    question=
                        record["question"],

                    reference_answer=
                        record["answer"],

                    model_response=
                        ori_response
                )

                print(
                    f"    ORI -> "
                    f"A={ori_result['A']} "
                    f"I={ori_result['I']:.4f}"
                )

            except JudgeError as e:

                api_error_ids.append(
                    rid
                )

                append_jsonl(
                    errors_path,
                    {
                        "id":
                            rid,

                        "stage":
                            "ori",

                        "error":
                            str(e),

                        "fatal":
                            e.fatal,
                    }
                )

                print(
                    f"    ORI -> "
                    f"API ERROR: {e}"
                )

                if e.fatal:

                    summary = build_summary(
                        records,
                        judged,
                        api_error_ids,
                        judge.alpha
                    )

                    save_json(
                        summary_path,
                        summary
                    )

                    raise

                ori_result = None

        # ----------------------------------------------------
        # CoT
        # ----------------------------------------------------

        think_response = record[
            "think_response"
        ]

        if not think_response:

            print(
                "    CoT -> EMPTY"
            )

            cot_result = None

        else:

            try:

                cot_result = judge.judge(

                    question=
                        record["question"],

                    reference_answer=
                        record["answer"],

                    model_response=
                        think_response
                )

                print(
                    f"    CoT -> "
                    f"A={cot_result['A']} "
                    f"I={cot_result['I']:.4f}"
                )

            except JudgeError as e:

                api_error_ids.append(
                    rid
                )

                append_jsonl(
                    errors_path,
                    {
                        "id":
                            rid,

                        "stage":
                            "cot",

                        "error":
                            str(e),

                        "fatal":
                            e.fatal,
                    }
                )

                print(
                    f"    CoT -> "
                    f"API ERROR: {e}"
                )

                if e.fatal:

                    summary = build_summary(
                        records,
                        judged,
                        api_error_ids,
                        judge.alpha
                    )

                    save_json(
                        summary_path,
                        summary
                    )

                    raise

                cot_result = None

        # ----------------------------------------------------
        # Construct output
        # ----------------------------------------------------

        output = {

            "id":
                rid,

            "question":
                record["question"],

            "answer":
                record["answer"],

            "type":
                record["type"],

            "ori_response":
                ori_response,

            "think_response":
                think_response,
        }

        # ====================================================
        # ORI metrics
        # ====================================================

        if ori_result is not None:

            ori_A = ori_result[
                "A"
            ]

            ori_I = ori_result[
                "I"
            ]

            ori_S = calculate_score(
                ori_A,
                ori_I,
                judge.alpha
            )

            output[
                "ori_A"
            ] = ori_A

            output[
                "ori_I"
            ] = ori_I

            output[
                "ori_S"
            ] = ori_S

            output[
                "ori_reason_A"
            ] = ori_result[
                "reason_A"
            ]

            output[
                "ori_reason_I"
            ] = ori_result[
                "reason_I"
            ]

        # ====================================================
        # CoT metrics
        # ====================================================

        if cot_result is not None:

            cot_A = cot_result[
                "A"
            ]

            cot_I = cot_result[
                "I"
            ]

            cot_S = calculate_score(
                cot_A,
                cot_I,
                judge.alpha
            )

            output[
                "cot_A"
            ] = cot_A

            output[
                "cot_I"
            ] = cot_I

            output[
                "cot_S"
            ] = cot_S

            output[
                "cot_reason_A"
            ] = cot_result[
                "reason_A"
            ]

            output[
                "cot_reason_I"
            ] = cot_result[
                "reason_I"
            ]

        # ====================================================
        # Comparison
        # ====================================================

        if (
            "ori_A" in output
            and
            "cot_A" in output
        ):

            output[
                "delta_A"
            ] = round(
                output["cot_A"]
                -
                output["ori_A"],
                6
            )

            output[
                "delta_I"
            ] = round(
                output["cot_I"]
                -
                output["ori_I"],
                6
            )

            output[
                "delta_S"
            ] = round(
                output["cot_S"]
                -
                output["ori_S"],
                6
            )

            # ------------------------------------------------
            # CoT-induced error
            # ------------------------------------------------

            output[
                "cot_induced_error"
            ] = (
                output["ori_A"] == 1
                and
                output["cot_A"] == 0
            )

            # ------------------------------------------------
            # CoT recovery
            # ------------------------------------------------

            output[
                "cot_recovery"
            ] = (
                output["ori_A"] == 0
                and
                output["cot_A"] == 1
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        append_jsonl(
            judgments_path,
            output
        )

        judged.append(
            output
        )

        print(
            f"    saved"
        )

        print()

    # ========================================================
    # Summary
    # ========================================================

    summary = build_summary(

        records,

        judged,

        api_error_ids,

        judge.alpha
    )

    save_json(
        summary_path,
        summary
    )

    # ========================================================
    # Result
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "RESULT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Total       : "
        f"{summary['total']}"
    )

    print(
        f"Evaluated   : "
        f"{summary['evaluated']}"
    )

    print(
        f"API Errors  : "
        f"{summary['api_errors']}"
    )

    print()

    # --------------------------------------------------------
    # ORI
    # --------------------------------------------------------

    print(
        "ORI"
    )

    print(
        "-" * 50
    )

    ori = summary[
        "ori"
    ]

    if ori["A"] is not None:

        print(
            f"Accuracy A : "
            f"{ori['A']:.6f}"
        )

        print(
            f"Irrelevance I : "
            f"{ori['I']:.6f}"
        )

        print(
            f"Score S    : "
            f"{ori['S']:.6f}"
        )

    else:

        print(
            "No ORI results."
        )

    print()

    # --------------------------------------------------------
    # CoT
    # --------------------------------------------------------

    print(
        "CoT"
    )

    print(
        "-" * 50
    )

    cot = summary[
        "cot"
    ]

    if cot["A"] is not None:

        print(
            f"Accuracy A : "
            f"{cot['A']:.6f}"
        )

        print(
            f"Irrelevance I : "
            f"{cot['I']:.6f}"
        )

        print(
            f"Score S    : "
            f"{cot['S']:.6f}"
        )

    else:

        print(
            "No CoT results."
        )

    print()

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    print(
        "ORI -> CoT"
    )

    print(
        "-" * 50
    )

    comparison = summary[
        "comparison"
    ]

    if comparison[
        "delta_A"
    ] is not None:

        print(
            f"Delta A : "
            f"{comparison['delta_A']:+.6f}"
        )

        print(
            f"Delta I : "
            f"{comparison['delta_I']:+.6f}"
        )

        print(
            f"Delta S : "
            f"{comparison['delta_S']:+.6f}"
        )

    print()

    # --------------------------------------------------------
    # CoT error analysis
    # --------------------------------------------------------

    error_analysis = summary[
        "cot_error_analysis"
    ]

    print(
        "CoT Error Analysis"
    )

    print(
        "-" * 50
    )

    print(
        f"Paired samples        : "
        f"{error_analysis['paired']}"
    )

    print(
        f"ORI correct           : "
        f"{error_analysis['ori_correct']}"
    )

    print(
        f"ORI incorrect         : "
        f"{error_analysis['ori_incorrect']}"
    )

    print(
        f"CoT correct           : "
        f"{error_analysis['cot_correct']}"
    )

    print(
        f"CoT incorrect         : "
        f"{error_analysis['cot_incorrect']}"
    )

    print(
        f"CoT induced errors    : "
        f"{error_analysis['cot_induced_error']}"
    )

    if (
        error_analysis[
            "cot_induced_error_rate"
        ]
        is not None
    ):

        print(
            f"CoT error rate        : "
            f"{error_analysis['cot_induced_error_rate']:.6f}"
        )

    print(
        f"CoT recoveries        : "
        f"{error_analysis['cot_recovery']}"
    )

    if (
        error_analysis[
            "cot_recovery_rate"
        ]
        is not None
    ):

        print(
            f"CoT recovery rate     : "
            f"{error_analysis['cot_recovery_rate']:.6f}"
        )

    # --------------------------------------------------------
    # By type
    # --------------------------------------------------------

    print()

    print(
        "By type:"
    )

    print()

    for typ, stat in (
        summary[
            "by_type"
        ].items()
    ):

        print(
            f"  {typ}"
        )

        print(
            f"      total   = "
            f"{stat['total']}"
        )

        if stat[
            "ori_A"
        ] is not None:

            print(
                f"      ORI A   = "
                f"{stat['ori_A']:.6f}"
            )

            print(
                f"      ORI I   = "
                f"{stat['ori_I']:.6f}"
            )

            print(
                f"      ORI S   = "
                f"{stat['ori_S']:.6f}"
            )

        if stat[
            "cot_A"
        ] is not None:

            print(
                f"      CoT A   = "
                f"{stat['cot_A']:.6f}"
            )

            print(
                f"      CoT I   = "
                f"{stat['cot_I']:.6f}"
            )

            print(
                f"      CoT S   = "
                f"{stat['cot_S']:.6f}"
            )

        if stat[
            "delta_A"
        ] is not None:

            print(
                f"      ΔA      = "
                f"{stat['delta_A']:+.6f}"
            )

            print(
                f"      ΔI      = "
                f"{stat['delta_I']:+.6f}"
            )

            print(
                f"      ΔS      = "
                f"{stat['delta_S']:+.6f}"
            )

        print()

    print(
        "Results saved to: "
        f"{judgments_path}"
    )

    print(
        "Summary saved to: "
        f"{summary_path}"
    )

    print(
        "API errors saved to: "
        f"{errors_path}"
    )

    print()

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "VQA ORI / CoT Semantic "
            "Evaluation using DeepSeek API"
        )
    )

    parser.add_argument(

        "--input",

        required=True,

        help=(
            "Input JSONL file"
        )
    )

    parser.add_argument(

        "--output-dir",

        default=None,

        help=(
            "Output directory"
        )
    )

    parser.add_argument(

        "--model",

        default=DEFAULT_MODEL,

        help=(
            "DeepSeek model"
        )
    )

    parser.add_argument(

        "--api-key",

        default=None,

        help=(
            "DeepSeek API key"
        )
    )

    parser.add_argument(

        "--alpha",

        type=float,

        default=DEFAULT_ALPHA,

        help=(
            "Irrelevance penalty alpha "
            "in S=A*(1-alpha*I)"
        )
    )

    parser.add_argument(

        "--limit",

        type=int,

        default=None,

        help=(
            "Only evaluate first N samples"
        )
    )

    parser.add_argument(

        "--retries",

        type=int,

        default=MAX_RETRIES,

        help=(
            "Number of retries after "
            "API failure"
        )
    )

    parser.add_argument(

        "--timeout",

        type=int,

        default=REQUEST_TIMEOUT,

        help=(
            "API timeout in seconds"
        )
    )

    parser.add_argument(

        "--dry-run",

        action="store_true",

        help=(
            "Only load input; "
            "do not call API"
        )
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validate alpha
    # --------------------------------------------------------

    if not (
        0.0
        <=
        args.alpha
        <=
        1.0
    ):

        raise ValueError(
            "--alpha must be between 0 and 1"
        )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_path = Path(
        args.input
    ).resolve()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if args.output_dir is None:

        output_dir = (

            input_path.parent
            /
            f"{input_path.stem}"
            "_evaluation"
        )

    else:

        output_dir = Path(
            args.output_dir
        ).resolve()

    # 防止覆盖原始数据

    if (
        output_dir
        ==
        input_path.parent
    ):

        raise ValueError(

            "Output directory "
            "cannot be the "
            "input directory"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_jsonl(
        input_path
    )

    if args.limit is not None:

        records = records[
            :args.limit
        ]

    print(
        f"Loaded "
        f"{len(records)} records"
    )

    # --------------------------------------------------------
    # Check CoT
    # --------------------------------------------------------

    cot_count = sum(

        1

        for r in records

        if r.get(
            "think_response"
        )
    )

    print(
        f"ORI responses : "
        f"{len(records)}"
    )

    print(
        f"CoT responses : "
        f"{cot_count}"
    )

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        print(
            "Dry run: "
            "no API request "
            "will be sent."
        )

        return 0

    # --------------------------------------------------------
    # API Key
    # --------------------------------------------------------

    api_key = (

        args.api_key

        or os.environ.get(
            "DEEPSEEK_API_KEY"
        )
    )

    if not api_key:

        raise RuntimeError(

            "DeepSeek API key "
            "not found.\n"

            "Use:\n"

            "    --api-key YOUR_KEY\n"

            "or:\n"

            "    export "
            "DEEPSEEK_API_KEY="
            "YOUR_KEY"
        )

    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------

    judge = DeepSeekJudge(

        api_key=api_key,

        model=args.model,

        retries=args.retries,

        timeout=args.timeout,

        alpha=args.alpha,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate(

        records=records,

        judge=judge,

        output_dir=output_dir,

        run_name="ori_vs_cot",

        limit=None
    )

    return 0


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )