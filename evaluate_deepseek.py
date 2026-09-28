import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# DeepSeek API
# ============================================================

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "DEEPSEEK_API_KEY is not set. "
        "Please run: export DEEPSEEK_API_KEY='your_key'"
    )


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)


MODEL = "deepseek-chat"

# Shared with score_parallel.py, so both scripts always judge identically.
JUDGE_SYSTEM_PROMPT = (
    "You are a strict and reliable VQA evaluator. "
    "Return only CORRECT or WRONG."
)


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Use DeepSeek to evaluate VQA results"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON or JSONL file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="direct",
        help=(
            "Mode used to read the model response, looked up as "
            "'{mode}_response', for example direct_response, "
            "cot_response or ccot_response."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file"
    )

    return parser.parse_args()


# ============================================================
# Load JSON / JSONL
# ============================================================

def load_data(file_path):

    print(f"Loading: {file_path}")

    if file_path.endswith(".jsonl"):

        data = []

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(f, 1):

                line = line.strip()

                if not line:
                    continue

                try:
                    data.append(json.loads(line))

                except json.JSONDecodeError as e:

                    print(
                        f"Warning: cannot parse line {line_number}: {e}"
                    )


    else:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


    if not isinstance(data, list):

        raise ValueError(
            "Input file must contain a list of samples."
        )


    print(f"Loaded {len(data)} samples")

    return data


# ============================================================
# Extract Model Response
# ============================================================

def get_model_response(item,mode):

    # Your VRGA files
    if item.get(f"{mode}_response"):

        response = item[f"{mode}_response"]

    # Common format
    elif item.get("response") is not None:

        response = item["response"]

    elif item.get("output") is not None:

        response = item["output"]

    elif item.get("prediction") is not None:

        response = item["prediction"]

    elif item.get("model_answer") is not None:

        response = item["model_answer"]
    elif item.get(f"{mode}"):
        response = item[f"{mode}"]
    else:

        raise KeyError(
            f"Cannot find model response field. "
            f"Available fields: {list(item.keys())}"
        )


    # Sometimes response is a list
    if isinstance(response, list):

        if len(response) == 0:
            return ""

        response = response[0]


    # Sometimes response is dict
    if isinstance(response, dict):

        response = (
            response.get("text")
            or response.get("content")
            or response.get("answer")
            or str(response)
        )


    return str(response)


# ============================================================
# Build Judge Prompt
# ============================================================

def build_prompt(question, ground_truth, model_response):

    prompt = f"""
You are a strict evaluator for Visual Question Answering (VQA).

Your task is to determine whether the MODEL ANSWER correctly answers
the QUESTION according to the GROUND TRUTH ANSWER.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

MODEL ANSWER:
{model_response}


Evaluation instructions:

1. Judge only whether the model's final answer is correct.
2. Ignore unnecessary explanations or additional descriptions.
3. Semantically equivalent answers should be considered CORRECT.
4. For yes/no questions:
   - "yes" means the model claims the answer is yes.
   - "no" means the model claims the answer is no.
   - If the model gives the opposite answer, it is WRONG.
5. Do not infer information that is not present in the model answer.
6. Do not judge the quality of the explanation.
7. The final decision must be based on whether the answer matches the
   ground truth.

Return ONLY one of the following two words:

CORRECT
WRONG
"""

    return prompt


# ============================================================
# DeepSeek Judge
# ============================================================

def judge(question, ground_truth, model_response):

    prompt = build_prompt(
        question,
        ground_truth,
        model_response
    )


    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[

                {
                    "role": "system",
                    "content": JUDGE_SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0,

        )


        result = (
            response
            .choices[0]
            .message
            .content
            .strip()
            .upper()
        )


        # Strict parsing
        if result == "CORRECT":

            return 1, result

        elif result == "WRONG":

            return 0, result

        else:

            # Try to handle cases such as:
            # "The answer is CORRECT."
            if result.startswith("CORRECT"):

                return 1, result

            elif result.startswith("WRONG"):

                return 0, result

            else:

                print(
                    f"Warning: unexpected DeepSeek output: {result}"
                )

                return 0, result


    except Exception as e:

        print(
            f"DeepSeek API error: {e}"
        )

        return 0, "ERROR"


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if args.output is None:

        if args.input.endswith(".jsonl"):

            output_file = args.input[:-6] + "_deepseek_eval.json"

        else:

            output_file = args.input[:-5] + "_deepseek_eval.json"

    else:

        output_file = args.output


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data(args.input)


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    correct = 0

    errors = 0


    for index, item in enumerate(
        tqdm(data, desc="DeepSeek Evaluation")
    ):

        try:

            question = item["question"]

            ground_truth = item["answer"]

            model_response = get_model_response(item,args.mode)


            score, judge_result = judge(
                question,
                ground_truth,
                model_response
            )


            if judge_result == "ERROR":

                errors += 1

            else:

                correct += score


            result_item = {

                "id": item.get(
                    "id",
                    index
                ),

                "question": question,

                "ground_truth": ground_truth,

                "model_response": model_response,

                "judge": judge_result,

                "score": score

            }


            results.append(result_item)


        except Exception as e:

            print(
                f"\nError processing sample {index}: {e}"
            )

            results.append({

                "id": item.get(
                    "id",
                    index
                ),

                "question": item.get(
                    "question",
                    ""
                ),

                "ground_truth": item.get(
                    "answer",
                    ""
                ),

                "model_response": "",

                "judge": "ERROR",

                "score": 0,

                "error": str(e)

            })

            errors += 1


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = len(data)

    valid = total - errors


    if valid > 0:

        accuracy = correct / valid

    else:

        accuracy = 0.0


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_data = {

        "input_file": args.input,

        "model": MODEL,

        "total_samples": total,

        "valid_samples": valid,

        "correct": correct,

        "errors": errors,

        "accuracy": accuracy,

        "results": results

    }


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=4,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DeepSeek VQA Evaluation")
    print("=" * 60)

    print(
        f"Input       : {args.input}"
    )

    print(
        f"Total       : {total}"
    )

    print(
        f"Valid       : {valid}"
    )

    print(
        f"Correct     : {correct}"
    )

    print(
        f"Errors      : {errors}"
    )

    print(
        f"Accuracy    : {accuracy:.6f}"
    )

    print(
        f"Accuracy %  : {accuracy * 100:.2f}%"
    )

    print(
        f"Output      : {output_file}"
    )

    print("=" * 60)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()