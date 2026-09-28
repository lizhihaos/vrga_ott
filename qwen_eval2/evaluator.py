import json
import os

from tqdm import tqdm

from .ccot import generate_ccot
from .generation import cleanup_gpu, generate_one, make_jsonable
from .icot import generate_icot, load_demo
from .modeling import load_model


CANONICAL_MODE_ORDER = ["direct", "cot", "ccot", "icot", "region_guided"]

MODE_LABELS = {
    "direct": "Direct",
    "cot": "CoT",
    "ccot": "CCoT",
    "icot": "ICoT",
    "region_guided": "Region-Guided",
}

# JSONL field name for each generated key. Keys that are not listed here
# are written under their own name, which is how the extra CCoT field
# (ccot_scene_graph) reaches the result file.
MODE_OUTPUT_FIELDS = {
    "direct": "direct_response",
    "cot": "cot_response",
    "ccot": "ccot_response",
    "icot": "icot_response",
}


def format_generation(output):
    """Readable console text for a generation result.

    Single-stage modes return one string. CCoT returns a dict of
    field name -> text, which is printed section by section so the
    intermediate scene graph can be inspected while the run goes on.
    """

    if not isinstance(output, dict):
        return output

    return "\n".join(
        f"{key}:\n{text}"
        for key, text in output.items()
    )
import os
import json
import gc
import torch

from tqdm import tqdm


# ============================================================
# Prompt modes
# ============================================================

def resolve_prompt_modes(prompt_modes="direct"):

    modes = [
        mode.strip().lower()
        for mode in prompt_modes.split(",")
        if mode.strip()
    ]

    aliases = {
        "direct": "direct",
        "think": "cot",
    }

    resolved = []

    for mode in modes:

        mode = aliases.get(mode, mode)

        if mode not in MODE_LABELS:
            raise ValueError(
                f"Unknown prompt mode: {mode}"
            )

        if mode not in resolved:
            resolved.append(mode)

    # Canonical order. The mode list ends up in the output file name, so
    # without this "direct,cot,ccot,icot" and "direct,cot,icot,ccot" would
    # write two different files for the same configuration and split the
    # resume state between them.
    resolved.sort(
        key=lambda mode: (
            CANONICAL_MODE_ORDER.index(mode)
            if mode in CANONICAL_MODE_ORDER
            else len(CANONICAL_MODE_ORDER)
        )
    )

    return resolved


# ============================================================
# GPU cleanup
# ============================================================

def cleanup_gpu(*objects):

    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# GPU memory information
# ============================================================

def print_gpu_memory(tag=""):

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"[GPU {tag}] "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={peak:.2f} GB"
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    data,
    data_name,
    model_name,
    max_new_tokens=2000,
    prompt_modes=None,
    device="cuda:0",
    ccot_scene_graph_max_new_tokens=256,
    ccot_question_style="full",
    icot_num_patches=16,
    icot_max_insertions=3,
    icot_insert_every=2,
    icot_patch_selector="tile",
    icot_vision_markers=False,
    icot_demo=None,
    vision_attn=None,
    load_4bit=False,
):
    """
    Generate VQA results.

    Features
    --------
    1. Model and processor remain on GPU throughout evaluation.
    2. Temporary generation tensors are cleaned after every sample.
    3. CUDA OOM triggers automatic cleanup and retry.
    4. max_new_tokens is reduced automatically after OOM.
    5. Actual successful token limit and retry count are saved.
    6. Results are appended to JSONL immediately.
    7. CCoT runs its two passes back to back per sample, so one retry
       covers both passes.
    """

    # ========================================================
    # Resolve prompt modes
    # ========================================================

    modes = resolve_prompt_modes(
        prompt_modes=prompt_modes
    )

    # ========================================================
    # Output tag
    #
    # A demonstration changes the prompt, so its runs get their own file.
    # ========================================================

    demo_tag = ""

    if "icot" in modes and icot_demo:

        demo_tag = f"_demo-{os.path.basename(os.path.normpath(icot_demo))}"

    tag = (
        f"{data_name}_{model_name}_"
        f"{'-'.join(modes)}"
        f"{demo_tag}_"
        f"maxNew{max_new_tokens}"
    )

    # One folder per dataset, one file per mode inside it: scoring,
    # re-running a single mode and the resume state all stay independent.
    tag_by_mode = {
        mode: f"{mode}{demo_tag}_maxNew{max_new_tokens}"
        for mode in modes
    }

    # ========================================================
    # Configuration
    # ========================================================

    print()
    print("=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)

    print(
        f"Dataset       : {data_name}"
    )

    print(
        f"Model         : {model_name}"
    )

    print(
        f"Prompt modes  : "
        f"{', '.join(MODE_LABELS[mode] for mode in modes)}"
    )

    print(
        f"Max New Token : {max_new_tokens}"
    )

    if "ccot" in modes:

        print(
            f"CCoT SG Token : "
            f"{ccot_scene_graph_max_new_tokens}"
        )

        print(
            f"CCoT Question : "
            f"{ccot_question_style}"
        )

    if "icot" in modes:

        print(
            f"ICoT Patches  : "
            f"{icot_num_patches} per insertion"
        )

        print(
            f"ICoT Inserts  : "
            f"up to {icot_max_insertions}, "
            f"every {icot_insert_every} newlines"
        )

        print(
            f"ICoT Markers  : "
            f"{icot_vision_markers}"
        )

        print(
            f"ICoT Selector : "
            f"{icot_patch_selector}"
        )

        print(
            f"ICoT Demo     : "
            f"{icot_demo or 'zero-shot'}"
        )

    print(
        f"Device        : {device}"
    )

    print(
        f"Tag           : {tag}"
    )

    print("=" * 80)

    # ========================================================
    # Output directory
    # ========================================================

    save_root = (
        "/home/lizhihao/phd/VRGA/rebuttal"
    )

    os.makedirs(
        save_root,
        exist_ok=True
    )

    # Layout: rebuttal/<dataset>/<model>/<mode>_maxNew<N>.jsonl
    save_dir = os.path.join(
        save_root,
        data_name,
        model_name,
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    save_paths = {
        mode: os.path.join(
            save_dir,
            f"{tag_by_mode[mode]}.jsonl",
        )
        for mode in modes
    }

    # ========================================================
    # Existing results
    # ========================================================

    have_judged = {mode: set() for mode in modes}

    for mode in modes:

        path = save_paths[mode]

        if not os.path.exists(path):

            print(
                f"No existing {MODE_LABELS[mode]} result, "
                f"starting that mode from scratch."
            )

            continue

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as handle:

            for line in handle:

                line = line.strip()

                if not line:
                    continue

                try:

                    item = json.loads(line)

                    if "id" in item:
                        have_judged[mode].add(
                            item["id"]
                        )

                except json.JSONDecodeError:

                    print(
                        "Warning: "
                        "skip invalid JSON line."
                    )

        print(
            f"Already evaluated {MODE_LABELS[mode]}: "
            f"{len(have_judged[mode])} "
            f"({path})"
        )

    # ========================================================
    # One-shot ICoT demonstration
    #
    # Loaded once and reused for every sample; it is part of the prompt.
    # ========================================================

    demo = None

    if "icot" in modes and icot_demo is not None:

        demo = load_demo(icot_demo)

        print(
            f"ICoT demo     : "
            f"{len(demo['crops'])} crops from {icot_demo}"
        )

    # ========================================================
    # Load model
    # ========================================================

    total = len(data)

    model = None
    processor = None

    print()

    model, processor = load_model(
        model_name,
        device,
        vision_attn=vision_attn,
        load_4bit=load_4bit,
    )

    model.eval()

    print_gpu_memory("after model loading")

    # ========================================================
    # Main evaluation
    # ========================================================

    try:

        with tqdm(
            total=total,
            desc=f"Evaluating {data_name}"
        ) as pbar:

            for sample in data:

                pbar.update(1)

                # ====================================================
                # Image check
                # ====================================================

                if sample.get("image") is None:

                    print(
                        f"\n[SKIP] "
                        f"id={sample.get('id')} "
                        f"has no image."
                    )

                    continue

                # ====================================================
                # Dataset ID
                # ====================================================

                dataset_id = sample["id"]

                # ====================================================
                # Skip existing
                # ====================================================

                pending_modes = [
                    mode
                    for mode in modes
                    if dataset_id not in have_judged[mode]
                ]

                if not pending_modes:
                    continue

                question = sample["question"]
                gt_answer = sample["answer"]

                # ====================================================
                # Generated outputs
                # ====================================================

                generated = {}

                # Actual successful token limit
                generated_tokens = {}

                # Number of OOM retries
                retry_counts = {}

                # ====================================================
                # Generate each mode
                # ====================================================

                for mode in pending_modes:

                    label = MODE_LABELS[mode]

                    print()
                    print(
                        f"[{dataset_id}] "
                        f"Generating {label}..."
                    )

                    # ------------------------------------------------
                    # Token fallback strategy
                    #
                    # Example:
                    #
                    # 2000
                    #   ↓ OOM
                    # 1000
                    #   ↓ OOM
                    # 512
                    #   ↓
                    # success
                    #
                    # ------------------------------------------------

                    token_candidates = [
                        max_new_tokens
                    ]

                    if max_new_tokens > 1000:

                        token_candidates.append(
                            1000
                        )

                    if max_new_tokens > 512:

                        token_candidates.append(
                            512
                        )

                    if max_new_tokens > 256:

                        token_candidates.append(
                            256
                        )

                    success = False
                    oom_count = 0

                    # ====================================================
                    # Retry loop
                    # ====================================================

                    for attempt, current_max_tokens in enumerate(
                        token_candidates,
                        start=1
                    ):

                        try:

                            print(
                                f"[{dataset_id}] "
                                f"{label} | "
                                f"attempt "
                                f"{attempt}/"
                                f"{len(token_candidates)} | "
                                f"max_new_tokens="
                                f"{current_max_tokens}"
                            )

                            print_gpu_memory(
                                f"before {label}"
                            )

                            # ------------------------------------------------
                            # Generate
                            #
                            # CCoT is a two-pass method and owns its own
                            # pipeline, every other mode is a single pass
                            # driven by the question alone.
                            # ------------------------------------------------

                            if mode == "ccot":

                                output = generate_ccot(
                                    processor,
                                    model,
                                    sample,
                                    max_new_tokens=(
                                        current_max_tokens
                                    ),
                                    scene_graph_max_new_tokens=(
                                        ccot_scene_graph_max_new_tokens
                                    ),
                                    question_style=(
                                        ccot_question_style
                                    ),
                                )

                            elif mode == "icot":

                                output = generate_icot(
                                    processor,
                                    model,
                                    sample,
                                    max_new_tokens=(
                                        current_max_tokens
                                    ),
                                    num_patches=(
                                        icot_num_patches
                                    ),
                                    max_insertions=(
                                        icot_max_insertions
                                    ),
                                    insert_every=(
                                        icot_insert_every
                                    ),
                                    patch_selector=(
                                        icot_patch_selector
                                    ),
                                    vision_markers=(
                                        icot_vision_markers
                                    ),
                                    demo=demo,
                                )

                            else:

                                output = generate_one(
                                    processor,
                                    model,
                                    sample,
                                    mode=mode,
                                    max_new_tokens=(
                                        current_max_tokens
                                    ),
                                )

                            # ------------------------------------------------
                            # Store generated text
                            #
                            # Single-pass modes give one string, CCoT gives a
                            # dict of field name -> text.
                            # ------------------------------------------------

                            # Everything lands under its output field name:
                            # single pass modes give one string, CCoT and
                            # ICoT give a dict of field name -> value.
                            if isinstance(output, dict):

                                generated.update(output)

                            else:

                                generated[
                                    MODE_OUTPUT_FIELDS.get(mode, mode)
                                ] = output

                            generated_tokens[mode] = (
                                current_max_tokens
                            )

                            retry_counts[mode] = (
                                oom_count
                            )

                            success = True

                            # ------------------------------------------------
                            # Print result
                            # ------------------------------------------------

                            print(
                                f"[{dataset_id}] "
                                f"{label}:"
                            )

                            print(
                                format_generation(output)
                            )

                            print(
                                f"[{dataset_id}] "
                                f"{label} succeeded with "
                                f"max_new_tokens="
                                f"{current_max_tokens}"
                            )

                            print_gpu_memory(
                                f"after {label}"
                            )

                            break

                        # ====================================================
                        # CUDA OOM
                        # ====================================================

                        except torch.cuda.OutOfMemoryError as exc:

                            oom_count += 1

                            print()
                            print(
                                "=" * 70
                            )

                            print(
                                f"[OOM] "
                                f"id={dataset_id}"
                            )

                            print(
                                f"[OOM] "
                                f"mode={label}"
                            )

                            print(
                                f"[OOM] "
                                f"max_new_tokens="
                                f"{current_max_tokens}"
                            )

                            print(
                                f"[OOM] "
                                f"attempt="
                                f"{attempt}/"
                                f"{len(token_candidates)}"
                            )

                            print(
                                repr(exc)
                            )

                            print(
                                "=" * 70
                            )

                            # ------------------------------------------------
                            # Clean GPU
                            # ------------------------------------------------

                            cleanup_gpu()

                            print_gpu_memory(
                                "after OOM cleanup"
                            )

                            # ------------------------------------------------
                            # Retry with smaller token size
                            # ------------------------------------------------

                            if attempt < len(
                                token_candidates
                            ):

                                next_tokens = (
                                    token_candidates[
                                        attempt
                                    ]
                                )

                                print(
                                    f"[RETRY] "
                                    f"id={dataset_id}, "
                                    f"mode={label}"
                                )

                                print(
                                    f"[RETRY] "
                                    f"Using "
                                    f"max_new_tokens="
                                    f"{next_tokens}"
                                )

                                continue

                            # ------------------------------------------------
                            # All attempts failed
                            # ------------------------------------------------

                            print(
                                f"[FAILED] "
                                f"id={dataset_id}, "
                                f"mode={label}"
                            )

                            print(
                                f"[FAILED] "
                                f"All token limits failed."
                            )

                            break

                        # ====================================================
                        # Other exception
                        # ====================================================

                        except Exception as exc:

                            print()
                            print(
                                f"[ERROR] "
                                f"{label} failed "
                                f"for id={dataset_id}"
                            )

                            print(
                                repr(exc)
                            )

                            cleanup_gpu()

                            break

                    # ====================================================
                    # Mode failed
                    # ====================================================

                    if not success:

                        print(
                            f"[WARNING] "
                            f"No valid output for "
                            f"id={dataset_id}, "
                            f"mode={label}"
                        )

                        # Record failure information
                        retry_counts[mode] = oom_count

                # ========================================================
                # Build result
                # ========================================================

                # ========================================================
                # One record per mode, each written into its own file
                # ========================================================

                settings = {
                    "ccot": {
                        "ccot_scene_graph_max_new_tokens": (
                            ccot_scene_graph_max_new_tokens
                        ),
                        "ccot_question_style": ccot_question_style,
                    },
                    "icot": {
                        "icot_num_patches": icot_num_patches,
                        "icot_max_insertions": icot_max_insertions,
                        "icot_insert_every": icot_insert_every,
                        "icot_patch_selector": icot_patch_selector,
                        "icot_vision_markers": icot_vision_markers,
                        "icot_demo": icot_demo or "",
                    },
                }

                extra_fields = {
                    "ccot": ("ccot_scene_graph",),
                    "icot": ("icot_insertions",),
                }

                for mode in pending_modes:

                    record = {
                        "id": dataset_id,

                        "question": question,

                        "answer": gt_answer,

                        "type": sample.get(
                            "type",
                            ""
                        ),

                        "mode": mode,

                        "model_name": model_name,

                        # Original requested token limit
                        "max_new_tokens": max_new_tokens,

                        # Actual successful token limit
                        "actual_max_new_tokens": {
                            mode: generated_tokens.get(mode),
                        },

                        # Number of OOM retries
                        "retry_count": {
                            mode: retry_counts.get(mode, 0),
                        },
                    }

                    record.update(
                        settings.get(mode, {})
                    )

                    for key in extra_fields.get(mode, ()):

                        if key in generated:
                            record[key] = generated[key]

                    output_field = MODE_OUTPUT_FIELDS.get(mode, mode)

                    record[output_field] = generated.get(
                        output_field,
                        "",
                    )

                    # ------------------------------------------------
                    # Save this mode's JSONL immediately
                    # ------------------------------------------------

                    with open(
                        save_paths[mode],
                        "a",
                        encoding="utf-8"
                    ) as handle:

                        json.dump(
                            make_jsonable(record),
                            handle,
                            ensure_ascii=False
                        )

                        handle.write("\n")

                        handle.flush()

                        os.fsync(
                            handle.fileno()
                        )

                    have_judged[mode].add(
                        dataset_id
                    )

                    print(
                        f"[{dataset_id}] Saved "
                        f"{MODE_LABELS[mode]}."
                    )

                # ========================================================
                # Cleanup CPU-side temporary objects
                # ========================================================

                del generated
                del generated_tokens
                del retry_counts

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

    # ================================================================
    # Final cleanup
    # ================================================================

    finally:

        print()
        print(
            "Cleaning model from GPU..."
        )

        # ------------------------------------------------------------
        # Explicitly delete model / processor
        # ------------------------------------------------------------

        if model is not None:
            del model

        if processor is not None:
            del processor

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        print(
            "GPU memory cleaned."
        )

    # ================================================================
    # Final summary
    # ================================================================

    print()

    print("=" * 80)
    print("Evaluation Finished")
    print("=" * 80)

    for mode in modes:

        print(
            f"Output {MODE_LABELS[mode]}: "
            f"{save_paths[mode]}"
        )

    print(
        f"Total samples: {total}"
    )

    print(
        f"Completed samples: "
        f"{max((len(have_judged[mode]) for mode in modes), default=0)}"
    )

    print("=" * 80)