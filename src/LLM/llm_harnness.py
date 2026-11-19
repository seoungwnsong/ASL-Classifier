"""
CSC211 Lab 8: test harness for the LLM approach to our ASL project.

Task: a short video clip shows one person signing one ASL word. Classify it as one of 20 words.
The LLM sees N_FRAMES still frames of the clip (made once by save_frames.py), sent as images.

    python src/llm/llm_harness.py            develop: score every prompt in PROMPTS on a fixed training subset
    python src/llm/llm_harness.py val v0 v1  compare your best few prompts on the validation set
    python src/llm/llm_harness.py test v1    run the prompt "v1" once on the test set, and write eval/llm.csv

Based on the course's demo harness (movie-review sentiment).
"""

import csv
import json
import sys
import time
from pathlib import Path

import ollama
import pandas as pd

from fewshot import FEWSHOT  # src/LLM/fewshot.py: the labelled example clips
from save_frames import N_FRAMES, clip_dir, training_subset  # where save_frames.py put the frames

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA = REPO_ROOT / "data" / "metadata.csv"  # example_id, split, word, source, clip_path
RUNS_DIR = REPO_ROOT / "results" / "llm_runs"   # saved responses, one .jsonl per version and split
OUTPUT_CSV = REPO_ROOT / "eval" / "llm.csv"     # final test predictions

MODEL = "gemma4:e4b"
OPTIONS = {
    "temperature": 0,  # always choose the most likely next token, so runs are repeatable
    "seed": 211,
    "num_ctx": 8192,   # the context window, in tokens; every image takes up a few hundred
}

# N_FRAMES and the frame folders come from save_frames.py, so the two always match:
# data/llm_frames/n<N_FRAMES>/<fewshot|train|val|test>/<example_id>/
SPLIT_OF = pd.read_csv(METADATA).set_index("example_id")["split"]  # which folder each clip is in

# The 20 words, in the order of SELECTED_CLASSES in src/preprocessing/filter_dataset.py.
LABELS = ["hello", "yes", "no", "please", "need", "good", "nice", "eat", "tired", "want",
          "like", "deaf", "mother", "father", "teacher", "learn", "school", "happy", "sad", "finish"]


def label_schema(with_reasoning=False):
    """The JSON the model must answer in: exactly one of the 20 labels.
    With with_reasoning=True, a "reasoning" field comes BEFORE "answer", so the model
    describes the sign first and only then picks a label (chain-of-thought, as in
    Lab 7's judge.py). This makes each call slower."""
    properties = {}
    if with_reasoning:
        properties["reasoning"] = {"type": "string"}
    properties["answer"] = {"type": "string", "enum": LABELS}
    return {"type": "object", "properties": properties, "required": list(properties)}


# Develop prompts on a fixed subset of the TRAINING clips: training_subset() and PER_CLASS
# are in save_frames.py, which saves frames only for those clips (in train/).

# Versions that use the schema with a "reasoning" field before "answer".
REASONING_VERSIONS = set()

# Versions that show the FEWSHOT clips, each with its correct answer, before the clip to classify.
FEWSHOT_VERSIONS = set()

# One entry per prompt version. {n_frames} is replaced by N_FRAMES; the frames
# themselves are attached as images, never written in the text.
# Write the change and the hypothesis BEFORE running a new version.
# Never edit a version after it has been scored: copy it to a new version instead.
PROMPTS = {
    # baseline
    "v0": """<task>
You will see {n_frames} frames, in order, taken evenly from a short video of one person
signing a single word in American Sign Language (ASL).
Which word is being signed?
</task>

<labels>
hello, yes, no, please, need, good, nice, eat, tired, want,
like, deaf, mother, father, teacher, learn, school, happy, sad, finish
</labels>

Answer with exactly one of the labels.""",

    # change (control): list the labels in alphabetical order instead
    # hypothesis: the order of the labels should not matter, so any score change
    #             is noise; |v1 - v0| is our noise floor
    "v1": """<task>
You will see {n_frames} frames, in order, taken evenly from a short video of one person
signing a single word in American Sign Language (ASL).
Which word is being signed?
</task>

<labels>
deaf, eat, father, finish, good, happy, hello, learn, like, mother,
need, nice, no, please, sad, school, teacher, tired, want, yes
</labels>

Answer with exactly one of the labels.""",

    # v2: copy v1 (or v0) here, make ONE change, and write the change and hypothesis above it.
    # For chain-of-thought, also add "v2" to REASONING_VERSIONS; for few-shot, to FEWSHOT_VERSIONS.
}


def load_frames(example_id):
    """The saved frames of one clip, as image bytes (made by save_frames.py)."""
    folder = clip_dir(example_id, SPLIT_OF[example_id])
    paths = sorted(folder.glob("*.jpg"))
    if len(paths) != N_FRAMES:
        sys.exit(f"Expected {N_FRAMES} frames in {folder}, found {len(paths)}. "
                 "Run: python src/LLM/save_frames.py")
    # Send the bytes, not the paths: the model must never see a file name.
    return [p.read_bytes() for p in paths]


def build_prompt(version):
    return PROMPTS[version].replace("{n_frames}", str(N_FRAMES))


def build_messages(version, example_id):
    """The chat sent for one clip. Few-shot versions first show each FEWSHOT clip as a
    user turn, followed by its correct answer as the model's own reply."""
    if version not in FEWSHOT_VERSIONS:
        return [{"role": "user", "content": build_prompt(version), "images": load_frames(example_id)}]
    messages = []
    for k, example in enumerate(FEWSHOT):
        intro = build_prompt(version) + "\n\n" if k == 0 else ""
        messages.append({"role": "user", "content": f"{intro}Example clip {k + 1}:",
                         "images": load_frames(example["example_id"])})
        messages.append({"role": "assistant", "content": json.dumps({"answer": example["word"]})})
    messages.append({"role": "user", "content": "New clip:", "images": load_frames(example_id)})
    return messages


def predict(version, example_id):
    """Ask the model about one clip. Returns (prediction, response, seconds).
    The prediction is None if the output is not one of the labels."""
    start = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=build_messages(version, example_id),
        format=label_schema(version in REASONING_VERSIONS),
        options=OPTIONS,
    )
    seconds = time.perf_counter() - start
    try:
        answer = json.loads(response.message.content)["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        answer = None
    return (answer if answer in LABELS else None), response, seconds


def run(version, rows, out_file):
    """Run one prompt on every row, and save every response to out_file.
    The first line of out_file records the setup. If out_file already has the
    outputs for exactly this setup and these rows, they are loaded instead."""
    setup = {"model": MODEL, "options": OPTIONS, "n_frames": N_FRAMES,
             "schema": label_schema(version in REASONING_VERSIONS), "prompt": build_prompt(version),
             "fewshot": FEWSHOT if version in FEWSHOT_VERSIONS else []}
    ids = [row["example_id"] for row in rows]
    if Path(out_file).exists():
        with open(out_file, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if lines and lines[0] == setup and [r["id"] for r in lines[1:]] == ids:
            print(f"\n{version}: nothing changed, so using the saved outputs in {out_file.name}")
            return lines[1:]

    print(f"\n===== {version}: the exact prompt for {ids[0]} =====")
    print(build_prompt(version))
    print(f"[+ {N_FRAMES} images attached, from {clip_dir(ids[0], SPLIT_OF[ids[0]]).relative_to(REPO_ROOT)}]")
    if version in FEWSHOT_VERSIONS:
        print(f"[shown first: {len(FEWSHOT)} example clips with their answers, "
              f"{', '.join(example['word'] for example in FEWSHOT)}]")
    print("=====")
    results = []
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(setup) + "\n")
        for row in rows:
            prediction, response, seconds = predict(version, row["example_id"])
            result = {
                "id": row["example_id"],
                "prediction": prediction,
                "raw": response.message.content,
                "seconds": round(seconds, 2),
                "prompt_tokens": response.prompt_eval_count,
            }
            f.write(json.dumps(result) + "\n")
            results.append(result)
            warning = ""
            if response.prompt_eval_count and response.prompt_eval_count > 0.9 * OPTIONS["num_ctx"]:
                warning = "  WARNING: close to num_ctx, the start of the prompt may be cut off"
            print(f"{row['example_id']}: {prediction or 'INVALID'}  "
                  f"({seconds:.1f} s, {response.prompt_eval_count} prompt tokens){warning}")
    return results


def split_rows(split):
    """All clips of one split, as dicts with example_id and word."""
    metadata = pd.read_csv(METADATA)
    return metadata[metadata["split"] == split][["example_id", "word"]].to_dict("records")


def compare(versions, rows, split):
    """Score each version on the same rows, and list fixes and breaks vs. the first version."""
    unknown = [v for v in versions if v not in PROMPTS]
    if unknown:
        sys.exit(f"No such prompt(s) in PROMPTS: {unknown}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Majority-class baseline: {max(sum(r['word'] == w for r in rows) for w in LABELS) / len(rows):.3f} "
          f"(always guessing the most common word); chance = {1 / len(LABELS):.3f}")
    correct = {}  # version -> [True/False for each clip]
    for version in versions:
        results = run(version, rows, RUNS_DIR / f"{version}_{split}.jsonl")
        correct[version] = [r["prediction"] == row["word"] for r, row in zip(results, rows)]
        invalid = sum(r["prediction"] is None for r in results)
        seconds = sum(r["seconds"] for r in results) / len(results)
        print(f"{version}: accuracy {sum(correct[version]) / len(rows):.3f} "
              f"({sum(correct[version])}/{len(rows)}) on {split} clips, "
              f"{invalid} invalid, {seconds:.1f} s per clip")

    # Fixes and breaks: clips that became right or wrong, compared with the first version.
    print()
    base = versions[0]
    for version in versions[1:]:
        pairs = list(zip(rows, correct[base], correct[version]))
        fixes = [row["example_id"] for row, old, new in pairs if new and not old]
        breaks = [row["example_id"] for row, old, new in pairs if old and not new]
        print(f"{version} vs. {base}: {len(fixes)} fixes {fixes or ''}, {len(breaks)} breaks {breaks or ''}")


def test(version):
    """Run ONCE, with the version chosen on the validation set. The test labels are never read."""
    if version not in PROMPTS:
        sys.exit(f"No such prompt in PROMPTS: {version}")
    metadata = pd.read_csv(METADATA)
    # all 136 test clips (= the human evaluation subset too); only the IDs, never the words
    rows = metadata[metadata["split"] == "test"][["example_id"]].to_dict("records")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = run(version, rows, RUNS_DIR / f"{version}_test.jsonl")
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Example ID", "LLM Prediction"])
        for r in results:
            writer.writerow([r["id"], r["prediction"] or "INVALID"])
    print(f"\nWrote {len(results)} predictions to {OUTPUT_CSV.relative_to(REPO_ROOT)}.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        compare(list(PROMPTS), training_subset(), "train")
    elif len(sys.argv) >= 3 and sys.argv[1] == "val":
        compare(sys.argv[2:], split_rows("val"), "val")
    elif len(sys.argv) == 3 and sys.argv[1] == "test":
        test(sys.argv[2])
    else:
        sys.exit(__doc__)