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

from fewshot import FEWSHOT, FEWSHOT_REASONING  # src/LLM/fewshot.py: the labelled example clips
from save_frames import N_FRAMES, clip_dir, training_subset  # where save_frames.py put the frames

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA = REPO_ROOT / "data" / "metadata.csv"  # example_id, split, word, source, clip_path
RUNS_DIR = REPO_ROOT / "results" / "llm_runs"   # saved responses, one .jsonl per version and split
OUTPUT_CSV = REPO_ROOT / "eval" / "llm.csv"     # final test predictions

# The private Ollama 0.40.16 server (start it with ~/ollama-0.40.16/start.sh).
# The lab's system Ollama (0.33.3, port 11434) breaks image prompts, so it is not used.
OLLAMA_HOST = "http://127.0.0.1:11435"
CLIENT = ollama.Client(host=OLLAMA_HOST)
# gemma4 "thinks" (hidden reasoning) before answering. Keep it on: with think=False
# the model answers "hello" for nearly every clip.
THINK = True

MODEL = "gemma4:e4b"
OPTIONS = {
    "temperature": 0,  # always choose the most likely next token, so runs are repeatable
    "seed": 211,
    # The context window, in tokens. A 288x288 frame really takes ~81 tokens, but Ollama
    # over-estimates images when deciding what fits, and SILENTLY drops the earlier messages
    # if it thinks they do not: at 8192 (or 16384, 32768) few-shot lost the instructions and
    # all example clips. 65536 keeps the whole few-shot chat (~5900 tokens).
    "num_ctx": 65536,
    # Cap on output tokens. gemma4 "thinks" before answering (up to ~1300 tokens seen),
    # so leave plenty of room; the prompt (up to ~6000 tokens with few-shot) + this must fit in num_ctx.
    "num_predict": 2048,
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
REASONING_VERSIONS = {"v5", "v6", "v7"}

# Versions that show the FEWSHOT clips, each with its correct answer, before the clip to classify.
FEWSHOT_VERSIONS = {"v4", "v5", "v6", "v7"}

# Few-shot versions whose example replies also show the reasoning (chain-of-thought examples),
# from FEWSHOT_REASONING in fewshot.py.
COT_EXAMPLE_VERSIONS = {"v6"}

# Versions that repeat the key instructions AFTER the clip to classify (sandwich technique).
# Ollama places a message's images before its text, so this lands right after the 16 frames.
SANDWICH_VERSIONS = {"v7"}
SANDWICH_REMINDER = (
    "Now classify this new clip. Work through the 6 steps in <steps> in order, using the "
    "label descriptions in <labels>, then respond in JSON with \"reasoning\" (one short line "
    "per step) and \"answer\" (exactly one of the 20 labels).")

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

    # change: builds on v0, but with SEVERAL changes at once:
    #         1. a <role> (expert ASL interpreter)
    #         2. the frames are numbered in the task (Frame 1 to Frame {n_frames})
    #         3. the labels in a new order (neither v0's nor v1's)
    #         4. a <format> section asking for JSON, with an example answer ("hello")
    # hypothesis: a clearer role, frame order and answer format help the model read the
    #             signs, so accuracy should go up over v0; since several things changed,
    #             a score change cannot be pinned on one of them
    "v2": """<role>
You are an expert American Sign Language (ASL) interpreter who can recognize signs
from still frames of a video.
</role>

<task>
You will see {n_frames} frames, in order (Frame 1 to Frame {n_frames}), taken evenly from a short
video of one person signing a single word in ASL. Identify which word is being signed.
</task>

<labels>
hello, father, learn, teacher, mother, eat, deaf, good, sad, no,
yes, need, please, like, happy, tired, school, want, finish, nice
</labels>

<format>
Respond in JSON with one field, "answer", whose value is exactly one of the labels above.
Example: {"answer": "hello"}
</format>""",

    # change: builds on v2; each label now has a one-line description of how the sign is
    #         made (from src/algorithmic/rules.py: the comments plus the location, one or two
    #         hands, and direction values), in v2's label order; one sentence in <task> says so
    # hypothesis: the model sees the hands but does not know these ASL signs, and mostly
    #             guesses from the prompt (versions agree on only ~26/120 clips); telling it
    #             what each sign looks like should raise accuracy over v2 by more than the
    #             noise (v1 vs v0 moved 8 clips each way). If it only piles answers onto one
    #             word (e.g. the first, "hello"), it is copying descriptions, not using them.
    "v3": """<role>
You are an expert American Sign Language (ASL) interpreter who can recognize signs
from still frames of a video.
</role>

<task>
You will see {n_frames} frames, in order (Frame 1 to Frame {n_frames}), taken evenly from a short
video of one person signing a single word in ASL. Identify which word is being signed.
Each label below is followed by a short description of how its sign is made.
</task>

<labels>
hello: one flat hand starts at the forehead and moves outward, like a salute
father: one open hand at the forehead; the thumb taps the forehead, repeatedly
learn: one hand with the fingers bunched together is lifted up to the forehead, once
teacher: both hands at the temples, fingers bunched, move forward, repeatedly
mother: one open hand at the chin; the thumb taps the chin, repeatedly
eat: one hand with the fingertips bunched together taps the lips, repeatedly
deaf: one index finger touches near the ear, then moves to near the mouth
good: one flat hand starts at the chin and moves down
sad: both open hands in front of the face slide down
no: one hand near the face or shoulder; the index and middle fingers snap closed onto the thumb, repeatedly
yes: one fist at chest height nods up and down, repeatedly
need: one hooked index finger at chest height bends down, repeatedly
please: one flat hand rubs a circle on the chest, repeatedly
like: one hand at the chest pulls away while the thumb and middle finger pinch together
happy: both flat hands brush upward on the chest, repeatedly
tired: both bent hands on the chest drop downward
school: both flat hands in front of the chest; one palm claps onto the other, twice
want: both hands, palms up and fingers curved like claws, pull in toward the body
finish: both open hands at chest height, palms facing in, flip quickly outward
nice: both flat hands in front of the chest; one palm slides across the other
</labels>

<format>
Respond in JSON with one field, "answer", whose value is exactly one of the labels above.
Example: {"answer": "hello"}
</format>""",

    # change: builds on v3; few-shot: before the clip to classify, the model sees the 3 example
    #         clips in src/LLM/fewshot.py (sad, deaf, eat; 16 frames each, from
    #         data/llm_frames/n16/fewshot/), each followed by its correct answer; one sentence
    #         in <task> says so
    # hypothesis: seeing real clips of a sign with its answer teaches the model what that sign
    #             looks like in these videos, so accuracy should rise over v3, mostly on the
    #             3 example words (v3: sad 0/6, eat 2/6, deaf 5/6). The other 17 words may not
    #             improve, and answers may lean toward the 3 shown words.
    "v4": """<role>
You are an expert American Sign Language (ASL) interpreter who can recognize signs
from still frames of a video.
</role>

<task>
You will see {n_frames} frames, in order (Frame 1 to Frame {n_frames}), taken evenly from a short
video of one person signing a single word in ASL. Identify which word is being signed.
Each label below is followed by a short description of how its sign is made.
First you will see a few example clips, each followed by its correct answer;
then you will see the new clip to classify.
</task>

<labels>
hello: one flat hand starts at the forehead and moves outward, like a salute
father: one open hand at the forehead; the thumb taps the forehead, repeatedly
learn: one hand with the fingers bunched together is lifted up to the forehead, once
teacher: both hands at the temples, fingers bunched, move forward, repeatedly
mother: one open hand at the chin; the thumb taps the chin, repeatedly
eat: one hand with the fingertips bunched together taps the lips, repeatedly
deaf: one index finger touches near the ear, then moves to near the mouth
good: one flat hand starts at the chin and moves down
sad: both open hands in front of the face slide down
no: one hand near the face or shoulder; the index and middle fingers snap closed onto the thumb, repeatedly
yes: one fist at chest height nods up and down, repeatedly
need: one hooked index finger at chest height bends down, repeatedly
please: one flat hand rubs a circle on the chest, repeatedly
like: one hand at the chest pulls away while the thumb and middle finger pinch together
happy: both flat hands brush upward on the chest, repeatedly
tired: both bent hands on the chest drop downward
school: both flat hands in front of the chest; one palm claps onto the other, twice
want: both hands, palms up and fingers curved like claws, pull in toward the body
finish: both open hands at chest height, palms facing in, flip quickly outward
nice: both flat hands in front of the chest; one palm slides across the other
</labels>

<format>
Respond in JSON with one field, "answer", whose value is exactly one of the labels above.
Example: {"answer": "hello"}
</format>""",

    # change: builds on v4; task decomposition: the task is split into 6 steps (where are the
    #         hands, how many, handshape, movement, narrow the labels down by location and number
    #         of hands, then choose by handshape and movement), and the model writes each step's
    #         result in a "reasoning" field before "answer" (v5 is in REASONING_VERSIONS)
    # hypothesis: v3/v4 usually find the right location but confuse signs made in the same place
    #             (eat/good/no vs. mother at the chin; please/school vs. happy/nice at the chest).
    #             Making the model first narrow the labels by location and number of hands, then
    #             decide by handshape and movement, should fix some of those confusions and raise
    #             accuracy over v4. The written steps also show which step goes wrong.
    "v5": """<role>
You are an expert American Sign Language (ASL) interpreter who can recognize signs
from still frames of a video.
</role>

<task>
You will see {n_frames} frames, in order (Frame 1 to Frame {n_frames}), taken evenly from a short
video of one person signing a single word in ASL. Identify which word is being signed.
Each label below is followed by a short description of how its sign is made.
First you will see a few example clips, each followed by its correct answer;
then you will see the new clip to classify.
</task>

<labels>
hello: one flat hand starts at the forehead and moves outward, like a salute
father: one open hand at the forehead; the thumb taps the forehead, repeatedly
learn: one hand with the fingers bunched together is lifted up to the forehead, once
teacher: both hands at the temples, fingers bunched, move forward, repeatedly
mother: one open hand at the chin; the thumb taps the chin, repeatedly
eat: one hand with the fingertips bunched together taps the lips, repeatedly
deaf: one index finger touches near the ear, then moves to near the mouth
good: one flat hand starts at the chin and moves down
sad: both open hands in front of the face slide down
no: one hand near the face or shoulder; the index and middle fingers snap closed onto the thumb, repeatedly
yes: one fist at chest height nods up and down, repeatedly
need: one hooked index finger at chest height bends down, repeatedly
please: one flat hand rubs a circle on the chest, repeatedly
like: one hand at the chest pulls away while the thumb and middle finger pinch together
happy: both flat hands brush upward on the chest, repeatedly
tired: both bent hands on the chest drop downward
school: both flat hands in front of the chest; one palm claps onto the other, twice
want: both hands, palms up and fingers curved like claws, pull in toward the body
finish: both open hands at chest height, palms facing in, flip quickly outward
nice: both flat hands in front of the chest; one palm slides across the other
</labels>

<steps>
Work through these steps for the new clip, in order:
1. Location: where are the hands during the sign? (forehead or temples, ear, chin or mouth,
   in front of the face, or chest)
2. Hands: is the sign made with one hand or two?
3. Handshape: what shape is the active hand? (for example flat, open with spread fingers,
   fist, index finger, hooked finger, bunched fingertips, claw, pinch)
4. Movement: how does the hand move across the frames? (for example taps, moves down,
   moves up, moves outward, circles, slides, flips, claps, pulls in), and is it repeated?
5. Candidates: list the labels whose description matches the location from step 1 and
   the number of hands from step 2.
6. Choice: from those candidates, choose the one whose handshape and movement best match
   steps 3 and 4.
</steps>

<format>
Respond in JSON with two fields. In "reasoning", write one short line for each of the
6 steps. In "answer", give exactly one of the labels above.
Example: {"reasoning": "1. ... 2. ... 3. ... 4. ... 5. ... 6. ...", "answer": "hello"}
</format>""",

    # change: builds on v5; chain-of-thought examples: each few-shot example's reply now shows
    #         the full 6-step reasoning (written for each clip in fewshot.py, FEWSHOT_REASONING)
    #         before its answer, instead of the answer alone; one sentence in <task> says so
    # hypothesis: v5 asks for step-by-step reasoning but its examples only show answers. In
    #             v5's wrong answers the location is usually right but the handshape/movement
    #             is wrong (deaf -> eat: "bunched fingertips tap the lips"). Worked examples that
    #             contrast exactly these signs (index finger vs. bunched fingertips vs. thumb)
    #             should teach the model to look at the handshape, and raise accuracy over v5,
    #             most of all on the chin/mouth signs.
    "v6": """<role>
You are an expert American Sign Language (ASL) interpreter who can recognize signs
from still frames of a video.
</role>

<task>
You will see {n_frames} frames, in order (Frame 1 to Frame {n_frames}), taken evenly from a short
video of one person signing a single word in ASL. Identify which word is being signed.
Each label below is followed by a short description of how its sign is made.
First you will see a few example clips, each followed by its step-by-step reasoning and
its correct answer; then you will see the new clip to classify.
</task>

<labels>
hello: one flat hand starts at the forehead and moves outward, like a salute
father: one open hand at the forehead; the thumb taps the forehead, repeatedly
learn: one hand with the fingers bunched together is lifted up to the forehead, once
teacher: both hands at the temples, fingers bunched, move forward, repeatedly
mother: one open hand at the chin; the thumb taps the chin, repeatedly
eat: one hand with the fingertips bunched together taps the lips, repeatedly
deaf: one index finger touches near the ear, then moves to near the mouth
good: one flat hand starts at the chin and moves down
sad: both open hands in front of the face slide down
no: one hand near the face or shoulder; the index and middle fingers snap closed onto the thumb, repeatedly
yes: one fist at chest height nods up and down, repeatedly
need: one hooked index finger at chest height bends down, repeatedly
please: one flat hand rubs a circle on the chest, repeatedly
like: one hand at the chest pulls away while the thumb and middle finger pinch together
happy: both flat hands brush upward on the chest, repeatedly
tired: both bent hands on the chest drop downward
school: both flat hands in front of the chest; one palm claps onto the other, twice
want: both hands, palms up and fingers curved like claws, pull in toward the body
finish: both open hands at chest height, palms facing in, flip quickly outward
nice: both flat hands in front of the chest; one palm slides across the other
</labels>

<steps>
Work through these steps for the new clip, in order:
1. Location: where are the hands during the sign? (forehead or temples, ear, chin or mouth,
   in front of the face, or chest)
2. Hands: is the sign made with one hand or two?
3. Handshape: what shape is the active hand? (for example flat, open with spread fingers,
   fist, index finger, hooked finger, bunched fingertips, claw, pinch)
4. Movement: how does the hand move across the frames? (for example taps, moves down,
   moves up, moves outward, circles, slides, flips, claps, pulls in), and is it repeated?
5. Candidates: list the labels whose description matches the location from step 1 and
   the number of hands from step 2.
6. Choice: from those candidates, choose the one whose handshape and movement best match
   steps 3 and 4.
</steps>

<format>
Respond in JSON with two fields. In "reasoning", write one short line for each of the
6 steps. In "answer", give exactly one of the labels above.
Example: {"reasoning": "1. ... 2. ... 3. ... 4. ... 5. ... 6. ...", "answer": "hello"}
</format>""",
}

# change: builds on v5 (not v6); sandwich: the same prompt as v5, plus a short reminder of the
#         task, steps and format (SANDWICH_REMINDER) placed after the new clip's frames, so the
#         instructions come both before and after the input
# hypothesis: in v5 the instructions are ~4,000 tokens (3 example clips) away from the clip to
#             classify; repeating them right after it should keep the model on task. But v5
#             already followed the 6 steps and the JSON format on 118/120 clips, and its errors
#             are about what it sees (handshape, ear vs. temple), so we expect little or no
#             change; a result within noise would show the bottleneck is perception, not
#             instruction following.
PROMPTS["v7"] = PROMPTS["v5"]


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
        reply = {"answer": example["word"]}
        if version in COT_EXAMPLE_VERSIONS:
            reply = {"reasoning": FEWSHOT_REASONING[example["example_id"]], "answer": example["word"]}
        messages.append({"role": "assistant", "content": json.dumps(reply)})
    final = "New clip:"
    if version in SANDWICH_VERSIONS:
        final += "\n\n" + SANDWICH_REMINDER
    messages.append({"role": "user", "content": final, "images": load_frames(example_id)})
    return messages


def predict(version, example_id):
    """Ask the model about one clip. Returns (prediction, response, seconds).
    The prediction is None if the output is not one of the labels."""
    start = time.perf_counter()
    response = CLIENT.chat(
        model=MODEL,
        messages=build_messages(version, example_id),
        format=label_schema(version in REASONING_VERSIONS),
        options=OPTIONS,
        think=THINK,
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
    setup = {"model": MODEL, "ollama_host": OLLAMA_HOST, "think": THINK, "options": OPTIONS, "n_frames": N_FRAMES,
             "schema": label_schema(version in REASONING_VERSIONS), "prompt": build_prompt(version),
             "fewshot": ([dict(e, reasoning=FEWSHOT_REASONING[e["example_id"]]) for e in FEWSHOT]
                         if version in COT_EXAMPLE_VERSIONS else
                         FEWSHOT if version in FEWSHOT_VERSIONS else [])}
    if version in SANDWICH_VERSIONS:
        setup["sandwich"] = SANDWICH_REMINDER
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
            # Guard: Ollama drops earlier messages without an error when it thinks the chat is
            # too long. Each frame takes ~81 tokens, so a prompt far smaller than the images
            # sent means part of the chat (the instructions, or few-shot examples) was cut.
            n_images = N_FRAMES * (len(FEWSHOT) + 1 if version in FEWSHOT_VERSIONS else 1)
            if response.prompt_eval_count and response.prompt_eval_count < 50 * n_images:
                sys.exit(f"{row['example_id']}: only {response.prompt_eval_count} prompt tokens for "
                         f"{n_images} images, so Ollama cut part of the chat. Raise num_ctx in OPTIONS.")
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