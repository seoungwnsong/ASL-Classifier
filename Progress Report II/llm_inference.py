"""LLM approach: predict the ASL word signed in one video clip with our best prompt (v6).

    video -> 16 frames (captions covered, cropped to the signer, 288 x 288), the same frames
             the deep learning model and the human participants saw
          -> gemma4:e4b (through Ollama), prompt v6 from llm_harness.py:
             3 example clips, each with its step-by-step reasoning and correct answer,
             then the new clip
          -> JSON {"reasoning": ..., "answer": one of the 20 words}

This file prepares the frames like save_frames.py and sends the same chat as llm_harness.py,
so it runs on its own. Needs an Ollama server with gemma4:e4b at OLLAMA_HOST. Run it from the
folder that holds sample.mp4, the 3 fewshot_*.mp4 clips and fewshot.py:
    python llm_inference.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import ollama

from fewshot import FEWSHOT_REASONING  # the worked reasoning shown with each example clip

HERE = Path(__file__).resolve().parent
OLLAMA_HOST = "http://127.0.0.1:11435"  # our Ollama 0.40.16 server (the lab's 0.33.3 breaks image prompts)
MODEL = "gemma4:e4b"
THINK = True  # gemma4's hidden reasoning; with it off the model answers "hello" for nearly every clip
OPTIONS = {"temperature": 0, "seed": 211, "num_ctx": 65536, "num_predict": 2048}  # as in llm_harness.py
NUM_FRAMES, FRAME_SIZE = 16, 288
BLACK_FRAME_MAX = 30  # a frame whose brightest pixel is below this is completely black

LABELS = ["hello", "yes", "no", "please", "need", "good", "nice", "eat", "tired", "want",
          "like", "deaf", "mother", "father", "teacher", "learn", "school", "happy", "sad", "finish"]

# The answer must be JSON: "reasoning" first, then "answer", which must be one of the 20 labels
SCHEMA = {"type": "object",
          "properties": {"reasoning": {"type": "string"}, "answer": {"type": "string", "enum": LABELS}},
          "required": ["reasoning", "answer"]}

# Prompt v6, exactly as in llm_harness.py (with {n_frames} = 16)
PROMPT = """<role>
You are an expert American Sign Language (ASL) interpreter who can recognize signs
from still frames of a video.
</role>

<task>
You will see 16 frames, in order (Frame 1 to Frame 16), taken evenly from a short
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
</format>"""

# The 3 few-shot example clips (training clips, never scored) with their signer box
# [top, left, bottom, right] and the black rectangles [left, top, right, bottom] over
# on-screen text, as fractions of the frame (from data/train.json and data/caption_covers.csv)
FEWSHOT = [
    {"example_id": "train_0547", "word": "sad", "video": HERE / "fewshot_1_sad.mp4",
     "box": [0.14938220381736755, 0.2206026315689087, 1.0, 0.6445947289466858], "covers": []},
    {"example_id": "train_0011", "word": "deaf", "video": HERE / "fewshot_2_deaf.mp4",
     "box": [0.17408037185668945, 0.3118080496788025, 1.0, 0.712863028049469], "covers": [[0.66, 0.1, 0.92, 0.33]]},
    {"example_id": "train_0083", "word": "eat", "video": HERE / "fewshot_3_eat.mp4",
     "box": [0.048593372106552124, 0.24541544914245605, 1.0, 0.6785926818847656], "covers": [[0.78, 0.03, 0.92, 0.18]]},
]

# The test sample (example j001), with its box and text cover, as in ml_inference.py
SAMPLE = HERE / "sample.mp4"
SAMPLE_BOX = [0.05659377574920654, 0.06644341349601746, 0.9990754723548889, 0.7649063467979431]
SAMPLE_COVERS = [[0.0, 0.78, 0.36, 1.0]]


def read_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def prepare_frame(frame, box, covers):
    """Cover the text, crop to the signer, pad to a square with black and resize one frame."""
    frame = frame.copy()
    height, width = frame.shape[:2]
    for left, top, right, bottom in covers:
        frame[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 0
    top, left, bottom, right = box
    frame = frame[int(top * height):int(bottom * height), int(left * width):int(right * width)]
    height, width = frame.shape[:2]
    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    frame = cv2.copyMakeBorder(frame, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
                               cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))


def clip_images(video_path, box, covers):
    """16 evenly spaced frames (skipping completely black ones), as JPEG bytes like save_frames.py.
    Only the image bytes are sent, never a file name, so the label cannot leak."""
    frames = read_frames(video_path)
    frames = [f for f in frames if f.max() >= BLACK_FRAME_MAX] or frames
    picks = np.linspace(0, len(frames) - 1, NUM_FRAMES).round().astype(int)
    return [cv2.imencode(".jpg", prepare_frame(frames[i], box, covers))[1].tobytes() for i in picks]


def build_messages(images):
    """The chat for one clip: each example clip as a user turn, followed by its reasoning and
    correct answer as the model's own reply; then the new clip."""
    messages = []
    for k, example in enumerate(FEWSHOT):
        intro = PROMPT + "\n\n" if k == 0 else ""
        messages.append({"role": "user", "content": f"{intro}Example clip {k + 1}:",
                         "images": clip_images(example["video"], example["box"], example["covers"])})
        reply = {"reasoning": FEWSHOT_REASONING[example["example_id"]], "answer": example["word"]}
        messages.append({"role": "assistant", "content": json.dumps(reply)})
    messages.append({"role": "user", "content": "New clip:", "images": images})
    return messages


def predict(video_path, box, covers):
    """The predicted word (None if the answer is not one of the 20 words) and the reasoning."""
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(model=MODEL, messages=build_messages(clip_images(video_path, box, covers)),
                           format=SCHEMA, options=OPTIONS, think=THINK)
    try:
        output = json.loads(response.message.content)
    except json.JSONDecodeError:
        return None, response.message.content
    answer = output.get("answer")
    return (answer if answer in LABELS else None), output.get("reasoning")


if __name__ == "__main__":
    word, reasoning = predict(SAMPLE, SAMPLE_BOX, SAMPLE_COVERS)
    print(f"Reasoning: {reasoning}")
    print(f"Prediction for {SAMPLE.name}: {word or 'INVALID'}")
