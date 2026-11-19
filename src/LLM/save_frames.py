"""Input conversion for the LLM approach: save N_FRAMES still frames per clip, ONCE.

The LLM cannot watch a video, so each clip becomes a few still frames, sent as images.
The frames are prepared exactly like the deep learning input (src/deep_learning/frames.py),
which is also the view the human participants saw:
    1. pick N_FRAMES frames spread evenly over the clip (skipping completely black frames)
    2. black out on-screen text (the answer word is written on some source videos)
    3. crop to the signer box, pad to a square, resize

Only the clips of SPLITS are saved: the same train, val and test clips as the deep
learning approach, all with the same number of frames, so llm_harnness.py reads
N_FRAMES frames per clip whichever split it runs on.

Saves data/llm_frames/n<N_FRAMES>/<folder>/<example_id>/00.jpg, 01.jpg, ...
where <folder> is one of
    fewshot   the few-shot example clips listed in fewshot.py (training clips, never scored)
    train     every other training clip
    val       the validation clips
    test      the test clips
The clip folders are named by example ID only (e.g. train_0012, a001), never by the word,
so nothing about the label can reach the model.

Run from the repository root (after make_metadata.py):
    python src/LLM/save_frames.py
To try a different number of frames, change N_FRAMES and run again: each number of
frames is saved in its own folder, so the old frames are kept.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "deep_learning"))
import settings as dl_settings  # noqa: E402  (deep learning settings: paths, black-frame threshold)
from frames import load_clip_info, prepare_frame, read_frames  # noqa: E402
from fewshot import FEWSHOT  # noqa: E402  (src/LLM/fewshot.py)

N_FRAMES = 16  # same number of frames per clip as the deep learning input
SPLITS = ["train", "val", "test"]  # same clips as the deep learning approach
FRAMES_DIR = REPO_ROOT / "data" / "llm_frames" / f"n{N_FRAMES}"
FEWSHOT_IDS = {example["example_id"] for example in FEWSHOT}


def clip_dir(example_id, split):
    """The folder of one clip's frames: few-shot examples in fewshot/, every other clip
    in the folder of its split. llm_harnness.py reads the frames from here too."""
    return FRAMES_DIR / ("fewshot" if example_id in FEWSHOT_IDS else split) / example_id


def save_clip(clip_path, out_dir, clip_info):
    frames = read_frames(REPO_ROOT / clip_path)
    # skip completely black frames (some videos end on one), as frames.py does
    frames = [f for f in frames if f.max() >= dl_settings.BLACK_FRAME_MAX] or frames
    picks = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
    info = clip_info[clip_path]
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(picks):
        frame = prepare_frame(frames[i], info["box"], info["covers"])  # BGR, as cv2.imwrite expects
        cv2.imwrite(str(out_dir / f"{n:02d}.jpg"), frame)


def main():
    metadata = pd.read_csv(dl_settings.METADATA)
    metadata = metadata[metadata["split"].isin(SPLITS)]
    clip_info = load_clip_info()
    done = 0
    for clip in metadata.itertuples():
        out_dir = clip_dir(clip.example_id, clip.split)
        if len(list(out_dir.glob("*.jpg"))) == N_FRAMES:
            continue  # already saved: never redo the conversion
        save_clip(clip.clip_path, out_dir, clip_info)
        done += 1
        if done % 50 == 0:
            print(f"  saved {done} clips", flush=True)
    print(f"Saved {done} new clips; {len(metadata)} clips in {FRAMES_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
