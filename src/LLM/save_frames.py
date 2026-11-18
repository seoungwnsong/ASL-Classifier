"""Input conversion for the LLM approach: save N_FRAMES still frames per clip, ONCE.

The LLM cannot watch a video, so each clip becomes a few still frames, sent as images.
The frames are prepared exactly like the deep learning input (src/deep_learning/frames.py),
which is also the view the human participants saw:
    1. pick N_FRAMES frames spread evenly over the clip (skipping completely black frames)
    2. black out on-screen text (the answer word is written on some source videos)
    3. crop to the signer box, pad to a square, resize

Saves data/llm_frames/n<N_FRAMES>/<example_id>/00.jpg, 01.jpg, ...
The folders are named by example ID only (e.g. train_0012, a001), never by the word,
so nothing about the label can reach the model.

Run from the repository root (after make_metadata.py):
    python src/llm/save_frames.py
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

N_FRAMES = 8
FRAMES_DIR = REPO_ROOT / "data" / "llm_frames" / f"n{N_FRAMES}"


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
    clip_info = load_clip_info()
    done = 0
    for clip in metadata.itertuples():
        out_dir = FRAMES_DIR / clip.example_id
        if len(list(out_dir.glob("*.jpg"))) == N_FRAMES:
            continue  # already saved: never redo the conversion
        save_clip(clip.clip_path, out_dir, clip_info)
        done += 1
        if done % 50 == 0:
            print(f"  saved {done} clips", flush=True)
    print(f"Saved {done} new clips; {len(metadata)} clips in {FRAMES_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
