"""Check that VideoPrism loads and runs on one TRAINING clip (Phase 1 setup test).

Loads the pretrained VideoPrism base model from keras_hub (the weights download
from Kaggle the first time), runs one clip through it, and prints the shapes and
how long it took. Nothing is trained or saved.

Run from the repository root:
    python src/deep_learning/check_videoprism.py
"""

import time
from pathlib import Path

import cv2
import keras_hub
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_FRAMES = 16   # VideoPrism base was trained on 16-frame clips
FRAME_SIZE = 288  # ... of 288 x 288 pixels


def read_clip(video_path):
    """16 evenly spaced frames of the clip, as RGB values between 0 and 1."""
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    picks = np.linspace(0, len(frames) - 1, NUM_FRAMES).round().astype(int)
    clip = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (FRAME_SIZE, FRAME_SIZE))
            for i in picks]
    return np.array(clip, dtype=np.float32) / 255.0


def main():
    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    clip = metadata[metadata["split"] == "train"].iloc[0]
    print(f"Clip: {clip['example_id']} ({clip['word']}), {clip['clip_path']}")

    video = read_clip(REPO_ROOT / clip["clip_path"])
    print(f"Input:  shape {video.shape}, values {video.min():.2f} to {video.max():.2f}")

    start = time.time()
    backbone = keras_hub.models.VideoPrismBackbone.from_preset("videoprism_public_v1_base")
    print(f"Loaded VideoPrism in {time.time() - start:.1f} s")

    start = time.time()
    output = backbone.predict(video[np.newaxis], verbose=0)  # add a batch dimension of 1
    print(f"Output: shape {output.shape}  (clips, frames, patches, numbers per patch)")
    print(f"One clip took {time.time() - start:.1f} s")

    features = output.mean(axis=(1, 2))[0]  # average over frames and patches
    print(f"Averaged: {features.shape[0]} numbers for this clip")


if __name__ == "__main__":
    main()
