"""Show what the model actually receives, for a few random TRAINING clips.

Each row is one clip: 4 of its 16 frames after cropping, padding and resizing.
Saves results/figures/dl_input_check.png.

Run from the repository root:
    python src/deep_learning/show_frames.py        (6 clips)
    python src/deep_learning/show_frames.py 10     (10 clips)
"""

import sys

import cv2
import numpy as np
import pandas as pd

import settings
from frames import clip_input, load_clip_info

OUT_FILE = settings.REPO_ROOT / "results" / "figures" / "dl_input_check.png"


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    metadata = pd.read_csv(settings.METADATA)
    train = metadata[metadata["split"] == "train"]
    sample = train.sample(n=count, random_state=0)
    clip_info = load_clip_info()

    rows = []
    for _, clip in sample.iterrows():
        video = clip_input(clip["clip_path"], clip_info)
        print(f"{clip['example_id']} {clip['word']:<8} {clip['source']:<7} "
              f"shape {video.shape}, values {video.min():.2f} to {video.max():.2f}")
        # back to 0-255 BGR so OpenCV can save it; frames 0, 5, 10, 15 of the 16
        picked = [cv2.cvtColor((video[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for i in (0, 5, 10, 15)]
        row = np.hstack(picked)
        label = f"{clip['example_id']} {clip['word']} ({clip['source']})"
        cv2.putText(row, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        rows.append(row)

    cv2.imwrite(str(OUT_FILE), np.vstack(rows))
    print(f"Saved {OUT_FILE.relative_to(settings.REPO_ROOT)}")


if __name__ == "__main__":
    main()
