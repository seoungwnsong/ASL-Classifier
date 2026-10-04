"""Draw the motion map (and the head) on TRAINING clips, to check that motion.py works.

The motion map is shown as a heat map over the middle frame: red/yellow where
the hands moved a lot, no colour where nothing moved. The green circle is the head.

Saves one picture per clip in results/figures/motion_debug/ and one overview
picture as results/figures/motion_debug.png. Uses the same random training
clips as show_head_debug.py.

Run from the repository root:
    python src/algorithmic/show_motion_debug.py          # 20 random training clips
    python src/algorithmic/show_motion_debug.py 40       # 40 clips
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from head import find_head
from motion import motion_map, motion_masks, read_frames
from show_head_debug import make_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "figures" / "motion_debug"


def draw_motion(frame, energy, head):
    """Blend the motion map over the frame as a heat map, and draw the head circle."""
    picture = frame.copy()
    if energy is not None and energy.max() > 0:
        scaled = (energy / energy.max() * 255).astype(np.uint8)    # brightest motion = 255
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        moved = scaled > 20                                        # only colour where something moved
        picture[moved] = cv2.addWeighted(frame, 0.4, heat, 0.6, 0)[moved]
    if head is not None:
        x, y, radius = (int(v) for v in head)
        cv2.circle(picture, (x, y), radius, (0, 255, 0), 3)
    return picture


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    train = metadata[metadata["split"] == "train"]
    sample = train.sample(n=count, random_state=0)  # same clips as show_head_debug.py

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pictures = []
    for _, row in sample.iterrows():
        video_path = REPO_ROOT / row["clip_path"]
        frames = read_frames(video_path)
        energy = motion_map(motion_masks(frames))
        picture = draw_motion(frames[len(frames) // 2], energy, find_head(video_path))
        label = f"{row['example_id']} {row['word']}"
        cv2.putText(picture, label, (10, picture.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / f"{row['example_id']}.png"), picture)
        pictures.append(picture)

    cv2.imwrite(str(OUT_DIR.parent / "motion_debug.png"), make_grid(pictures))
    print(f"Saved pictures in {OUT_DIR.relative_to(REPO_ROOT)} and the overview "
          f"{(OUT_DIR.parent / 'motion_debug.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
