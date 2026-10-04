"""Draw the detected head on TRAINING clips, to check that head.py works.

Saves one picture per clip in results/figures/head_debug/ and one overview
picture (all clips in a grid) as results/figures/head_debug.png.

Only training clips are used: the test set must not be looked at while the
rules are being developed.

Run from the repository root:
    python src/algorithmic/show_head_debug.py          # 20 random training clips
    python src/algorithmic/show_head_debug.py 40       # 40 clips
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from head import find_head, read_first_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "figures" / "head_debug"
TILE_HEIGHT = 240  # size of each picture in the overview grid
GRID_COLUMNS = 5


def draw_head(frame, head):
    """Draw the head circle, plus lines at the forehead and chin height."""
    picture = frame.copy()
    if head is None:
        cv2.putText(picture, "no head found", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return picture
    x, y, radius = (int(v) for v in head)
    cv2.circle(picture, (x, y), radius, (0, 255, 0), 3)          # head outline
    cv2.circle(picture, (x, y), 4, (0, 255, 0), -1)               # head centre
    width = picture.shape[1]
    forehead_y = y - radius // 2
    chin_y = y + radius
    cv2.line(picture, (0, forehead_y), (width, forehead_y), (255, 200, 0), 2)  # forehead (blue)
    cv2.line(picture, (0, chin_y), (width, chin_y), (0, 165, 255), 2)          # chin (orange)
    return picture


def make_grid(pictures):
    """Resize the pictures to the same height and arrange them in a grid."""
    tiles = []
    for picture in pictures:
        scale = TILE_HEIGHT / picture.shape[0]
        tiles.append(cv2.resize(picture, (int(picture.shape[1] * scale), TILE_HEIGHT)))
    tile_width = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, tile_width - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % GRID_COLUMNS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + GRID_COLUMNS]) for i in range(0, len(tiles), GRID_COLUMNS)]
    return np.vstack(rows)


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    train = metadata[metadata["split"] == "train"]
    sample = train.sample(n=count, random_state=0)  # same clips every run

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pictures = []
    found = 0
    for _, row in sample.iterrows():
        video_path = REPO_ROOT / row["clip_path"]
        head = find_head(video_path)
        found += head is not None
        frame = read_first_frames(video_path, 1)[0]
        picture = draw_head(frame, head)
        label = f"{row['example_id']} {row['word']}"
        cv2.putText(picture, label, (10, picture.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / f"{row['example_id']}.png"), picture)
        pictures.append(picture)

    cv2.imwrite(str(OUT_DIR.parent / "head_debug.png"), make_grid(pictures))
    print(f"Head found in {found}/{count} training clips")
    print(f"Saved pictures in {OUT_DIR.relative_to(REPO_ROOT)} and the overview "
          f"{(OUT_DIR.parent / 'head_debug.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
