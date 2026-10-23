"""Figure 1 of the progress report: two failure cases of the algorithmic approach.

(a) q002, "teacher" predicted as "tired": the hands are at the head, but the motion
    of the forearms below them puts most of the motion in the chest row.
(b) j001, "mother" predicted as "no": only the moving fingers of the open hand are
    detected (the still palm and thumb are lost), so the hand is measured as small.

Saves "Progress Report/figure_1a_q002.png" and "Progress Report/figure_1b_j001.png".

Run from the repository root:
    python src/algorithmic/make_failure_figure.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import settings
from features import frame_blobs, middle_frames
from head import find_head
from motion import motion_map, motion_masks, read_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Progress Report"
PANEL_HEIGHT = 720  # both images are saved at the same height
GREEN, ORANGE, RED, WHITE = (0, 200, 0), (0, 140, 255), (40, 40, 230), (255, 255, 255)


def clip_path(example_id):
    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv").set_index("example_id")
    return REPO_ROOT / metadata.loc[example_id, "clip_path"]


def draw_head(picture, head):
    x, y, radius = (int(v) for v in head)
    cv2.circle(picture, (x, y), radius, GREEN, 3)


def panel_rows(example_id):
    """Middle frame with the motion map (heat), the head circle and the three rows."""
    path = clip_path(example_id)
    frames = read_frames(path)
    head = find_head(path)
    energy = motion_map(motion_masks(middle_frames(frames)))
    picture = frames[len(frames) // 2].copy()

    scaled = (energy / energy.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    moved = scaled > 20
    picture[moved] = cv2.addWeighted(picture, 0.4, heat, 0.6, 0)[moved]

    x, y, radius = head
    for name, (upper, lower) in settings.ROWS.items():
        for edge in (upper, lower):
            row_y = int(y + edge * radius)
            cv2.line(picture, (0, row_y), (picture.shape[1], row_y), ORANGE, 3)
        label_y = int(y + (upper + lower) / 2 * radius)
        cv2.putText(picture, name, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, ORANGE, 4)
    draw_head(picture, head)
    return picture


def panel_hand(example_id):
    """The middle-part frame with the most motion, with the detected moving skin area in red."""
    path = clip_path(example_id)
    frames = middle_frames(read_frames(path))
    head = find_head(path)
    masks = motion_masks(frames)
    head_area = np.pi * head[2] ** 2
    sizes = [frame_blobs(m, settings.MIN_BLOB * head_area) for m in masks]
    best = max(range(len(masks)), key=lambda i: sizes[i][0][0] if sizes[i] else 0)

    picture = frames[best + 1].copy()  # masks[i] compares frame i with frame i + 1
    picture[masks[best] > 0] = RED
    contours, _ = cv2.findContours(masks[best], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(picture, contours, -1, WHITE, 1)
    draw_head(picture, head)
    return picture


def resized(picture):
    """Scale the picture to PANEL_HEIGHT pixels tall."""
    scale = PANEL_HEIGHT / picture.shape[0]
    return cv2.resize(picture, (int(picture.shape[1] * scale), PANEL_HEIGHT))


def main():
    for name, picture in [("figure_1a_q002.png", panel_rows("q002")),
                          ("figure_1b_j001.png", panel_hand("j001"))]:
        cv2.imwrite(str(OUT_DIR / name), resized(picture))
        print(f"Saved Progress Report/{name}")


if __name__ == "__main__":
    main()
