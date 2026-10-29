"""Every number and path used by the deep learning approach, in one place.

These values are chosen with the TRAINING and VALIDATION clips only.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- Pretrained model ----

# VideoPrism base from keras_hub, kept frozen (transfer learning, no fine-tuning).
PRESET = "videoprism_public_v1_base"

# VideoPrism base was trained on 16-frame clips of 288 x 288 pixels.
NUM_FRAMES = 16
FRAME_SIZE = 288

# A frame is "completely black" if even its brightest pixel is below this (0-255).
# 49 of the 1002 train/val clips have one such frame, usually the last one.
# (Clips with a black background are not affected: the signer is still bright.)
BLACK_FRAME_MAX = 30

# ---- Input (same view of the signer that the human participants saw) ----

# Many MS-ASL source videos have the answer word written on screen. Like the human
# quiz (src/human_baseline/quiz/app.py), we crop to the signer box and draw a black
# rectangle over the text. data/caption_covers.csv has one rectangle per YouTube video:
# [left, top, right, bottom] as fractions of the full frame. The 4 test rows are the
# same ones the quiz used; the train and val rows were found by looking at 2 clips
# of every source video (frames at 0%, 50% and 100%), then checked again on the
# cropped, covered model input (2 clips x 2 frames per video).

# ---- Files ----

METADATA = REPO_ROOT / "data" / "metadata.csv"
CAPTION_COVERS = REPO_ROOT / "data" / "caption_covers.csv"
SPLIT_JSONS = [REPO_ROOT / "data" / f"{split}.json" for split in ("train", "val", "test")]
WLASL_METADATA = REPO_ROOT / "data" / "raw" / "WLASL_v0.3.json"
