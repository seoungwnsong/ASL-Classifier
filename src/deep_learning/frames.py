"""Turn one clip into the input VideoPrism expects: 16 frames of 288 x 288 RGB pixels.

For each clip:
1. pick 16 frames spread evenly over the whole clip (skipping completely black frames)
2. black out on-screen text (for the source videos listed in data/caption_covers.csv)
3. crop to the signer box
4. pad with black to a square (so the signer is not stretched), then resize to 288 x 288
5. convert BGR (OpenCV's colour order) to RGB, and scale the values from 0-255 to 0-1
"""

import json

import cv2
import numpy as np
import pandas as pd

import settings


def load_clip_info():
    """For every clip: its signer box [top, left, bottom, right] (fractions of the frame)
    and its caption covers. Returns a dict keyed by clip_path, as in data/metadata.csv."""
    covers = pd.read_csv(settings.CAPTION_COVERS)
    cover_of = {row.video_id: [row.left, row.top, row.right, row.bottom] for row in covers.itertuples()}

    info = {}
    wlasl_paths = []
    for split_json in settings.SPLIT_JSONS:
        for entry in json.loads(split_json.read_text()):
            if entry["source"] == "MS-ASL":
                clip_covers = [rect for video_id, rect in cover_of.items() if video_id in entry["url"]]
                info[entry["clip_path"]] = {"box": entry["box"], "covers": clip_covers}
            else:
                wlasl_paths.append(entry["clip_path"])

    # WLASL clips: the box comes from WLASL's own metadata, as [x1, y1, x2, y2] in pixels
    wlasl_boxes = {}
    for gloss in json.loads(settings.WLASL_METADATA.read_text()):
        for instance in gloss["instances"]:
            wlasl_boxes[instance["video_id"]] = instance["bbox"]
    for clip_path in wlasl_paths:
        x1, y1, x2, y2 = wlasl_boxes[clip_path.split("/")[-1].removesuffix(".mp4")]
        capture = cv2.VideoCapture(str(settings.REPO_ROOT / clip_path))
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        capture.release()
        info[clip_path] = {"box": [y1 / height, x1 / width, y2 / height, x2 / width], "covers": []}
    return info


def read_frames(video_path):
    """All frames of the video, as a list of BGR images."""
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
    """Cover the text, crop to the signer, pad to a square and resize one frame."""
    frame = frame.copy()
    height, width = frame.shape[:2]
    for left, top, right, bottom in covers:
        frame[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 0

    top, left, bottom, right = box
    frame = frame[int(top * height):int(bottom * height), int(left * width):int(right * width)]

    # pad the shorter side with black (equally on both sides) to make a square
    height, width = frame.shape[:2]
    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    frame = cv2.copyMakeBorder(frame, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
                               cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(frame, (settings.FRAME_SIZE, settings.FRAME_SIZE))


def clip_input(clip_path, clip_info):
    """The model input for one clip: shape (16, 288, 288, 3), values between 0 and 1."""
    frames = read_frames(settings.REPO_ROOT / clip_path)
    # skip completely black frames (some videos end on one); dark backgrounds still have bright pixels
    frames = [f for f in frames if f.max() >= settings.BLACK_FRAME_MAX] or frames
    picks = np.linspace(0, len(frames) - 1, settings.NUM_FRAMES).round().astype(int)
    info = clip_info[clip_path]
    clip = [cv2.cvtColor(prepare_frame(frames[i], info["box"], info["covers"]), cv2.COLOR_BGR2RGB)
            for i in picks]
    return np.array(clip, dtype=np.float32) / 255.0
