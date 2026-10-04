"""Find where things move in a video, with frame differencing.

No learned model is used: a pixel "moved" if its brightness changed a lot
between two frames in a row (and, optionally, if it is skin-coloured).

The motion map adds up the moving pixels of every frame into one picture:
bright where the hands moved often, black where nothing moved. This is a
"motion energy image" (Bobick & Davis, 2001, "The recognition of human
movement using temporal templates", IEEE PAMI).
"""

import cv2
import numpy as np

import settings
from head import skin_mask


def read_frames(video_path):
    """Return every frame of the video as a list."""
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def motion_masks(frames):
    """For each pair of frames in a row, a black-and-white picture of what moved.

    Returns a list with one mask per frame after the first (white = moved).
    """
    kernel = np.ones((settings.MOTION_CLEAN, settings.MOTION_CLEAN), np.uint8)
    masks = []
    previous = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (settings.MOTION_BLUR, settings.MOTION_BLUR), 0)
        if previous is not None:
            difference = cv2.absdiff(gray, previous)
            _, moved = cv2.threshold(difference, settings.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            if settings.MOTION_SKIN_ONLY:
                moved = cv2.bitwise_and(moved, skin_mask(frame))
            moved = cv2.morphologyEx(moved, cv2.MORPH_OPEN, kernel)  # remove tiny specks
            masks.append(moved)
        previous = gray
    return masks


def motion_map(masks):
    """Add up the motion masks: for each pixel, the share of frames in which it moved (0 to 1)."""
    if not masks:
        return None
    total = np.zeros(masks[0].shape, np.float32)
    for mask in masks:
        total += mask > 0
    return total / len(masks)
