"""Turn one clip into 7 simple measurements of where and how the hands move.

1. row          top / face / chest      where the hand is (highest busy row around the head)
2. hands        one / two               motion on one side of the head, or both
3. repeated     yes / no                does the amount of motion rise and fall several times
4. direction    up / down / none        does the hand end higher or lower than it started
5. axis         up-down / side          does the hand travel more vertically or sideways
6. hand_size    small / big             size of the moving hand compared to the head
7. hands_apart  close / medium / far    distance between the two moving hands

Only the middle part of the clip is used, and every distance is measured in
head radii, using the head found by head.py. No learned model is used.
"""

import cv2
import numpy as np

import settings
from head import find_head
from motion import motion_map, motion_masks, read_frames


def middle_frames(frames):
    """The middle part of the clip (skips the hands rising and dropping)."""
    start = int(len(frames) * settings.MIDDLE_START)
    end = int(len(frames) * settings.MIDDLE_END)
    return frames[start:end]


# ---- 1 and 2: from the motion map (all middle frames added up) ----

def measure_row(energy, head):
    """The highest row (top, face, chest) with a fair share of the motion."""
    x, y, radius = head
    density = {}
    for name, (upper, lower) in settings.ROWS.items():
        top = max(0, int(y + upper * radius))
        bottom = min(energy.shape[0], int(y + lower * radius))
        # motion divided by the row's height, so the tall chest row does not win by size
        density[name] = energy[top:bottom].sum() / (bottom - top) if bottom > top else 0.0
    busiest = max(density.values())
    # ROWS is listed top to bottom; the hand is at the top of the moving arm
    return next(name for name in settings.ROWS if density[name] >= settings.ROW_MIN_SHARE * busiest)


def measure_hands(energy, head):
    """"two" if the motion left and right of the head's centre line is similar, else "one"."""
    x = int(head[0])
    left = energy[:, :x].sum()
    right = energy[:, x:].sum()
    if max(left, right) == 0:
        return "one"
    return "two" if min(left, right) / max(left, right) >= settings.TWO_HANDS_BALANCE else "one"


# ---- 3 to 7: from each frame's moving blobs ----

def frame_blobs(mask, min_area):
    """The moving blobs in one motion mask, biggest first: list of (area, centre_x, centre_y)."""
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centres[i][0], centres[i][1])
             for i in range(1, count)  # label 0 is the background
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return sorted(blobs, reverse=True)


def measure_repeated(masks):
    """"yes" if the amount of motion rises above BURST_LEVEL x its maximum at least twice."""
    amount = np.array([np.count_nonzero(m) for m in masks], dtype=float)
    smooth = np.convolve(amount, np.ones(settings.SMOOTH_FRAMES) / settings.SMOOTH_FRAMES, mode="same")
    above = smooth > settings.BURST_LEVEL * smooth.max()
    bursts = np.count_nonzero(above[1:] & ~above[:-1]) + int(above[0])  # count each rise
    return "yes" if bursts >= settings.REPEATED_MIN_BURSTS else "no"


def measure_direction(positions, radius):
    """Compare the hand's height in the first and second half of the middle part."""
    heights = [y for _, y in positions]
    half = len(heights) // 2
    change = (np.median(heights[half:]) - np.median(heights[:half])) / radius  # down = positive
    if change > settings.DIRECTION_MIN:
        return "down"
    if change < -settings.DIRECTION_MIN:
        return "up"
    return "none"


def measure_axis(positions):
    """"side" if the hand travels further left-right than up-down, else "up-down"."""
    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    sideways = np.percentile(xs, 90) - np.percentile(xs, 10)   # ignores the most extreme 10%
    vertical = np.percentile(ys, 90) - np.percentile(ys, 10)
    return "side" if sideways > vertical else "up-down"


def measure_hand_size(blobs_per_frame, head_area):
    """"big" if the biggest moving blob is usually large compared to the head."""
    sizes = [blobs[0][0] for blobs in blobs_per_frame if blobs]
    return "big" if np.median(sizes) / head_area >= settings.HAND_SIZE_BIG else "small"


def measure_hands_apart(blobs_per_frame, radius):
    """Typical distance between the two biggest moving blobs, in head widths."""
    gaps = [np.hypot(b[0][1] - b[1][1], b[0][2] - b[1][2]) / (2 * radius)
            for b in blobs_per_frame if len(b) >= 2]
    if not gaps:
        return "close"  # the two hands were never seen apart
    gap = np.median(gaps)
    if gap < settings.HANDS_CLOSE:
        return "close"
    if gap > settings.HANDS_FAR:
        return "far"
    return "medium"


def clip_features(video_path):
    """Return the 7 measurements for one clip, or None if no head or no motion was found."""
    head = find_head(video_path)
    masks = motion_masks(middle_frames(read_frames(video_path)))
    if head is None or not masks:
        return None
    energy = motion_map(masks)
    if energy.sum() == 0:
        return None

    radius = head[2]
    head_area = np.pi * radius ** 2
    blobs_per_frame = [frame_blobs(m, settings.MIN_BLOB * head_area) for m in masks]
    # the hand's position in each frame = centre of the biggest moving blob
    positions = [(blobs[0][1], blobs[0][2]) for blobs in blobs_per_frame if blobs]
    if len(positions) < 2:
        return None

    return {
        "row": measure_row(energy, head),
        "hands": measure_hands(energy, head),
        "repeated": measure_repeated(masks),
        "direction": measure_direction(positions, radius),
        "axis": measure_axis(positions),
        "hand_size": measure_hand_size(blobs_per_frame, head_area),
        "hands_apart": measure_hands_apart(blobs_per_frame, radius),
    }
