"""Turn one clip into two simple answers: WHERE the motion is, and ONE or TWO hands.

1. Find the head (head.py) and the motion in the middle part of the clip (motion.py).
2. Split the area around the head into three rows (top, face, chest), sized by the head.
3. Row  = the highest row with a fair share of the motion (the hand is at the
          top of the moving arm).
4. Hands = "two" if there is similar motion left and right of the head's centre line.
"""

import settings
from head import find_head
from motion import motion_map, motion_masks, read_frames


def middle_frames(frames):
    """The middle part of the clip (skips the hands rising and dropping)."""
    start = int(len(frames) * settings.MIDDLE_START)
    end = int(len(frames) * settings.MIDDLE_END)
    return frames[start:end]


def row_motion(energy, head):
    """Motion per row, divided by the row's height: {"top": ..., "face": ..., "chest": ...}.

    Dividing by the height stops the tall chest row from winning just because
    it is bigger.
    """
    x, y, radius = head
    height, width = energy.shape
    totals = {}
    for name, (upper, lower) in settings.ROWS.items():
        top = max(0, int(y + upper * radius))
        bottom = min(height, int(y + lower * radius))
        totals[name] = float(energy[top:bottom, :].sum()) / (bottom - top) if bottom > top else 0.0
    return totals


def left_right_balance(energy, head):
    """Motion on the weaker side divided by motion on the stronger side (0 to 1)."""
    x, y, radius = head
    upper = min(lower for lower, _ in settings.ROWS.values())   # top of the top row
    lower = max(lower for _, lower in settings.ROWS.values())   # bottom of the chest row
    area = energy[max(0, int(y + upper * radius)):int(y + lower * radius), :]
    left = float(area[:, :int(x)].sum())
    right = float(area[:, int(x):].sum())
    if max(left, right) == 0:
        return 0.0
    return min(left, right) / max(left, right)


def clip_zones(video_path):
    """Return {"row", "hands", plus the raw numbers} for one clip.

    "row" is None when no head or no motion was found (the rule then uses its fallback).
    """
    head = find_head(video_path)
    frames = middle_frames(read_frames(video_path))
    energy = motion_map(motion_masks(frames))
    if head is None or energy is None or energy.sum() == 0:
        return {"row": None, "hands": None}

    totals = row_motion(energy, head)
    balance = left_right_balance(energy, head)

    # The hand is at the top of the moving arm: take the highest row (ROWS is
    # listed top to bottom) with a fair share of the busiest row's motion.
    busiest = max(totals.values())
    row = next(name for name in settings.ROWS if totals[name] >= settings.ROW_MIN_SHARE * busiest)

    return {
        "row": row,
        "hands": "two" if balance >= settings.TWO_HANDS_BALANCE else "one",
        "top": round(totals["top"], 1),
        "face": round(totals["face"], 1),
        "chest": round(totals["chest"], 1),
        "balance": round(balance, 2),
    }
