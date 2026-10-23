""" Algorthmic Approach to ASL Classification

Main Pipeline:
    Step 1. Detect head: Hough circle transform + a fixed skin-colour range
    Step 2. Finding motion: Comparing Frames + skin-coloured pixels (Focus on the hands)
    Step 3. Extracting 7 measurements: row, hands, repeated, direction, axis, hand size, hands apart
    Step 4. Result: Output the word whose hand-written description matches the most measurements

"""

import sys
import cv2
import numpy as np
from pathlib import Path


HEAD_FRAMES = 5 # Look for the head in the first 5 frames of the clip
# Radius of the head, 6% to 16% of the frame height, and the head centre must be in the top 60%
#  of the frame. At least 30% of the circle must be skin-coloured.
HEAD_MIN_RADIUS = 0.06
HEAD_MAX_RADIUS = 0.16
HEAD_MAX_CENTER_Y = 0.6
HEAD_MIN_SKIN = 0.3

# skin colour range in YCrCb (Chai & Ngan, 1999)
SKIN_LOWER = (0, 133, 77)
SKIN_UPPER = (255, 173, 127)

MOTION_THRESHOLD = 25 # Thresholds for motion detection for brighness change. Range (0-255)
MIDDLE_START, MIDDLE_END = 0.2, 0.8 # We only use the middle 60% of the clip.

ROWS = {"top": (-2.0, -0.3), "face": (-0.3, 1.3), "chest": (1.3, 5.0)} # 3 horizontal rows: top, face, and chest.
ROW_MIN_SHARE = 0.5 # highest row with at least half the busiest row's motion
TWO_HANDS_BALANCE = 0.5 # two hands if the weaker side has at least half the motion
BURST_LEVEL = 0.7 # a burst = motion rising above 70% of its maximum
DIRECTION_MIN = 0.5 # up/down if the hand moves more than half a head radius
MIN_BLOB = 0.05 # ignore moving blobs smaller than 5% of the head area
HAND_SIZE_BIG = 0.16 # Threshold for a big hand: 16% of the head area
HANDS_CLOSE, HANDS_FAR = 1.0, 2.5 # distance between hands, in head widths

# Rules for 20 Words using 7 measurements 
# 7 Measurements: row, hands, repeated, direction, axis, hand size, hands apart
# None = not used - considered hands apart only for two-handed signs
MEASUREMENTS = ["row", "hands", "repeated", "direction", "axis", "hand_size", "hands_apart"]
SIGNS = {
    "hello":   ("top",   "one", "no",  "none", "side",    "big",   None),
    "father":  ("top",   "one", "yes", "none", "up-down", "big",   None),
    "learn":   ("top",   "one", "no",  "up",   "up-down", "small", None),
    "teacher": ("top",   "two", "yes", "none", "up-down", "small", "medium"),
    "mother":  ("face",  "one", "yes", "none", "up-down", "big",   None),
    "eat":     ("face",  "one", "yes", "none", "up-down", "small", None),
    "deaf":    ("face",  "one", "no",  "none", "side",    "small", None),
    "good":    ("face",  "one", "no",  "down", "up-down", "big",   None),
    "sad":     ("face",  "two", "no",  "down", "up-down", "big",   "medium"),
    "no":      ("face",  "one", "yes", "none", "side",    "small", None),
    "yes":     ("chest", "one", "yes", "none", "up-down", "small", None),
    "need":    ("chest", "one", "yes", "down", "up-down", "small", None),
    "please":  ("chest", "one", "yes", "none", "side",    "big",   None),
    "like":    ("chest", "one", "no",  "none", "side",    "small", None),
    "happy":   ("chest", "two", "yes", "up",   "up-down", "big",   "medium"),
    "tired":   ("chest", "two", "no",  "down", "up-down", "big",   "medium"),
    "school":  ("chest", "two", "yes", "none", "up-down", "big",   "close"),
    "want":    ("chest", "two", "no",  "none", "side",    "small", "medium"),
    "finish":  ("chest", "two", "no",  "none", "side",    "big",   "far"),
    "nice":    ("chest", "two", "no",  "none", "side",    "big",   "close"),
}
DEFAULT_WORD = "yes"  # This is used when there is no head or motion is found.

# Step 1. Detect head: Hough circle transform + a fixed skin-colour range
def skin_mask(frame):
    """White (255) where the pixel colour is in the skin range."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, SKIN_LOWER, SKIN_UPPER)

def find_head_in_frame(frame):
    """The strongest head-sized, skin-coloured circle in the frame: (x, y, radius), or None."""
    height = frame.shape[0]
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.1 * height,
                               param1=100, param2=30,
                               minRadius=int(HEAD_MIN_RADIUS * height),
                               maxRadius=int(HEAD_MAX_RADIUS * height))
    if circles is None:
        return None
    skin = skin_mask(frame)
    for x, y, radius in circles[0]:
        inside = np.zeros_like(skin)
        cv2.circle(inside, (int(x), int(y)), int(radius), 255, -1)
        skin_share = np.count_nonzero(skin & inside) / np.count_nonzero(inside)
        if y <= HEAD_MAX_CENTER_Y * height and skin_share >= HEAD_MIN_SKIN:
            return float(x), float(y), float(radius)
    return None

def find_head(frames):
    """Median of the head circles found in the first few frames, or None."""
    circles = [c for c in (find_head_in_frame(f) for f in frames[:HEAD_FRAMES]) if c is not None]
    if not circles:
        return None
    x, y, radius = np.median(np.array(circles), axis=0)
    return float(x), float(y), float(radius)


# Step 2. Finding motion: Comparing Frames + skin-coloured pixels (Focus on the hands)
def motion_masks(frames):
    """For each frame, white where a skin-coloured pixel changed a lot since the previous frame."""
    masks = []
    previous = None
    for frame in frames:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if previous is not None:
            _, moved = cv2.threshold(cv2.absdiff(gray, previous), MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            moved = cv2.bitwise_and(moved, skin_mask(frame))
            moved = cv2.morphologyEx(moved, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            masks.append(moved)
        previous = gray
    return masks

def frame_blobs(mask, min_area):
    """The moving blobs in one mask, biggest first: (area, centre x, centre y)."""
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centres[i][0], centres[i][1])
             for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return sorted(blobs, reverse=True)


# Step 3. Extracting 7 measurements: row, hands, repeated, direction, axis, hand size, hands apart
def measure(frames):
    """The 7 measurements for a clip, or None if no head or no motion was found."""
    head = find_head(frames)
    middle = frames[int(len(frames) * MIDDLE_START):int(len(frames) * MIDDLE_END)]
    masks = motion_masks(middle)
    if head is None or not masks:
        return None
    energy = sum((m > 0).astype(np.float32) for m in masks) / len(masks) 
    if energy.sum() == 0:
        return None
    x, y, radius = head
    head_area = np.pi * radius ** 2
    blobs = [frame_blobs(m, MIN_BLOB * head_area) for m in masks]
    positions = [(b[0][1], b[0][2]) for b in blobs if b]
    if len(positions) < 2:
        return None
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # 1. row
    density = {}
    for name, (upper, lower) in ROWS.items():
        top = max(0, int(y + upper * radius))
        bottom = min(energy.shape[0], int(y + lower * radius))
        density[name] = energy[top:bottom].sum() / (bottom - top) if bottom > top else 0.0
    row = next(name for name in ROWS if density[name] >= ROW_MIN_SHARE * max(density.values()))

    # 2. hands
    left, right = energy[:, :int(x)].sum(), energy[:, int(x):].sum()
    hands = "two" if max(left, right) > 0 and min(left, right) / max(left, right) >= TWO_HANDS_BALANCE else "one"

    # 3. repeated
    amount = np.array([np.count_nonzero(m) for m in masks], dtype=float)
    smooth = np.convolve(amount, np.ones(3) / 3, mode="same")
    above = smooth > BURST_LEVEL * smooth.max()
    bursts = np.count_nonzero(above[1:] & ~above[:-1]) + int(above[0])
    repeated = "yes" if bursts >= 2 else "no"

    # 4. direction
    half = len(ys) // 2
    change = (np.median(ys[half:]) - np.median(ys[:half])) / radius
    direction = "down" if change > DIRECTION_MIN else "up" if change < -DIRECTION_MIN else "none"

    # 5. axis
    sideways = np.percentile(xs, 90) - np.percentile(xs, 10)
    vertical = np.percentile(ys, 90) - np.percentile(ys, 10)
    axis = "side" if sideways > vertical else "up-down"

    # 6. hand size
    sizes = [b[0][0] for b in blobs if b]
    hand_size = "big" if np.median(sizes) / head_area >= HAND_SIZE_BIG else "small"

    # 7. hands apart
    gaps = [np.hypot(b[0][1] - b[1][1], b[0][2] - b[1][2]) / (2 * radius) for b in blobs if len(b) >= 2]
    gap = np.median(gaps) if gaps else 0.0
    hands_apart = "close" if gap < HANDS_CLOSE else "far" if gap > HANDS_FAR else "medium"

    return {"row": row, "hands": hands, "repeated": repeated, "direction": direction,
            "axis": axis, "hand_size": hand_size, "hands_apart": hands_apart}


# Step 4. Result: Output the word whose hand-written description matches the most measurements
def predict(video_path):
    """Read the video and return the predicted word."""
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    features = measure(frames)
    if features is None:
        return DEFAULT_WORD
    def score(word):
        """Share of the word's measurements (ignoring None) that match the clip."""
        wanted = [(m, v) for m, v in zip(MEASUREMENTS, SIGNS[word]) if v is not None]
        return sum(features[m] == v for m, v in wanted) / len(wanted)
    return max(SIGNS, key=score)  # if two words tie, the one listed first wins


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        video = Path(__file__).parent / "sample_test_clip.mp4"
    print(f"{video.name}: predicted word: {predict(video)}")
