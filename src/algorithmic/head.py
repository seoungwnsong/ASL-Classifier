"""Find the signer's head in a video with the Hough circle transform.

No learned model is used: the head is the strongest circle of a head-like size,
in the upper part of the frame, whose inside is mostly skin-coloured
(cv2.HoughCircles = Canny edges + circle voting; skin = a fixed colour range).

References:
- OpenCV Hough Circle Transform tutorial,
  https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
- Chai & Ngan (1999), skin colour range in YCrCb (see settings.py)
"""

import cv2
import numpy as np

import settings


def read_first_frames(video_path, count):
    """Return up to `count` frames from the start of the video."""
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def skin_mask(frame):
    """Black-and-white picture: white (255) where the pixel colour is skin-like."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, settings.SKIN_LOWER, settings.SKIN_UPPER)


def skin_fraction(skin, x, y, radius):
    """Share of the pixels inside the circle that are skin-coloured (0 to 1)."""
    inside = np.zeros_like(skin)
    cv2.circle(inside, (int(x), int(y)), int(radius), 255, -1)  # filled circle
    total = np.count_nonzero(inside)
    return np.count_nonzero(skin & inside) / total if total else 0.0


def find_head_in_frame(frame):
    """Return the head circle (x, y, radius) in pixels for one frame, or None."""
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, settings.HEAD_BLUR)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=settings.HOUGH_DP,
        minDist=settings.HOUGH_MIN_DISTANCE * height,  # allow several candidates
        param1=settings.HOUGH_EDGE,
        param2=settings.HOUGH_VOTES,
        minRadius=int(settings.HEAD_MIN_RADIUS * height),
        maxRadius=int(settings.HEAD_MAX_RADIUS * height),
    )
    if circles is None:
        return None

    # OpenCV returns the circles with the most votes first: keep the first one
    # that is high enough in the frame and mostly skin-coloured inside.
    skin = skin_mask(frame)
    for x, y, radius in circles[0]:
        if y > settings.HEAD_MAX_CENTER_Y * height:
            continue
        if skin_fraction(skin, x, y, radius) < settings.HEAD_MIN_SKIN:
            continue
        return float(x), float(y), float(radius)
    return None


def find_head(video_path):
    """Return the head circle (x, y, radius) for a video, or None if not found.

    The head is searched in the first few frames, and the median of the
    circles found is used, so one bad frame does not move the result.
    """
    frames = read_first_frames(video_path, settings.HEAD_FRAMES)
    circles = [c for c in (find_head_in_frame(f) for f in frames) if c is not None]
    if not circles:
        return None
    x, y, radius = np.median(np.array(circles), axis=0)
    return float(x), float(y), float(radius)
