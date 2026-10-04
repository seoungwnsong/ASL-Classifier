"""Every hand-chosen number used by the algorithmic approach, in one place.

These values were picked by looking at TRAINING clips only.
"""

# ---- Head detection (Hough circle transform) ----

# Look for the head in the first few frames, before the hands move in front of it.
HEAD_FRAMES = 5

# Blur strength before edge detection (must be odd). Blurring removes small
# details (hair, texture) so the head outline gives cleaner edges.
HEAD_BLUR = 5

# Head radius as a fraction of the frame height (ignore circles that are too
# small or too big to be a head).
HEAD_MIN_RADIUS = 0.06
HEAD_MAX_RADIUS = 0.16  # 0.25 picked the shoulders/neckline as a big "circle"

# The head centre must be in the top part of the frame (fraction of the height).
HEAD_MAX_CENTER_Y = 0.6

# cv2.HoughCircles parameters:
#   HOUGH_DP:      1 = vote at full resolution, larger = coarser (faster, looser)
#   HOUGH_EDGE:    upper threshold for the Canny edge detector run inside HoughCircles
#   HOUGH_VOTES:   how many votes a circle needs; lower = more (and weaker) circles
HOUGH_DP = 1.2
HOUGH_EDGE = 100
HOUGH_VOTES = 30

# Candidate circles must be at least this far apart (fraction of the height).
HOUGH_MIN_DISTANCE = 0.1

# ---- Skin colour (a fixed colour threshold) ----

# A pixel is skin if its colour, in the YCrCb colour space, is inside this box.
# YCrCb separates brightness (Y) from colour (Cr, Cb), so the same range works
# in brighter and darker lighting. Range from Chai & Ngan (1999),
# "Face segmentation using skin-color map in videophone applications".
SKIN_LOWER = (0, 133, 77)    # (Y, Cr, Cb)
SKIN_UPPER = (255, 173, 127)

# A head circle must be at least this skin-coloured inside (fraction of pixels).
HEAD_MIN_SKIN = 0.3
