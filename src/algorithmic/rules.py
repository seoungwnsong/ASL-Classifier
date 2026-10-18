"""The hand-written rule: each of the 20 words described by the 7 measurements.

Each description comes from how the sign is made (ASL dictionaries such as
Lifeprint and Signing Savvy) and was checked on TRAINING clips with
show_feature_table.py. Every word has a different description.

A clip gets one point for each measurement that matches a word's description;
the word with the highest share of matching measurements wins. A blank (None)
means "not used for this word" (hands_apart only matters for two-handed signs).
If two words tie, the one listed first wins.
"""

MEASUREMENTS = ["row", "hands", "repeated", "direction", "axis", "hand_size", "hands_apart"]

SIGNS = {
    #           row      hands  repeated direction axis       hand_size hands_apart
    "hello":   ("top",   "one", "no",  "none", "side",    "big",   None),      # salute out from the forehead
    "father":  ("top",   "one", "yes", "none", "up-down", "big",   None),      # open hand, thumb taps the forehead
    "learn":   ("top",   "one", "no",  "up",   "up-down", "small", None),      # bunched hand lifted to the forehead
    "teacher": ("top",   "two", "yes", "none", "up-down", "small", "medium"),  # both hands at the temples
    "mother":  ("face",  "one", "yes", "none", "up-down", "big",   None),      # open hand, thumb taps the chin
    "eat":     ("face",  "one", "yes", "none", "up-down", "small", None),      # bunched fingertips tap the lips
    "deaf":    ("face",  "one", "no",  "none", "side",    "small", None),      # finger moves from ear to mouth
    "good":    ("face",  "one", "no",  "down", "up-down", "big",   None),      # flat hand from chin down
    "sad":     ("face",  "two", "no",  "down", "up-down", "big",   "medium"),  # both hands slide down the face
    "no":      ("face",  "one", "yes", "none", "side",    "small", None),      # fingers snap to the thumb
    "yes":     ("chest", "one", "yes", "none", "up-down", "small", None),      # fist nods
    "need":    ("chest", "one", "yes", "down", "up-down", "small", None),      # hooked finger bends down
    "please":  ("chest", "one", "yes", "none", "side",    "big",   None),      # flat hand circles on the chest
    "like":    ("chest", "one", "no",  "none", "side",    "small", None),      # pinch pulls away from the chest
    "happy":   ("chest", "two", "yes", "up",   "up-down", "big",   "medium"),  # flat hands brush up the chest
    "tired":   ("chest", "two", "no",  "down", "up-down", "big",   "medium"),  # bent hands drop on the chest
    "school":  ("chest", "two", "yes", "none", "up-down", "big",   "close"),   # palms clap twice
    "want":    ("chest", "two", "no",  "none", "side",    "small", "medium"),  # claw hands pull in
    "finish":  ("chest", "two", "no",  "none", "side",    "big",   "far"),     # open hands flip outward
    "nice":    ("chest", "two", "no",  "none", "side",    "big",   "close"),   # flat hand slides across the palm
}

# Used when no head or no motion was found in the clip.
FALLBACK = "yes"


def score(features, description):
    """Share of the word's measurements (ignoring blanks) that match the clip."""
    pairs = [(features[name], wanted) for name, wanted in zip(MEASUREMENTS, description) if wanted is not None]
    return sum(measured == wanted for measured, wanted in pairs) / len(pairs)


def predict_word(features):
    """Return the word whose description best matches the clip's 7 measurements."""
    if features is None:
        return FALLBACK
    # max() keeps the first word when scores tie, so the order of SIGNS breaks ties
    return max(SIGNS, key=lambda word: score(features, SIGNS[word]))
