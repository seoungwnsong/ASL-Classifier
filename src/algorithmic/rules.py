"""The hand-written rule: one word for each (row, hands) group.

Each word was chosen from how the sign is made (ASL dictionaries such as
Lifeprint / Signing Savvy) and checked on TRAINING clips with show_zone_table.py.
With 3 rows x 2 hand counts, the rule can answer 6 of the 20 words.
"""

RULE_TABLE = {
    ("top", "one"): "father",    # thumb taps the forehead
    ("top", "two"): "teacher",   # both hands at the temples
    ("face", "one"): "mother",   # thumb taps the chin
    ("face", "two"): "sad",      # both hands slide down the face
    ("chest", "one"): "yes",     # fist nods in front of the chest
    ("chest", "two"): "happy",   # hands brush up the chest
}

# Used when no head or no motion was found in the clip.
FALLBACK = "yes"


def predict_word(zones):
    """Return the predicted word for a clip's zones ({"row": ..., "hands": ...})."""
    return RULE_TABLE.get((zones["row"], zones["hands"]), FALLBACK)
