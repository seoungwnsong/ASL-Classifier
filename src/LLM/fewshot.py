"""Few-shot examples for the LLM approach: training clips shown to the model WITH their word.

Each example is sent as its own turn before the clip to classify: the N_FRAMES frames of the
example (from data/llm_frames/n<N_FRAMES>/), then the correct answer as the model's reply.
llm_harnness.py uses them only for the versions in FEWSHOT_VERSIONS, and never scores these
clips (they are left out of the training subset).

Only 3 examples: each clip of 16 frames costs about 1,300 prompt tokens, so 3 examples plus
the clip to classify fit in num_ctx = 8192.

How they were picked (once, then fixed):
    - 3 different words, chosen with random.Random(211).sample(LABELS, 3)
    - one MS-ASL training clip of each word, with the same random generator
      (MS-ASL, because every val and test clip is MS-ASL)
    - checked by eye: the sign is clearly visible and no text is on screen
Never change this list after a few-shot version has been scored: make a new version instead.
"""

FEWSHOT = [
    {"example_id": "train_0547", "word": "sad"},   # both hands move down in front of the face
    {"example_id": "train_0011", "word": "deaf"},  # index finger from the ear to the mouth
    {"example_id": "train_0083", "word": "eat"},   # fingertips tap the mouth
]

# Step-by-step reasoning for each example, in the 6 steps of v5's <steps> (written by looking
# at the 16 frames of each clip). Used only by versions in COT_EXAMPLE_VERSIONS (llm_harnness.py).
FEWSHOT_REASONING = {
    "train_0547": (  # sad
        "1. Location: in front of the face; the hands rest in frames 1-6 and 13-16, so the sign is "
        "in frames 7-12. 2. Hands: two hands. 3. Handshape: both hands open, fingers spread, palms "
        "facing the face. 4. Movement: both hands slide down from the eyes to the chin, once. "
        "5. Candidates (in front of the face, two hands): sad. 6. Choice: two open hands sliding "
        "down in front of the face match sad."),
    "train_0011": (  # deaf
        "1. Location: beside the mouth and cheek, then up near the ear. 2. Hands: one hand. "
        "3. Handshape: index finger pointing up, not bunched fingertips and not an open hand. "
        "4. Movement: the index finger moves between the mouth and the ear (the sign can go "
        "either way), once. 5. Candidates (chin or mouth, one hand): mother, eat, deaf, good, no. "
        "6. Choice: an index finger moving between the mouth and the ear matches deaf; eat would "
        "need bunched fingertips tapping the lips, and mother an open hand with the thumb on the chin."),
    "train_0083": (  # eat
        "1. Location: at the mouth, on the lips. 2. Hands: one hand. 3. Handshape: fingertips "
        "bunched together against the thumb. 4. Movement: the bunched fingertips tap the lips "
        "repeatedly; the hand drops in frames 9-11 and comes back to tap again. 5. Candidates "
        "(chin or mouth, one hand): mother, eat, deaf, good, no. 6. Choice: bunched fingertips "
        "tapping the lips match eat; deaf would show one index finger, and mother an open hand "
        "with the thumb on the chin."),
}
