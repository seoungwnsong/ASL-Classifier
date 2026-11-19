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
