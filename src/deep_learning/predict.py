"""Run the final deep learning model ONCE on the test set (do not change the model afterwards).

Uses the saved VideoPrism features of the test clips (data/features/test.npz, made by
extract_features.py) and the final head chosen on the validation clips (settings.FINAL_HEAD).

    -> eval/ml_eval.csv  (Example ID, Deep Learning Prediction)
       same example IDs as eval/human_test_set.csv; no score is printed (eval.py computes it)

Run from the repository root:
    python src/deep_learning/predict.py
"""

import keras
import numpy as np
import pandas as pd

import settings

TEST_OUT = settings.REPO_ROOT / "eval" / "ml_eval.csv"

words = sorted(pd.read_csv(settings.METADATA)["word"].unique())  # word number i = words[i], as in train.py
saved = np.load(settings.FEATURES_DIR / "test.npz")
head = keras.models.load_model(settings.MODELS_DIR / settings.FINAL_HEAD)

probabilities = head.predict(saved["features"], verbose=0)  # (136 clips, 20 words)
predictions = [words[i] for i in probabilities.argmax(axis=1)]

pd.DataFrame({"Example ID": saved["example_id"], "Deep Learning Prediction": predictions}).to_csv(TEST_OUT, index=False)
print(f"wrote {len(predictions)} predictions to {TEST_OUT.relative_to(settings.REPO_ROOT)} using {settings.FINAL_HEAD}")
