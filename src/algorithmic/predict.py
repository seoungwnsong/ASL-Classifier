"""Run the algorithmic approach on a split and save one prediction per clip.

Training (default, use this while developing the rule):
    python src/algorithmic/predict.py
    -> results/algorithmic_train.csv  (Example ID, Algorithmic Prediction, True Word)
       and prints the training accuracy

Test (run ONCE, when the rule is final; do not change the rule afterwards):
    python src/algorithmic/predict.py --test
    -> eval/algorithmic.csv  (Example ID, Algorithmic Prediction)
       same example IDs as eval/human_test_set.csv; no score is printed
"""

import argparse
from pathlib import Path

import pandas as pd

from rules import predict_word
from zones import clip_zones

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_OUT = REPO_ROOT / "results" / "algorithmic_train.csv"
TEST_OUT = REPO_ROOT / "eval" / "algorithmic.csv"


def predict_split(split):
    """Return a table with one row per clip of the split: example ID, prediction, true word."""
    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    clips = metadata[metadata["split"] == split]
    rows = []
    for i, (_, clip) in enumerate(clips.iterrows(), start=1):
        zones = clip_zones(REPO_ROOT / clip["clip_path"])
        rows.append({
            "Example ID": clip["example_id"],
            "Algorithmic Prediction": predict_word(zones),
            "True Word": clip["word"],
        })
        if i % 100 == 0:
            print(f"  {i}/{len(clips)} clips", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="run once on the test set (final run)")
    args = parser.parse_args()

    if args.test:
        results = predict_split("test")
        results[["Example ID", "Algorithmic Prediction"]].to_csv(TEST_OUT, index=False)
        print(f"Wrote {len(results)} predictions to {TEST_OUT.relative_to(REPO_ROOT)}")
        return

    results = predict_split("train")
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TRAIN_OUT, index=False)
    correct = (results["Algorithmic Prediction"] == results["True Word"]).mean()
    print(f"Training accuracy: {correct:.1%} ({len(results)} clips, chance = 5%)")
    print(f"Saved {TRAIN_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
