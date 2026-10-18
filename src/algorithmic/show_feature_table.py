"""Check the rule table on TRAINING clips: do real clips match each word's description?

For a few training clips per word, computes the 7 measurements, saves them to
results/algorithmic_train_features.csv, and prints for each word the most common
measured value next to the value written in rules.py ("*" = they differ).

Run from the repository root:
    python src/algorithmic/show_feature_table.py        # 10 clips per word
    python src/algorithmic/show_feature_table.py 20     # 20 clips per word
"""

import sys
from pathlib import Path

import pandas as pd

from features import clip_features
from rules import MEASUREMENTS, SIGNS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = REPO_ROOT / "results" / "algorithmic_train_features.csv"


def main():
    per_word = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    train = metadata[metadata["split"] == "train"]
    sample = train.groupby("word").sample(n=per_word, random_state=0)  # same clips every run

    rows = []
    for _, clip in sample.iterrows():
        features = clip_features(REPO_ROOT / clip["clip_path"]) or {}
        rows.append({"example_id": clip["example_id"], "word": clip["word"], **features})
    results = pd.DataFrame(rows)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_FILE, index=False)

    print("most common measured value (share of clips) | value in rules.py, * = different\n")
    for word, description in SIGNS.items():
        clips = results[results["word"] == word]
        cells = []
        for name, wanted in zip(MEASUREMENTS, description):
            counts = clips[name].value_counts(normalize=True)
            measured, share = counts.index[0], counts.iloc[0]
            mark = "" if wanted is None or measured == wanted else "*"
            cells.append(f"{name}={measured} ({share:.0%}) | {wanted}{mark}")
        print(f"{word}:\n  " + "\n  ".join(cells))
    print(f"\nSaved every clip's measurements to {OUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
