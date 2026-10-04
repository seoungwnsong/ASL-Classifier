"""Check zones.py on TRAINING clips: which (row, hands) group does each word land in?

Takes a few training clips per word, computes their row (top / face / chest) and
hands (one / two), saves every clip's result to results/algorithmic_train_zones.csv
and prints a table: one line per word, counting how its clips were grouped.

This table is only used to CHECK the hand-written rule table in rules.py.

Run from the repository root:
    python src/algorithmic/show_zone_table.py        # 10 clips per word
    python src/algorithmic/show_zone_table.py 20     # 20 clips per word
"""

import sys
from pathlib import Path

import pandas as pd

from zones import clip_zones

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = REPO_ROOT / "results" / "algorithmic_train_zones.csv"


def main():
    per_word = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    metadata = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    train = metadata[metadata["split"] == "train"]
    sample = train.groupby("word").sample(n=per_word, random_state=0)  # same clips every run

    rows = []
    for _, clip in sample.iterrows():
        zones = clip_zones(REPO_ROOT / clip["clip_path"])
        rows.append({"example_id": clip["example_id"], "word": clip["word"], **zones})
    results = pd.DataFrame(rows)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_FILE, index=False)

    results["group"] = results["row"].fillna("none") + "+" + results["hands"].fillna("none")
    table = pd.crosstab(results["word"], results["group"])
    print(table.to_string())
    print(f"\nSaved every clip's result to {OUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
