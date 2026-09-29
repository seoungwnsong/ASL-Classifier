"""Write data/metadata.csv: one row per video clip, with a fixed example ID.

Columns: example_id, split, word, source, clip_path

IDs:
- test:  same IDs as the human baseline (eval/human_test_set.csv).
         Letter = word in alphabetical order (deaf = a, eat = b, ...),
         number = the clip's position when that word's file names are sorted.
         Example: a001 = the first "deaf" clip.
- train: train_0001, train_0002, ... (sorted by word, then file name)
- val:   val_0001, val_0002, ...     (sorted by word, then file name)

Run from the repository root (after filter_dataset.py):
    python src/preprocessing/make_metadata.py
"""

import csv
import json

from filter_dataset import OUT_DIR, REPO_ROOT

METADATA_FILE = OUT_DIR / "metadata.csv"
HUMAN_TEST_FILE = REPO_ROOT / "eval" / "human_test_set.csv"


def load_split(split):
    """One row per video file (a few train clips are listed twice in MS-ASL)."""
    with open(OUT_DIR / f"{split}.json", encoding="utf-8") as f:
        entries = json.load(f)
    unique = {e["clip_path"]: e for e in entries}
    return sorted(unique.values(), key=lambda e: (e["text"], e["clip_path"]))


def test_ids(entries):
    """IDs in the human-baseline style: word letter + position, e.g. a001."""
    words = sorted({e["text"] for e in entries})
    counts = {}
    ids = []
    for e in entries:  # already sorted by word, then file name
        counts[e["text"]] = counts.get(e["text"], 0) + 1
        letter = chr(ord("a") + words.index(e["text"]))
        ids.append(f"{letter}{counts[e['text']]:03d}")
    return ids


def check_against_human(rows):
    """Make sure every test ID matches the human baseline file exactly."""
    if not HUMAN_TEST_FILE.exists():
        print(f"WARNING: {HUMAN_TEST_FILE.name} not found, could not check test IDs")
        return
    with open(HUMAN_TEST_FILE, encoding="utf-8") as f:
        human = {r["video id"]: r["video file"] for r in csv.DictReader(f)}
    ours = {r["example_id"]: r["clip_path"].removeprefix("data/")
            for r in rows if r["split"] == "test"}
    if ours != human:
        raise SystemExit(f"Test IDs do not match {HUMAN_TEST_FILE.name}")
    print(f"Test IDs match {HUMAN_TEST_FILE.name} ({len(human)} clips)")


def main():
    rows = []
    for split in ["train", "val", "test"]:
        entries = load_split(split)
        if split == "test":
            ids = test_ids(entries)
        else:
            ids = [f"{split}_{n:04d}" for n in range(1, len(entries) + 1)]
        for example_id, e in zip(ids, entries):
            rows.append({
                "example_id": example_id,
                "split": split,
                "word": e["text"],
                "source": e["source"],
                "clip_path": e["clip_path"],
            })

    check_against_human(rows)

    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for split in ["train", "val", "test"]:
        print(f"{split}: {sum(r['split'] == split for r in rows)} clips")
    print(f"Wrote {METADATA_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
