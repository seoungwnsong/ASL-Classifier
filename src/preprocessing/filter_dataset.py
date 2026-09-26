"""Build the train/val/test splits for the 20 selected ASL classes.

Steps:
1. Keep MS-ASL entries whose "text" exactly matches a selected class and whose
   YouTube video is still playable (see check_availability.py).
2. Train = MS-ASL train + every WLASL video in data/wlasl/ (selected classes only).
3. MS-ASL val and test are pooled and re-split so that the test set has
   TEST_PER_CLASS clips per class. Clips from the same YouTube video always
   stay on the same side, so val and test never share a recording.

All original MS-ASL metadata is kept. Every entry gets a "source" field
("MS-ASL" or "WLASL"). Nothing in data/raw/ is modified.

Run from the repository root:
    python src/preprocessing/check_availability.py   # once, needs internet
    python src/preprocessing/filter_dataset.py
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SELECTED_CLASSES = [
    "hello",
    "yes",
    "no",
    "please",
    "need",
    "good",
    "nice",
    "eat",
    "tired",
    "want",
    "like",
    "deaf",
    "mother",
    "father",
    "teacher",
    "learn",
    "school",
    "happy",
    "sad",
    "finish",
]

TEST_PER_CLASS = 7
SEED = 211
MAX_SPLIT_ATTEMPTS = 1000

# Resolve paths relative to the repo root so the script works from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data"
AVAILABILITY_FILE = OUT_DIR / "video_availability.json"
# WLASL videos for the selected classes only, as data/wlasl/<word>/<id>.mp4.
WLASL_DIR = OUT_DIR / "wlasl"

# (split name, MS-ASL input file, output file)
SPLITS = [
    ("train", "MSASL_train.json", "train.json"),
    ("val", "MSASL_val.json", "val.json"),
    ("test", "MSASL_test.json", "test.json"),
]


def video_id(url):
    """Extract the 11-character YouTube video id from an MS-ASL url."""
    return re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url).group(1)


def load_availability():
    if not AVAILABILITY_FILE.exists():
        raise SystemExit(
            f"{AVAILABILITY_FILE} not found. "
            "Run: python src/preprocessing/check_availability.py"
        )
    with open(AVAILABILITY_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_msasl(input_name, availability):
    """Selected-class MS-ASL entries whose video is still playable."""
    with open(RAW_DIR / input_name, encoding="utf-8") as f:
        entries = json.load(f)

    kept = []
    for entry in entries:
        if entry["text"] not in SELECTED_CLASSES:
            continue
        status = availability.get(video_id(entry["url"]))
        if status not in ("ok", "unavailable", "private"):
            raise SystemExit(
                f"Availability of {entry['url']} is not known. "
                "Run: python src/preprocessing/check_availability.py"
            )
        if status == "ok":
            kept.append({**entry, "source": "MS-ASL"})
    return kept


def load_wlasl():
    """One entry per WLASL video file of a selected class."""
    if not WLASL_DIR.is_dir():
        raise SystemExit(f"WLASL videos not found at {WLASL_DIR}.")
    entries = []
    for word in SELECTED_CLASSES:
        for path in sorted((WLASL_DIR / word).glob("*.mp4")):
            entries.append({
                "text": word,
                "file": path.relative_to(REPO_ROOT).as_posix(),
                "video_id": path.stem,
                "source": "WLASL",
            })
    return entries


def split_val_test(entries):
    """Split pooled entries into (val, test) with TEST_PER_CLASS test clips per class.

    Whole YouTube videos are assigned to one side. Videos are added to test in
    a random order whenever they do not push any class over TEST_PER_CLASS;
    the most balanced of several seeded attempts is kept.
    """
    clips_by_video = defaultdict(list)
    for entry in entries:
        clips_by_video[video_id(entry["url"])].append(entry)
    videos = sorted(clips_by_video)

    rng = random.Random(SEED)
    best_shortfall, best_test_videos = None, None
    for _ in range(MAX_SPLIT_ATTEMPTS):
        rng.shuffle(videos)
        counts = Counter()
        test_videos = set()
        for vid in videos:
            video_counts = Counter(e["text"] for e in clips_by_video[vid])
            if all(counts[w] + n <= TEST_PER_CLASS for w, n in video_counts.items()):
                counts.update(video_counts)
                test_videos.add(vid)

        shortfall = sum(TEST_PER_CLASS - counts[w] for w in SELECTED_CLASSES)
        if best_shortfall is None or shortfall < best_shortfall:
            best_shortfall, best_test_videos = shortfall, test_videos
        if shortfall == 0:
            break

    val = [e for e in entries if video_id(e["url"]) not in best_test_videos]
    test = [e for e in entries if video_id(e["url"]) in best_test_videos]
    return val, test


def write_json_lines(entries, path):
    """Write a JSON array with one compact entry per line."""
    lines = [json.dumps(entry, separators=(",", ":")) for entry in entries]
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(lines) + "\n]\n")


def print_summary(splits):
    counts = {name: Counter(e["text"] for e in entries) for name, entries in splits.items()}
    wlasl = Counter(e["text"] for e in splits["train"] if e["source"] == "WLASL")

    print(f"{'word':<10}{'train (WLASL)':>15}{'val':>6}{'test':>6}")
    for word in SELECTED_CLASSES:
        train = f"{counts['train'][word]} ({wlasl[word]})"
        print(f"{word:<10}{train:>15}{counts['val'][word]:>6}{counts['test'][word]:>6}")
    train_total = f"{len(splits['train'])} ({sum(wlasl.values())})"
    print(f"{'Total':<10}{train_total:>15}{len(splits['val']):>6}{len(splits['test']):>6}")

    for name in splits:
        for word in SELECTED_CLASSES:
            if counts[name][word] == 0:
                print(f"WARNING: '{word}' has 0 examples in {name}")
    for word in SELECTED_CLASSES:
        if counts["test"][word] != TEST_PER_CLASS:
            print(f"NOTE: '{word}' has {counts['test'][word]} test clips "
                  f"(target {TEST_PER_CLASS})")


def main():
    availability = load_availability()
    msasl = {name: load_msasl(input_name, availability) for name, input_name, _ in SPLITS}

    train = msasl["train"] + load_wlasl()
    val, test = split_val_test(msasl["val"] + msasl["test"])
    splits = {"train": train, "val": val, "test": test}

    for name, _, output_name in SPLITS:
        write_json_lines(splits[name], OUT_DIR / output_name)

    print_summary(splits)


if __name__ == "__main__":
    main()
