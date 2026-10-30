"""Run the frozen VideoPrism model ONCE on every clip and save its features.

VideoPrism is frozen (never trained), so a clip always gives the same output.
We compute it once here, and train.py trains only the new last layers on the
saved numbers. Takes about 1 hour (about 3.3 s per clip on the CPU).

For each clip:
    16 frames (frames.py) -> VideoPrism -> (16 frames, 256 patches, 768)
    -> average over frames and patches -> 768 numbers

Saves data/features/train.npz, val.npz and test.npz, each with:
    example_id   (clips,)       which clip each row is, as in data/metadata.csv
    features     (clips, 768)   the VideoPrism numbers
No words (labels) are saved: train.py looks them up in data/metadata.csv.

Progress is saved every 50 clips in <split>_partial.npz. If the run stops,
running it again continues from there.

Run from the repository root:
    python src/deep_learning/extract_features.py
"""

import time

import keras_hub
import numpy as np
import pandas as pd

import settings
from frames import clip_input, load_clip_info

SAVE_EVERY = 50


def save(path, ids, features):
    np.savez(path, example_id=np.array(ids), features=np.array(features, dtype=np.float32))


def extract_split(split, clips, backbone, clip_info):
    """Features for every clip of one split, saved to data/features/<split>.npz."""
    final_path = settings.FEATURES_DIR / f"{split}.npz"
    partial_path = settings.FEATURES_DIR / f"{split}_partial.npz"
    if final_path.exists():
        print(f"{split}: already done ({final_path.name}), skipping")
        return

    # continue from a stopped run, if there is one
    ids, features = [], []
    if partial_path.exists():
        saved = np.load(partial_path)
        ids, features = list(saved["example_id"]), list(saved["features"])
    todo = clips[~clips["example_id"].isin(ids)]
    print(f"{split}: {len(clips)} clips, {len(ids)} already done, {len(todo)} to go", flush=True)

    start = time.time()
    for i, clip in enumerate(todo.itertuples(), start=1):
        video = clip_input(clip.clip_path, clip_info)            # (16, 288, 288, 3)
        output = backbone.predict(video[np.newaxis], verbose=0)  # (1, 16, 256, 768)
        features.append(output.mean(axis=(1, 2))[0])             # (768,)
        ids.append(clip.example_id)

        if i % SAVE_EVERY == 0 or i == len(todo):
            save(partial_path, ids, features)
            minutes_left = (time.time() - start) / i * (len(todo) - i) / 60
            print(f"  {len(ids)}/{len(clips)} clips, about {minutes_left:.0f} min left for {split}", flush=True)

    partial_path.rename(final_path)
    print(f"{split}: saved {final_path.relative_to(settings.REPO_ROOT)}, features {np.array(features).shape}")


def main():
    metadata = pd.read_csv(settings.METADATA)
    clip_info = load_clip_info()
    backbone = keras_hub.models.VideoPrismBackbone.from_preset(settings.PRESET)
    settings.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        extract_split(split, metadata[metadata["split"] == split], backbone, clip_info)


if __name__ == "__main__":
    main()
