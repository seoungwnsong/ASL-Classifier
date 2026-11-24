""" Machine Learning Approach to ASL Classification: Training

Main Pipeline:
    Step 1. Preprocessing: 16 frames per clip, text covered, cropped to the signer, 288 x 288
    Step 2. Features: frozen VideoPrism (pretrained, from keras_hub) -> average -> 768 numbers per clip
    Step 3. Training: head trained from scratch
            Normalization -> Dropout -> Dense(H, relu) -> Dropout -> Dense(20, softmax)
    Step 4. Result: save the weights, the tuning table and the training curve

Tuned on the validation set: H in {0, 64, 128, 256, 512} and dropout in {0, 0.3, 0.5}.
Best: H = 64, dropout = 0.5.

Run: python ml_train.py
     python ml_train.py --hidden 128 --dropout 0.3
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd()
PRESET = "videoprism_public_v1_base" # VideoPrism base, kept frozen
NUM_FRAMES, FRAME_SIZE = 16, 288 # VideoPrism input: 16 frames of 288 x 288
BLACK_FRAME_MAX = 30 # skip completely black frames (brightest pixel below 30)

HIDDEN_UNITS = 64 # best H on the validation set
DROPOUT = 0.5 # best dropout rate on the validation set
EPOCHS = 300 # at most 300 epochs
PATIENCE = 30 # early stopping: stop after 30 epochs with no better validation loss
BATCH_SIZE = 32
SEED = 0 # same random start every run

METADATA = REPO_ROOT / "data" / "metadata.csv" # example_id, split, word, source, clip_path
SPLIT_JSONS = [REPO_ROOT / "data" / f"{split}.json" for split in ("train", "val", "test")]
WLASL_METADATA = REPO_ROOT / "data" / "raw" / "WLASL_v0.3.json"
CAPTION_COVERS = REPO_ROOT / "data" / "caption_covers.csv" # black boxes over on-screen text
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
TUNING_RESULTS = REPO_ROOT / "results" / "dl_tuning.csv"


# Step 1. Preprocessing: 16 frames per clip, text covered, cropped to the signer, 288 x 288
def load_clip_info():
    """Signer box and text covers of every clip."""
    covers = pd.read_csv(CAPTION_COVERS)
    cover_of = {row.video_id: [row.left, row.top, row.right, row.bottom] for row in covers.itertuples()}

    info = {}
    wlasl_paths = []
    for split_json in SPLIT_JSONS:
        for entry in json.loads(split_json.read_text()):
            if entry["source"] == "MS-ASL":
                clip_covers = [rect for video_id, rect in cover_of.items() if video_id in entry["url"]]
                info[entry["clip_path"]] = {"box": entry["box"], "covers": clip_covers}
            else:
                wlasl_paths.append(entry["clip_path"])

    # WLASL clips: box in pixels from WLASL's metadata, changed to fractions of the frame
    wlasl_boxes = {}
    for gloss in json.loads(WLASL_METADATA.read_text()):
        for instance in gloss["instances"]:
            wlasl_boxes[instance["video_id"]] = instance["bbox"]
    for clip_path in wlasl_paths:
        x1, y1, x2, y2 = wlasl_boxes[clip_path.split("/")[-1].removesuffix(".mp4")]
        capture = cv2.VideoCapture(str(REPO_ROOT / clip_path))
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        capture.release()
        info[clip_path] = {"box": [y1 / height, x1 / width, y2 / height, x2 / width], "covers": []}
    return info

def read_frames(video_path):
    """All frames of the video."""
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames

def prepare_frame(frame, box, covers):
    """Cover the text, crop to the signer, pad to a square and resize one frame."""
    frame = frame.copy()
    height, width = frame.shape[:2]
    for left, top, right, bottom in covers:
        frame[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 0

    top, left, bottom, right = box
    frame = frame[int(top * height):int(bottom * height), int(left * width):int(right * width)]

    height, width = frame.shape[:2]
    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    frame = cv2.copyMakeBorder(frame, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
                               cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

def clip_input(clip_path, clip_info):
    """16 evenly spaced frames of one clip: shape (16, 288, 288, 3), values 0 to 1."""
    frames = read_frames(REPO_ROOT / clip_path)
    frames = [f for f in frames if f.max() >= BLACK_FRAME_MAX] or frames
    picks = np.linspace(0, len(frames) - 1, NUM_FRAMES).round().astype(int)
    info = clip_info[clip_path]
    clip = [cv2.cvtColor(prepare_frame(frames[i], info["box"], info["covers"]), cv2.COLOR_BGR2RGB)
            for i in picks]
    return np.array(clip, dtype=np.float32) / 255.0


# Step 2. Features: frozen VideoPrism (pretrained, from keras_hub) -> average -> 768 numbers per clip
def extract_features():
    """Run VideoPrism once on every clip and save the features (skipped if already saved)."""
    metadata = pd.read_csv(METADATA)
    splits = [s for s in ("train", "val", "test") if not (FEATURES_DIR / f"{s}.npz").exists()]
    if not splits:
        return
    clip_info = load_clip_info()
    backbone = keras_hub.models.VideoPrismBackbone.from_preset(PRESET)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        clips = metadata[metadata["split"] == split]
        features = []
        for clip in clips.itertuples():
            video = clip_input(clip.clip_path, clip_info) # (16, 288, 288, 3)
            output = backbone.predict(video[np.newaxis], verbose=0) # (1, 16 frames, 256 patches, 768)
            features.append(output.mean(axis=(1, 2))[0]) # (768,)
        np.savez(FEATURES_DIR / f"{split}.npz", example_id=clips["example_id"].to_numpy(),
                 features=np.array(features, dtype=np.float32))
        print(f"{split}: saved features of {len(clips)} clips")

def load_split(split, words):
    """Features (clips, 768) and word numbers (clips,) of one split."""
    saved = np.load(FEATURES_DIR / f"{split}.npz")
    word_of = pd.read_csv(METADATA).set_index("example_id")["word"]
    labels = [words.index(word_of[example_id]) for example_id in saved["example_id"]]
    return saved["features"], np.array(labels)


# Step 3. Training: head trained from scratch
def build_model(H, DROPOUT, train_features):
    """Normalization -> Dropout -> Dense(H, relu) -> Dropout -> Dense(20, softmax)."""
    normalize = keras.layers.Normalization() # uses the mean and variance of the training features
    normalize.adapt(train_features)

    layers = [keras.Input(shape=(768,)), # input layer
              normalize,
              keras.layers.Dropout(DROPOUT)] # set some numbers to 0 during training, against overfitting
    if H > 0:
        layers += [keras.layers.Dense(H, activation="relu"), # hidden layer
                   keras.layers.Dropout(DROPOUT)]
    layers += [keras.layers.Dense(20, activation="softmax")] # output layer: one probability per word

    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=HIDDEN_UNITS, help="neurons in the hidden layer")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="dropout rate (0 to 1)")
    args = parser.parse_args()
    H = args.hidden # neurons in the hidden layer (0 = no hidden layer)
    DROPOUT = args.dropout

    # Load the features (the test set is not used here)
    extract_features()
    words = sorted(pd.read_csv(METADATA)["word"].unique()) # word number i = words[i]
    x_train, y_train = load_split("train", words)
    x_val, y_val = load_split("val", words)
    print(f"train {x_train.shape}, val {x_val.shape}, H = {H}, dropout = {DROPOUT}")

    # Check 1: must beat always guessing the most common word
    most_common = np.bincount(y_train).argmax()
    baseline_accuracy = np.mean(y_val == most_common)
    print(f"majority baseline ('{words[most_common]}'): val accuracy {baseline_accuracy:.1%}")

    # Check 2: must be able to learn 10 training clips perfectly
    keras.utils.set_random_seed(SEED)
    picks = np.random.default_rng(SEED).choice(len(x_train), 10, replace=False)
    model = build_model(H, DROPOUT, x_train[picks])
    model.fit(x_train[picks], y_train[picks], epochs=200, verbose=0)
    _, overfit_accuracy = model.evaluate(x_train[picks], y_train[picks], verbose=0)
    print(f"overfit 10 training clips: accuracy {overfit_accuracy:.0%}")

    # Train on all training clips, keep the weights of the best epoch (lowest validation loss)
    keras.utils.set_random_seed(SEED)
    model = build_model(H, DROPOUT, x_train)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    start = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS,
                        batch_size=BATCH_SIZE, callbacks=[stop], verbose=0).history
    training_seconds = time.perf_counter() - start
    best = stop.best_epoch
    trained_parameters = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    print(f"{trained_parameters:,} trained parameters, training took {training_seconds:.1f} s")
    train_accuracy = history["accuracy"][best] # dropout on, as in the training curve
    val_accuracy = history["val_accuracy"][best]
    _, train_accuracy_off = model.evaluate(x_train, y_train, verbose=0) # dropout off, like validation
    print(f"best epoch {best + 1} of {len(history['loss'])}: train accuracy {train_accuracy:.1%} "
          f"(dropout off: {train_accuracy_off:.1%}), "
          f"val accuracy {val_accuracy:.1%} ({round(val_accuracy * len(y_val))}/{len(y_val)} clips)")

    # Step 4. Result: save the weights, the tuning table and the training curve
    name = f"H{H}_D{DROPOUT}" # e.g. H64_D0.5
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / f"head_{name}.keras") # the weights ml_inference.py loads

    # Tuning table: one row per (H, dropout), a rerun replaces its row
    row = pd.DataFrame([{"H": H, "dropout": DROPOUT, "trained parameters": trained_parameters,
                         "best epoch": best + 1, "epochs run": len(history["loss"]),
                         "train accuracy (dropout on)": round(train_accuracy, 4),
                         "train accuracy (dropout off)": round(train_accuracy_off, 4),
                         "val accuracy": round(val_accuracy, 4),
                         "val correct": f"{round(val_accuracy * len(y_val))}/{len(y_val)}",
                         "training seconds": round(training_seconds, 1),
                         "overfit 10 clips": round(overfit_accuracy, 4),
                         "majority baseline val": round(baseline_accuracy, 4)}])
    if TUNING_RESULTS.exists():
        old = pd.read_csv(TUNING_RESULTS)
        same_pair = (old["H"] == H) & (old["dropout"] == DROPOUT)
        row = pd.concat([old[~same_pair], row])
    TUNING_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    row.sort_values(["H", "dropout"]).to_csv(TUNING_RESULTS, index=False)

    # Training curve: (a) loss and (b) accuracy per epoch, train vs validation
    TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834" # blue / orange: colour-blind safe
    INK, GREY, GRID = "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False})

    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panels = [(ax_loss, "loss", 1, "Cross-entropy loss", "(a)"), (ax_acc, "accuracy", 100, "Accuracy (%)", "(b)")]
    for ax, key, scale, ylabel, letter in panels:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(best + 1, color=GREY, linewidth=0.8, linestyle=":") # best epoch
        ax.plot(epochs, np.array(history[key]) * scale, color=TRAIN_COLOR, linewidth=1.5)
        ax.plot(epochs, np.array(history[f"val_{key}"]) * scale, color=VAL_COLOR, linewidth=1.5,
                linestyle=(0, (4, 2))) # validation is dashed
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, len(epochs) + 1)
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold", color=INK)
    ax_acc.set_ylim(0, 105)

    # accuracies at the best epoch
    ax_acc.annotate(f"{train_accuracy:.1%}", (best + 1, train_accuracy * 100), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax_acc.annotate(f"{val_accuracy:.1%}", (best + 1, val_accuracy * 100), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=INK)

    legend = [plt.Line2D([], [], color=TRAIN_COLOR, linewidth=1.5, label="Train"),
              plt.Line2D([], [], color=VAL_COLOR, linewidth=1.5, linestyle=(0, (4, 2)), label="Validation"),
              plt.Line2D([], [], color=GREY, linewidth=0.8, linestyle=":", label=f"Best epoch ({best + 1})")]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = FIGURES_DIR / f"dl_training_curve_{name}"
    fig.savefig(f"{curve_path}.png", dpi=300)
    fig.savefig(f"{curve_path}.pdf")
    print(f"saved models/head_{name}.keras, results/figures/dl_training_curve_{name}.png/.pdf, "
          f"results/dl_tuning.csv")
