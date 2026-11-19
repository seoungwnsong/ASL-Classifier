"""Train the new last layers (the "head") on the saved VideoPrism features.

VideoPrism is not used here: extract_features.py already turned every clip into
768 numbers. The head learns to map those 768 numbers to one of the 20 words:

    768 numbers -> Normalization -> Dropout -> Dense(H, relu) -> Dropout -> Dense(20, softmax)

Only TRAINING and VALIDATION clips are used.

Two hyperparameters are tuned on the validation clips: H (neurons in the hidden layer)
and the dropout rate.

Saves (D = dropout rate):
    models/head_H<H>_D<D>.keras                       the trained head
    results/figures/dl_training_curve_H<H>_D<D>.png   loss and accuracy per epoch, train vs val (also .pdf)
    results/dl_tuning.csv                             one row per (H, D) pair: the tuning table

Run from the repository root (without options, H and D come from settings.py):
    python src/deep_learning/train.py
    python src/deep_learning/train.py --hidden 64 --dropout 0.5
"""

import argparse
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import settings

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=settings.HIDDEN_UNITS, help="neurons in the hidden layer")
parser.add_argument("--dropout", type=float, default=settings.DROPOUT, help="dropout rate (0 to 1)")
args = parser.parse_args()
H = args.hidden         # neurons in the hidden layer (0 = no hidden layer)
DROPOUT = args.dropout  # fraction of numbers set to 0 during training


def load_split(split, words):
    """Features (clips, 768) and word numbers (clips,) of one split."""
    saved = np.load(settings.FEATURES_DIR / f"{split}.npz")
    word_of = pd.read_csv(settings.METADATA).set_index("example_id")["word"]
    labels = [words.index(word_of[example_id]) for example_id in saved["example_id"]]
    return saved["features"], np.array(labels)


def build_model(H, DROPOUT, train_features):
    # Normalization learns the mean and spread of each of the 768 numbers from the
    # training clips, so all of them are on a similar scale
    normalize = keras.layers.Normalization()
    normalize.adapt(train_features)

    layers = [keras.Input(shape=(768,)), # input layer
              normalize, # first normalize so that the 768 numbers are on a similar scale
              keras.layers.Dropout(DROPOUT)] # drop out some numbers to avoid overfitting
    if H > 0:
        layers += [keras.layers.Dense(H, activation="relu"), # Hidden layer
                   keras.layers.Dropout(DROPOUT)] # drop out some numbers to avoid overfitting
    layers += [keras.layers.Dense(20, activation="softmax")]  # Output layer

    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---- 1. Load the features ----
words = sorted(pd.read_csv(settings.METADATA)["word"].unique())  # word number i = words[i]
x_train, y_train = load_split("train", words)
x_val, y_val = load_split("val", words)
print(f"train {x_train.shape}, val {x_val.shape}, H = {H}, dropout = {DROPOUT}")

# ---- 2. Check: the head must beat always guessing the most common training word ----
most_common = np.bincount(y_train).argmax()
baseline_accuracy = np.mean(y_val == most_common)
print(f"majority baseline ('{words[most_common]}'): val accuracy {baseline_accuracy:.1%}")

# ---- 3. Check: the head must be able to learn 10 training clips perfectly ----
keras.utils.set_random_seed(settings.SEED)
picks = np.random.default_rng(settings.SEED).choice(len(x_train), 10, replace=False)
model = build_model(H, DROPOUT, x_train[picks])
model.fit(x_train[picks], y_train[picks], epochs=200, verbose=0)
_, overfit_accuracy = model.evaluate(x_train[picks], y_train[picks], verbose=0)
print(f"overfit 10 training clips: accuracy {overfit_accuracy:.0%}")

# ---- 4. Train on all training clips ----
# stop when the validation loss has not improved for PATIENCE epochs, and go back to the best epoch
keras.utils.set_random_seed(settings.SEED)
model = build_model(H, DROPOUT, x_train)
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings.PATIENCE, restore_best_weights=True)
start = time.perf_counter()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=settings.EPOCHS,
                    batch_size=settings.BATCH_SIZE, callbacks=[stop], verbose=0).history
training_seconds = time.perf_counter() - start
best = stop.best_epoch
# weights that training changes (the Dense layers); Normalization's mean and spread are not counted
trained_parameters = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
print(f"{trained_parameters:,} trained parameters, training took {training_seconds:.1f} s")
# accuracies at the best epoch, as logged during training (train accuracy is logged with dropout on)
train_accuracy = history["accuracy"][best]
val_accuracy = history["val_accuracy"][best]
# the same best-epoch weights scored on the training clips with dropout OFF, like validation,
# so rows with different dropout rates can be compared (evaluate always turns dropout off)
_, train_accuracy_off = model.evaluate(x_train, y_train, verbose=0)
print(f"best epoch {best + 1} of {len(history['loss'])}: train accuracy {train_accuracy:.1%} "
      f"(dropout off: {train_accuracy_off:.1%}), "
      f"val accuracy {val_accuracy:.1%} ({round(val_accuracy * len(y_val))}/{len(y_val)} clips)")

# ---- 5. Save the model and add this (H, dropout) pair to the tuning table ----
name = f"H{H}_D{DROPOUT}"  # e.g. H128_D0.3
settings.MODELS_DIR.mkdir(exist_ok=True)
model.save(settings.MODELS_DIR / f"head_{name}.keras")

row = pd.DataFrame([{"H": H, "dropout": DROPOUT, "trained parameters": trained_parameters,
                     "best epoch": best + 1, "epochs run": len(history["loss"]),
                     "train accuracy (dropout on)": round(train_accuracy, 4),  # as in the training curve
                     "train accuracy (dropout off)": round(train_accuracy_off, 4),
                     "val accuracy": round(val_accuracy, 4),
                     "val correct": f"{round(val_accuracy * len(y_val))}/{len(y_val)}",
                     "training seconds": round(training_seconds, 1),
                     "overfit 10 clips": round(overfit_accuracy, 4),          # sanity check 2
                     "majority baseline val": round(baseline_accuracy, 4)}])  # sanity check 1
if settings.TUNING_RESULTS.exists():
    old = pd.read_csv(settings.TUNING_RESULTS)
    same_pair = (old["H"] == H) & (old["dropout"] == DROPOUT)
    row = pd.concat([old[~same_pair], row])  # running the same pair again replaces its row
row.sort_values(["H", "dropout"]).to_csv(settings.TUNING_RESULTS, index=False)

# ---- 6. Training curve (research-paper style: no title, the caption goes in the report) ----
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"  # blue / orange: colour-blind safe pair
INK, GREY, GRID = "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.spines.top": False, "axes.spines.right": False})

epochs = np.arange(1, len(history["loss"]) + 1)
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(7.0, 2.6))  # full width of a report page
panels = [(ax_loss, "loss", 1, "Cross-entropy loss", "(a)"), (ax_acc, "accuracy", 100, "Accuracy (%)", "(b)")]
for ax, key, scale, ylabel, letter in panels:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axvline(best + 1, color=GREY, linewidth=0.8, linestyle=":")
    ax.plot(epochs, np.array(history[key]) * scale, color=TRAIN_COLOR, linewidth=1.5)
    # validation is dashed so the two lines can be told apart in black-and-white print too
    ax.plot(epochs, np.array(history[f"val_{key}"]) * scale, color=VAL_COLOR, linewidth=1.5, linestyle=(0, (4, 2)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(epochs) + 1)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold", color=INK)
ax_acc.set_ylim(0, 105)

# write the best epoch's accuracies next to the dotted line
ax_acc.annotate(f"{train_accuracy:.1%}", (best + 1, train_accuracy * 100), xytext=(6, 4),
                textcoords="offset points", fontsize=8, color=INK)
ax_acc.annotate(f"{val_accuracy:.1%}", (best + 1, val_accuracy * 100), xytext=(6, -12),
                textcoords="offset points", fontsize=8, color=INK)

legend = [plt.Line2D([], [], color=TRAIN_COLOR, linewidth=1.5, label="Train"),
          plt.Line2D([], [], color=VAL_COLOR, linewidth=1.5, linestyle=(0, (4, 2)), label="Validation"),
          plt.Line2D([], [], color=GREY, linewidth=0.8, linestyle=":", label=f"Best epoch ({best + 1})")]
fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92))
curve_path = settings.FIGURES_DIR / f"dl_training_curve_{name}"
fig.savefig(f"{curve_path}.png", dpi=300)
fig.savefig(f"{curve_path}.pdf")  # vector version for the report / poster
print(f"saved models/head_{name}.keras, results/figures/dl_training_curve_{name}.png/.pdf, results/dl_tuning.csv")
