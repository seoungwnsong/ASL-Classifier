"""Train the new last layers (the "head") on the saved VideoPrism features.

VideoPrism is not used here: extract_features.py already turned every clip into
768 numbers. The head learns to map those 768 numbers to one of the 20 words:

    768 numbers -> Normalization -> Dropout -> Dense(H, relu) -> Dropout -> Dense(20, softmax)

Only TRAINING and VALIDATION clips are used.

Saves:
    models/head_H<H>.keras                  the trained head
    results/figures/dl_training_curve.png   loss and accuracy per epoch, train vs val

Run from the repository root:
    python src/deep_learning/train.py
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import settings

H = settings.HIDDEN_UNITS  # neurons in the hidden layer (0 = no hidden layer)


def load_split(split, words):
    """Features (clips, 768) and word numbers (clips,) of one split."""
    saved = np.load(settings.FEATURES_DIR / f"{split}.npz")
    word_of = pd.read_csv(settings.METADATA).set_index("example_id")["word"]
    labels = [words.index(word_of[example_id]) for example_id in saved["example_id"]]
    return saved["features"], np.array(labels)


def build_model(H, train_features):
    # Normalization learns the mean and spread of each of the 768 numbers from the
    # training clips, so all of them are on a similar scale
    normalize = keras.layers.Normalization()
    normalize.adapt(train_features)

    layers = [keras.Input(shape=(768,)), # input layer
              normalize, # first normalize so that the 768 numbers are on a similar scale
              keras.layers.Dropout(settings.DROPOUT)] # drop out some numbers to avoid overfitting
    if H > 0: 
        layers += [keras.layers.Dense(H, activation="relu"), # Hidden layer
                   keras.layers.Dropout(settings.DROPOUT)] # drop out some numbers to avoid overfitting
    layers += [keras.layers.Dense(20, activation="softmax")]  # Output layer

    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---- 1. Load the features ----
words = sorted(pd.read_csv(settings.METADATA)["word"].unique())  # word number i = words[i]
x_train, y_train = load_split("train", words)
x_val, y_val = load_split("val", words)
print(f"train {x_train.shape}, val {x_val.shape}, H = {H}")

# ---- 2. Check: the head must beat always guessing the most common training word ----
most_common = np.bincount(y_train).argmax()
print(f"majority baseline ('{words[most_common]}'): val accuracy {np.mean(y_val == most_common):.1%}")

# ---- 3. Check: the head must be able to learn 10 training clips perfectly ----
keras.utils.set_random_seed(settings.SEED)
picks = np.random.default_rng(settings.SEED).choice(len(x_train), 10, replace=False)
model = build_model(H, x_train[picks])
model.fit(x_train[picks], y_train[picks], epochs=200, verbose=0)
_, accuracy = model.evaluate(x_train[picks], y_train[picks], verbose=0)
print(f"overfit 10 training clips: accuracy {accuracy:.0%}")

# ---- 4. Train on all training clips ----
# stop when the validation loss has not improved for PATIENCE epochs, and go back to the best epoch
keras.utils.set_random_seed(settings.SEED)
model = build_model(H, x_train)
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings.PATIENCE, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=settings.EPOCHS,
                    batch_size=settings.BATCH_SIZE, callbacks=[stop], verbose=0).history
best = stop.best_epoch
print(f"best epoch {best + 1} of {len(history['loss'])}: train accuracy {history['accuracy'][best]:.1%}, "
      f"val accuracy {history['val_accuracy'][best]:.1%} "
      f"({round(history['val_accuracy'][best] * len(y_val))}/{len(y_val)} clips)")

# ---- 5. Save the model ----
settings.MODELS_DIR.mkdir(exist_ok=True)
model.save(settings.MODELS_DIR / f"head_H{H}.keras")

# ---- 6. Training curve ----
epochs = np.arange(1, len(history["loss"]) + 1)
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
for ax, key in [(ax_loss, "loss"), (ax_acc, "accuracy")]:
    ax.plot(epochs, history[key], label="train")
    ax.plot(epochs, history[f"val_{key}"], label="validation")
    ax.axvline(best + 1, color="gray", linestyle="--", label=f"best epoch ({best + 1})")
    ax.set_xlabel("epoch")
    ax.set_title(key)
    ax.legend()
fig.suptitle(f"Deep learning head training (H = {H})")
fig.tight_layout()
fig.savefig(settings.FIGURES_DIR / "dl_training_curve.png", dpi=120)
print(f"saved models/head_H{H}.keras and results/figures/dl_training_curve.png")
