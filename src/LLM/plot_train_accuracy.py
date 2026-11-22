"""Accuracy of every LLM prompt version on the training subset, and a bar chart of it.

Reads the saved runs in results/llm_runs/<version>_train.jsonl (made by llm_harnness.py,
which scores every version on the same fixed training subset). No model is called.
An invalid answer (not one of the 20 labels) counts as wrong.

Prints one row per version and saves:
    results/figures/llm_train_accuracy.png   bar chart of training accuracy per version (also .pdf)

Run from the repository root:
    python src/LLM/plot_train_accuracy.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA = REPO_ROOT / "data" / "metadata.csv"
RUNS_DIR = REPO_ROOT / "results" / "llm_runs"
FIGURE = REPO_ROOT / "results" / "figures" / "llm_train_accuracy"
FINAL_VERSION = "v6"  # the version chosen on the validation set (marked on the chart)
CHANCE = 1 / 20


def load_run(path):
    """The saved predictions of one run, as a list of dicts (the first line is the setup)."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[1:]


def main():
    word_of = pd.read_csv(METADATA).set_index("example_id")["word"]
    paths = sorted(RUNS_DIR.glob("v*_train.jsonl"), key=lambda p: int(p.name.split("_")[0][1:]))
    rows, clip_ids = [], None
    for path in paths:
        results = load_run(path)
        ids = [r["id"] for r in results]
        if clip_ids is not None and ids != clip_ids:
            print(f"WARNING: {path.name} was scored on different clips from the other versions")
        clip_ids = clip_ids or ids
        correct = sum(r["prediction"] == word_of[r["id"]] for r in results)
        rows.append({"version": path.name.split("_")[0], "correct": correct, "clips": len(results),
                     "accuracy": correct / len(results),
                     "invalid": sum(r["prediction"] is None for r in results)})
    scores = pd.DataFrame(rows)
    print(scores.to_string(index=False, formatters={"accuracy": "{:.1%}".format}))

    # ---- Bar chart (research-paper style, as in train.py: no title, the caption goes in the report) ----
    BAR_COLOR = "#2a78d6"  # the same blue as the deep learning training curve
    INK, GREY, GRID = "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    percent = scores["accuracy"] * 100
    bars = ax.bar(scores["version"], percent, width=0.6, color=BAR_COLOR)
    for bar, value in zip(bars, percent):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}",
                ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(CHANCE * 100, color=GREY, linewidth=0.8, linestyle=":")
    # label the chance line just right of the last bar, so it never sits on top of a bar
    ax.text(len(scores) - 0.45, CHANCE * 100, "chance\n(5%)", ha="left", va="center", fontsize=7.5,
            color=GREY, clip_on=False)
    ax.set_xlim(-0.6, len(scores) - 0.4)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([f"{v}\n(final)" if v == FINAL_VERSION else v for v in scores["version"]])
    ax.set_xlabel("Prompt version")
    ax.set_ylabel(f"Training accuracy (%, n = {scores['clips'].iloc[0]})")
    ax.set_ylim(0, max(percent.max() * 1.2, 10))
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{FIGURE}.png", dpi=300)
    fig.savefig(f"{FIGURE}.pdf")  # vector version for the report / poster
    print(f"saved {FIGURE.relative_to(REPO_ROOT)}.png/.pdf")


if __name__ == "__main__":
    main()
