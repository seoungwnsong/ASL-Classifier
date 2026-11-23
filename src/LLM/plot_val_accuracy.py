"""Compare the LLM prompt versions on the validation set, with a bar chart.

Reads the saved runs in results/llm_runs/<version>_val.jsonl (made by
`python src/LLM/llm_harnness.py val v3 v5 v6`). No model is called.
An invalid answer (not one of the 20 labels) counts as wrong.

The validation set is uneven (4 to 20 clips per word), so besides accuracy this also
reports balanced accuracy (the average over the 20 words of the share of that word's
clips answered correctly) and macro-F1; both count every word equally.

Prints one row per version and saves:
    results/figures/llm_val_accuracy.png   accuracy and balanced accuracy per version (also .pdf)

Run from the repository root:
    python src/LLM/plot_val_accuracy.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from plot_train_accuracy import (BAR_COLOR, CHANCE, FINAL_VERSION, GREY, GRID, INK, METADATA, REPO_ROOT,
                                 RUNS_DIR, SECOND_COLOR, apply_style, load_run, version_number)

FIGURE = REPO_ROOT / "results" / "figures" / "llm_val_accuracy"


def scores_of(results, word_of):
    """Accuracy, balanced accuracy and macro-F1 of one run."""
    truth = [word_of[r["id"]] for r in results]
    guess = [r["prediction"] for r in results]
    recalls, f1s = [], []
    for word in sorted(set(truth)):
        hits = sum(t == word and g == word for t, g in zip(truth, guess))
        guessed = sum(g == word for g in guess)
        actual = sum(t == word for t in truth)
        recalls.append(hits / actual)
        f1s.append(2 * hits / (guessed + actual) if hits else 0.0)
    correct = sum(t == g for t, g in zip(truth, guess))
    return {"correct": correct, "clips": len(results), "accuracy": correct / len(results),
            "balanced accuracy": sum(recalls) / len(recalls), "macro-F1": sum(f1s) / len(f1s),
            "invalid": guess.count(None)}


def main():
    word_of = pd.read_csv(METADATA).set_index("example_id")["word"]
    paths = sorted(RUNS_DIR.glob("v*_val.jsonl"), key=version_number)
    if not paths:
        raise SystemExit("No validation runs found. Run: python src/LLM/llm_harnness.py val v3 v5 v6")
    rows, clip_ids = [], None
    for path in paths:
        results = load_run(path)
        ids = [r["id"] for r in results]
        if clip_ids is not None and ids != clip_ids:
            print(f"WARNING: {path.name} was scored on different clips from the other versions")
        clip_ids = clip_ids or ids
        rows.append({"version": path.name.split("_")[0], **scores_of(results, word_of)})
    scores = pd.DataFrame(rows)
    percent = "{:.1%}".format
    print(scores.to_string(index=False, formatters={"accuracy": percent, "balanced accuracy": percent,
                                                    "macro-F1": "{:.3f}".format}))

    # ---- Bar chart: accuracy and balanced accuracy side by side for each version ----
    apply_style()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    width = 0.36
    positions = range(len(scores))
    for offset, key, color in [(-width / 2, "accuracy", BAR_COLOR), (width / 2, "balanced accuracy", SECOND_COLOR)]:
        values = scores[key] * 100
        # bars a bit narrower than their slot, so the two bars of a version have a small gap
        bars = ax.bar([p + offset for p in positions], values, width=width * 0.92, color=color,
                      label=key.capitalize())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(CHANCE * 100, color=GREY, linewidth=0.8, linestyle=":")
    # label the chance line just right of the last bars, so it never sits on top of a bar
    ax.text(len(scores) - 0.45, CHANCE * 100, "chance\n(5%)", ha="left", va="center", fontsize=7.5,
            color=GREY, clip_on=False)
    ax.set_xlim(-0.6, len(scores) - 0.4)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{v}\n(final)" if v == FINAL_VERSION else v for v in scores["version"]])
    ax.set_xlabel("Prompt version")
    ax.set_ylabel(f"Validation score (%, n = {scores['clips'].iloc[0]})")
    ax.set_ylim(0, max(scores[["accuracy", "balanced accuracy"]].max().max() * 100 * 1.3, 10))
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{FIGURE}.png", dpi=300)
    fig.savefig(f"{FIGURE}.pdf")  # vector version for the report / poster
    print(f"saved {FIGURE.relative_to(REPO_ROOT)}.png/.pdf")


if __name__ == "__main__":
    main()
