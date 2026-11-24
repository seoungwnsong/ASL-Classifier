# Computes every number in our results table (Table 3) from the five csv files.
# Our human evaluation subset is the full test set (136 clips), so one table covers both.
# Run: python eval.py

from pathlib import Path

import pandas as pd

folder = Path(__file__).parent
truth = pd.read_csv(folder / "ground_truth.csv")
human = pd.read_csv(folder / "human_eval.csv", keep_default_na=False)
symbolic = pd.read_csv(folder / "symbolic_eval.csv")
ml = pd.read_csv(folder / "ml_eval.csv")
llm = pd.read_csv(folder / "llm_eval.csv", keep_default_na=False)

# put the true word and every prediction for each clip in one table
data = (truth.merge(human, on="Example ID").merge(symbolic, on="Example ID")
        .merge(ml, on="Example ID").merge(llm, on="Example ID"))
n = len(data)
print("Human evaluation subset = full test set, n =", n)

def accuracy(column):
    correct = (data[column] == data["Ground Truth"]).sum()
    return correct, correct / n * 100

p1_correct, p1_acc = accuracy("Human Prediction (Person 1)")
p2_correct, p2_acc = accuracy("Human Prediction (Person 2)")
alg_correct, alg_acc = accuracy("Algorithmic Prediction")
ml_correct, ml_acc = accuracy("Deep Learning Prediction")
llm_correct, llm_acc = accuracy("LLM Prediction")
# answers that were not one of the 20 words are saved as INVALID, and count as wrong
llm_invalid = (data["LLM Prediction"] == "INVALID").sum()

print(f"Human Participant 1:  {p1_acc:.1f}% ({p1_correct}/{n})")
print(f"Human Participant 2:  {p2_acc:.1f}% ({p2_correct}/{n})")
print(f"Human Average:        {(p1_acc + p2_acc) / 2:.1f}% ({(p1_correct + p2_correct) / 2}/{n})")
print(f"Algorithmic Approach: {alg_acc:.1f}% ({alg_correct}/{n})")
print(f"ML Approach:          {ml_acc:.1f}% ({ml_correct}/{n})")
print(f"LLM Approach:         {llm_acc:.1f}% ({llm_correct}/{n}), {llm_invalid} unparseable")
