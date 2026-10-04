"""Tiny web server for the ASL human-baseline quiz.

Run from the repo root:   python quiz/app.py
Then open:                http://127.0.0.1:5050
"""
import csv
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

QUIZ_DIR = Path(__file__).resolve().parent  # .../ASL-Classifier/quiz
REPO_DIR = QUIZ_DIR.parent                   # .../ASL-Classifier

TEST_JSON = REPO_DIR / "data" / "test.json"
# Maps each test clip to a short id (a001, a002, ...) used in the results CSV.
TEST_IDS_CSV = REPO_DIR / "eval" / "human_test_set.csv"
RESULTS_CSV = QUIZ_DIR / "results" / "test_results.csv"

# Some source videos have the answer word written on screen. During the test, each clip is
# cropped to the signer ("box" in test.json), which hides most of that text. For these
# sources the text is inside the crop, so the page also draws a black rectangle over it.
# Rectangles are [left, top, right, bottom] as fractions of the full video frame.
# (Found by checking every test clip at 10%, 50% and 90% of its length.)
CAPTION_COVERS = {
    "9g-hioQapYE": [0.00, 0.78, 0.36, 1.00],  # "WANT", "MOTHER", ... bottom-left
    "GmxS5HkNc3o": [0.52, 0.12, 0.85, 0.38],  # "Like", "Finish", ... top-right
    "ybhXZy32d1E": [0.00, 0.40, 0.565, 0.62], # "Father/Daddy", "Mother/Mommy" left
    "G77ZoILMYw4": [0.07, 0.80, 0.33, 0.95],  # "TEACHER" bottom-left
}

RESULT_COLUMNS = [
    "participant", "timestamp", "question_number", "video_id", "video_file",
    "correct_answer", "user_answer", "is_correct", "time_spent_seconds",
]

# Files in quiz/static/ (index.html, style.css, quiz.js) are served at the site root,
# e.g. quiz/static/quiz.js -> http://127.0.0.1:5050/quiz.js
app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/practice.json")
def practice_list():
    return send_from_directory(QUIZ_DIR, "practice.json")


@app.route("/test.json")
def test_list():
    """The 136 test clips: same fields as practice.json, plus video_id, box and covers."""
    with open(TEST_IDS_CSV, newline="") as f:
        ids = {row["video file"]: row["video id"] for row in csv.DictReader(f)}

    questions = []
    for entry in json.loads(TEST_JSON.read_text()):
        # test.json says "data/msasl/...", but the clips live in "msasl/..." at the repo root.
        clip_path = entry["clip_path"].removeprefix("data/")
        questions.append({
            "video_id": ids[clip_path],
            "text": entry["text"],
            "label": entry["label"],
            "clip_path": clip_path,
            "box": entry["box"],  # signer area: [top, left, bottom, right], fractions of the frame
            "covers": [rect for src, rect in CAPTION_COVERS.items() if src in entry["url"]],
        })
    return jsonify(questions)


@app.route("/api/results", methods=["POST"])
def save_results():
    """Append one test's answers (one row per question) to results/test_results.csv."""
    rows = request.get_json()["rows"]

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in RESULT_COLUMNS})

    return jsonify({"saved": len(rows), "file": str(RESULTS_CSV.relative_to(REPO_DIR))})


# Videos: each entry's clip_path is relative to the repo root, e.g. "msasl/no/xyz.mp4",
# so the browser can request it unchanged as "/msasl/no/xyz.mp4".
@app.route("/msasl/<path:path>")
def msasl_video(path):
    return send_from_directory(REPO_DIR / "msasl", path)


@app.route("/data/<path:path>")
def data_file(path):
    return send_from_directory(REPO_DIR / "data", path)


if __name__ == "__main__":
    app.run(debug=True, port=5050)  # not 5000: macOS AirPlay uses that port
