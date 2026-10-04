# ASL Sign Quiz (human baseline)

A small website for measuring human accuracy and speed on the same 136-clip test set the models use.

```
quiz/
├── app.py            # Flask server: serves the pages, the question lists and the videos,
│                     #   and appends test results to results/test_results.csv
├── practice.json     # 20 practice clips (1 per sign, none from the test set)
├── results/          # created on the first saved test
│   └── test_results.csv
└── static/
    ├── index.html    # the screens (start / question / end)
    ├── style.css
    └── quiz.js       # all quiz logic (shuffle, show video, check answer, timing, saving)
```

## Run it

From the repo root:

```bash
pip install flask
python quiz/app.py
```

Then open http://127.0.0.1:5050

The videos must be in `msasl/<word>/...mp4` at the repo root.

## Modes

- **Practice**: the 20 clips in `practice.json`, shuffled. You get "Correct" / "Wrong" after each answer.
- **Test**: enter a participant id (e.g. Person 1, not a real name), then all 136 clips from `data/test.json`, shuffled, with a visible timer.
  There's no feedback until the end. "Don't know" records a blank answer.
  When you finish, the results are appended to `results/test_results.csv`, one row per video:

  | column | meaning |
  |---|---|
  | participant | the participant id entered at the start (e.g. Person 1) |
  | timestamp | local time the answer was submitted |
  | question_number | 1–136, in the order this person saw them |
  | video_id | a001–a136, same ids as `eval/human_test_set.csv` |
  | video_file | clip path |
  | correct_answer / user_answer | the sign word / what was typed |
  | is_correct | 1 or 0 (case and extra spaces ignored) |
  | time_spent_seconds | from the video appearing to the answer being submitted |

Answers are free text, so a typo counts as wrong.

## Hidden captions in the test

Some test clips have the answer written on screen (e.g. "WANT" in a corner). In test mode each video is
cropped to the signer's bounding box (`box` in `test.json`), and clips from four source videos also get a
black rectangle over the text (`CAPTION_COVERS` in `app.py`). Every test clip was checked at 10%, 50% and
90% of its length. The videos have no player controls (so no fullscreen), and clicking the video replays it.

## Changing the practice videos

Each entry in `practice.json` looks like:

```json
{"text": "hello", "label": 0, "clip_path": "msasl/hello/yG09SQB0Hds_1874_1903.mp4"}
```

Change `clip_path` to swap a video (you can also list more than one per sign).
Avoid clips that are in `data/test.json`, or you'll have seen those test clips during practice.
