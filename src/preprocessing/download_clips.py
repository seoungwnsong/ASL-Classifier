"""Download and trim the MS-ASL clips listed in data/train.json, val.json, test.json.

Each YouTube video is downloaded once into data/msasl_downloads/ (not in git),
then every clip from it is cut out with ffmpeg and saved at the entry's
"clip_path" (data/msasl/<word>/<video id>_<start frame>_<end frame>.mp4).

Already-downloaded clips are skipped, so the script can be stopped and re-run.
Videos that fail are recorded in data/msasl_download_failures.json.
If YouTube starts blocking requests, the script stops; wait a while and re-run.

Needs yt-dlp and imageio-ffmpeg (pip install -r requirements.txt).

Run from the repository root:
    python src/preprocessing/download_clips.py            # test, then val, then train
    python src/preprocessing/download_clips.py test       # one split only
"""

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict

import imageio_ffmpeg
import yt_dlp

from filter_dataset import DOWNLOAD_FAILURES_FILE as FAILURES_FILE
from filter_dataset import OUT_DIR, REPO_ROOT, SELECTED_CLASSES, video_id

SPLIT_ORDER = ["test", "val", "train"]
DOWNLOAD_DIR = OUT_DIR / "msasl_downloads"
SECONDS_BETWEEN_VIDEOS = 3
PLAYER_CLIENTS = [None, "android"]  # None = yt-dlp's default
# YouTube's block shows up as ordinary per-video errors ("Please sign in"),
# so many failures in a row means we are blocked, not that the videos are broken.
MAX_FAILURES_IN_A_ROW = 5

# Error messages that mean YouTube is blocking us, not that the video is broken.
BLOCKED_MESSAGES = ["not a bot", "HTTP Error 429", "Too Many Requests"]


class Blocked(Exception):
    pass


def download_video(vid):
    """Download one YouTube video (video only, at most 720p) and return its path."""
    existing = list(DOWNLOAD_DIR.glob(f"{vid}.*"))
    if existing:
        return existing[0]

    # yt-dlp's default YouTube clients fail for some videos ("not available",
    # HTTP 403) that still play fine; the Android client usually works for them.
    for client in PLAYER_CLIENTS:
        options = {
            "format": "bv*[height<=720][ext=mp4]/b[height<=720]/b",
            "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        if client:
            options["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            return next(DOWNLOAD_DIR.glob(f"{vid}.*"))
        except yt_dlp.utils.DownloadError as error:
            if any(message in str(error) for message in BLOCKED_MESSAGES):
                raise Blocked(str(error)) from error
            if client == PLAYER_CLIENTS[-1]:
                raise


def trim_clip(video_path, entry):
    """Cut [start_time, end_time] out of the video and save it at entry["clip_path"]."""
    out_path = REPO_ROOT / entry["clip_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-ss", str(entry["start_time"]),
            "-i", str(video_path),
            "-t", str(entry["end_time"] - entry["start_time"]),
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(tmp_path),
        ],
        check=True,
    )
    tmp_path.rename(out_path)  # only a finished clip gets the real name


def load_failures():
    if FAILURES_FILE.exists():
        with open(FAILURES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_failures(failures):
    with open(FAILURES_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(failures.items())), f, indent=0)


def download_split(split, failures):
    with open(OUT_DIR / f"{split}.json", encoding="utf-8") as f:
        entries = [e for e in json.load(f) if e["source"] == "MS-ASL"]

    todo = defaultdict(list)
    for entry in entries:
        if not (REPO_ROOT / entry["clip_path"]).exists():
            todo[video_id(entry["url"])].append(entry)
    print(f"{split}: {len(entries)} MS-ASL clips, {sum(map(len, todo.values()))} "
          f"to download from {len(todo)} videos")

    failures_in_a_row = 0
    for i, (vid, clips) in enumerate(sorted(todo.items()), start=1):
        if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
            save_failures(failures)
            raise SystemExit(f"{failures_in_a_row} videos failed in a row, so YouTube is "
                             "probably blocking downloads. Wait an hour or more and run again.")
        try:
            video_path = download_video(vid)
            for entry in clips:
                trim_clip(video_path, entry)
            failures.pop(vid, None)
            failures_in_a_row = 0
            print(f"  [{i}/{len(todo)}] {vid}: {len(clips)} clips")
        except Blocked as error:
            save_failures(failures)
            raise SystemExit(f"YouTube is blocking downloads ({error}). "
                             "Wait a while (an hour or more) and run again.")
        except (yt_dlp.utils.DownloadError, subprocess.CalledProcessError, StopIteration) as error:
            failures[vid] = str(error).strip().splitlines()[-1][:300]
            failures_in_a_row += 1
            print(f"  [{i}/{len(todo)}] {vid}: FAILED - {failures[vid]}")
        save_failures(failures)
        time.sleep(SECONDS_BETWEEN_VIDEOS)

    missing = Counter(e["text"] for e in entries if not (REPO_ROOT / e["clip_path"]).exists())
    print(f"{split}: {len(entries) - sum(missing.values())}/{len(entries)} clips present")
    for word in SELECTED_CLASSES:
        if missing[word]:
            print(f"  {word}: {missing[word]} missing")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress live, even in a log file
    splits = sys.argv[1:] or SPLIT_ORDER
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    failures = load_failures()
    for split in splits:
        download_split(split, failures)


if __name__ == "__main__":
    main()
