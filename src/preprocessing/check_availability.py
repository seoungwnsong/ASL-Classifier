"""Check which MS-ASL YouTube videos are still playable.

Many MS-ASL links are dead (removed or private). This script checks every
video used by the selected classes and saves the result to
data/video_availability.json, which filter_dataset.py reads.

Results are cached: videos already marked "ok", "unavailable" or "private"
are not checked again. Delete the file to re-check everything.

Run from the repository root (needs internet access):
    python src/preprocessing/check_availability.py
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from filter_dataset import RAW_DIR, SELECTED_CLASSES, SPLITS, AVAILABILITY_FILE, video_id

FINAL_STATUSES = {"ok", "unavailable", "private"}


def fetch(request, timeout):
    """urlopen that waits and retries when YouTube rate-limits us (HTTP 429)."""
    for wait in [10, 30, 60, 120]:
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(wait)
    return urllib.request.urlopen(request, timeout=timeout)


def check_video(vid):
    """Return "ok", "unavailable", "private" or "unknown" for one YouTube video."""
    # oEmbed answers 200 for public, playable videos.
    url = urllib.parse.quote(f"https://www.youtube.com/watch?v={vid}", safe="")
    try:
        fetch(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=15)
        return "ok"
    except urllib.error.HTTPError as error:
        if error.code == 429:
            return "unknown"
    except OSError:
        return "unknown"

    # Otherwise read the watch page: videos with embedding disabled still play.
    request = urllib.request.Request(
        f"https://www.youtube.com/watch?v={vid}",
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US"},
    )
    try:
        html = fetch(request, timeout=20).read().decode("utf-8", "ignore")
    except OSError:
        return "unknown"
    match = re.search(r'"playabilityStatus":\{"status":"(\w+)"', html)
    if not match:
        return "unknown"
    return {"OK": "ok", "LOGIN_REQUIRED": "private"}.get(match.group(1), "unavailable")


def main():
    video_ids = set()
    for _, input_name, _ in SPLITS:
        with open(RAW_DIR / input_name, encoding="utf-8") as f:
            for entry in json.load(f):
                if entry["text"] in SELECTED_CLASSES:
                    video_ids.add(video_id(entry["url"]))

    availability = {}
    if AVAILABILITY_FILE.exists():
        with open(AVAILABILITY_FILE, encoding="utf-8") as f:
            availability = json.load(f)

    to_check = sorted(v for v in video_ids if availability.get(v) not in FINAL_STATUSES)
    print(f"{len(video_ids)} videos used, checking {len(to_check)}...")

    with ThreadPoolExecutor(max_workers=4) as pool:
        availability.update(zip(to_check, pool.map(check_video, to_check)))

    with open(AVAILABILITY_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(availability.items())), f, indent=0)

    statuses = Counter(availability[v] for v in video_ids)
    print(statuses)
    if statuses["unknown"]:
        print("Some videos could not be checked (network or rate limit). Run again later.")


if __name__ == "__main__":
    main()
