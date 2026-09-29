"""Extract MediaPipe keypoints (hands, body, face) from every clip in the splits.

For each clip in data/train.json, val.json and test.json this saves one file
    data/keypoints/<source>/<word>/<clip name>.npy
with shape (frames, 63, 2): the (x, y) position of 63 landmarks in each frame,
as fractions of the frame width/height (0 to 1). Landmarks that were not found
in a frame are NaN. The landmark order is written to data/keypoints/landmarks.json.

"left"/"right" always mean the SIGNER's left/right hand (matched to the body's
wrists), not MediaPipe's own handedness label, which assumes a mirrored camera.

A quality report (share of frames where each part was found) is written to
data/keypoints/quality.csv.

Already-extracted clips are skipped, so the script can be stopped and re-run.

Run from the repository root:
    python src/preprocessing/extract_keypoints.py --limit 5     # try a few clips first
    python src/preprocessing/extract_keypoints.py               # all clips
    python src/preprocessing/extract_keypoints.py --preview data/msasl/no/0HvgswZfDGI_0_35.mp4
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from filter_dataset import OUT_DIR, REPO_ROOT

KEYPOINTS_DIR = OUT_DIR / "keypoints"
MODELS_DIR = REPO_ROOT / "models" / "mediapipe"
MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task",
}

# Body landmarks to keep (MediaPipe Pose index -> name).
POSE_POINTS = {
    0: "nose", 7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
}
# Face landmarks to keep (MediaPipe Face Mesh index -> name): the places signs are made.
FACE_POINTS = {
    10: "forehead_top", 151: "forehead", 1: "nose_tip",
    13: "upper_lip", 14: "lower_lip", 61: "mouth_right", 291: "mouth_left",
    152: "chin", 33: "right_eye", 263: "left_eye",
    234: "right_cheek", 454: "left_cheek",
}
HAND_POINTS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
LANDMARK_NAMES = (
    [f"pose_{name}" for name in POSE_POINTS.values()]
    + [f"face_{name}" for name in FACE_POINTS.values()]
    + [f"left_hand_{name}" for name in HAND_POINTS]
    + [f"right_hand_{name}" for name in HAND_POINTS]
)
POSE_SLICE = slice(0, len(POSE_POINTS))
FACE_SLICE = slice(POSE_SLICE.stop, POSE_SLICE.stop + len(FACE_POINTS))
LEFT_HAND_SLICE = slice(FACE_SLICE.stop, FACE_SLICE.stop + len(HAND_POINTS))
RIGHT_HAND_SLICE = slice(LEFT_HAND_SLICE.stop, LEFT_HAND_SLICE.stop + len(HAND_POINTS))

# Lower than MediaPipe's default (0.5) because many MS-ASL clips are only 360p.
MIN_CONFIDENCE = 0.3


def keypoints_path(entry):
    """data/msasl/mother/x.mp4 -> data/keypoints/msasl/mother/x.npy"""
    relative = Path(entry["clip_path"]).relative_to("data")
    return KEYPOINTS_DIR / relative.with_suffix(".npy")


def model_path(name):
    """Local path of a MediaPipe model, downloading it the first time."""
    path = MODELS_DIR / Path(MODEL_URLS[name]).name
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(MODEL_URLS[name], path)
    return str(path)


def create_landmarkers():
    CPU = BaseOptions.Delegate.CPU  # the GPU (Metal) delegate crashes on some Macs
    video = vision.RunningMode.VIDEO
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("hand"), delegate=CPU), running_mode=video,
        num_hands=2, min_hand_detection_confidence=MIN_CONFIDENCE,
        min_hand_presence_confidence=MIN_CONFIDENCE, min_tracking_confidence=MIN_CONFIDENCE))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("pose"), delegate=CPU), running_mode=video,
        min_pose_detection_confidence=MIN_CONFIDENCE, min_pose_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("face"), delegate=CPU), running_mode=video,
        min_face_detection_confidence=MIN_CONFIDENCE, min_face_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE))
    return hand, pose, face


def assign_hands(hands, pose_frame):
    """Decide which detected hand is the signer's left and which is the right.

    Each hand goes to the side whose body wrist is closer. Without a body,
    MediaPipe's label is used (flipped, because it assumes a mirrored image).
    """
    left = right = None
    body_wrists = pose_frame[[list(POSE_POINTS.values()).index("left_wrist"),
                              list(POSE_POINTS.values()).index("right_wrist")]]
    for points, label in hands:
        if not np.isnan(body_wrists).any():
            to_left, to_right = np.linalg.norm(body_wrists - points[0], axis=1)
            is_left = to_left < to_right
        else:
            is_left = label == "Right"
        if is_left and left is None:
            left = points
        elif not is_left and right is None:
            right = points
        elif left is None:
            left = points
        else:
            right = points
    return left, right


def extract_clip(video_path):
    """Return an array (frames, 63, 2) of landmark positions for one video."""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    hand_model, pose_model, face_model = create_landmarkers()
    frames = []
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(index * 1000 / fps)
        index += 1

        frame = np.full((len(LANDMARK_NAMES), 2), np.nan, dtype=np.float32)
        pose = pose_model.detect_for_video(image, timestamp_ms)
        if pose.pose_landmarks:
            points = pose.pose_landmarks[0]
            frame[POSE_SLICE] = [(points[i].x, points[i].y) for i in POSE_POINTS]
        face = face_model.detect_for_video(image, timestamp_ms)
        if face.face_landmarks:
            points = face.face_landmarks[0]
            frame[FACE_SLICE] = [(points[i].x, points[i].y) for i in FACE_POINTS]
        hand = hand_model.detect_for_video(image, timestamp_ms)
        hands = [(np.array([(p.x, p.y) for p in points], dtype=np.float32), handedness[0].category_name)
                 for points, handedness in zip(hand.hand_landmarks, hand.handedness)]
        left, right = assign_hands(hands, frame[POSE_SLICE])
        if left is not None:
            frame[LEFT_HAND_SLICE] = left
        if right is not None:
            frame[RIGHT_HAND_SLICE] = right
        frames.append(frame)

    capture.release()
    for model in (hand_model, pose_model, face_model):
        model.close()
    return np.stack(frames) if frames else np.empty((0, len(LANDMARK_NAMES), 2), np.float32)


def quality_row(entry, keypoints):
    """Share of frames in which each body part was found."""
    def found(part):
        if len(keypoints) == 0:
            return 0.0
        return float(np.mean(~np.isnan(keypoints[:, part, 0]).any(axis=1)))
    either_hand = ~(np.isnan(keypoints[:, LEFT_HAND_SLICE, 0]).any(axis=1)
                    & np.isnan(keypoints[:, RIGHT_HAND_SLICE, 0]).any(axis=1))
    return {
        "clip_path": entry["clip_path"], "split": entry["split"], "text": entry["text"],
        "source": entry["source"], "frames": len(keypoints),
        "pose": round(found(POSE_SLICE), 3), "face": round(found(FACE_SLICE), 3),
        "left_hand": round(found(LEFT_HAND_SLICE), 3), "right_hand": round(found(RIGHT_HAND_SLICE), 3),
        "any_hand": round(float(either_hand.mean()) if len(keypoints) else 0.0, 3),
    }


def write_quality_report(entries):
    rows = []
    for entry in entries:
        path = keypoints_path(entry)
        if path.exists():
            rows.append(quality_row(entry, np.load(path)))
    with open(KEYPOINTS_DIR / "quality.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    any_hand = np.array([row["any_hand"] for row in rows])
    print(f"\nQuality ({len(rows)} clips): hands found in {any_hand.mean():.0%} of frames on average")
    print(f"  clips with hands in < 50% of frames: {(any_hand < 0.5).sum()}")
    print(f"  pose found: {np.mean([r['pose'] for r in rows]):.0%}, "
          f"face found: {np.mean([r['face'] for r in rows]):.0%} of frames on average")


def save_preview(video_path, out_path):
    """Draw the landmarks on the middle frame of a clip, to check they are in the right place."""
    keypoints = extract_clip(video_path)
    capture = cv2.VideoCapture(str(video_path))
    middle = len(keypoints) // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, middle)
    _, bgr = capture.read()
    capture.release()
    height, width = bgr.shape[:2]
    colors = [((255, 200, 0), POSE_SLICE), ((0, 200, 255), FACE_SLICE),
              ((0, 255, 0), LEFT_HAND_SLICE), ((255, 0, 255), RIGHT_HAND_SLICE)]
    for color, part in colors:
        for x, y in keypoints[middle, part]:
            if not np.isnan(x):
                cv2.circle(bgr, (int(x * width), int(y * height)), 3, color, -1)
    cv2.imwrite(str(out_path), bgr)
    print(f"Saved {out_path} (blue=body, orange=face, green=left hand, pink=right hand)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process this many clips (for testing)")
    parser.add_argument("--preview", help="draw landmarks on one clip's middle frame and exit")
    args = parser.parse_args()

    if args.preview:
        save_preview(REPO_ROOT / args.preview, REPO_ROOT / "keypoints_preview.png")
        return

    entries = []
    for split in ["test", "val", "train"]:
        with open(OUT_DIR / f"{split}.json", encoding="utf-8") as f:
            entries += [{**e, "split": split} for e in json.load(f)]
    entries = list({e["clip_path"]: e for e in entries}.values())  # a few clips are listed twice
    if args.limit:
        entries = entries[:args.limit]

    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(KEYPOINTS_DIR / "landmarks.json", "w", encoding="utf-8") as f:
        json.dump(LANDMARK_NAMES, f, indent=0)

    todo = [e for e in entries if not keypoints_path(e).exists()]
    print(f"{len(entries)} clips, {len(todo)} to extract")
    for i, entry in enumerate(todo, start=1):
        keypoints = extract_clip(REPO_ROOT / entry["clip_path"])
        path = keypoints_path(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            np.save(f, keypoints)
        tmp_path.rename(path)  # only a finished file gets the real name
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {entry['clip_path']}", flush=True)

    write_quality_report(entries)


if __name__ == "__main__":
    main()
