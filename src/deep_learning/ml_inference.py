"""Deep learning approach: predict the ASL word signed in one video clip.

    video -> 16 frames (captions covered, cropped to the signer, 288 x 288)
          -> VideoPrism base, frozen (weights downloaded from keras_hub, not trained by us)
          -> average over the 16 frames and 256 patches -> 768 numbers
          -> our trained head (head_H64_D0.5.keras) -> 20 probabilities -> the word

This file does the same preprocessing as frames.py and extract_features.py, for one clip,
so it runs on its own. Run it from the folder that holds sample.mp4 and the head file:
    python ml_inference.py
"""

from pathlib import Path

import cv2
import keras
import keras_hub
import numpy as np

HERE = Path(__file__).resolve().parent
HEAD_FILE = HERE / "head_H64_D0.5.keras"
PRESET = "videoprism_public_v1_base"
NUM_FRAMES, FRAME_SIZE = 16, 288
BLACK_FRAME_MAX = 30  # a frame whose brightest pixel is below this is completely black

# word number i = WORDS[i] (alphabetical, as in training)
WORDS = ["deaf", "eat", "father", "finish", "good", "happy", "hello", "learn", "like", "mother",
         "need", "nice", "no", "please", "sad", "school", "teacher", "tired", "want", "yes"]

# The test sample (example j001): its signer box [top, left, bottom, right] and the black
# rectangle [left, top, right, bottom] over the on-screen text, as fractions of the frame
# (from data/test.json and data/caption_covers.csv, the same values used for training data)
SAMPLE = HERE / "sample.mp4"
SAMPLE_BOX = [0.05659377574920654, 0.06644341349601746, 0.9990754723548889, 0.7649063467979431]
SAMPLE_COVERS = [[0.0, 0.78, 0.36, 1.0]]


def read_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def prepare_frame(frame, box, covers):
    """Cover the text, crop to the signer, pad to a square with black and resize one frame."""
    frame = frame.copy()
    height, width = frame.shape[:2]
    for left, top, right, bottom in covers:
        frame[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 0
    top, left, bottom, right = box
    frame = frame[int(top * height):int(bottom * height), int(left * width):int(right * width)]
    height, width = frame.shape[:2]
    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    frame = cv2.copyMakeBorder(frame, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
                               cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))


def clip_input(video_path, box, covers):
    """16 evenly spaced frames (skipping completely black ones), RGB, values 0 to 1: (16, 288, 288, 3)."""
    frames = read_frames(video_path)
    frames = [f for f in frames if f.max() >= BLACK_FRAME_MAX] or frames
    picks = np.linspace(0, len(frames) - 1, NUM_FRAMES).round().astype(int)
    clip = [cv2.cvtColor(prepare_frame(frames[i], box, covers), cv2.COLOR_BGR2RGB) for i in picks]
    return np.array(clip, dtype=np.float32) / 255.0


def predict(video_path, box, covers):
    """The predicted word and its probability."""
    backbone = keras_hub.models.VideoPrismBackbone.from_preset(PRESET)  # frozen, pretrained
    head = keras.models.load_model(HEAD_FILE)                           # trained by ml_train.py

    video = clip_input(video_path, box, covers)
    output = backbone.predict(video[np.newaxis], verbose=0)  # (1, 16 frames, 256 patches, 768)
    features = output.mean(axis=(1, 2))                      # (1, 768)
    probabilities = head.predict(features, verbose=0)[0]     # (20,)
    best = probabilities.argmax()
    return WORDS[best], probabilities[best]


if __name__ == "__main__":
    word, probability = predict(SAMPLE, SAMPLE_BOX, SAMPLE_COVERS)
    print(f"Prediction for {SAMPLE.name}: {word} (probability {probability:.2f})")
