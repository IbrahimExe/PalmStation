
# File to collect create and save my own dataset of hand gesture landmarks using MediaPipe.
#   r - RIGHT
#   l - LEFT
#   u - UP
#   d - DOWN
#   s - Save dataset to data/gestures.csv
#   q - Quit
# Each recorded sample appends a row to the CSV with 42 features (x,y for 21 landmarks) + label.
# 100 - 150 samples per gesture is a good enough size for training.

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from collections import defaultdict

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "gestures.csv")

# store rows as (features..., label)
data = []
counts = defaultdict(int)

def landmarks_to_features(landmarks):
    # returns flat list [x1,y1,x2,y2,...] normalized (already in 0..1)
    feats = []
    for lm in landmarks:
        feats.append(lm.x)
        feats.append(lm.y)
    return feats

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())
print("Press r/l/u/d to record. s to save. q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    display_text = "Counts: " + ", ".join(f"{k}:{v}" for k,v in counts.items())
    cv2.putText(frame, display_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collect Landmarks (r/l/u/d, s save, q quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('r'), ord('l'), ord('u'), ord('d')]:
        label = {ord('r'):'right', ord('l'):'left', ord('u'):'up', ord('d'):'down'}[key]
        if res.multi_hand_landmarks:
            feats = landmarks_to_features(hand.landmark)
            data.append(feats + [label])
            counts[label] += 1
            print(f"Recorded {label} sample. Totals: {counts}")
        else:
            print("No hand detected — not recorded.")
    elif key == ord('s'):
        # save to CSV
        header = [f"x{i}" if i%2==0 else f"y{i//2}" for i in range(42)]
        # header above isn't perfect readable but fine; we'll write generic names
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}"]
        header.append("label")
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        print(f"Saved {len(data)} rows to {CSV_PATH}")

cap.release()
cv2.destroyAllWindows()