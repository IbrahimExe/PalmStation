# snake_gesture_opencv.py
print("SCRIPT STARTED")

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
from collections import deque, Counter
import joblib
import os

# ------------ config ------------
MODEL_PATH = "models/gesture_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"
CONF_THRESH = 0.70       # minimum classifier probability to accept prediction
VOTE_LEN = 5             # majority vote window for classifier predictions
move_interval = 0.20     # seconds per move (adjustable with +/- during play)

# ------------ load model if present ------------
clf = None
le = None
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    print("Loaded classifier:", MODEL_PATH)
else:
    print("No classifier found at", MODEL_PATH, "- the game will fall back to angle method.")

# ------------ MediaPipe ------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

# ------------ Game settings ------------
WIDTH, HEIGHT = 640, 480
CELL = 20
GRID_W = WIDTH // CELL
GRID_H = HEIGHT // CELL

snake = [(GRID_W//2, GRID_H//2)]
direction = (1, 0)  # cell steps
food = (random.randrange(0, GRID_W), random.randrange(0, GRID_H))

last_move = time.time()

# ------------ smoothing & helpers ------------
class AngleEMA:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None
    def update(self, new_value):
        if new_value is None:
            return self.value
        if self.value is None:
            self.value = new_value
            return self.value
        # wrap-safe averaging of angles
        a = math.radians(self.value)
        b = math.radians(new_value)
        x = math.cos(b) * self.alpha + math.cos(a) * (1 - self.alpha)
        y = math.sin(b) * self.alpha + math.sin(a) * (1 - self.alpha)
        self.value = math.degrees(math.atan2(y, x))
        return self.value

angle_filter = AngleEMA(alpha=0.35)
recent_votes = deque(maxlen=VOTE_LEN)

def landmarks_to_features(landmarks):
    feats = []
    for lm in landmarks:
        feats.append(lm.x)
        feats.append(lm.y)
    return np.array(feats).reshape(1, -1)

def finger_angle(landmarks, tip_index, base_index):
    tip = landmarks[tip_index]
    base = landmarks[base_index]
    dx = tip.x - base.x
    dy = base.y - tip.y
    return math.degrees(math.atan2(dy, dx))

def angle_to_direction(angle):
    if angle is None:
        return None
    if -45 <= angle <= 45:
        return "right"
    elif 45 < angle <= 135:
        return "up"
    elif angle > 135 or angle < -135:
        return "left"
    else:
        return "down"

def spawn_food():
    while True:
        p = (random.randrange(0, GRID_W), random.randrange(0, GRID_H))
        if p not in snake:
            return p

# ---------- functions needed by choose_pointing_finger ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_size(landmarks):
    # approximate palm size using wrist (0) to middle_mcp (9)
    return dist(landmarks[0], landmarks[9])

def finger_angle_from(tip_landmark, base_landmark):
    dx = tip_landmark.x - base_landmark.x
    dy = base_landmark.y - tip_landmark.y  # invert y for screen coords
    return math.degrees(math.atan2(dy, dx))

# -------------------------
# Gesture decision logic
# -------------------------
def choose_pointing_finger(landmarks):
    """
    Decide whether user is pointing with index or thumb.
    Returns tuple (finger, angle) where finger is "index" or "thumb" or None.
    Angle is the computed angle (degrees) for the chosen finger, or None.
    """
    psize = palm_size(landmarks)
    if psize == 0:
        return None, None

    # indexes for mediapipe landmarks
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]

    index_tip_to_pip = dist(index_tip, index_pip) / psize
    index_tip_to_mcp = dist(index_tip, index_mcp) / psize
    thumb_tip_to_mcp = dist(thumb_tip, thumb_mcp) / psize

    # heuristics (tweakable)
    INDEX_EXTENDED = 0.50    # index tip well away from pip/mcp
    INDEX_FOLDED = 0.30      # index tip close -> folded
    THUMB_EXTENDED = 0.45    # thumb tip distance threshold

    index_extended = (index_tip_to_mcp >= INDEX_EXTENDED)
    index_folded = (index_tip_to_mcp <= INDEX_FOLDED)
    thumb_extended = (thumb_tip_to_mcp >= THUMB_EXTENDED)

    # prefer index if clearly extended
    if index_extended:
        angle = finger_angle_from(index_tip, index_mcp)
        return "index", angle

    # if index folded and thumb extended -> thumbs-out (fist + thumb) or horizontal thumb
    if index_folded and thumb_extended:
        angle = finger_angle_from(thumb_tip, thumb_mcp)
        return "thumb", angle

    # if thumb extended and index not strongly extended, accept thumb
    if (thumb_extended and not index_extended):
        angle = finger_angle_from(thumb_tip, thumb_mcp)
        return "thumb", angle

    return None, None

# ------------ main loop ------------
print("Starting Gesture Snake. Keys: q quit, +/- change speed.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    chosen_gesture = None
    classifier_conf = 0.0
    chosen_finger = None  # "index" or "thumb" if chosen

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # 1) classifier path (preferred if model exists)
        if clf is not None:
            feats = landmarks_to_features(hand.landmark)
            try:
                probs = clf.predict_proba(feats)[0]
                idx = probs.argmax()
                label = le.inverse_transform([idx])[0]
                conf = probs[idx]
                classifier_conf = float(conf)
                if conf >= CONF_THRESH:
                    recent_votes.append(label)
                    # majority vote
                    vote = Counter(recent_votes).most_common(1)[0][0]
                    chosen_gesture = vote
                else:
                    # don't append low-confidence predictions
                    pass
            except Exception:
                # prediction failed; fall back
                classifier_conf = 0.0

        # 2) fallback: angle method using chosen finger (index or thumb)
        if chosen_gesture is None:
            chosen_finger, raw_angle = choose_pointing_finger(hand.landmark)
            if raw_angle is not None:
                smooth_angle = angle_filter.update(raw_angle)
                chosen_gesture = angle_to_direction(smooth_angle)
            else:
                # if no finger clearly detected, do nothing (keep previous direction)
                chosen_gesture = None

        # display which finger is used
        if chosen_finger:
            cv2.putText(frame, f"Using: {chosen_finger}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    # map to candidate direction vector (cells)
    candidate = None
    if chosen_gesture == "up":
        candidate = (0, -1)
    elif chosen_gesture == "down":
        candidate = (0, 1)
    elif chosen_gesture == "left":
        candidate = (-1, 0)
    elif chosen_gesture == "right":
        candidate = (1, 0)

    # accept candidate if not 180 reversal
    if candidate:
        if (candidate[0] != -direction[0]) or (candidate[1] != -direction[1]) or len(snake) == 1:
            direction = candidate

    # movement timer
    now = time.time()
    if now - last_move >= move_interval:
        last_move = now
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        head = (head[0] % GRID_W, head[1] % GRID_H)
        if head in snake:
            snake = [(GRID_W//2, GRID_H//2)]
            direction = (1, 0)
            food = spawn_food()
        else:
            snake.insert(0, head)
            if head == food:
                food = spawn_food()
            else:
                snake.pop()

    # draw food & snake (pixel coords)
    fx, fy = food[0]*CELL, food[1]*CELL
    cv2.rectangle(frame, (fx, fy), (fx+CELL, fy+CELL), (0,0,255), -1)
    for i, (gx, gy) in enumerate(snake):
        px, py = gx*CELL, gy*CELL
        color = (0,200,0) if i!=0 else (0,255,100)
        cv2.rectangle(frame, (px, py), (px+CELL-1, py+CELL-1), color, -1)

    # HUD: score, speed, classifier confidence
    cv2.putText(frame, f"Score: {len(snake)-1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Speed: {move_interval:.2f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(frame, f"Model conf: {classifier_conf:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    cv2.imshow("Gesture Snake (Classifier+Angle+Thumb)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        move_interval = max(0.05, move_interval - 0.02)
    elif key == ord('-') or key == ord('_'):
        move_interval = min(1.0, move_interval + 0.02)

cap.release()
cv2.destroyAllWindows()