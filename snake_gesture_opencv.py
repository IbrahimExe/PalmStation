# snake_gesture_opencv.py
print("SCRIPT STARTED")

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# -------------------------
# MediaPipe setup
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)

# -------------------------
# Game settings
# -------------------------
WIDTH, HEIGHT = 640, 480
CELL = 20

snake = [(320, 240)]
direction = (CELL, 0)
food = (random.randrange(0, WIDTH, CELL),
        random.randrange(0, HEIGHT, CELL))

move_interval = 0.20   # Seconds per movement step [smaller = faster moving snake]
last_move = time.time()

# -------------------------
# Helpers: smoothing / geometry
# -------------------------
class AngleEMA:
    """Exponential moving average for smoothing angles"""
    def __init__(self, alpha=0.3): # Smoothing Factor [0 ~ 1] - If twitchy, lower to ~0.2
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value
        if self.value is None:
            self.value = new_value
        else:
            # wrap-around safe update for angles
            a = math.radians(self.value)
            b = math.radians(new_value)
            x = math.cos(b) * self.alpha + math.cos(a) * (1 - self.alpha)
            y = math.sin(b) * self.alpha + math.sin(a) * (1 - self.alpha)
            self.value = math.degrees(math.atan2(y, x))
        return self.value

angle_filter = AngleEMA(alpha=0.35)

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_size(landmarks):
    # approximate palm size using wrist (0) to middle_mcp (9)
    return dist(landmarks[0], landmarks[9])

def finger_angle_from( tip_landmark, base_landmark ):
    # returns angle in degrees, where 0 = right, positive = up
    dx = tip_landmark.x - base_landmark.x
    dy = base_landmark.y - tip_landmark.y  # invert y for screen coords
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
        p = (random.randrange(0, WIDTH, CELL), random.randrange(0, HEIGHT, CELL))
        if p not in snake:
            return p

# -------------------------
# Gesture decision logic
# -------------------------
def choose_pointing_finger(landmarks):
    """
    Decide whether user is pointing with index or thumb.
    Returns tuple (finger, angle) where finger is "index" or "thumb" or None.
    Angle is the computed angle (degrees) for the chosen finger, or None.
    Logic:
      - compute palm size to normalize distances
      - if index is extended -> choose index
      - elif thumb is extended and index folded -> choose thumb (useful for thumbs-out)
      - else -> None (no clear pointing)
    """
    # landmarks: list-like with attributes x,y
    psize = palm_size(landmarks)
    if psize == 0:
        return None, None

    # distances normalized by palm size
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
                             # [0.35 ~ 0.55] Lower = Easier to trigger

    # determine states
    index_extended = (index_tip_to_mcp >= INDEX_EXTENDED)
    index_folded = (index_tip_to_mcp <= INDEX_FOLDED)
    thumb_extended = (thumb_tip_to_mcp >= THUMB_EXTENDED)

    # prefer index if clearly extended
    if index_extended:
        angle = finger_angle_from(index_tip, index_mcp)
        return "index", angle

    # if index folded and thumb extended -> thumbs-out / horizontal thumb
    if index_folded and thumb_extended:
        angle = finger_angle_from(thumb_tip, thumb_mcp)
        return "thumb", angle

    # if thumb extended and index not strongly extended, accept thumb
    if (thumb_extended and not index_extended):
        angle = finger_angle_from(thumb_tip, thumb_mcp)
        return "thumb", angle

    return None, None

# -------------------------
# Main loop
# -------------------------
print("Starting Gesture Snake. Press 'q' in the window to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    chosen_angle = None
    chosen_finger = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        chosen_finger, raw_angle = choose_pointing_finger(hand.landmark)
        if raw_angle is not None:
            chosen_angle = angle_filter.update(raw_angle)
        # draw small label to help testing:
        if chosen_finger:
            cv2.putText(frame, f"Using: {chosen_finger}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    # compute gesture from smoothed angle if available
    gesture = None
    if chosen_angle is not None:
        gesture = angle_to_direction(chosen_angle)

    # map gesture to candidate direction (in pixels)
    if gesture == "up":
        candidate = (0, -CELL)
    elif gesture == "down":
        candidate = (0, CELL)
    elif gesture == "left":
        candidate = (-CELL, 0)
    elif gesture == "right":
        candidate = (CELL, 0)
    else:
        candidate = None

    # apply candidate if valid and not 180 reversal
    if candidate:
        if (candidate[0] != -direction[0]) or (candidate[1] != -direction[1]) or len(snake) == 1:
            direction = candidate

    # move snake on timer
    now = time.time()
    if now - last_move >= move_interval:
        last_move = now
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        head = (head[0] % WIDTH, head[1] % HEIGHT)

        if head in snake:
            # reset
            snake = [(WIDTH//2, HEIGHT//2)]
            direction = (CELL, 0)
            food = spawn_food()
        else:
            snake.insert(0, head)
            if head == food:
                food = spawn_food()
            else:
                snake.pop()

    # Draw food and snake
    fx, fy = food
    cv2.rectangle(frame, (fx, fy), (fx + CELL, fy + CELL), (0, 0, 255), -1)

    for i, (sx, sy) in enumerate(snake):
        color = (0, 200, 0) if i != 0 else (0, 255, 100)
        cv2.rectangle(frame, (sx, sy), (sx + CELL - 1, sy + CELL - 1), color, -1)

    cv2.putText(frame, "Gesture Snake (Q to quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Gesture Snake", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()