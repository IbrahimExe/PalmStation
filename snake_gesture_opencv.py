print("SCRIPT STARTED")

import cv2
import mediapipe as mp
import random
import time
from collections import deque, Counter

# ------------ MediaPipe setup ------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ------------ Game settings ------------
WIDTH, HEIGHT = 640, 480
CELL = 20

GRID_W = WIDTH // CELL
GRID_H = HEIGHT // CELL

snake = [(GRID_W // 2, GRID_H // 2)]
direction = (1, 0)  # grid direction: (dx, dy) in cells
food = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))

move_interval = 0.12  # seconds per move
last_move = time.time()

# ------------ Gesture smoothing ------------
recent_gestures = deque(maxlen=7)  # collect last N gestures
STABLE_THRESHOLD = 3  # require majority of last N to be same to change

def landmark_direction_from_landmarks(landmarks):
    """
    Given a mediapipe hand.landmark list, return 'up','down','left' or 'right'.
    Uses index finger tip (8) vs index pip (5) local vector.
    Landmarks are normalized [0..1] in x,y relative to image.
    """
    tip = landmarks[8]
    base = landmarks[5]
    dx = tip.x - base.x
    dy = tip.y - base.y

    # choose major axis
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"

def most_common_stable(gestures):
    if not gestures:
        return None
    c = Counter(gestures)
    label, count = c.most_common(1)[0]
    if count >= STABLE_THRESHOLD:
        return label
    return None

# ------------ Helper functions ------------
def spawn_food():
    while True:
        p = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
        if p not in snake:
            return p

def grid_wrap(head):
    return (head[0] % GRID_W, head[1] % GRID_H)

# ------------ Main loop ------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Starting Gesture Snake. Press 'q' in the window to quit.")

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # default: keep old direction unless we get a stable new one
    new_gesture = None
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        new_gesture = landmark_direction_from_landmarks(hand.landmark)
        recent_gestures.append(new_gesture)
    else:
        # no hand detected: don't append; keep previous gestures buffer for smoothing
        pass

    stable = most_common_stable(recent_gestures)
    if stable:
        # map stable gesture to direction vector but prevent 180deg reversals
        if stable == "up":
            candidate = (0, -1)
        elif stable == "down":
            candidate = (0, 1)
        elif stable == "left":
            candidate = (-1, 0)
        else:
            candidate = (1, 0)

        # prevent immediate 180° turn
        if (candidate[0] != -direction[0]) or (candidate[1] != -direction[1]) or len(snake) == 1:
            direction = candidate

    # move snake on a timed interval
    now = time.time()
    if now - last_move >= move_interval:
        last_move = now
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        head = grid_wrap(head)
        if head in snake:
            # reset on collision
            snake = [(GRID_W // 2, GRID_H // 2)]
            direction = (1, 0)
            food = spawn_food()
        else:
            snake.insert(0, head)
            if head == food:
                food = spawn_food()
            else:
                snake.pop()

    # draw food and snake
    # convert grid coords to pixel coords
    fx, fy = food[0] * CELL, food[1] * CELL
    cv2.rectangle(frame, (fx, fy), (fx + CELL, fy + CELL), (0, 0, 255), -1)

    for i, (gx, gy) in enumerate(snake):
        px, py = gx * CELL, gy * CELL
        color = (0, 200, 0) if i != 0 else (0, 255, 100)
        cv2.rectangle(frame, (px, py), (px + CELL - 1, py + CELL - 1), color, -1)

    # status text
    cv2.putText(frame, f"Score: {len(snake)-1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Gesture Snake", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False

cap.release()
cv2.destroyAllWindows()
