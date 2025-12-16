print("SCRIPT STARTED")

import cv2
import mediapipe as mp
import numpy as np
import random

# -------------------------
# MediaPipe setup
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

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

# -------------------------
# Helper: gesture → direction
# -------------------------
def get_direction(hand_landmarks):
    tip = hand_landmarks.landmark[8]
    base = hand_landmarks.landmark[5]

    dx = tip.x - base.x
    dy = tip.y - base.y

    if abs(dx) > abs(dy):
        return (CELL, 0) if dx > 0 else (-CELL, 0)
    else:
        return (0, CELL) if dy > 0 else (0, -CELL)

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Gesture detection
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        direction = get_direction(hand)
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Move snake
    head = (snake[0][0] + direction[0],
            snake[0][1] + direction[1])

    head = (head[0] % WIDTH, head[1] % HEIGHT)

    if head in snake:
        snake = [(320, 240)]  # reset on collision
    else:
        snake.insert(0, head)

    if head == food:
        food = (random.randrange(0, WIDTH, CELL),
                random.randrange(0, HEIGHT, CELL))
    else:
        snake.pop()

    # Draw food
    cv2.rectangle(frame, food,
                  (food[0] + CELL, food[1] + CELL),
                  (0, 0, 255), -1)

    # Draw snake
    for s in snake:
        cv2.rectangle(frame, s,
                      (s[0] + CELL, s[1] + CELL),
                      (0, 255, 0), -1)

    cv2.putText(frame, "Gesture Snake (Q to quit)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.imshow("Gesture Snake", frame)

    if cv2.waitKey(80) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
