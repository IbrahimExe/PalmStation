# rps_gesture_improved.py
"""
Gesture-based Rock-Paper-Scissors (improved UX)
- Slower, clearer round flow
- Shows heuristic "confidence" for predicted gesture
- Shows AI choice, result, pauses so player can see outcome
- "Get Ready" countdown before next round
- Uses MediaPipe hands landmarks (heuristic detection)
Run with: python rps_gesture_improved.py
"""

import cv2
import mediapipe as mp
import time
import random
import math

# -------- Settings (tweakable) ----------
STABLE_DURATION = 0.9      # seconds candidate must persist to accept (increase for more stability)
RESULT_PAUSE = 2.5         # seconds to show result screen
GET_READY_SECONDS = 3      # countdown seconds before new round
PALM_SCALE_FACTOR = 0.45   # sensitivity to detect extended fingers (0.35 - 0.6)
LOOP_WAIT_MS = 30          # main loop sleep (ms) for cv2.waitKey

# -------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

# -------- helpers ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_size(landmarks):
    return dist(landmarks[0], landmarks[9]) + 1e-8

def finger_extended(landmarks, tip_idx, pip_idx, palm_size_val, threshold_factor=PALM_SCALE_FACTOR):
    d = dist(landmarks[tip_idx], landmarks[pip_idx]) / palm_size_val
    return d > threshold_factor, d

def thumb_extended(landmarks, tip_idx=4, mcp_idx=2, palm_size_val=None, threshold_factor=PALM_SCALE_FACTOR):
    d = dist(landmarks[tip_idx], landmarks[mcp_idx]) / (palm_size_val or 1.0)
    return d > (threshold_factor * 0.9), d

def detect_rps_from_landmarks(landmarks):
    psize = palm_size(landmarks)
    idx_ext, idx_d = finger_extended(landmarks, 8, 6, psize)
    mid_ext, mid_d = finger_extended(landmarks, 12, 10, psize)
    ring_ext, ring_d = finger_extended(landmarks, 16, 14, psize)
    pinky_ext, pinky_d = finger_extended(landmarks, 20, 18, psize)
    thumb_ext, thumb_d = thumb_extended(landmarks, 4, 2, psize)

    ext_count = sum([1 if v else 0 for v in [idx_ext, mid_ext, ring_ext, pinky_ext]])

    # Rules (same as before, but includes thresholds)
    if ext_count >= 3:
        return "paper"
    if idx_ext and mid_ext and (not ring_ext) and (not pinky_ext):
        return "scissors"
    if ext_count == 0:
        return "rock"
    if (not idx_ext) and (not mid_ext) and (not ring_ext) and (not pinky_ext) and thumb_ext:
        return "rock"
    return None

def compute_confidence(landmarks, predicted):
    """Return heuristic confidence 0..1 for predicted class based on normalized distances."""
    psize = palm_size(landmarks)
    idx_ext, idx_d = finger_extended(landmarks, 8, 6, psize)
    mid_ext, mid_d = finger_extended(landmarks, 12, 10, psize)
    ring_ext, ring_d = finger_extended(landmarks, 16, 14, psize)
    pinky_ext, pinky_d = finger_extended(landmarks, 20, 18, psize)
    thumb_ext, thumb_d = thumb_extended(landmarks, 4, 2, psize)

    # clamp helper
    clamp = lambda x: max(0.0, min(1.0, x))

    if predicted == "paper":
        # confidence increases with number of extended fingers and their distances
        score = ( (1 if idx_ext else 0) + (1 if mid_ext else 0) + (1 if ring_ext else 0) + (1 if pinky_ext else 0) ) / 4.0
        # modest weight for thumb being extended
        score = 0.8 * score + 0.2 * (1.0 if thumb_ext else 0.0)
        return clamp(score)
    if predicted == "scissors":
        # index+middle should be clearly extended, ring+pinky folded
        score = 0.0
        score += 0.4 * (idx_d / (PALM_SCALE_FACTOR*2))  # distance normalized roughly
        score += 0.4 * (mid_d / (PALM_SCALE_FACTOR*2))
        score += 0.1 * (1.0 - ring_d/(PALM_SCALE_FACTOR*2))
        score += 0.1 * (1.0 - pinky_d/(PALM_SCALE_FACTOR*2))
        return clamp(score)
    if predicted == "rock":
        # rock: fingers folded -> distances small; if thumb-out treat as rock too
        folded_score = ( (0 if idx_ext else 1) + (0 if mid_ext else 1) + (0 if ring_ext else 1) + (0 if pinky_ext else 1) ) / 4.0
        # if thumb extended and others folded, boost score
        if thumb_ext and folded_score >= 0.75:
            return clamp(0.9)
        return clamp(0.7 * folded_score)
    return 0.0

def resolve(player, comp):
    if player == comp:
        return "Draw"
    beats = {"rock":"scissors", "scissors":"paper", "paper":"rock"}
    return "You Win!" if beats[player] == comp else "You Lose"

# -------- Game state ----------
player_score = 0
computer_score = 0

# round state machine: "idle" (waiting), "result" (showing result), "countdown" (get ready)
state = "idle"
current_candidate = None
candidate_start = 0.0
result_start = 0.0
result_player = None
result_comp = None
result_text = ""
result_conf = 0.0
countdown_start = 0.0

print("Press 'q' to quit")

# -------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = time.time()

    gesture = None
    predicted_conf = 0.0

    results = hands.process(rgb)

    if state == "idle":
        # detect candidate gesture
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            candidate = detect_rps_from_landmarks(hand.landmark)
            if candidate is not None:
                # started a candidate or continuing
                if candidate == current_candidate:
                    # continuing same candidate; check duration
                    if now - candidate_start >= STABLE_DURATION:
                        # accept
                        gesture = candidate
                        predicted_conf = compute_confidence(hand.landmark, gesture)
                        # move to result state
                        result_player = gesture
                        result_comp = random.choice(["rock","paper","scissors"])
                        outcome = resolve(result_player, result_comp)
                        result_text = outcome
                        # update scores
                        if outcome == "You Win!":
                            player_score += 1
                        elif outcome == "You Lose":
                            computer_score += 1
                        result_conf = predicted_conf
                        result_start = now
                        state = "result"
                else:
                    # new candidate begins
                    current_candidate = candidate
                    candidate_start = now
            else:
                # no candidate
                current_candidate = None
                candidate_start = 0.0
        else:
            current_candidate = None
            candidate_start = 0.0

        # HUD while waiting
        status = "Hold a stable gesture: rock / paper / scissors"
        if current_candidate:
            elapsed = now - candidate_start
            pct = min(1.0, max(0.0, elapsed / STABLE_DURATION))
            bar = "[" + "#"*int(pct*12) + "-"*(12-int(pct*12)) + "]"
            status = f"Holding: {current_candidate} {bar} {int(pct*100)}%"

    elif state == "result":
        # show result screen until RESULT_PAUSE elapsed
        # ensure we draw landmarks if hand present
        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        # compute elapsed
        if now - result_start >= RESULT_PAUSE:
            # move to countdown
            countdown_start = time.time()
            state = "countdown"

    elif state == "countdown":
        # draw countdown, when finished go back to idle
        if now - countdown_start >= GET_READY_SECONDS:
            state = "idle"
            current_candidate = None
            candidate_start = 0.0

    # -------- Drawing / HUD --------
    # background overlay for text readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,70), (0,0,0), -1)
    alpha = 0.35
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Score
    cv2.putText(frame, f"Player: {player_score}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"CPU: {computer_score}", (150,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if state == "idle":
        # top status
        cv2.putText(frame, status, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    elif state == "result":
        # Big centered result display
        center_x = w // 2
        # player predicted + confidence
        pred_text = f"You: {result_player.upper()}  ({int(result_conf*100)}%)"
        ai_text = f"AI: {result_comp.upper()}"
        # result line
        cv2.putText(frame, pred_text, (center_x - 220, h//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, ai_text, (center_x - 220, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200,200,200), 2)
        cv2.putText(frame, result_text, (center_x - 220, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0) if result_text=="You Win!" else (0,165,255) if result_text=="Draw" else (0,60,255), 3)

    elif state == "countdown":
        secs_left = int(GET_READY_SECONDS - (now - countdown_start)) + 1
        ready_text = f"Get ready... {secs_left}"
        cv2.putText(frame, ready_text, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,180,255), 2)

    # footer help
    cv2.putText(frame, "Press 'q' to quit", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    cv2.imshow("RPS Gesture (Improved UX)", frame)
    key = cv2.waitKey(LOOP_WAIT_MS) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
