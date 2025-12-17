import mediapipe as mp

print("Installed MediaPipe version:", mp.__version__)

try:
    hands = mp.solutions.hands.Hands()
    print("Legacy Hands API is available!")
except AttributeError as e:
    print("Legacy Hands API NOT found:", e)
except Exception as e:
    print("Other error while initializing Hands API:", e)
