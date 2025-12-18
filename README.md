# PalmStation
A gesture controlled environment where you can play classic games like Snake!

## Overview

A pair of small, beginner-friendly projects that demonstrate **real-time gesture recognition** using MediaPipe Hands and OpenCV.

PalmStation contains two playable demos: a gesture‑controlled Snake game and a gesture‑controlled Rock–Paper–Scissors game. The RPS game includes an improved user experience with hold-to-confirm and result screens. An optional small transfer‑learning pipeline (collect → train → use) allows you to improve gesture accuracy by training a classifier on MediaPipe landmarks using data you can collect on your own!

## Features

* Real-time hand landmark detection using MediaPipe Hands
* Gesture controlled Snake (index or thumb gestures) with smoothing and classifier support
* Gesture controlled Rock–Paper‑Scissors with stable-hold confirmation, confidence score, and result screens
* Optional data collection and training pipeline (collect_landmarks.py → train_classifier.py)
* Easy to run in a Python virtual environment (Windows / macOS / Linux)

## Project structure

```
PalmStation/
├─ .venv/                 # virtual environment (not committed)
├─ models/                # trained classifier + optional mediapipe models
├─ data/                  # personally collected landmark CSVs
├─ snake_gesture_opencv.py
├─ rps_gesture_improved.py
├─ collect_landmarks.py
├─ train_classifier.py
├─ requirements.txt
├─ README.md
```

## Installation

### Prerequisites

* Python 3.11 (recommended) — keep venv per project
* Webcam for real-time detection
* Windows users: install Visual C++ Redistributable if builds fail

### Setup (create & activate venv then install)

```bash
# create venv (one-time)
py -3.11 -m venv .venv
# activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` should include (example):

```
mediapipe==0.10.21
opencv-python
numpy
scikit-learn
pandas
joblib
```

> If you have trouble installing `mediapipe` on your system, check Python version (we recommend 3.11) and platform wheel availability.

## Usage

### Play Gesture Snake (camera)

```bash
python snake_gesture_opencv.py
```

### Play Gesture Rock–Paper–Scissors (camera)

```bash
python rps_gesture_improved.py
```

### Collect landmark data (for transfer learning)

```bash
python collect_landmarks.py
# Keys: r=right, l=left, u=up, d=down, s=save, q=quit
```

### Train gesture classifier

```bash
python train_classifier.py
# produces models/gesture_model.joblib and models/label_encoder.joblib
```

### Run snake (with classifier if available)

`snake_gesture_opencv.py` will automatically load `models/gesture_model.joblib` if present and use it; otherwise it falls back to the angle+thumb heuristics.

## How it works

* The system uses **MediaPipe Hands** to extract 21 hand landmarks per detected hand (x,y normalized coordinates). These landmarks are used directly for heuristic rules (finger extended / folded) or flattened into a feature vector for training a small classifier (RandomForest by default).
* Gesture smoothing uses an **exponential moving average (EMA)** on angles and short voting buffers to avoid flicker and accidental flips.
* The RPS game implements a stable‑hold confirmation (player must hold the same pose for a short duration) and a visible result + countdown flow to prioritize the user experience.

## Model information

* **Pretrained model(s)**: MediaPipe Hands / Hand Landmarker (pretrained hand landmark model). The project uses MediaPipe’s prebuilt models for landmark extraction.
* **Custom classifier (optional)**: RandomForest pipeline trained on flattened MediaPipe landmark X/Y coordinates (42 features). Saved with `joblib`.

## Performance

* **Accuracy:** depends on data collected (no fixed number shipped). When you train your own classifier on 100–300 samples per class you can expect good personal accuracy for your camera/lighting.
* **FPS / Latency:** depends on CPU and camera; MediaPipe Hands runs in real time on modern laptops (tens of FPS). You can tune the scripts to show FPS or reduce processing load.

> Add your measured values here (FPS, inference time, accuracy) after you run experiments on your machine.

## Results

Add screenshots, GIFs or demo videos to the `docs/` or `assets/` folder and link them here once available.

## Challenges & Solutions

* **Mediapipe / wheel compatibility:** fixed by using Python 3.11 and the mediapipe 0.10.21 wheel on Windows.
* **Gesture jitter & misclassification:** solved with EMA smoothing, majority voting, and an optional transfer‑learning classifier trained on user‑collected landmarks.
* **Thumb vs index pointing:** resolved with a simple heuristic that compares normalized distances relative to palm size (thumb & index selection logic).

## Future improvements

* Add a small CNN or lightweight neural network trained on image crops (or landmark sequences) for even better robustness
* Add user calibration flow to personalize thresholds automatically
* Add better visuals, menus, sound effects and a high-score persistence layer
* Export a web demo (WebRTC) using TensorFlow.js + MediaPipe on the browser

## Troubleshooting

* If `mediapipe` import has a yellow squiggle in VS Code but the script runs, reselect the `.venv` interpreter and reload the window. Ensure the terminal is activated with `.\.venv\Scripts\Activate.ps1` on Windows.
* If OpenCV camera doesn’t open, try different camera indices (`cv2.VideoCapture(1)`), close other apps using the camera (Zoom/Discord), and check Windows camera privacy settings.

## Acknowledgments

* MediaPipe Hand Landmarker / Hands (used for landmark extraction and real-time tracking).
* Google Colab notebook I used: [Your Colab Notebook — replace this with your actual URL](LINK_TO_YOUR_COLAB_NOTEBOOK)

### Kaggle datasets you could use for training / augmentation

* Rock Paper Scissors dataset — 2,892 images (example). [https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)
* Rock-Paper-Scissors images collection. [https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* Stone Paper Scissors hand-landmarks dataset (landmark-based). [https://www.kaggle.com/datasets/aryan7781/stone-paper-scissors-hand-landmarks-dataset](https://www.kaggle.com/datasets/aryan7781/stone-paper-scissors-hand-landmarks-dataset)

Thanks to the MediaPipe team, example notebooks and community tutorials that helped build this project.

---

https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf
