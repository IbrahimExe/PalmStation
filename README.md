<div align="center">

<img width="473" height="474" alt="PalmStationLogoSmall" src="https://github.com/user-attachments/assets/09026364-cb37-403b-9270-d374ee234426" />

A gesture controlled console environment where you can play classic two classic games!

</div>

## Overview

A small, Python & AI project that demonstrates **real-time gesture recognition** using Google's MediaPipe Hands and OpenCV to open webcams!

> ⚠️ *This is a student project built to further learn data manipulation with NumPy & Pandas, practice and understand the how Artificial Intelligence actually works with neural networks (MLPs & CNNs), and how we can use Transfer Learning to edit models to better suit our final classes and needs.

> The original model was trained in a Google Collab Notebook [Trained Collab Model Attempt](https://github.com/IbrahimExe/Kotlin_ToDoList_App), however, this build uses Google's pre-built MediaPipe model, while also allowing you, the user, to input and save your very own dataset!*

Like the name suggests, this is supposed to be a console that allows you to play *hopefully down the line* a plethora of games.
For now, however, you may try two playable demos: a gesture‑controlled Snake game and a gesture‑controlled Rock–Paper–Scissors game!

The optional small transfer‑learning pipeline (collect → train → use) mentioned above allows you to improve gesture accuracy by training a classifier on MediaPipe landmarks using data you can collect on your own!

## Features

* Real-time hand landmark detection using MediaPipe Hands
* Gesture controlled Snake (index or thumb gestures) with smoothing and classifier support for better playing comfort
* Gesture controlled Rock–Paper‑Scissors with stable-hold confirmation, confidence score, and result screens
* Optional data collection and training pipeline (collect_landmarks.py → train_classifier.py)
* Easy to run in a Python virtual environment (Windows / macOS / Linux)

<div align="center">
  Place the gif/ video here
</div>

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

`requirements.txt` should include:

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

### Command to Play Gesture Snake:

```bash
python snake_gesture.py
```

### Command to Play Rock–Paper–Gesture:

```bash
python rock_paper_gesture.py
```

## Collect your own Data:

### Collect landmark data (for transfer learning)

```bash
python collect_data.py
# Keys: r=right, l=left, u=up, d=down, s=save, q=quit
```

### Train Gesture Classifier

```bash
python train_classifier.py
# produces models/gesture_model.joblib and models/label_encoder.joblib
```

### Running Snake (with classifier if available)

`snake_gesture.py` will automatically load `models/gesture_model.joblib` if present and use it; otherwise it falls back to the angle+thumb heuristics.

## How it works

* The system uses **MediaPipe Hands** to extract **21 hand landmarks** per detected hand (x,y normalized coordinates). These landmarks are used directly for heuristic rules (finger extended / folded) or flattened into a feature vector for training a small classifier ([RandomForest by default](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)).
* Gesture smoothing uses an **exponential moving average (EMA)** on angles and short voting buffers to avoid flicker and accidental flips, this was done to mainly allow players to use thier thumbs to navigate smoother as well.
* The RPS game implements a stable‑hold confirmation (player must hold the same pose for a short duration) and a visible result + countdown flow to prioritize the user experience.

## Model information

* **Pretrained model**: [Google's MediaPipe Hand Landmarker](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf) (pretrained hand landmark model). The project uses MediaPipe’s prebuilt models for landmark extraction.
* **Custom classifier (optional)**: RandomForest pipeline trained on flattened MediaPipe landmark X/Y coordinates (42 features). Saved with `joblib`.

## Performance

* **Accuracy:** Original accuracy using a [Kaggle Rock Paper Scissors dataset:](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* Custom data would obviously depends per user.
* **FPS / Latency:** Depends on CPU and camera; MediaPipe Hands runs in real time on modern laptops (tens of FPS).

> See the Training Models used on my [Collab Notebook:](https://colab.research.google.com/drive/1NEygtlssISnd27ubzyKEnV75lAWFR5F_#scrollTo=UTOUrhrprJnA)

## Results
**3 Training Models Created**:
* 1 - Model 1 (4200811 trainable parameters) & 10 Epochs at a learning rate of 0.001.
* 2 - Model 2 (2417547 trainable parameters) & 15 Epochs at a learning rate of 0.1.
* 3 - Model 3 (4794739 trainable parameters) & 20 Epochs at a learning rate of 0.001

| Experiment | Train Batch Size | Test Batch Size | Parameters  | Num Conv Layers | Padding Used | Learning Rate | Epochs | Final Train Acc | Final Val Acc |
|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 64 | 16 | 4,200,811 | 2 | 0 | 0.001 | 10 | 98.40% | 98.26% |
| Model 2 | 128 | 32 | 2,417,547 | 3 | 1 | 0.1 | 15 | 34.28% | 34.28% |
| Model 3 | 64 | 16 | 4794739 | 3 | 1 | 0.01 | 20 | 98.97% | 99.32% |

## Challenges & Solutions

* Python version compatability - Newer Python versions (13 and >) pose issues when using models, as they are usually built on older versions of Python.
* **Mediapipe / wheel compatibility:** fixed by using Python 3.11 and the mediapipe 0.10.21 wheel on Windows.
* **Gesture jitter & misclassification:** solved with EMA smoothing, majority voting, and an optional transfer‑learning classifier trained on user‑collected landmarks.
* **Thumb vs index pointing:** resolved with a simple heuristic that compares normalized distances relative to palm size (thumb & index selection logic).
* **Datasets Used:** Both datasets used were relativley small, this allowed training to be quick, however, at the cost of accuracy and high amounts of loss.
* Particullarly with the Rock Paper Scissors dataset, as the data had the palms on a greenscreen, but a webcam would usually not have just a palm and greenscreen being recorded.
* This meant it was just better and more efficient to use the pre-trained MediaPipe model.

## Future improvements

* Add better visuals, menus, sound effects and a high-scores and other UI related stuff.
* Add a small CNN or lightweight neural network trained on image crops (or landmark sequences) for even better robustness
* Export a web demo (WebRTC) using TensorFlow.js + MediaPipe on the browser perhaps.

## Troubleshooting

* If `mediapipe` import has a yellow squiggle in VS Code but the script runs, reselect the `.venv` interpreter and reload the window. Ensure the terminal is activated with `.\.venv\Scripts\Activate.ps1` on Windows.
* If OpenCV camera doesn’t open, try different camera indices (`cv2.VideoCapture(1)`), close other apps using the camera (Zoom/Discord), and check Windows camera privacy settings.

## Acknowledgments

* [Google's MediaPipe Hand Landmarker / Hands](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf) (used for landmark extraction and real-time tracking).
* Google Collab notebook I used to try different Models: [Collab Notebook](https://colab.research.google.com/drive/1NEygtlssISnd27ubzyKEnV75lAWFR5F_#scrollTo=UTOUrhrprJnA)

### Kaggle datasets used:

* [Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) — 2,188 images
* [Finger Direction Detection Dataset](https://www.kaggle.com/datasets/ushnish/finger-direction-detection) — 132 images

---
