import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_CSV = "data/gestures.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)
print("Loaded data shape:", df.shape)

X = df.drop(columns=["label"]).values
y = df["label"].values

# label encode
le = LabelEncoder()
y_enc = le.fit_transform(y)

# simple split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# pipeline (scaler optional for RF, but good practice)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

print("Training classifier...")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# save pipeline and encoder
joblib.dump(pipe, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print("Saved model to", MODEL_PATH)
print("Saved label encoder to", ENCODER_PATH)
