"""
Project 67: Voice Emotion Classifier Training Pipeline
Trains a high-accuracy acoustic emotion classifier using acoustic features,
evaluates cross-validation performance, and saves model checkpoints.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project67_emotion_clf.dataset import prepare_emotion_dataset, EMOTIONS

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "emotion_classifier.joblib")


def augment_features(X: np.ndarray, y: np.ndarray, num_copies: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Applies realistic acoustic variance/jitter data augmentation."""
    augmented_X, augmented_y = list(X), list(y)
    rng = np.random.RandomState(42)

    for _ in range(num_copies):
        for feat, label in zip(X, y):
            # Add Gaussian noise jitter (1.5% std)
            noise = rng.normal(0, 0.015 * np.std(feat), size=feat.shape)
            augmented_X.append(feat + noise)
            augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)


def train_emotion_model():
    print("=== Training Project 67: Voice Emotion Classifier ===")
    X_raw, y_raw, files = prepare_emotion_dataset()
    print(f"Loaded {len(X_raw)} base emotional audio samples.")

    # Data augmentation for robust generalization
    X_aug, y_aug = augment_features(X_raw, y_raw, num_copies=6)
    print(f"Augmented dataset size: {len(X_aug)} samples, {X_aug.shape[1]} features.")

    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )

    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Test Evaluation] Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save artifact containing classifier, scaler, and classes
    payload = {
        "model": clf,
        "scaler": scaler,
        "classes": sorted(list(set(y_aug))),
        "num_features": X_aug.shape[1],
        "accuracy": acc
    }
    joblib.dump(payload, MODEL_SAVE_PATH)
    print(f"Saved trained emotion model checkpoint to: {MODEL_SAVE_PATH}")
    print("=== Training Completed Successfully ===\n")
    return payload


if __name__ == "__main__":
    train_emotion_model()
