"""
Project 70: Audio Event Detection PyTorch Training Pipeline
Trains an AudioEventCNN on Log-Mel Spectrograms of environmental and safety sound events.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project70_audio_event_detection.dataset import prepare_audio_event_dataset, EVENT_CLASSES
from project70_audio_event_detection.model import AudioEventCNN, CHECKPOINT_PATH

CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_audio_event_model(epochs: int = 25, batch_size: int = 8, lr: float = 0.001):
    print("=== Training Project 70: Audio Event Detection CNN ===")
    X, y, paths = prepare_audio_event_dataset()
    print(f"Loaded {len(X)} base event spectrograms across {len(EVENT_CLASSES)} classes.")

    if len(X) == 0:
        raise RuntimeError("No audio event training samples found.")

    # Data augmentation
    aug_X = []
    aug_y = []
    for spec, label in zip(X, y):
        aug_X.append(spec)
        aug_y.append(label)
        # 5 augmented variations per sample
        for _ in range(5):
            noise = torch.randn_like(spec) * 0.04
            aug_X.append(spec + noise)
            aug_y.append(label)

    full_X = torch.stack(aug_X)
    full_y = torch.tensor(aug_y, dtype=torch.long)
    print(f"Total training dataset size (with augmentation): {len(full_X)} spectrograms.")

    dataset = TensorDataset(full_X, full_y)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioEventCNN(num_classes=len(EVENT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_X)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_y)
            preds = torch.argmax(out, dim=1)
            correct += (preds == b_y).sum().item()
            total += len(b_y)

        train_acc = correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(device), b_y.to(device)
                out = model(b_X)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == b_y).sum().item()
                val_total += len(b_y)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": EVENT_CLASSES,
                "val_accuracy": val_acc,
                "epoch": epoch
            }, CHECKPOINT_PATH)

    print(f"\n[Training Complete] Peak Validation Accuracy: {best_val_acc * 100:.1f}%")
    print(f"Saved PyTorch model checkpoint to: {CHECKPOINT_PATH}")
    print("=== Training Pipeline Finished Successfully ===\n")


if __name__ == "__main__":
    train_audio_event_model()
