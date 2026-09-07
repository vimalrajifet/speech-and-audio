"""
Project 64: Standalone Model Training Script
Trains SoundCNN on 10 UrbanSound8K classes using PyTorch and saves checkpoints/sound_cnn.pth.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project64_sound_clf.model import SoundCNN, CLASSES
from project64_sound_clf.dataset import UrbanSoundSyntheticDataset, prepare_sample_files

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sound_cnn.pth")

def train_model(epochs: int = 15, batch_size: int = 16, lr: float = 0.001):
    print("=== Training Project 64: UrbanSound8K SoundCNN ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare sample demo audio files
    prepare_sample_files()

    # Load dataset
    full_dataset = UrbanSoundSyntheticDataset(samples_per_class=35)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SoundCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"\n[OK] Training completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved model checkpoint to: {CHECKPOINT_PATH}")
    print("===================================================\n")

if __name__ == "__main__":
    train_model(epochs=12)
