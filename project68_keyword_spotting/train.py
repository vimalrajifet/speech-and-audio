"""
Project 68: Audio Keyword Spotting PyTorch Training Pipeline
Trains a 2D-CNN classifier on Mel-spectrograms of Google Speech Commands keywords.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project68_keyword_spotting.dataset import prepare_keyword_dataset, KEYWORDS
from project68_keyword_spotting.model import KeywordCNN, CHECKPOINT_PATH

CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_keyword_spotter(epochs: int = 25, batch_size: int = 8, lr: float = 0.001):
    print("=== Training Project 68: Audio Keyword Spotting CNN ===")
    X, y, paths = prepare_keyword_dataset()
    print(f"Loaded {len(X)} speech command spectrograms. Vocab size: {len(KEYWORDS)}")

    if len(X) == 0:
        raise RuntimeError("No keyword training samples found.")

    # Augment data with subtle jitter
    augmented_X = []
    augmented_y = []
    for t, label in zip(X, y):
        augmented_X.append(t)
        augmented_y.append(label)
        # Jitter copies
        for _ in range(4):
            jitter = torch.randn_like(t) * 0.05
            augmented_X.append(t + jitter)
            augmented_y.append(label)

    full_X = torch.stack(augmented_X)
    full_y = torch.tensor(augmented_y, dtype=torch.long)
    print(f"Total training dataset (with augmentation): {len(full_X)} spectrograms.")

    dataset = TensorDataset(full_X, full_y)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeywordCNN(num_classes=len(KEYWORDS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_acc = correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "keywords": KEYWORDS,
                "val_accuracy": val_acc,
                "epoch": epoch
            }, CHECKPOINT_PATH)

    print(f"\n[Training Complete] Peak Validation Accuracy: {best_val_acc * 100:.1f}%")
    print(f"Saved PyTorch model checkpoint to: {CHECKPOINT_PATH}")
    print("=== Training Pipeline Finished Successfully ===\n")


if __name__ == "__main__":
    train_keyword_spotter()
