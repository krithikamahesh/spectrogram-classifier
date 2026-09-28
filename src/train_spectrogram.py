from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_files(root: Path):
    """Binary: drone=1, background=0."""
    records = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        label = 1 if class_dir.name.lower() == "drone" else 0
        for f in class_dir.rglob("*.png"):
            records.append({"path": str(f), "label": label})
    return records


class SpectrogramDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(rec["label"], dtype=torch.float32)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs_spectrogram")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_files(Path(args.data_root))
    labels = [r["label"] for r in records]
    print(f"Files: {len(records)} | Drone: {sum(labels)} | Background: {len(labels)-sum(labels)}")

    train, test = train_test_split(records, test_size=0.2, stratify=labels, random_state=args.seed)
    train, val = train_test_split(train, test_size=0.2, stratify=[r["label"] for r in train], random_state=args.seed)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    train_loader = DataLoader(SpectrogramDataset(train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset(val), batch_size=args.batch_size)
    test_loader = DataLoader(SpectrogramDataset(test), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x.to(device)), y.to(device)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = (torch.sigmoid(model(x.to(device))) > 0.5).int().cpu()
            y_true.extend(y.int().tolist())
            y_pred.extend(preds.tolist())

    print("\n" + classification_report(y_true, y_pred, target_names=["background", "drone"], zero_division=0))
    print(confusion_matrix(y_true, y_pred))

    torch.save({"model_state_dict": best_state}, output_dir / "model.pt")
    print(f"\nModel saved to {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
