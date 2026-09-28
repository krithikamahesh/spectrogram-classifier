from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_files(root: Path, drone_class: str = "drone"):
    records = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        label = 1 if class_dir.name.lower() == drone_class else 0
        for f in class_dir.rglob("*.np[yz]"):
            records.append({"path": str(f), "label": label})
    return records


def load_iq(path: str) -> np.ndarray:
    p = Path(path)
    arr = np.load(p) if p.suffix == ".npy" else np.load(p)[list(np.load(p).files)[0]]
    if np.iscomplexobj(arr):
        return arr.flatten().astype(np.complex64)
    if arr.ndim == 2:
        arr = arr.T if arr.shape[0] == 2 else arr
        return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    return arr.flatten().astype(np.complex64)


def normalize_iq(iq: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.abs(iq) ** 2)) + 1e-8
    iq = iq / scale
    return np.stack([iq.real, iq.imag], axis=0).astype(np.float32)


def balance_records(records, seed=42):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for r in records:
        by_label[r["label"]].append(r)
    
    max_count = max(len(by_label[0]), len(by_label[1]))
    balanced = []
    for label, recs in by_label.items():
        if len(recs) < max_count:
            recs = recs + rng.choices(recs, k=max_count - len(recs))
        balanced.extend(recs)
    rng.shuffle(balanced)
    return balanced


class IQDataset(Dataset):
    def __init__(self, records, window_size: int):
        self.records = records
        self.window_size = window_size
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if rec["path"] not in self.cache:
            self.cache[rec["path"]] = load_iq(rec["path"])
        iq = self.cache[rec["path"]]
        
        if len(iq) < self.window_size:
            iq = np.pad(iq, (0, self.window_size - len(iq)))
        else:
            iq = iq[:self.window_size]
        
        x = normalize_iq(iq)
        return torch.from_numpy(x), torch.tensor(rec["label"], dtype=torch.float32)


class SmallIQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs_binary")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_files(data_root)
    labels = [r["label"] for r in records]
    print(f"Files: {len(records)} | Drone: {sum(labels)} | Background: {len(labels)-sum(labels)}")

    # Sanity check: print sample stats
    print("\n--- Data sanity check ---")
    for lbl, name in [(0, "background"), (1, "drone")]:
        samples = [r for r in records if r["label"] == lbl][:3]
        for s in samples:
            iq = load_iq(s["path"])
            print(f"{name}: len={len(iq)}, power={np.mean(np.abs(iq)**2):.4f}, path={Path(s['path']).name}")
    print("-------------------------\n")

    train, test = train_test_split(records, test_size=0.2, stratify=labels, random_state=args.seed)
    train, val = train_test_split(train, test_size=0.2, stratify=[r["label"] for r in train], random_state=args.seed)

    train = balance_records(train, args.seed)
    print(f"Train (balanced): {len(train)} | Val: {len(val)} | Test: {len(test)}")

    train_loader = DataLoader(IQDataset(train, args.window_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(IQDataset(val, args.window_size), batch_size=args.batch_size)
    test_loader = DataLoader(IQDataset(test, args.window_size), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallIQCNN().to(device)
    
    # NO pos_weight since we're oversampling
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
    
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            scores = torch.sigmoid(model(x.to(device))).cpu()
            preds = (scores > 0.5).int()
            y_true.extend(y.int().tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(scores.tolist())

    print(classification_report(y_true, y_pred, target_names=["background", "drone"], zero_division=0))
    print(confusion_matrix(y_true, y_pred))
    
    # Check prediction distribution
    print(f"\nPrediction scores - min: {min(y_scores):.3f}, max: {max(y_scores):.3f}, mean: {np.mean(y_scores):.3f}")

    torch.save({"model_state_dict": best_state, "window_size": args.window_size}, output_dir / "model.pt")


if __name__ == "__main__":
    main()
