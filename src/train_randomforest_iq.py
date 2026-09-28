from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def extract_features(iq: np.ndarray) -> np.ndarray:
    """Extract statistical features from IQ samples."""
    mag = np.abs(iq)
    phase = np.angle(iq)
    power = mag ** 2
    
    # Magnitude features
    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    mag_max = np.max(mag)
    mag_min = np.min(mag)
    
    # Power features
    power_mean = np.mean(power)
    power_std = np.std(power)
    power_max = np.max(power)
    
    # Phase features
    phase_std = np.std(phase)
    phase_diff_std = np.std(np.diff(phase)) if len(phase) > 1 else 0
    
    # I/Q features
    i_mean, q_mean = np.mean(iq.real), np.mean(iq.imag)
    i_std, q_std = np.std(iq.real), np.std(iq.imag)
    iq_corr = np.corrcoef(iq.real, iq.imag)[0, 1] if len(iq) > 1 else 0
    iq_corr = 0 if np.isnan(iq_corr) else iq_corr
    
    # Length as feature (might be discriminative)
    length = len(iq)
    
    # Frequency domain (simple)
    fft_mag = np.abs(np.fft.fft(iq))
    fft_peak = np.max(fft_mag)
    fft_mean = np.mean(fft_mag)
    spectral_centroid = np.sum(np.arange(len(fft_mag)) * fft_mag) / (np.sum(fft_mag) + 1e-8)
    
    return np.array([
        mag_mean, mag_std, mag_max, mag_min,
        power_mean, power_std, power_max,
        phase_std, phase_diff_std,
        i_mean, q_mean, i_std, q_std, iq_corr,
        length,
        fft_peak, fft_mean, spectral_centroid
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    records = discover_files(Path(args.data_root))
    print(f"Files: {len(records)} | Drone: {sum(r['label'] for r in records)} | Background: {sum(1 for r in records if r['label']==0)}")

    # Extract features
    print("Extracting features...")
    X, y = [], []
    for rec in records:
        iq = load_iq(rec["path"])
        X.append(extract_features(iq))
        y.append(rec["label"])
    
    X = np.array(X)
    y = np.array(y)
    
    # Check for NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    print(f"Feature matrix: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=args.seed
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train Random Forest (handles imbalance well)
    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=["background", "drone"], zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    feature_names = [
        "mag_mean", "mag_std", "mag_max", "mag_min",
        "power_mean", "power_std", "power_max",
        "phase_std", "phase_diff_std",
        "i_mean", "q_mean", "i_std", "q_std", "iq_corr",
        "length",
        "fft_peak", "fft_mean", "spectral_centroid"
    ]
    importance = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    print("\nTop features:")
    for name, imp in importance[:5]:
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
