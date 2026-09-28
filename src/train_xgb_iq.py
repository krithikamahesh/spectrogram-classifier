from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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
    """Extract richer statistical features from IQ samples."""
    n = len(iq)
    mag = np.abs(iq)
    phase = np.angle(iq)
    power = mag ** 2
    
    # Basic stats
    features = [
        np.mean(mag), np.std(mag), np.max(mag), np.min(mag), np.median(mag),
        np.mean(power), np.std(power), np.max(power),
        np.mean(iq.real), np.std(iq.real),
        np.mean(iq.imag), np.std(iq.imag),
    ]
    
    # Phase features
    phase_diff = np.diff(phase) if n > 1 else np.array([0])
    features += [
        np.std(phase),
        np.mean(np.abs(phase_diff)),
        np.std(phase_diff),
    ]
    
    # Correlation
    iq_corr = np.corrcoef(iq.real, iq.imag)[0, 1] if n > 1 else 0
    features.append(0 if np.isnan(iq_corr) else iq_corr)
    
    # Higher order stats
    features += [
        np.mean(mag ** 3),  # skewness proxy
        np.mean(mag ** 4),  # kurtosis proxy
    ]
    
    # Zero crossing rate
    zcr_i = np.sum(np.diff(np.sign(iq.real)) != 0) / max(n - 1, 1)
    zcr_q = np.sum(np.diff(np.sign(iq.imag)) != 0) / max(n - 1, 1)
    features += [zcr_i, zcr_q]
    
    # FFT features
    fft = np.fft.fft(iq)
    fft_mag = np.abs(fft)
    fft_power = fft_mag ** 2
    
    features += [
        np.max(fft_mag),
        np.mean(fft_mag),
        np.std(fft_mag),
        np.argmax(fft_mag) / max(n, 1),  # normalized peak index
    ]
    
    # Spectral features
    freqs = np.arange(n)
    spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-8)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-8))
    spectral_flatness = np.exp(np.mean(np.log(fft_mag + 1e-8))) / (np.mean(fft_mag) + 1e-8)
    
    features += [spectral_centroid / max(n, 1), spectral_spread / max(n, 1), spectral_flatness]
    
    # Energy ratio (low vs high freq)
    mid = n // 2
    if mid > 0:
        low_energy = np.sum(fft_power[:mid])
        high_energy = np.sum(fft_power[mid:])
        energy_ratio = low_energy / (high_energy + 1e-8)
    else:
        energy_ratio = 1.0
    features.append(energy_ratio)
    
    # Length (normalized)
    features.append(n / 150.0)  # normalize by max expected length
    
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    records = discover_files(Path(args.data_root))
    print(f"Files: {len(records)} | Drone: {sum(r['label'] for r in records)} | Background: {sum(1 for r in records if r['label']==0)}")

    print("Extracting features...")
    X, y = [], []
    for rec in records:
        iq = load_iq(rec["path"])
        X.append(extract_features(iq))
        y.append(rec["label"])
    
    X = np.nan_to_num(np.array(X), nan=0, posinf=0, neginf=0)
    y = np.array(y)
    print(f"Feature matrix: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost with tuned params
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\n" + classification_report(y_test, y_pred, target_names=["background", "drone"], zero_division=0))
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nPrediction probs - min: {y_prob.min():.3f}, max: {y_prob.max():.3f}, mean: {y_prob.mean():.3f}")

    # Top features
    feature_names = [
        "mag_mean", "mag_std", "mag_max", "mag_min", "mag_median",
        "power_mean", "power_std", "power_max",
        "i_mean", "i_std", "q_mean", "q_std",
        "phase_std", "phase_diff_mean", "phase_diff_std",
        "iq_corr",
        "mag_cube", "mag_quad",
        "zcr_i", "zcr_q",
        "fft_max", "fft_mean", "fft_std", "fft_peak_idx",
        "spectral_centroid", "spectral_spread", "spectral_flatness",
        "energy_ratio", "length"
    ]
    
    importance = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 features:")
    for name, imp in importance[:10]:
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
