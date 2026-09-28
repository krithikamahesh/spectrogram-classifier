import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tempfile


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


def load_iq_csv(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    iq = data[:, 0] + 1j * data[:, 1]
    return iq.astype(np.complex64)


def iq_to_spectrogram(iq: np.ndarray, nfft: int = 256, noverlap: int = 128):
    _, _, Sxx = signal.spectrogram(
        iq, nperseg=nfft, noverlap=noverlap,
        return_onesided=False, mode='magnitude'
    )
    Sxx = np.fft.fftshift(Sxx, axes=0)
    return 10 * np.log10(Sxx + 1e-10)


def spectrogram_to_tensor(Sxx_db: np.ndarray, transform):
    """Convert spectrogram array to tensor without saving to disk."""
    # Normalize to 0-255
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)
    Sxx_uint8 = (Sxx_norm * 255).astype(np.uint8)
    
    # Convert to PIL Image (apply colormap)
    plt.figure(figsize=(4, 4))
    plt.imshow(Sxx_db, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to buffer
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        img = Image.open(tmp.name).convert('RGB')
    
    return transform(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True, help="Path to CSV file with I,Q columns")
    parser.add_argument("--model-path", required=True, help="Path to trained model.pt")
    parser.add_argument("--window-size", type=int, default=8192, help="Samples per window")
    parser.add_argument("--hop-size", type=int, default=4096, help="Hop between windows")
    parser.add_argument("--nfft", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load IQ data
    print(f"Loading {args.csv_file}...")
    iq = load_iq_csv(args.csv_file)
    print(f"Loaded {len(iq):,} samples")

    # Sliding window inference
    results = []
    num_windows = (len(iq) - args.window_size) // args.hop_size + 1
    
    print(f"Processing {num_windows} windows...")
    
    with torch.no_grad():
        for i in range(0, len(iq) - args.window_size + 1, args.hop_size):
            window = iq[i:i + args.window_size]
            Sxx_db = iq_to_spectrogram(window, args.nfft)
            x = spectrogram_to_tensor(Sxx_db, transform).unsqueeze(0).to(device)
            
            prob = torch.sigmoid(model(x)).item()
            pred = "drone" if prob > args.threshold else "background"
            
            results.append({
                "start": i,
                "end": i + args.window_size,
                "prob": prob,
                "pred": pred
            })
            
            if len(results) % 50 == 0:
                print(f"  Processed {len(results)}/{num_windows} windows")

    # Summary
    drone_count = sum(1 for r in results if r["pred"] == "drone")
    print(f"\n=== Results ===")
    print(f"Total windows: {len(results)}")
    print(f"Drone detected: {drone_count} ({100*drone_count/len(results):.1f}%)")
    print(f"Background: {len(results) - drone_count} ({100*(len(results)-drone_count)/len(results):.1f}%)")

    # Show drone detections
    drone_windows = [r for r in results if r["pred"] == "drone"]
    if drone_windows:
        print(f"\nDrone detections (top 10 by confidence):")
        for r in sorted(drone_windows, key=lambda x: -x["prob"])[:10]:
            print(f"  Samples {r['start']:,}-{r['end']:,}: {r['prob']:.3f}")


if __name__ == "__main__":
    main()
