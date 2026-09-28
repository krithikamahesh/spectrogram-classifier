import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


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
    parser.add_argument("--images", required=True, help="Path to image or folder of images")
    parser.add_argument("--model-path", required=True, help="Path to trained model.pt")
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

    # Get image paths
    path = Path(args.images)
    if path.is_file():
        image_paths = [path]
    else:
        image_paths = list(path.rglob("*.png")) + list(path.rglob("*.jpg"))

    print(f"Processing {len(image_paths)} images...\n")

    results = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            
            prob = torch.sigmoid(model(x)).item()
            pred = "drone" if prob > args.threshold else "background"
            
            results.append({"path": img_path.name, "prob": prob, "pred": pred})
            print(f"{img_path.name}: {pred} ({prob:.3f})")

    # Summary
    drone_count = sum(1 for r in results if r["pred"] == "drone")
    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"Drone: {drone_count}")
    print(f"Background: {len(results) - drone_count}")


if __name__ == "__main__":
    main()
