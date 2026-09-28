from pathlib import Path
import argparse
import numpy as np


def load_array(path):
    array = np.load(path, allow_pickle=False)

    if np.iscomplexobj(array):
        return array.reshape(-1).astype(np.complex64)

    if array.ndim == 2 and array.shape[1] == 2:
        return (
            array[:, 0].astype(np.float32)
            + 1j * array[:, 1].astype(np.float32)
        ).astype(np.complex64)

    if array.ndim == 2 and array.shape[0] == 2:
        return (
            array[0].astype(np.float32)
            + 1j * array[1].astype(np.float32)
        ).astype(np.complex64)

    return array.reshape(-1).astype(np.float32).astype(np.complex64)


def describe(path):
    try:
        x = load_array(path)

        power = np.abs(x) ** 2
        magnitude = np.abs(x)

        print(f"{path}")
        print(f"  samples:       {len(x):,}")
        print(f"  dtype:         {x.dtype}")
        print(f"  mean I:        {x.real.mean():.6g}")
        print(f"  mean Q:        {x.imag.mean():.6g}")
        print(f"  mean power:    {power.mean():.6g}")
        print(f"  median power:  {np.median(power):.6g}")
        print(f"  p01 power:     {np.percentile(power, 1):.6g}")
        print(f"  p99 power:     {np.percentile(power, 99):.6g}")
        print(f"  mean magnitude:{magnitude.mean():.6g}")
        print(f"  finite:        {np.isfinite(x).all()}")

    except Exception as exc:
        print(f"FAILED {path}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root)

    class_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    for class_dir in class_dirs:
        files = sorted(class_dir.rglob("*.npy"))

        print(f"CLASS: {class_dir.name}")
        print(f"FILES: {len(files)}")

        for path in files[:args.limit]:
            describe(path)


if __name__ == "__main__":
    main() 
