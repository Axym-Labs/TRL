import argparse
from pathlib import Path

import torchvision


def try_dataset(ds_name: str, root: str):
    ds_cls = torchvision.datasets.CIFAR10 if ds_name == "cifar10" else torchvision.datasets.CIFAR100
    print(f"\n=== {ds_name.upper()} ===")
    print(f"data_path: {Path(root).resolve()}")

    try:
        _ = ds_cls(root=root, train=True, download=True)
        _ = ds_cls(root=root, train=False, download=True)
        print("download=True: OK")
    except Exception as exc:
        print(f"download=True: FAIL -> {type(exc).__name__}: {exc}")

    try:
        train_ds = ds_cls(root=root, train=True, download=False)
        val_ds = ds_cls(root=root, train=False, download=False)
        print(f"local load (download=False): OK -> train={len(train_ds)}, val={len(val_ds)}")
    except Exception as exc:
        print(f"local load (download=False): FAIL -> {type(exc).__name__}: {exc}")


def try_yesno(root: str):
    print("\n=== YESNO ===")
    root_path = Path(root).resolve()
    print(f"data_path: {root_path}")
    try:
        from torchaudio.datasets import YESNO
    except Exception as exc:
        print(f"torchaudio YESNO import: FAIL -> {type(exc).__name__}: {exc}")
        return

    # Ensure root exists to avoid partial-file path errors on some systems.
    Path(root).mkdir(parents=True, exist_ok=True)

    try:
        ds = YESNO(root=root, download=True)
        print(f"download=True: OK -> n={len(ds)}")
    except Exception as exc:
        print(f"download=True: FAIL -> {type(exc).__name__}: {exc}")

    try:
        ds = YESNO(root=root, download=False)
        print(f"local load (download=False): OK -> n={len(ds)}")
    except Exception as exc:
        print(f"local load (download=False): FAIL -> {type(exc).__name__}: {exc}")

def try_speechcommands(root: str):
    print("\n=== SPEECHCOMMANDS ===")
    root_path = Path(root).resolve()
    print(f"data_path: {root_path}")
    try:
        from torchaudio.datasets import SPEECHCOMMANDS
    except Exception as exc:
        print(f"torchaudio SPEECHCOMMANDS import: FAIL -> {type(exc).__name__}: {exc}")
        return

    Path(root).mkdir(parents=True, exist_ok=True)
    try:
        train_ds = SPEECHCOMMANDS(root=root, subset="training", download=True)
        val_ds = SPEECHCOMMANDS(root=root, subset="validation", download=True)
        print(f"download=True: OK -> train={len(train_ds)}, val={len(val_ds)}")
    except Exception as exc:
        print(f"download=True: FAIL -> {type(exc).__name__}: {exc}")

    try:
        train_ds = SPEECHCOMMANDS(root=root, subset="training", download=False)
        val_ds = SPEECHCOMMANDS(root=root, subset="validation", download=False)
        print(f"local load (download=False): OK -> train={len(train_ds)}, val={len(val_ds)}")
    except Exception as exc:
        print(f"local load (download=False): FAIL -> {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data", help="Dataset root directory")
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "yesno", "speechcommands", "both", "all"],
        default="both",
        help="Which dataset to test",
    )
    args = parser.parse_args()

    if args.dataset in ("cifar10", "both", "all"):
        try_dataset("cifar10", args.data_path)
    if args.dataset in ("cifar100", "both", "all"):
        try_dataset("cifar100", args.data_path)
    if args.dataset in ("yesno", "all"):
        try_yesno(str(Path(args.data_path) / "yesno"))
    if args.dataset in ("speechcommands", "all"):
        try_speechcommands(str(Path(args.data_path) / "speechcommands"))


if __name__ == "__main__":
    main()
