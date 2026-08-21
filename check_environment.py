from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata
from pathlib import Path


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "mss": "mss",
    "Pillow": "PIL",
    "rapidocr": "rapidocr",
    "onnxruntime": "onnxruntime",
    "PyDirectInput": "pydirectinput",
}

PROJECT_FILES = (
    Path("detect_game_yolo.py"),
    Path("capture_game_window.py"),
    Path("maplestory_02.pt"),
    Path("assests/scroll/10.png"),
    Path("assests/scroll/60.png"),
    Path("assests/scroll/100.png"),
    Path("C:/Windows/Fonts/msjhbd.ttc"),
)


def package_version(distribution_name: str, module: object) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    print("=== Python environment ===")
    print(f"Python:     {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Platform:   {platform.platform()}")

    in_virtual_environment = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Virtualenv: {'YES' if in_virtual_environment else 'NO'}")
    print(f"Prefix:     {sys.prefix}")

    print("\n=== Required packages ===")
    imported_modules: dict[str, object] = {}
    missing_or_broken: list[str] = []

    for distribution_name, import_name in PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            imported_modules[import_name] = module
            version = package_version(distribution_name, module)
            print(f"[OK]   {distribution_name:<16} {version}")
        except Exception as error:
            missing_or_broken.append(distribution_name)
            print(f"[FAIL] {distribution_name:<16} {type(error).__name__}: {error}")

    print("\n=== PyTorch GPU ===")
    torch = imported_modules.get("torch")
    cuda_ready = False

    if torch is None:
        print("[FAIL] PyTorch could not be imported, so CUDA cannot be checked.")
    else:
        cuda_ready = bool(torch.cuda.is_available())
        print(f"PyTorch build CUDA: {torch.version.cuda}")
        print(f"CUDA available:     {cuda_ready}")
        print(f"cuDNN available:    {torch.backends.cudnn.is_available()}")
        print(f"cuDNN version:      {torch.backends.cudnn.version()}")

        if cuda_ready:
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            memory_gib = properties.total_memory / (1024**3)
            capability = torch.cuda.get_device_capability(device_index)
            print(f"GPU:                {properties.name}")
            print(f"VRAM:               {memory_gib:.1f} GiB")
            print(f"Compute capability: {capability[0]}.{capability[1]}")

            try:
                value = torch.ones(1, device="cuda") * 2
                torch.cuda.synchronize()
                print(f"CUDA tensor test:   PASS ({value.item():.1f})")
            except Exception as error:
                cuda_ready = False
                print(f"CUDA tensor test:   FAIL ({type(error).__name__}: {error})")
        else:
            print("GPU:                Not available to PyTorch")

    print("\n=== Required files ===")
    missing_files = []
    for path in PROJECT_FILES:
        if path.is_file():
            print(f"[OK]   {path}")
        else:
            missing_files.append(path)
            print(f"[FAIL] {path}")

    print("\n=== Result ===")
    if missing_or_broken:
        print("FAILED: missing or broken packages:")
        for package in missing_or_broken:
            print(f"  - {package}")
        return 1

    if not cuda_ready:
        print("FAILED: packages are installed, but PyTorch cannot use CUDA.")
        return 1

    if missing_files:
        print("FAILED: required project files or fonts are missing.")
        return 1

    if not in_virtual_environment:
        print("PASSED: CUDA is ready, but Python is not running inside a virtualenv.")
        return 0

    print("PASSED: virtualenv, packages, CUDA, cuDNN, and GPU are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
