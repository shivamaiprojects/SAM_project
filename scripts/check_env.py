"""Module 0 verification script. Run this to confirm the environment is sane."""

import importlib
import sys
from pathlib import Path

CHECKS_PASSED, CHECKS_FAILED = [], []


def check(label: str, condition: bool, detail: str = "") -> None:
    (CHECKS_PASSED if condition else CHECKS_FAILED).append(label)
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ->  {detail}" if detail else ""))


def main() -> None:
    print("=" * 60)
    print("MODULE 0 ENVIRONMENT CHECK")
    print("=" * 60)

    # 1. Python version
    v = sys.version_info
    check("Python >= 3.12", v >= (3, 12), f"found {v.major}.{v.minor}.{v.micro}")

    # 2. Core third-party libs
    for lib in ["numpy", "cv2", "PIL", "matplotlib", "pandas", "yaml", "tqdm"]:
        try:
            importlib.import_module(lib)
            check(f"import {lib}", True)
        except ImportError as e:
            check(f"import {lib}", False, str(e))

    # 3. PyTorch + CUDA
    try:
        import torch

        check("import torch", True, f"version {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        check("CUDA available", cuda_ok)
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"       GPU: {name}  |  VRAM: {vram:.1f} GB")
            # A real tensor op on the GPU — proves the driver stack works
            x = torch.randn(1000, 1000, device="cuda")
            _ = (x @ x).sum().item()
            check("GPU matmul", True)
    except ImportError as e:
        check("import torch", False, str(e))

    # 4. Our own package is installed and importable
    try:
        from sam3_seg.utils.config import load_config, project_root

        check("import sam3_seg", True, f"root = {project_root()}")
        cfg = load_config()
        check("load configs/default.yaml", isinstance(cfg, dict),
            f"{len(cfg)} top-level keys")
    except Exception as e:
        check("import sam3_seg", False, str(e))

    # 5. Directory tree
    from pathlib import Path as P
    root = P(__file__).resolve().parents[1]
    for d in ["src", "configs", "scripts", "tests", "data/raw", "outputs", "external"]:
        check(f"dir exists: {d}", (root / d).is_dir())

    print("=" * 60)
    print(f"{len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed")
    if CHECKS_FAILED:
        print("Failed:", ", ".join(CHECKS_FAILED))
        sys.exit(1)
    print("Module 0 environment OK.")


if __name__ == "__main__":
    main()