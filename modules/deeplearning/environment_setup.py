"""
Environment readiness checks for Temporal Fusion Transformer roadmap (Section 1).

Run:
    python -m modules.deeplearning_environment_setup
to verify dependencies, GPU availability, and requirements entries.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REQUIRED_PACKAGES: List[str] = [
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch-lightning",
    "pytorch-forecasting",
    "tensorboard",
    "pandas",
    "scikit-learn",
]

PACKAGE_IMPORT_MAP: Dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "pytorch-lightning": "pytorch_lightning",
    "pytorch-forecasting": "pytorch_forecasting",
    "tensorboard": "tensorboard",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
}


@dataclass
class PackageStatus:
    name: str
    installed: bool
    version: Optional[str]
    error: Optional[str] = None


def check_package(package: str) -> PackageStatus:
    try:
        module_name = PACKAGE_IMPORT_MAP.get(package, package.replace("-", "_"))
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return PackageStatus(name=package, installed=True, version=str(version))
    except ModuleNotFoundError as exc:
        return PackageStatus(
            name=package, installed=False, version=None, error=str(exc)
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PackageStatus(
            name=package, installed=False, version=None, error=str(exc)
        )


def check_gpu() -> str:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return "PyTorch not installed; cannot check CUDA."

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
        return f"CUDA available ({device_count} device(s)): {', '.join(devices)}"
    return "CUDA not available. Falling back to CPU."


def ensure_requirements(entries: Iterable[str], path: Path) -> List[str]:
    if not path.exists():
        return [f"requirements file {path} not found."]
    content = path.read_text().splitlines()
    missing = [pkg for pkg in entries if pkg not in content]
    return missing


def format_report(statuses: List[PackageStatus], missing_requirements: List[str]) -> str:
    lines = ["=== Dependency Report ==="]
    for status in statuses:
        if status.installed:
            lines.append(f"[OK] {status.name} ({status.version})")
        else:
            lines.append(f"[MISSING] {status.name}: {status.error}")

    if missing_requirements:
        lines.append("\nAdd to requirements.txt:")
        lines.extend(f"  - {pkg}" for pkg in missing_requirements)
    else:
        lines.append("\nrequirements.txt already includes TensorFlow section entries.")

    lines.append(f"\nGPU status: {check_gpu()}")
    return "\n".join(lines)


def main(requirements_path: str = "requirements.txt") -> int:
    statuses = [check_package(pkg) for pkg in REQUIRED_PACKAGES]
    missing_req = ensure_requirements(REQUIRED_PACKAGES, Path(requirements_path))
    print(format_report(statuses, missing_req))
    missing_pkgs = [s for s in statuses if not s.installed]
    return 1 if missing_pkgs else 0


if __name__ == "__main__":
    sys.exit(main())

