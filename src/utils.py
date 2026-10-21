"""Seeds, devices, JSON helpers, and environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEDULER_ORDER = (
    "constant",
    "step",
    "cosine",
    "linear",
    "plateau",
    "ablr",
    "ablr_recovery",
    "ablr_hybrid",
)

SCHEDULER_LABELS = {
    "constant": "Constant",
    "step": "Step decay",
    "cosine": "Cosine annealing",
    "linear": "Linear to zero",
    "plateau": "ReduceLROnPlateau",
    "ablr": "ABLR",
    "ablr_recovery": "ABLR + recovery",
    "ablr_hybrid": "ABLR + linear envelope",
}


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch.

    CUDA convolution algorithms can still be nondeterministic. When
    ``deterministic`` is true, cuDNN benchmarking is disabled and PyTorch is
    asked to prefer deterministic kernels. That request is best-effort:
    ``warn_only=True`` avoids crashing on kernels that have no deterministic
    implementation. DataLoader worker ordering is another residual source and
    is documented in the README.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def select_device(preference: str | None = None) -> torch.device:
    """Select CUDA when it is available, otherwise CPU.

    ``preference`` may be ``"auto"``, ``"cpu"``, ``"cuda"``, or a device
    string such as ``"cuda:0"``. A CUDA request on a machine without CUDA
    falls back to CPU and does not raise.
    """
    choice = (preference or "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(choice if choice != "cuda" else "cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_builtin(value: Any) -> Any:
    """Convert NumPy and PyTorch scalars into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict[str, Any]:
    """Record software versions and the visible accelerator."""
    gpu_name = None
    gpu_memory_gb = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = round(props.total_memory / (1024**3), 2)
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "PyYAML",
        "pytest",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "packages": {name: package_version(name) for name in packages},
    }


def config_fingerprint(payload: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(to_builtin(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
