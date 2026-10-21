"""Run one campaign from a YAML config.

Set the cuBLAS workspace before PyTorch is imported so deterministic mode has
a chance to apply. On Windows, run this script from the project root.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import apply_overrides, load_config, run_campaign  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Adaptive Bounce LR experiments")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--schedulers", nargs="+", default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda")
    parser.add_argument("--force", action="store_true", help="Overwrite existing runs")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config = apply_overrides(
        config,
        epochs=args.epochs,
        seeds=args.seeds,
        schedulers=args.schedulers,
        device=args.device,
    )
    run_campaign(config, force=args.force, plot=not args.no_plot)


if __name__ == "__main__":
    main()
