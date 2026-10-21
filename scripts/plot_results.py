"""Plot a finished campaign. This script does not train and does not invent runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import load_completed_summaries, write_aggregate  # noqa: E402
from src.evaluate import format_results_markdown  # noqa: E402
from src.plotting import plot_campaign  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completed scheduler runs")
    parser.add_argument("--config-name", required=True, help="Campaign name, such as quick or benchmark")
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    results = ROOT / args.results
    figures = ROOT / args.figures
    summaries = load_completed_summaries(results, args.config_name)
    aggregate = write_aggregate(results, args.config_name, summaries)
    print(format_results_markdown(aggregate, config_name=args.config_name))
    written = plot_campaign(results, figures, args.config_name)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
