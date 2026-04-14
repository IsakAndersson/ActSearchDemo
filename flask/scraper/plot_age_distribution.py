"""Plot a histogram of document age distribution from Docplus metadata."""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import statistics


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of document ages from metadata publish_date values.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default="output/plots/document_age_distribution.png",
        help="Path where the histogram PNG will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins.",
    )
    return parser.parse_args(argv)


def _parse_publish_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        return datetime.strptime(trimmed, "%Y-%m-%d").date()
    except ValueError:
        return None


def collect_document_ages_years(metadata_dir: str) -> list[float]:
    today = date.today()
    ages_years: list[float] = []

    for path in sorted(Path(metadata_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        publish_date = _parse_publish_date(metadata_dict.get("publish_date"))
        if publish_date is None:
            continue

        age_days = (today - publish_date).days
        ages_years.append(age_days / 365.25)

    return ages_years


def plot_age_distribution(ages_years: list[float], output_path: str, bins: int) -> None:
    if not ages_years:
        raise ValueError("No valid publish_date values found in metadata.")

    median_age = statistics.median(ages_years)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    max_age = max(ages_years)
    bins = np.arange(0, (int(max_age) + 2)*2, 1)

    ax.hist(ages_years, bins=bins, edgecolor="white", alpha=0.9)

    ax.set_xticks(np.arange(0, int(max_age) + 1, 1))
    ax.set_xlim(0, int(max_age) + 1)
   
    ax.axvline(2.0, color="#c23b22", linestyle="--", linewidth=2, label="Outdated threshold (2 years)")
    ax.axvline(median_age, color="#1f5f3f", linestyle="-", linewidth=2, label=f"Median ({median_age:.2f} years)")

    ax.set_title("Document Age Distribution")
    ax.set_xlabel("Document age (years)")
    ax.set_ylabel("Number of documents")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    ages_years = collect_document_ages_years(args.metadata_dir)
    plot_age_distribution(ages_years, args.output_path, args.bins)
    print(f"Wrote age distribution histogram to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
