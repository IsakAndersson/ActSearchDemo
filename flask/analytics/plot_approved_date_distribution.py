"""Plot a histogram of document age distribution based on approved dates in parsed text."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional
import matplotlib.patches as mpatches

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import statistics

APPROVED_DATE_REGEX = re.compile(r"Godkänt den:\s*(?:\n\s*)*(\d{4}-\d{2}-\d{2})")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of document ages from approved dates extracted from parsed text.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default="output/plots/approved_date_distribution.png",
        help="Path where the histogram PNG will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--head-chars",
        type=int,
        default=500,
        help="How many characters from the start of text to inspect for approved date.",
    )
    return parser.parse_args(argv)


def _extract_approved_date(text: object, head_chars: int) -> date | None:
    if not isinstance(text, str):
        return None
    match = APPROVED_DATE_REGEX.search(text[:head_chars])
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def collect_document_ages_years(parsed_dir: str, head_chars: int) -> list[float]:
    today = date.today()
    ages_years: list[float] = []

    for path in sorted(Path(parsed_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        approved_date = _extract_approved_date(payload.get("text"), head_chars=head_chars)
        if approved_date is None:
            continue

        age_days = (today - approved_date).days
        ages_years.append(age_days / 365.25)

    return ages_years


def plot_approved_date_distribution(ages_years: list[float], output_path: str, bins: int) -> None:
    if not ages_years:
        raise ValueError("No valid approved dates found in parsed documents.")

    import numpy as np
    import statistics
    from pathlib import Path
    import matplotlib.pyplot as plt

    median_age = statistics.median(ages_years)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Beräkna histogram utan att plotta direkt
    counts, bin_edges = np.histogram(ages_years, bins=bins)

    # Rita varje bin manuellt
    for i in range(len(counts)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        width = right - left

        # Kolla om medianen ligger i denna bin
        if left <= median_age < right:
            color = "#9f8765"  # mörkare blå för median-bin
        else:
            color = "#d7b98e"

        ax.bar(left, counts[i], width=width, align="edge",
               color=color, edgecolor="white", alpha=0.9)

    # Outdated threshold
    outdated_line = ax.axvline(2.0, color="#c23b22", linestyle="--", linewidth=2,
               label="Outdated threshold (2 years)")

    ax.set_title("Approved Date Age Distribution")
    ax.set_xlabel("Document age from approved date (years)")
    ax.set_ylabel("Number of documents")
    # Legend
    median_patch = mpatches.Patch(
        color="#9f8765",
        label=f"Median ({median_age:.2f} years)"
    )
    ax.legend(handles=[median_patch, outdated_line])
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    ages_years = collect_document_ages_years(args.parsed_dir, args.head_chars)
    plot_approved_date_distribution(ages_years, args.output_path, args.bins)
    print(f"Wrote approved-date age distribution histogram to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
