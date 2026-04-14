"""Plot a histogram of document length distribution from parsed page_count values."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import statistics


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of document length from metadata.page_count values.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default="output/plots/document_length_distribution.png",
        help="Path where the histogram PNG will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins.",
    )
    return parser.parse_args(argv)


def collect_page_counts(parsed_dir: str) -> list[int]:
    page_counts: list[int] = []

    for path in sorted(Path(parsed_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        page_count = metadata_dict.get("page_count")
        if isinstance(page_count, int):
            page_counts.append(page_count)

    return page_counts


def plot_document_length_distribution(page_counts: list[int], output_path: str, bins: int) -> None:
    if not page_counts:
        raise ValueError("No valid metadata.page_count values found in parsed documents.")

    median_pages = statistics.median(page_counts)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(page_counts, bins=bins, color="#a8c686", edgecolor="white", alpha=0.9)
    ax.axvline(50, color="#cc7a00", linestyle="--", linewidth=2, label="50 pages")
    ax.axvline(100, color="#c23b22", linestyle="--", linewidth=2, label="100 pages")
    ax.axvline(median_pages, color="#2e5d8a", linestyle="-", linewidth=2, label=f"Median ({median_pages:.1f} pages)")

    ax.set_title("Document Length Distribution")
    ax.set_xlabel("Document length (pages)")
    ax.set_ylabel("Number of documents")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    page_counts = collect_page_counts(args.parsed_dir)
    plot_document_length_distribution(page_counts, args.output_path, args.bins)
    print(f"Wrote document length histogram to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
