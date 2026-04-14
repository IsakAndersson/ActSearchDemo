"""Create box plot, violin plot, and ECDF for document length."""
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
import numpy as np
import statistics


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create box plot, violin plot, and ECDF for document length from metadata.page_count values.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/plots",
        help="Directory where plot PNG files will be written.",
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


def _prepare_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _add_reference_lines(ax, median_pages: float) -> None:
    ax.axvline(50, color="#cc7a00", linestyle="--", linewidth=1.5, label="50 pages")
    ax.axvline(100, color="#c23b22", linestyle="--", linewidth=1.5, label="100 pages")
    ax.axvline(median_pages, color="#2e5d8a", linestyle="-", linewidth=2, label=f"Median ({median_pages:.1f} pages)")


def plot_histogram(page_counts: list[int], output_dir: Path, bins: int = 30) -> Path:
    median_pages = statistics.median(page_counts)
    output_path = output_dir / "document_length_distribution.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(page_counts, bins=bins, color="#a8c686", edgecolor="white", alpha=0.9)
    _add_reference_lines(ax, median_pages)
    ax.set_title("Document Length Distribution")
    ax.set_xlabel("Document length (pages)")
    ax.set_ylabel("Number of documents")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_box(page_counts: list[int], output_dir: Path) -> Path:
    median_pages = statistics.median(page_counts)
    output_path = output_dir / "document_length_boxplot.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(page_counts, vert=False, patch_artist=True, boxprops={"facecolor": "#a8c686"})
    _add_reference_lines(ax, median_pages)
    ax.set_title("Document Length Box Plot")
    ax.set_xlabel("Document length (pages)")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_violin(page_counts: list[int], output_dir: Path) -> Path:
    median_pages = statistics.median(page_counts)
    output_path = output_dir / "document_length_violin.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    violin = ax.violinplot(page_counts, vert=False, showmeans=False, showmedians=False, showextrema=True)
    for body in violin["bodies"]:
        body.set_facecolor("#7aa6c2")
        body.set_edgecolor("#4e748b")
        body.set_alpha(0.9)

    _add_reference_lines(ax, median_pages)
    ax.set_title("Document Length Violin Plot")
    ax.set_xlabel("Document length (pages)")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_ecdf(page_counts: list[int], output_dir: Path) -> Path:
    median_pages = statistics.median(page_counts)
    output_path = output_dir / "document_length_ecdf.png"

    sorted_counts = np.sort(np.array(page_counts, dtype=float))
    y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(sorted_counts, y, where="post", color="#3b6f9c", linewidth=2)
    _add_reference_lines(ax, median_pages)
    ax.set_title("Document Length ECDF")
    ax.set_xlabel("Document length (pages)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    page_counts = collect_page_counts(args.parsed_dir)
    if not page_counts:
        raise ValueError("No valid metadata.page_count values found in parsed documents.")

    output_dir = _prepare_output_dir(args.output_dir)
    outputs = [
        plot_histogram(page_counts, output_dir),
        plot_box(page_counts, output_dir),
        plot_violin(page_counts, output_dir),
        plot_ecdf(page_counts, output_dir),
    ]
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
