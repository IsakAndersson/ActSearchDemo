"""Plot a histogram of document length distribution from parsed page_count values."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional
from collections import Counter

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

    # --- Ursprungligt diagram (oförändrat) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(page_counts, bins=bins, color="#a8c686", edgecolor="white", alpha=0.9)
    #ax.axvline(median_pages, color="#2e5d8a", linestyle="-", linewidth=2, label=f"Median ({median_pages:.1f} pages)")
    ax.set_title("Document Length Distribution")
    ax.set_xlabel("Document length (pages)")
    ax.set_ylabel("Number of documents")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

    # --- Subplot-version ---
    split = 30
    short = [p for p in page_counts if p <= split]
    long  = [p for p in page_counts if p > split]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(short, bins=range(0, split + 2), color="#a8c686", edgecolor="white", alpha=0.9)
    ax1.axvline(median_pages, color="#2e5d8a", linestyle="-", linewidth=2, label=f"Median ({median_pages:.1f} pages)")
    ax1.set_title(f"Short documents (0–{split} pages)")
    ax1.set_xlabel("Document length (pages)")
    ax1.set_ylabel("Number of documents")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    ax2.hist(long, bins=bins, color="#a8c686", edgecolor="white", alpha=0.9)
    ax2.set_title(f"Longer documents (>{split} pages)")
    ax2.set_xlabel("Document length (pages)")
    ax2.set_ylabel("Number of documents")
    ax2.grid(axis="y", alpha=0.25)

    fig2.suptitle("Document Length Distribution", fontsize=14)
    fig2.tight_layout()
    subplot_output = output.with_stem(output.stem + "_split")
    fig2.savefig(subplot_output, dpi=160)
    plt.close(fig2)
    
    # --- Separat plot: samma som första subplotten, avkortad vid 30 med diskreta staplar ---
    fig_short, ax_short = plt.subplots(figsize=(10, 6))

    from collections import Counter
    import matplotlib.patches as mpatches

    counts = Counter(short)
    x = list(range(0, split + 1))
    y = [counts.get(i, 0) for i in x]

    median_int = int(round(median_pages))
    colors = ["#788d60" if i == median_int else "#a8c686" for i in x]

    ax_short.bar(
        x,
        y,
        width=0.8,
        color=colors,
        edgecolor="white",
    )

    ax_short.set_title("Document Length Distribution (0-30 pages)")
    ax_short.set_xlabel("Document length (pages)")
    ax_short.set_ylabel("Number of documents")
    ax_short.set_xlim(-0.5, split + 0.5)
    ax_short.set_xticks(range(0, split + 1, 5))
    ax_short.grid(axis="y", alpha=0.25)

    median_patch = mpatches.Patch(
        color="#788d60",
        label=f"Median ({median_int} pages)",
    )
    ax_short.legend(handles=[median_patch])

    fig_short.tight_layout()
    short_output = output.with_stem(output.stem + "_0_30")
    fig_short.savefig(short_output, dpi=160)
    plt.close(fig_short)

    # --- Bruten y-axel ---
    from matplotlib.ticker import MaxNLocator

    fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 3]})
    fig3.subplots_adjust(hspace=0.05)

    for ax in (ax_top, ax_bot):
        ax.hist(page_counts, bins=bins*4, color="#a8c686", edgecolor="white", alpha=0.9)
        ax.axvline(median_pages, color="#2e5d8a", linestyle="-", linewidth=1,
                   label=f"Median ({median_pages:.1f} pages)")
        ax.grid(axis="y", alpha=0.25)

    # Justera gränserna efter dina faktiska värden
    ax_top.set_ylim(6000, 6100)
    ax_bot.set_ylim(0, 2000)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)

    d = 0.015
    kwargs = dict(transform=fig3.transFigure, color="k", clip_on=False, linewidth=1)
    left  = ax_top.get_position().x0
    right = ax_top.get_position().x1
    for y in (ax_top.get_position().y0, ax_bot.get_position().y1):
        fig3.add_artist(plt.Line2D([left  - d, left  + d], [y - d, y + d], **kwargs))
        fig3.add_artist(plt.Line2D([right - d, right + d], [y - d, y + d], **kwargs))

    ax_top.set_title("Document Length Distribution")
    ax_bot.set_xlabel("Document length (pages)")
    fig3.text(0.04, 0.5, "Number of documents", va="center", rotation="vertical")
    ax_top.legend()
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))

    fig3.tight_layout()
    broken_output = output.with_stem(output.stem + "_broken")
    fig3.savefig(broken_output, dpi=160)
    plt.close(fig3)
    
    # --- Vanlig skala, fler bins ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(page_counts, bins=bins * 10, color="#a8c686", edgecolor="white", alpha=0.9)
    ax4.axvline(median_pages, color="#2e5d8a", linestyle="-", linewidth=2, label=f"Median ({median_pages:.1f} pages)")
    ax4.set_title("Document Length Distribution")
    ax4.set_xlabel("Document length (pages)")
    ax4.set_ylabel("Number of documents")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.25)
    fig4.tight_layout()
    finebins_output = output.with_stem(output.stem + "_finebins")
    fig4.savefig(finebins_output, dpi=160)
    plt.close(fig4)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    page_counts = collect_page_counts(args.parsed_dir)
    plot_document_length_distribution(page_counts, args.output_path, args.bins)
    print(f"Wrote document length histogram to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
