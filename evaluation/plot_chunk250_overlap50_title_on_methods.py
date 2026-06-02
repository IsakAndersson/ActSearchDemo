from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_ROOT = EVALUATION_DIR / "experiments"
DEFAULT_OUTPUT_PATH = (
    EVALUATION_DIR / "plots" / "chunk250_overlap50_title_on_method_comparison.png"
)
SUMMARY_FILE_NAME = "evaluation_summary.csv"
DEFAULT_METHODS = ("bm25", "hybrid", "hybrid_e5")
METRIC_SPECS = (
    ("average_score", "RR@20"),
    ("average_ndcg@10", "nDCG@10"),
    ("average_recall@10", "Recall@10"),
)
METHOD_LABELS = {
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "hybrid_e5": "Hybrid E5",
}
METHOD_COLORS = {
    "bm25": "#2f7f5f",
    "hybrid": "#3269a8",
    "hybrid_e5": "#b15d2a",
}


def _is_header_row(row: list[str]) -> bool:
    row_set = {cell.strip() for cell in row}
    return "average_score" in row_set and "method" in row_set


def _read_summary_csv(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    headers: list[list[str]] = []
    current_header: list[str] | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            if _is_header_row(raw_row):
                current_header = [cell.strip() for cell in raw_row]
                headers.append(current_header)
                continue
            if current_header is None:
                raise ValueError(f"No header found before data rows in {path}")

            matching_header = current_header if len(raw_row) == len(current_header) else None
            if matching_header is None:
                for candidate in reversed(headers):
                    if len(raw_row) == len(candidate):
                        matching_header = candidate
                        break
            if matching_header is None:
                raise ValueError(f"Malformed summary row in {path}: {len(raw_row)} fields")
            rows.append(dict(zip(matching_header, raw_row)))

    return pd.DataFrame(rows)


def _experiment_from_path(path: Path, experiments_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(experiments_root.resolve())
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


def _load_run_summaries(experiments_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(experiments_root.glob(f"*/results/runs/**/{SUMMARY_FILE_NAME}")):
        df = _read_summary_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path.resolve())
        df["experiment_from_path"] = _experiment_from_path(path, experiments_root)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "experiment" not in combined.columns:
        combined["experiment"] = ""
    combined["experiment"] = combined["experiment"].fillna("").astype(str)
    combined["experiment"] = combined["experiment"].where(
        combined["experiment"].str.strip() != "",
        combined["experiment_from_path"].fillna("").astype(str),
    )
    return combined


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in (
        "average_score",
        "average_ndcg@10",
        "average_recall@10",
        "average_rank",
        "chunk_size",
        "chunk_overlap",
        "top_k",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "evaluated_at_cet" in df.columns:
        df["evaluated_at_cet"] = pd.to_datetime(df["evaluated_at_cet"], errors="coerce")
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].fillna("").astype(str)
    return df


def _filter_rows(
    df: pd.DataFrame,
    methods: tuple[str, ...],
    qrels_source: str,
) -> pd.DataFrame:
    required_columns = {
        "method",
        "chunk_size",
        "chunk_overlap",
        "include_title_chunk",
        *(column for column, _ in METRIC_SPECS),
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {', '.join(sorted(missing))}")

    filtered = df[
        df["method"].isin(methods)
        & (df["chunk_size"] == 250)
        & (df["chunk_overlap"] == 50)
        & df["include_title_chunk"].map(_as_bool)
    ].copy()

    if qrels_source != "all":
        if "qrels_source" not in filtered.columns:
            raise ValueError("Cannot filter by qrels_source because the column is missing")
        filtered = filtered[filtered["qrels_source"] == qrels_source].copy()

    metric_columns = [column for column, _ in METRIC_SPECS]
    filtered = filtered.dropna(subset=metric_columns)
    return filtered.reset_index(drop=True)


def _select_latest_per_method(df: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for method in methods:
        method_df = df[df["method"] == method].copy()
        if method_df.empty:
            continue
        method_df = method_df.sort_values(
            by=["evaluated_at_cet", "source_file"],
            ascending=[False, False],
            na_position="last",
        )
        rows.append(method_df.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        raise ValueError("No matching runs found to plot")

    labels = [METHOD_LABELS.get(method, method) for method in df["method"]]
    colors = [METHOD_COLORS.get(method, "#4b5563") for method in df["method"]]

    fig, axes = plt.subplots(nrows=1, ncols=len(METRIC_SPECS), figsize=(13, 4.8), sharey=False)
    if len(METRIC_SPECS) == 1:
        axes = [axes]

    for ax, (column, title) in zip(axes, METRIC_SPECS):
        values = df[column].astype(float)
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, float(values.max()) * 1.15))
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Chunk 250, overlap 50, title chunk: method comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a bar chart comparing BM25, hybrid, and hybrid_e5 runs for "
            "chunk_size=250, chunk_overlap=50, include_title_chunk=True."
        )
    )
    parser.add_argument(
        "--experiments-root",
        default=str(DEFAULT_EXPERIMENTS_ROOT),
        help="Root directory containing evaluation experiments.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="PNG path for the generated plot.",
    )
    parser.add_argument(
        "--qrels-source",
        default="form_submissions",
        choices=["form_submissions", "google_sheet", "all"],
        help="Only compare runs with this qrels_source. Use 'all' to disable filtering.",
    )
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    output_path = Path(args.output)

    summaries = _coerce_types(_load_run_summaries(experiments_root))
    matching = _filter_rows(
        summaries,
        methods=DEFAULT_METHODS,
        qrels_source=args.qrels_source,
    )
    selected = _select_latest_per_method(matching, methods=DEFAULT_METHODS)
    missing_methods = [method for method in DEFAULT_METHODS if method not in set(selected["method"])]
    if missing_methods:
        print(f"Missing matching runs for: {', '.join(missing_methods)}")
    _plot(selected, output_path)

    print(f"Wrote {output_path}")
    print(
        selected[
            [
                "method",
                "experiment",
                "average_score",
                "average_ndcg@10",
                "average_recall@10",
                "evaluated_at_cet",
                "source_file",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
