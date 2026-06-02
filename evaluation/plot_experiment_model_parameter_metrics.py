from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_ROOT = EVALUATION_DIR / "experiments"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots" / "experiment_model_parameter_metrics"
SUMMARY_FILE_NAME = "evaluation_summary.csv"
METRIC_SPECS = [
    ("average_score", "RR@20", "rr20"),
    ("average_ndcg@10", "nDCG@10", "ndcg10"),
    ("average_recall@10", "Recall@10", "recall10"),
]
MODEL_COLORS = [
    "#2f7f5f",
    "#3269a8",
    "#b15d2a",
    "#7b4fa3",
    "#c44e52",
    "#4b5563",
    "#d89c2b",
    "#008b8b",
]


def _is_header_row(row: list[str]) -> bool:
    row_set = {cell.strip() for cell in row}
    return "average_score" in row_set and (
        "evaluated_at_cet" in row_set or "method" in row_set
    )


def _read_appended_summary_csv(path: Path) -> pd.DataFrame:
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


def _experiment_name_from_summary_path(path: Path, experiments_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(experiments_root.resolve())
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


def _load_experiment_summaries(experiments_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(experiments_root.glob(f"*/results/aggregate/{SUMMARY_FILE_NAME}")):
        df = _read_appended_summary_csv(path)
        if df.empty:
            continue
        df["experiment_from_path"] = _experiment_name_from_summary_path(path, experiments_root)
        df["source_file"] = str(path.resolve())
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


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "evaluated_at_cet" in df.columns:
        df["evaluated_at_cet"] = pd.to_datetime(df["evaluated_at_cet"], errors="coerce")
    for column in [
        "average_score",
        "average_ndcg@10",
        "average_recall@10",
        "chunk_size",
        "chunk_overlap",
        "top_k",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].fillna("").astype(str)
    return df


def _short_model_name(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text.split("/")[-1]


def _bool_text(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"true", "1", "yes"}:
        return "title_on"
    if text.lower() in {"false", "0", "no"}:
        return "title_off"
    return text or "title_unknown"


def _build_parameter_label(row: pd.Series) -> str:
    parts: list[str] = []
    if pd.notna(row.get("chunk_size", pd.NA)):
        parts.append(f"chunk {int(row['chunk_size'])}")
    if pd.notna(row.get("chunk_overlap", pd.NA)):
        parts.append(f"overlap {int(row['chunk_overlap'])}")
    if str(row.get("include_title_chunk", "")).strip():
        parts.append(_bool_text(row.get("include_title_chunk")))
    text_source = str(row.get("text_source", "")).strip()
    if text_source:
        parts.append(text_source)
    return ", ".join(parts) if parts else "default parameters"


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = df.get("method", "").fillna("").astype(str)
    df["model_label"] = df.get("model_name", "").fillna("").astype(str).map(_short_model_name)
    df["model_label"] = df["model_label"].where(df["model_label"].str.strip() != "", df["method"])
    df["report_model_label"] = df.apply(_build_report_model_label, axis=1)
    df["parameter_label"] = df.apply(_build_parameter_label, axis=1)
    df["comparison_label"] = (
        df["method"].astype(str)
        + " | "
        + df["model_label"].astype(str)
        + " | "
        + df["parameter_label"].astype(str)
    )
    df["report_label"] = df.apply(_build_report_label, axis=1)
    return df


def _build_report_model_label(row: pd.Series) -> str:
    method = str(row.get("method", "")).strip()
    model = _short_model_name(row.get("model_name", ""))
    if not model or model == method:
        return method or "unknown"
    return f"{method} ({model})"


def _build_report_label(row: pd.Series) -> str:
    model = str(row.get("report_model_label", "")).strip() or _build_report_model_label(row)
    parts = [model]
    if pd.notna(row.get("chunk_size", pd.NA)):
        parts.append(f"chunk {int(row['chunk_size'])}")
    if pd.notna(row.get("chunk_overlap", pd.NA)):
        parts.append(f"overlap {int(row['chunk_overlap'])}")
    return " | ".join(parts)


def _deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    subset = [
        column
        for column in [
            "experiment",
            "method",
            "model_name",
            "config_slug",
            "qrels_path",
            "chunk_size",
            "chunk_overlap",
            "include_title_chunk",
            "text_source",
            "average_score",
            "average_ndcg@10",
            "average_recall@10",
        ]
        if column in df.columns
    ]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def _select_best_model_parameter_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {column for column, _, _ in METRIC_SPECS}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    plot_df = df.dropna(subset=["average_ndcg@10"]).copy()
    if plot_df.empty:
        return pd.DataFrame()

    group_columns = [
        column
        for column in [
            "method",
            "model_label",
            "chunk_size",
            "chunk_overlap",
            "include_title_chunk",
            "text_source",
        ]
        if column in plot_df.columns
    ]
    sort_columns = ["average_ndcg@10"]
    ascending = [False]
    if "evaluated_at_cet" in plot_df.columns:
        sort_columns.append("evaluated_at_cet")
        ascending.append(False)

    return (
        plot_df.sort_values(sort_columns, ascending=ascending, kind="stable")
        .drop_duplicates(subset=group_columns, keep="first")
        .sort_values("average_ndcg@10", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _limit_rows(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if top_n <= 0:
        return df
    return df.head(top_n).copy()


def _is_title_chunk_enabled(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _filter_title_chunk_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "include_title_chunk" not in df.columns:
        return pd.DataFrame()
    return df[df["include_title_chunk"].map(_is_title_chunk_enabled)].copy()


def _model_color_map(df: pd.DataFrame, color_column: str) -> dict[str, str]:
    if df.empty:
        return {}
    if color_column not in df.columns:
        return {}
    models = sorted({str(value) for value in df[color_column] if str(value).strip()})
    return {
        model: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, model in enumerate(models)
    }


def _add_model_legend(axis, color_map: dict[str, str]) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#25313d", label=model)
        for model, color in color_map.items()
    ]
    if handles:
        axis.legend(handles=handles, title="Model", loc="lower right", fontsize=8, title_fontsize=9)


def _plot_single_metric(
    df: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    output_path: Path,
    color_map: dict[str, str],
    label_column: str = "comparison_label",
    color_column: str = "model_label",
    title_suffix: str = "by Model and Parameter Values",
) -> None:
    if df.empty or metric_column not in df.columns:
        return
    plot_df = df.iloc[::-1].reset_index(drop=True)
    y_positions = list(range(len(plot_df)))
    labels = plot_df[label_column].tolist()
    height = max(7.0, len(plot_df) * 0.48)
    fig, axis = plt.subplots(figsize=(12.5, height))
    values = plot_df[metric_column].astype(float)
    colors = [
        color_map.get(str(model), "#4b5563")
        for model in plot_df[color_column].astype(str)
    ]

    axis.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="#25313d",
        alpha=0.9,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel(metric_label)
    axis.set_xlim(left=0, right=max(1.0, float(values.max()) * 1.08))
    axis.set_title(f"{metric_label} {title_suffix}")
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)

    for y_index, value in enumerate(values):
        axis.text(
            float(value) + 0.012,
            y_index,
            f"{float(value):.3f}",
            va="center",
            fontsize=8,
        )

    _add_model_legend(axis, color_map)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_comparisons(
    df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "experiment_model_parameter_metrics",
    label_column: str = "comparison_label",
    color_column: str = "model_label",
    title_suffix: str = "by Model and Parameter Values",
) -> list[Path]:
    color_map = _model_color_map(df, color_column=color_column)
    output_paths: list[Path] = []
    for metric_column, metric_label, metric_slug in METRIC_SPECS:
        output_path = output_dir / f"{file_prefix}_{metric_slug}.png"
        _plot_single_metric(
            df=df,
            metric_column=metric_column,
            metric_label=metric_label,
            output_path=output_path,
            color_map=color_map,
            label_column=label_column,
            color_column=color_column,
            title_suffix=title_suffix,
        )
        if output_path.exists():
            output_paths.append(output_path)
    return output_paths


def _write_markdown_summary(df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if df.empty:
        body = "No experiment summary rows found.\n"
    else:
        shown = len(df) if top_n <= 0 else min(top_n, len(df))
        body = (
            "# Experiment Model/Parameter Metrics\n\n"
            f"Rows plotted: {shown}. Rows are the best run per model and parameter combination, "
            "selected by nDCG@10.\n\n"
            "Metrics plotted as separate bar charts: RR@20, nDCG@10, Recall@10. "
            "Colors are assigned by model.\n\n"
            "Report-specific plots use only runs with title chunks enabled and compact labels: "
            "model, chunk size, and overlap.\n"
        )
    output_path.write_text(body, encoding="utf-8")


def _parse_methods(raw_methods: str | None) -> set[str]:
    if not raw_methods:
        return set()
    return {item.strip() for item in raw_methods.split(",") if item.strip()}


def _filter_methods(df: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    methods = set(methods)
    if not methods or "method" not in df.columns:
        return df
    return df[df["method"].astype(str).isin(methods)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RR@20, nDCG@10, and Recall@10 across models and parameter values "
            "from experiment aggregate summaries."
        )
    )
    parser.add_argument(
        "--experiments-root",
        default=str(DEFAULT_EXPERIMENTS_ROOT),
        help="Directory containing experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV and PNG outputs are written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of best model/parameter combinations to plot. Use 0 for all.",
    )
    parser.add_argument(
        "--methods",
        help="Optional comma-separated method filter, for example bm25,dense_e5,hybrid_e5.",
    )
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _load_experiment_summaries(experiments_root)
    raw_df = _add_labels(_coerce_types(raw_df))
    raw_df = _filter_methods(raw_df, _parse_methods(args.methods))
    raw_df = _deduplicate_rows(raw_df)
    comparison_df = _select_best_model_parameter_rows(raw_df)
    plotted_df = _limit_rows(comparison_df, args.top_n)
    report_df = _limit_rows(_filter_title_chunk_rows(comparison_df), args.top_n)

    raw_csv = output_dir / "experiment_model_parameter_metrics_all_runs.csv"
    comparison_csv = output_dir / "experiment_model_parameter_metrics_best.csv"
    plotted_csv = output_dir / "experiment_model_parameter_metrics_plotted.csv"
    report_csv = output_dir / "experiment_model_parameter_metrics_report_title_chunk.csv"
    summary_path = output_dir / "experiment_model_parameter_metrics.md"

    raw_df.to_csv(raw_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    plotted_df.to_csv(plotted_csv, index=False)
    report_df.to_csv(report_csv, index=False)
    png_paths = _plot_metric_comparisons(plotted_df, output_dir)
    report_png_paths = _plot_metric_comparisons(
        df=report_df,
        output_dir=output_dir,
        file_prefix="report_title_chunk_model_parameter_metrics",
        label_column="report_label",
        color_column="report_model_label",
        title_suffix="by Model and Chunking Parameters (Title Chunk Only)",
    )
    _write_markdown_summary(comparison_df, summary_path, args.top_n)

    for path in [
        raw_csv,
        comparison_csv,
        plotted_csv,
        report_csv,
        *png_paths,
        *report_png_paths,
        summary_path,
    ]:
        if path.exists():
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
