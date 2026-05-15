from __future__ import annotations

import argparse
import csv
import html
import os
import re
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .evaluation import load_qrels_dataset
except ImportError:
    from evaluation import load_qrels_dataset


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots" / "all_results"
SUMMARY_FILE_NAME = "evaluation_summary.csv"
RESULTS_FILE_NAME = "evaluation_results.csv"
RUN_FILE_NAME = "evaluation_run.csv"


def _is_header_row(row: list[str], required_columns: set[str]) -> bool:
    row_set = {cell.strip() for cell in row}
    return bool(required_columns & row_set) and (
        "evaluated_at_cet" in row_set or "method" in row_set or "query_type" in row_set
    )


def _read_appended_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Read CSVs that may contain repeated headers with changing metadata columns."""
    rows: list[dict[str, str]] = []
    seen_headers: list[list[str]] = []
    current_header: list[str] | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue

            if _is_header_row(raw_row, required_columns):
                current_header = [cell.strip() for cell in raw_row]
                seen_headers.append(current_header)
                continue

            if current_header is None:
                raise ValueError(f"No CSV header found before data row in {path}")

            matching_header = current_header if len(raw_row) == len(current_header) else None
            if matching_header is None:
                for candidate in reversed(seen_headers):
                    if len(raw_row) == len(candidate):
                        matching_header = candidate
                        break

            if matching_header is None:
                raise ValueError(
                    f"Malformed row in {path}: found {len(raw_row)} fields, "
                    f"but known headers have lengths {sorted({len(header) for header in seen_headers})}"
                )

            rows.append(dict(zip(matching_header, raw_row)))

    return pd.DataFrame(rows)


def _experiment_name_from_path(path: Path, experiments_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(experiments_root.resolve())
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


def _collect_csvs(
    evaluation_dir: Path,
    experiments_root: Path,
    file_name: str,
    include_experiment_run_files: bool,
) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []

    root_path = evaluation_dir / file_name
    if root_path.exists():
        paths.append((root_path, "terminal"))

    aggregate_paths = sorted(experiments_root.glob(f"*/results/aggregate/{file_name}"))
    paths.extend((path, "experiment_aggregate") for path in aggregate_paths)

    if include_experiment_run_files:
        run_paths = sorted(experiments_root.glob(f"*/results/runs/**/{file_name}"))
        paths.extend((path, "experiment_run_file") for path in run_paths)

    return paths


def _load_kind(
    evaluation_dir: Path,
    experiments_root: Path,
    file_name: str,
    required_columns: set[str],
    include_experiment_run_files: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, source_kind in _collect_csvs(
        evaluation_dir,
        experiments_root,
        file_name,
        include_experiment_run_files,
    ):
        df = _read_appended_csv(path, required_columns)
        if df.empty:
            continue
        df["source_kind"] = source_kind
        df["source_file"] = str(path.resolve())
        df["experiment_from_path"] = (
            _experiment_name_from_path(path, experiments_root)
            if source_kind.startswith("experiment")
            else ""
        )
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


def _build_qrels_table(qrels_source: str, qrels_path: str | None) -> pd.DataFrame:
    frames = load_qrels_dataset(qrels_source=qrels_source, qrels_path=qrels_path)
    rows: list[dict[str, object]] = []

    for query_type, frame in frames.items():
        qrels_df = frame["qrels"].copy()
        qrels_df = qrels_df[qrels_df["relevance"] > 0]

        for query_id in frame["queries"]["query_id"].drop_duplicates():
            query_id_text = str(query_id)
            matching = qrels_df[qrels_df["query_id"].astype(str) == query_id_text]
            relevant_doc_ids = sorted(
                {str(doc_id) for doc_id in matching["doc_id"] if str(doc_id).strip()}
            )
            rows.append(
                {
                    "query_type": str(query_type),
                    "query_id": query_id_text,
                    "relevant_doc_ids": relevant_doc_ids,
                }
            )

    return pd.DataFrame(rows)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "evaluated_at_cet" in df.columns:
        df["evaluated_at_cet"] = pd.to_datetime(df["evaluated_at_cet"], errors="coerce")
    for column in [
        "average_rank",
        "average_score",
        "RR@20",
        "top_k",
        "num_queries_total",
        "num_query_types",
        "chunk_size",
        "chunk_overlap",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _canonicalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].fillna("").astype(str)
    return df


def _drop_duplicate_metric_rows(df: pd.DataFrame, metric_columns: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df

    candidate_columns = [
        column
        for column in [
            *metric_columns,
            "evaluated_at_cet",
            "method",
            "model_name",
            "experiment",
            "config_slug",
            "query_type",
            "top_k",
            "qrels_source",
            "data_source",
            "chunk_size",
            "chunk_overlap",
            "include_title_chunk",
            "text_source",
        ]
        if column in df.columns
    ]
    return df.drop_duplicates(subset=candidate_columns, keep="first").reset_index(drop=True)


def _format_timestamp(value: object) -> str:
    if pd.isna(value):
        return "no_timestamp"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def _short_model(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text.split("/")[-1]


def _build_run_label(row: pd.Series, include_timestamp: bool = True) -> str:
    parts: list[str] = []
    for column in ["method", "experiment", "config_slug"]:
        value = str(row.get(column, "")).strip()
        if value:
            parts.append(value)

    model_name = _short_model(row.get("model_name", ""))
    if model_name and model_name not in parts:
        parts.append(model_name)

    param_parts: list[str] = []
    if pd.notna(row.get("chunk_size", pd.NA)):
        param_parts.append(f"chunk={int(row['chunk_size'])}")
    if pd.notna(row.get("chunk_overlap", pd.NA)):
        param_parts.append(f"overlap={int(row['chunk_overlap'])}")
    include_title = str(row.get("include_title_chunk", "")).strip()
    if include_title:
        param_parts.append(f"title={include_title}")
    if param_parts:
        parts.append(", ".join(param_parts))

    if include_timestamp:
        parts.append(_format_timestamp(row.get("evaluated_at_cet")))

    return " | ".join(parts) if parts else "run"


def _add_run_labels(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = summary_df.copy()
    results_df = results_df.copy()

    if not summary_df.empty:
        summary_df["run_label"] = summary_df.apply(_build_run_label, axis=1)

    if not results_df.empty:
        results_df["run_label"] = results_df.apply(_build_run_label, axis=1)

    return summary_df, results_df


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unnamed"


def _prepare_plot_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = summary_df.dropna(subset=["evaluated_at_cet"]).copy()
    if plot_df.empty:
        return plot_df
    sort_columns = [column for column in ["evaluated_at_cet", "method", "experiment", "config_slug"] if column in plot_df]
    return plot_df.sort_values(sort_columns, kind="stable")


def _plot_metric_over_time(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plot_df = _prepare_plot_df(summary_df)
    if plot_df.empty or metric not in plot_df.columns:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    for method, method_df in plot_df.groupby("method", dropna=False, sort=True):
        label = str(method) if str(method).strip() else "unknown"
        ax.plot(
            method_df["evaluated_at_cet"],
            method_df[metric],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=label,
        )

    ax.set_title(f"{ylabel} over time")
    ax.set_xlabel("Evaluated at")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate(rotation=35, ha="right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_runs(summary_df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if summary_df.empty or "average_score" not in summary_df.columns:
        return

    plot_df = summary_df.dropna(subset=["average_score"]).copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("average_score", ascending=False, kind="stable").head(top_n)
    plot_df = plot_df.sort_values("average_score", ascending=True, kind="stable")

    labels = [_build_run_label(row, include_timestamp=True) for _, row in plot_df.iterrows()]
    height = max(6, len(plot_df) * 0.45)
    fig, ax = plt.subplots(figsize=(14, height))
    ax.barh(range(len(plot_df)), plot_df["average_score"], color="#2f7f5f", edgecolor="#1e513d")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Average RR@20")
    ax.set_title(f"Top {len(plot_df)} evaluation runs by average RR@20")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)

    for index, value in enumerate(plot_df["average_score"]):
        ax.text(float(value) + 0.005, index, f"{value:.3f}", va="center", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_query_type_heatmap(results_df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if results_df.empty or "RR@20" not in results_df.columns or "query_type" not in results_df.columns:
        return

    run_scores = (
        results_df.groupby("run_label", dropna=False)["RR@20"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    selected = results_df[results_df["run_label"].isin(run_scores.index)].copy()
    if selected.empty:
        return

    pivot = selected.pivot_table(
        index="run_label",
        columns="query_type",
        values="RR@20",
        aggfunc="mean",
    ).loc[run_scores.index]

    fig_width = max(10, len(pivot.columns) * 1.45)
    fig_height = max(7, len(pivot.index) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_title(f"RR@20 by query type for top {len(pivot.index)} runs")
    ax.set_xlabel("Query type")
    ax.set_ylabel("Run")

    for y in range(len(pivot.index)):
        for x in range(len(pivot.columns)):
            value = pivot.iat[y, x]
            if pd.notna(value):
                text_color = "white" if float(value) < 0.45 else "black"
                ax.text(x, y, f"{float(value):.2f}", ha="center", va="center", color=text_color, fontsize=7)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("RR@20")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_parameter_scatter(summary_df: pd.DataFrame, output_path: Path) -> None:
    required = {"chunk_size", "chunk_overlap", "average_score"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return

    plot_df = summary_df.dropna(subset=list(required)).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    methods = list(dict.fromkeys(plot_df["method"].fillna("unknown").astype(str)))
    for method in methods:
        method_df = plot_df[plot_df["method"].astype(str) == method]
        ax.scatter(
            method_df["chunk_size"],
            method_df["average_score"],
            s=(method_df["chunk_overlap"].fillna(0) + 20).clip(lower=20, upper=120),
            alpha=0.75,
            label=method,
        )

    ax.set_title("Average RR@20 by chunk size")
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Average RR@20")
    ax.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_total_hits_df(run_df: pd.DataFrame, qrels_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty or qrels_df.empty:
        return pd.DataFrame()
    if not {"run_label", "query_id", "doc_id"}.issubset(run_df.columns):
        return pd.DataFrame()

    run_df = run_df.drop_duplicates(
        subset=[
            column
            for column in ["run_label", "query_type", "query_id", "doc_id"]
            if column in run_df.columns
        ],
        keep="first",
    ).copy()
    run_df["query_id"] = run_df["query_id"].astype(str)
    run_df["doc_id"] = run_df["doc_id"].astype(str)
    if "query_type" in run_df.columns:
        run_df["query_type"] = run_df["query_type"].astype(str)

    metadata_columns = [
        column
        for column in [
            "run_label",
            "evaluated_at_cet",
            "method",
            "model_name",
            "experiment",
            "config_slug",
            "top_k",
            "qrels_source",
            "data_source",
            "chunk_size",
            "chunk_overlap",
            "include_title_chunk",
            "text_source",
            "source_kind",
            "source_file",
        ]
        if column in run_df.columns
    ]
    metadata_df = run_df[metadata_columns].drop_duplicates(subset=["run_label"], keep="first")

    rows: list[dict[str, object]] = []
    for run_label, run_group in run_df.groupby("run_label", sort=False):
        total_hits = 0
        for _, qrel_row in qrels_df.iterrows():
            query_matches = run_group[run_group["query_id"] == str(qrel_row["query_id"])]
            if "query_type" in query_matches.columns:
                query_matches = query_matches[
                    query_matches["query_type"].astype(str) == str(qrel_row["query_type"])
                ]
            retrieved_doc_ids = {str(doc_id) for doc_id in query_matches["doc_id"] if str(doc_id).strip()}
            total_hits += int(bool(retrieved_doc_ids & set(qrel_row["relevant_doc_ids"])))

        rows.append(
            {
                "run_label": run_label,
                "total_hits": total_hits,
                "total_queries": len(qrels_df),
                "hit_rate": total_hits / len(qrels_df) if len(qrels_df) else 0.0,
            }
        )

    hits_df = pd.DataFrame(rows)
    if hits_df.empty:
        return hits_df

    hits_df = hits_df.merge(metadata_df, on="run_label", how="left")
    sort_columns = [column for column in ["total_hits", "hit_rate", "evaluated_at_cet"] if column in hits_df.columns]
    ascending = [False, False, True][: len(sort_columns)]
    return hits_df.sort_values(sort_columns, ascending=ascending, kind="stable").reset_index(drop=True)


def _plot_total_hits(total_hits_df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if total_hits_df.empty or "total_hits" not in total_hits_df.columns:
        return

    plot_df = total_hits_df.head(top_n).copy()
    plot_df = plot_df.sort_values("total_hits", ascending=True, kind="stable")

    height = max(7, len(plot_df) * 0.45)
    fig, ax = plt.subplots(figsize=(14, height))
    ax.barh(range(len(plot_df)), plot_df["total_hits"], color="#3269a8", edgecolor="#21466f")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["run_label"].tolist(), fontsize=8)
    ax.set_xlabel("Total hits")
    ax.set_title(f"Total number of hits across all query types for top {len(plot_df)} runs")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)

    for index, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            float(row.total_hits) + 0.25,
            index,
            f"{int(row.total_hits)}/{int(row.total_queries)}",
            va="center",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _html_table(df: pd.DataFrame, max_rows: int) -> str:
    if df.empty:
        return "<p>No rows found.</p>"

    display_df = df.head(max_rows).copy()
    for column in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[column]):
            display_df[column] = display_df[column].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in display_df.columns)
    rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _write_html_report(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    total_hits_df: pd.DataFrame,
    output_dir: Path,
    image_paths: list[Path],
    max_table_rows: int,
) -> Path:
    report_path = output_dir / "all_results_report.html"
    relative_images = [path.name for path in image_paths if path.exists()]

    top_summary = (
        summary_df.sort_values("average_score", ascending=False, kind="stable")
        if "average_score" in summary_df.columns
        else summary_df
    )

    image_html = "\n".join(
        f'<section><h2>{html.escape(Path(image).stem.replace("_", " ").title())}</h2>'
        f'<img src="{html.escape(image)}" alt="{html.escape(image)}"></section>'
        for image in relative_images
    )

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evaluation Results Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 0 0 32px; }}
    img {{ max-width: 100%; border: 1px solid #d9d9d9; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d9d9d9; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 640px; overflow: auto; border: 1px solid #d9d9d9; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Evaluation Results Report</h1>
  <section>
    <p>Summary rows: {len(summary_df)}. Per-query-type rows: {len(results_df)}.</p>
    <p>Normalized data files: <code>all_evaluation_summary.csv</code> and <code>all_evaluation_results.csv</code>.</p>
  </section>
  {image_html}
  <section>
    <h2>Top Runs With Metadata</h2>
    <div class="table-wrap">{_html_table(top_summary, max_table_rows)}</div>
  </section>
  <section>
    <h2>Total Hits With Metadata</h2>
    <div class="table-wrap">{_html_table(total_hits_df, max_table_rows)}</div>
  </section>
  <section>
    <h2>Per Query Type Results With Metadata</h2>
    <div class="table-wrap">{_html_table(results_df, max_table_rows)}</div>
  </section>
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create visualizations across terminal evaluations and experiment results."
    )
    parser.add_argument(
        "--evaluation-dir",
        default=str(EVALUATION_DIR),
        help="Directory containing root evaluation_*.csv files.",
    )
    parser.add_argument(
        "--experiments-root",
        default=str(EVALUATION_DIR / "experiments"),
        help="Directory containing experiment result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where normalized CSVs, PNGs, and HTML report are written.",
    )
    parser.add_argument(
        "--include-experiment-run-files",
        action="store_true",
        help=(
            "Also scan results/runs/** CSVs. By default only aggregate experiment CSVs are used "
            "to avoid duplicate rows."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top runs included in dense plots and report highlights.",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=250,
        help="Maximum rows shown per HTML table. Full rows are always written to CSV.",
    )
    parser.add_argument(
        "--qrels-source",
        choices=["google_sheet", "form_submissions"],
        default="google_sheet",
        help="Qrels source used to count total hits from evaluation_run.csv.",
    )
    parser.add_argument(
        "--qrels-path",
        help="Optional qrels CSV path when --qrels-source=form_submissions.",
    )
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir).resolve()
    experiments_root = Path(args.experiments_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _load_kind(
        evaluation_dir=evaluation_dir,
        experiments_root=experiments_root,
        file_name=SUMMARY_FILE_NAME,
        required_columns={"average_rank", "average_score"},
        include_experiment_run_files=args.include_experiment_run_files,
    )
    results_df = _load_kind(
        evaluation_dir=evaluation_dir,
        experiments_root=experiments_root,
        file_name=RESULTS_FILE_NAME,
        required_columns={"query_type", "RR@20"},
        include_experiment_run_files=args.include_experiment_run_files,
    )
    run_df = _load_kind(
        evaluation_dir=evaluation_dir,
        experiments_root=experiments_root,
        file_name=RUN_FILE_NAME,
        required_columns={"doc_id", "query_id", "score"},
        include_experiment_run_files=args.include_experiment_run_files,
    )

    summary_df = _canonicalize_text_columns(_coerce_types(summary_df))
    results_df = _canonicalize_text_columns(_coerce_types(results_df))
    run_df = _canonicalize_text_columns(_coerce_types(run_df))
    summary_df = _drop_duplicate_metric_rows(summary_df, ["average_rank", "average_score"])
    results_df = _drop_duplicate_metric_rows(results_df, ["RR@20", "average_rank"])
    summary_df, results_df = _add_run_labels(summary_df, results_df)
    if not run_df.empty:
        run_df["run_label"] = run_df.apply(_build_run_label, axis=1)

    if not summary_df.empty and "evaluated_at_cet" in summary_df.columns:
        summary_df = summary_df.sort_values("evaluated_at_cet", kind="stable", na_position="last")
    if not results_df.empty and "evaluated_at_cet" in results_df.columns:
        results_df = results_df.sort_values(
            [column for column in ["evaluated_at_cet", "run_label", "query_type"] if column in results_df.columns],
            kind="stable",
            na_position="last",
        )

    qrels_df = _build_qrels_table(args.qrels_source, args.qrels_path)
    total_hits_df = _build_total_hits_df(run_df, qrels_df)

    summary_csv = output_dir / "all_evaluation_summary.csv"
    results_csv = output_dir / "all_evaluation_results.csv"
    total_hits_csv = output_dir / "all_run_total_hits.csv"
    summary_df.to_csv(summary_csv, index=False)
    results_df.to_csv(results_csv, index=False)
    total_hits_df.to_csv(total_hits_csv, index=False)

    image_paths = [
        output_dir / "average_rr20_over_time.png",
        output_dir / "average_rank_over_time.png",
        output_dir / "top_runs_by_average_rr20.png",
        output_dir / "rr20_by_query_type_heatmap.png",
        output_dir / "average_rr20_by_chunk_size.png",
        output_dir / "total_hits_all_runs.png",
    ]
    _plot_metric_over_time(summary_df, "average_score", "Average RR@20", image_paths[0])
    _plot_metric_over_time(summary_df, "average_rank", "Average rank", image_paths[1])
    _plot_best_runs(summary_df, image_paths[2], top_n=args.top_n)
    _plot_query_type_heatmap(results_df, image_paths[3], top_n=args.top_n)
    _plot_parameter_scatter(summary_df, image_paths[4])
    _plot_total_hits(total_hits_df, image_paths[5], top_n=args.top_n)

    report_path = _write_html_report(
        summary_df=summary_df,
        results_df=results_df,
        total_hits_df=total_hits_df,
        output_dir=output_dir,
        image_paths=image_paths,
        max_table_rows=args.max_table_rows,
    )

    written = [summary_csv, results_csv, total_hits_csv, report_path, *[path for path in image_paths if path.exists()]]
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
