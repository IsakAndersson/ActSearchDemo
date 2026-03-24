from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

try:
    from .evaluation import EVALUATION_DIR, load_qrels_dataset
except ImportError:
    from evaluation import EVALUATION_DIR, load_qrels_dataset


DEFAULT_METHOD_ORDER = [
    "bm25",
    "dense",
    "dense_e5",
    "hybrid",
    "hybrid_e5",
    "docplus_live",
    "sts_live",
]


def _read_appended_run_csv(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    current_header: List[str] | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            if raw_row[0] == "doc_id":
                current_header = raw_row
                continue
            if current_header is None:
                raise ValueError(f"No CSV header found in {path}")
            if len(raw_row) != len(current_header):
                raise ValueError(
                    f"Malformed row in {path}: expected {len(current_header)} fields, got {len(raw_row)}"
                )
            rows.append(dict(zip(current_header, raw_row)))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "evaluated_at_cet" in df.columns:
        df["evaluated_at_cet"] = pd.to_datetime(df["evaluated_at_cet"], errors="coerce")
    return df


def _build_qrels_table(qrels_source: str, qrels_path: str | None) -> pd.DataFrame:
    frames = load_qrels_dataset(qrels_source=qrels_source, qrels_path=qrels_path)
    ordered_rows: List[Dict[str, str]] = []

    for query_type, frame in frames.items():
        qrels_df = frame["qrels"].copy()
        qrels_df = qrels_df[qrels_df["relevance"] > 0]
        seen_queries = set()

        for query_id in frame["queries"]["query_id"]:
            query_id = str(query_id)
            if query_id in seen_queries:
                continue
            seen_queries.add(query_id)

            matching = qrels_df[qrels_df["query_id"].astype(str) == query_id]
            relevant_doc_ids = sorted({str(doc_id) for doc_id in matching["doc_id"] if str(doc_id).strip()})
            ordered_rows.append(
                {
                    "query_type": str(query_type),
                    "query_id": query_id,
                    "query_label": f"{query_type}: {query_id}",
                    "relevant_doc_ids": relevant_doc_ids,
                }
            )

    return pd.DataFrame(ordered_rows)


def _pick_latest_run_per_method(run_df: pd.DataFrame, methods: Sequence[str] | None) -> pd.DataFrame:
    if run_df.empty:
        return run_df

    if methods:
        wanted = {method.strip() for method in methods if method.strip()}
        run_df = run_df[run_df["method"].astype(str).isin(wanted)].copy()

    run_df = run_df.dropna(subset=["method", "evaluated_at_cet"]).copy()
    latest_per_method = (
        run_df.groupby("method", dropna=False)["evaluated_at_cet"].max().rename("latest_evaluated_at_cet")
    )
    selected = run_df.merge(
        latest_per_method,
        left_on=["method", "evaluated_at_cet"],
        right_on=["method", "latest_evaluated_at_cet"],
        how="inner",
    )
    return selected.drop(columns=["latest_evaluated_at_cet"])


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    method_list = list(dict.fromkeys(str(method) for method in methods))
    preferred = [method for method in DEFAULT_METHOD_ORDER if method in method_list]
    extras = sorted(method for method in method_list if method not in DEFAULT_METHOD_ORDER)
    return preferred + extras


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unnamed"


def _build_hit_matrix(qrels_df: pd.DataFrame, latest_runs_df: pd.DataFrame) -> pd.DataFrame:
    methods = _ordered_methods(latest_runs_df["method"].dropna().astype(str).tolist())
    hit_rows: List[Dict[str, object]] = []

    for _, qrel_row in qrels_df.iterrows():
        relevant_doc_ids = set(qrel_row["relevant_doc_ids"])
        row: Dict[str, object] = {
            "query_type": qrel_row["query_type"],
            "query_id": qrel_row["query_id"],
            "query_label": qrel_row["query_label"],
        }
        for method in methods:
            method_hits = latest_runs_df[
                (latest_runs_df["method"].astype(str) == method)
                & (latest_runs_df["query_id"].astype(str) == str(qrel_row["query_id"]))
            ]
            retrieved_doc_ids = {str(doc_id) for doc_id in method_hits["doc_id"] if str(doc_id).strip()}
            row[method] = int(bool(relevant_doc_ids & retrieved_doc_ids))
        hit_rows.append(row)

    matrix_df = pd.DataFrame(hit_rows)
    if not matrix_df.empty:
        matrix_df["query_type"] = pd.Categorical(
            matrix_df["query_type"],
            categories=list(dict.fromkeys(qrels_df["query_type"].tolist())),
            ordered=True,
        )
        matrix_df = matrix_df.sort_values(["query_type"], kind="stable").reset_index(drop=True)
    return matrix_df


def _build_named_hit_matrix(qrels_df: pd.DataFrame, run_df: pd.DataFrame, column_names: Sequence[str]) -> pd.DataFrame:
    hit_rows: List[Dict[str, object]] = []

    for _, qrel_row in qrels_df.iterrows():
        relevant_doc_ids = set(qrel_row["relevant_doc_ids"])
        row: Dict[str, object] = {
            "query_type": qrel_row["query_type"],
            "query_id": qrel_row["query_id"],
            "query_label": qrel_row["query_label"],
        }
        for column_name in column_names:
            run_hits = run_df[
                (run_df["run_label"].astype(str) == column_name)
                & (run_df["query_id"].astype(str) == str(qrel_row["query_id"]))
            ]
            retrieved_doc_ids = {str(doc_id) for doc_id in run_hits["doc_id"] if str(doc_id).strip()}
            row[column_name] = int(bool(relevant_doc_ids & retrieved_doc_ids))
        hit_rows.append(row)

    matrix_df = pd.DataFrame(hit_rows)
    if not matrix_df.empty:
        matrix_df["query_type"] = pd.Categorical(
            matrix_df["query_type"],
            categories=list(dict.fromkeys(qrels_df["query_type"].tolist())),
            ordered=True,
        )
        matrix_df = matrix_df.sort_values(["query_type"], kind="stable").reset_index(drop=True)
    return matrix_df


def _plot_hit_matrix(matrix_df: pd.DataFrame, x_labels: Sequence[str], output_path: Path, title: str) -> None:
    plot_data = matrix_df.loc[:, x_labels]
    query_labels = matrix_df["query_label"].tolist()

    fig_width = max(8, len(x_labels) * 1.25)
    fig_height = max(10, len(query_labels) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(plot_data.to_numpy(), aspect="auto", cmap=ListedColormap(["#d73027", "#1a9850"]), vmin=0, vmax=1)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(query_labels)))
    ax.set_yticklabels(query_labels, fontsize=8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Query")
    ax.set_title(title)

    ax.set_xticks([x - 0.5 for x in range(1, len(x_labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(query_labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.03, pad=0.02)
    colorbar.ax.set_yticklabels(["Miss", "Hit"])

    longest_label = max((len(label) for label in query_labels), default=0)
    left_margin = min(0.72, max(0.22, longest_label * 0.0045))
    fig.subplots_adjust(left=left_margin, bottom=0.12, right=0.95, top=0.94)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_total_hits_bar_chart(matrix_df: pd.DataFrame, x_labels: Sequence[str], output_path: Path, title: str) -> None:
    totals = matrix_df.loc[:, x_labels].sum(axis=0).sort_values(ascending=False)

    fig_width = max(10, len(x_labels) * 0.45)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.bar(range(len(totals)), totals.to_numpy(), color="#1a9850", edgecolor="#0f5c30", linewidth=0.6)
    ax.set_xticks(range(len(totals)))
    ax.set_xticklabels(totals.index.tolist(), rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Total hits")
    ax.set_xlabel("Run")
    ax.set_title(title)
    ax.set_ylim(0, max(1, int(totals.max())) * 1.1)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.08, bottom=0.42, right=0.98, top=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_experiment_run_rows(experiments_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(experiments_root.rglob("results/runs/**/evaluation_run.csv")):
        df = _read_appended_run_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path.resolve())
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "evaluated_at_cet" in combined.columns:
        combined["evaluated_at_cet"] = pd.to_datetime(combined["evaluated_at_cet"], errors="coerce")
    return combined


def _build_run_label(row: pd.Series) -> str:
    parts = []
    method = str(row.get("method", "")).strip()
    experiment = str(row.get("experiment", "")).strip()
    config_slug = str(row.get("config_slug", "")).strip()

    if method:
        parts.append(method)
    if experiment:
        parts.append(experiment)
    if config_slug:
        parts.append(config_slug)
    elif row.get("evaluated_at_cet") is not pd.NaT and pd.notna(row.get("evaluated_at_cet")):
        parts.append(pd.Timestamp(row["evaluated_at_cet"]).strftime("%Y-%m-%d %H:%M"))
    return " | ".join(parts) if parts else "run"


def _prepare_experiment_runs(experiment_runs_df: pd.DataFrame) -> pd.DataFrame:
    if experiment_runs_df.empty:
        return experiment_runs_df

    experiment_runs_df = experiment_runs_df.copy()
    if "model_name" in experiment_runs_df.columns:
        model_name = experiment_runs_df["model_name"].fillna("").astype(str).str.strip()
    else:
        model_name = pd.Series([""] * len(experiment_runs_df), index=experiment_runs_df.index)
    experiment_runs_df["model_key"] = model_name.where(model_name != "", experiment_runs_df["method"].astype(str))
    experiment_runs_df["run_label"] = experiment_runs_df.apply(_build_run_label, axis=1)
    return experiment_runs_df


def _ordered_run_labels(run_df: pd.DataFrame) -> List[str]:
    run_order_df = (
        run_df[["run_label", "evaluated_at_cet", "model_key", "method", "experiment", "config_slug"]]
        .drop_duplicates()
        .sort_values(
            by=["model_key", "evaluated_at_cet", "method", "experiment", "config_slug"],
            kind="stable",
            na_position="last",
        )
    )
    return run_order_df["run_label"].tolist()


def _write_model_run_matrices(qrels_df: pd.DataFrame, experiment_runs_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    if experiment_runs_df.empty:
        return []

    experiment_runs_df = _prepare_experiment_runs(experiment_runs_df)

    written_paths: List[Path] = []
    for model_key, model_df in experiment_runs_df.groupby("model_key", sort=True):
        run_labels = _ordered_run_labels(model_df)
        matrix_df = _build_named_hit_matrix(qrels_df, model_df, run_labels)
        model_slug = _slugify(str(model_key))

        csv_output_path = output_dir / f"query_run_hit_matrix__{model_slug}.csv"
        png_output_path = output_dir / f"query_run_hit_matrix__{model_slug}.png"
        matrix_df.to_csv(csv_output_path, index=False)
        _plot_hit_matrix(
            matrix_df,
            run_labels,
            png_output_path,
            title=f"Relevant document present in results: {model_key}",
        )
        written_paths.extend([csv_output_path, png_output_path])

    return written_paths


def _write_all_runs_matrix(qrels_df: pd.DataFrame, experiment_runs_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    if experiment_runs_df.empty:
        return []

    experiment_runs_df = _prepare_experiment_runs(experiment_runs_df)
    run_labels = _ordered_run_labels(experiment_runs_df)
    matrix_df = _build_named_hit_matrix(qrels_df, experiment_runs_df, run_labels)

    csv_output_path = output_dir / "query_run_hit_matrix__all_runs.csv"
    png_output_path = output_dir / "query_run_hit_matrix__all_runs.png"
    totals_csv_output_path = output_dir / "run_total_hits__all_runs.csv"
    totals_png_output_path = output_dir / "run_total_hits__all_runs.png"
    matrix_df.to_csv(csv_output_path, index=False)
    _plot_hit_matrix(
        matrix_df,
        run_labels,
        png_output_path,
        title="Relevant document present in results: all experiment runs",
    )
    totals_df = pd.DataFrame(
        {
            "run_label": run_labels,
            "total_hits": [int(matrix_df[label].sum()) for label in run_labels],
        }
    ).sort_values("total_hits", ascending=False, kind="stable")
    totals_df.to_csv(totals_csv_output_path, index=False)
    _plot_total_hits_bar_chart(
        matrix_df,
        run_labels,
        totals_png_output_path,
        title="Total hits across all queries for each run",
    )
    return [csv_output_path, png_output_path, totals_csv_output_path, totals_png_output_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a red/green query-method hit matrix from evaluation_run.csv."
    )
    parser.add_argument(
        "--run-csv",
        default=str(EVALUATION_DIR / "evaluation_run.csv"),
        help="Path to evaluation_run.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_DIR / "plots"),
        help="Directory where the plot and CSV are written.",
    )
    parser.add_argument(
        "--qrels-source",
        choices=["google_sheet", "form_submissions"],
        default="google_sheet",
        help="Qrels source used to decide which document is relevant for each query.",
    )
    parser.add_argument(
        "--qrels-path",
        help="Optional qrels CSV path when --qrels-source=form_submissions.",
    )
    parser.add_argument(
        "--methods",
        help="Optional comma-separated method list. Default: all methods found in the run CSV.",
    )
    parser.add_argument(
        "--experiments-root",
        default=str(EVALUATION_DIR / "experiments"),
        help="Root directory scanned for experiment run CSVs used for per-model run matrices.",
    )
    args = parser.parse_args()

    run_csv_path = Path(args.run_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    experiments_root = Path(args.experiments_root).resolve()
    selected_methods = args.methods.split(",") if args.methods else None

    run_df = _read_appended_run_csv(run_csv_path)
    if run_df.empty:
        raise ValueError(f"No run rows found in {run_csv_path}")
    if "method" not in run_df.columns:
        raise ValueError(f"{run_csv_path} does not contain a 'method' column")

    latest_runs_df = _pick_latest_run_per_method(run_df, selected_methods)
    if latest_runs_df.empty:
        raise ValueError("No matching methods found in the run CSV")

    qrels_df = _build_qrels_table(args.qrels_source, args.qrels_path)
    matrix_df = _build_hit_matrix(qrels_df, latest_runs_df)

    methods = _ordered_methods(latest_runs_df["method"].dropna().astype(str).tolist())
    csv_output_path = output_dir / "query_method_hit_matrix.csv"
    png_output_path = output_dir / "query_method_hit_matrix.png"

    matrix_df.to_csv(csv_output_path, index=False)
    _plot_hit_matrix(matrix_df, methods, png_output_path, title="Relevant document present in results")

    written_paths = [csv_output_path, png_output_path]
    experiment_runs_df = _load_experiment_run_rows(experiments_root)
    written_paths.extend(_write_all_runs_matrix(qrels_df, experiment_runs_df, output_dir))
    written_paths.extend(_write_model_run_matrices(qrels_df, experiment_runs_df, output_dir))

    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
