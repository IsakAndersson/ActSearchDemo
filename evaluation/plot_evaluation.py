from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .evaluation import load_qrels_dataset
except ImportError:
    from evaluation import load_qrels_dataset


plt.style.use("ggplot")
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots"


def load_mixed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []
    current_header = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            row = [cell.strip() for cell in row]
            if row[0] in {"average_rank", "query_type", "doc_id"}:
                current_header = row
                continue
            if current_header is None:
                continue
            if len(row) < len(current_header):
                row = row + [""] * (len(current_header) - len(row))
            row = row[: len(current_header)]
            rows.append(dict(zip(current_header, row)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df.iloc[:, 0] != df.columns[0]]
    return df


def discover_csv_paths(filename: str) -> list[Path]:
    base_candidates = [
        Path(filename),
        Path("evaluation") / filename,
        Path("../evaluation") / filename,
    ]

    experiment_candidates: list[Path] = []
    for root in (
        Path("evaluation/experiments"),
        Path("../evaluation/experiments"),
        Path("experiments"),
    ):
        if root.exists():
            experiment_candidates.extend(root.glob(f"*/results/aggregate/{filename}"))

    paths = []
    seen: set[Path] = set()
    for candidate in base_candidates + sorted(experiment_candidates):
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        paths.append(candidate)
    return paths


def load_mixed_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    parts = []
    for path in paths:
        df = load_mixed_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["file_source"] = str(path)
        if "data_source" not in df.columns and "qrels_source" not in df.columns:
            df["data_source"] = str(path)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def pick_time_col(df: pd.DataFrame) -> str | None:
    for col in ("evaluated_at_cet", "evaluated_at_utc", "evaluated_at"):
        if col in df.columns:
            return col
    return None


def pick_method_col(df: pd.DataFrame) -> str | None:
    for col in ("method", "search_method"):
        if col in df.columns:
            return col
    return None


def pick_source_col(df: pd.DataFrame) -> str | None:
    for col in ("qrels_source", "data_source", "file_source"):
        if col in df.columns:
            return col
    return None


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "plot"


def prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = load_mixed_csvs(discover_csv_paths("evaluation_summary.csv"))
    results_df = load_mixed_csvs(discover_csv_paths("evaluation_results.csv"))
    run_df = load_mixed_csvs(discover_csv_paths("evaluation_run.csv"))

    for numeric_col in (
        "average_rank",
        "average_score",
        "RR@20",
        "score",
        "top_k",
        "num_queries_total",
        "num_query_types",
        "chunk_size",
        "chunk_overlap",
    ):
        if numeric_col in summary_df.columns:
            summary_df[numeric_col] = pd.to_numeric(summary_df[numeric_col], errors="coerce")
        if numeric_col in results_df.columns:
            results_df[numeric_col] = pd.to_numeric(results_df[numeric_col], errors="coerce")
        if numeric_col in run_df.columns:
            run_df[numeric_col] = pd.to_numeric(run_df[numeric_col], errors="coerce")

    for df in (summary_df, results_df, run_df):
        time_col = pick_time_col(df)
        if time_col:
            df["evaluated_at"] = pd.to_datetime(df[time_col], errors="coerce")
        method_col = pick_method_col(df)
        source_col = pick_source_col(df)
        df["method_name"] = (
            df[method_col].fillna("").replace("", "unknown") if method_col else "unknown"
        )
        df["source_name"] = (
            df[source_col].fillna("").replace("", "unknown_source") if source_col else "unknown_source"
        )
        df["series_name"] = df["method_name"].astype(str) + " | " + df["source_name"].astype(str)

    return summary_df, results_df, run_df


def build_run_labels(df: pd.DataFrame, include_method: bool = True) -> pd.Series:
    labels = pd.Series(["run"] * len(df), index=df.index, dtype="object")
    if "config_slug" in df.columns:
        labels = df["config_slug"].fillna("").astype(str)
    if "text_source" in df.columns:
        text_suffix = df["text_source"].fillna("").astype(str)
        labels = labels.where(
            text_suffix.eq("") | labels.str.contains(text_suffix, regex=False),
            labels + " | " + text_suffix,
        )
    if "evaluated_at" in df.columns:
        time_suffix = pd.to_datetime(df["evaluated_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
        labels = labels.where(time_suffix.eq(""), labels + " | " + time_suffix)
    if include_method and "method_name" in df.columns:
        labels = df["method_name"].fillna("unknown").astype(str) + " | " + labels.astype(str)
    return labels.str.strip(" |")


def plot_average_score_over_time(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df.dropna(subset=["average_score"]).copy()
    if plot_df.empty:
        return
    if "evaluated_at" in plot_df.columns:
        plot_df = plot_df.sort_values("evaluated_at")

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, group in plot_df.groupby("series_name"):
        x_values = group["evaluated_at"] if "evaluated_at" in group.columns else range(len(group))
        ax.plot(x_values, group["average_score"], marker="o", label=method)
    ax.set_title("Average RR@20 / score over time")
    ax.set_ylabel("average_score")
    ax.set_xlabel("run time")
    if plot_df["series_name"].nunique() > 1:
        ax.legend(title="method | source")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir / "average_score_over_time.png")


def plot_best_run_per_method(summary_df: pd.DataFrame, output_dir: Path) -> None:
    best_per_method = (
        summary_df.dropna(subset=["average_score"])
        .sort_values("average_score", ascending=False)
        .groupby("series_name", as_index=False)
        .first()
    )
    if best_per_method.empty:
        return

    columns = [
        col
        for col in ("series_name", "average_score", "average_rank", "evaluated_at")
        if col in best_per_method.columns
    ]
    save_table(best_per_method[columns], output_dir / "best_run_per_method.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(best_per_method["series_name"], best_per_method["average_score"])
    ax.set_title("Best average_score per method/source")
    ax.set_ylabel("average_score")
    ax.set_xlabel("method | source")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir / "best_run_per_method.png")


def plot_query_type_profile(results_df: pd.DataFrame, output_dir: Path) -> None:
    if results_df.empty or "RR@20" not in results_df.columns:
        return
    plot_df = results_df.dropna(subset=["RR@20"]).copy()
    if plot_df.empty:
        return

    if "series_name" in plot_df.columns and plot_df["series_name"].nunique() > 1:
        pivot = plot_df.pivot_table(
            index="query_type",
            columns="series_name",
            values="RR@20",
            aggfunc="mean",
        ).sort_index()
        save_table(pivot.reset_index(), output_dir / "query_type_rr20_profile.csv")

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("RR@20 by query type and method/source")
        ax.set_ylabel("RR@20")
        ax.set_xlabel("query_type")
        ax.legend(title="method | source", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        save_figure(fig, output_dir / "query_type_rr20_profile.png")
    else:
        grouped = (
            plot_df.groupby("query_type", as_index=False)["RR@20"]
            .mean()
            .sort_values("RR@20", ascending=False)
        )
        save_table(grouped, output_dir / "query_type_rr20_profile.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(grouped["query_type"], grouped["RR@20"])
        ax.set_title("RR@20 by query type")
        ax.set_ylabel("RR@20")
        ax.set_xlabel("query_type")
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        save_figure(fig, output_dir / "query_type_rr20_profile.png")


def build_viz_df(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    viz_df = summary_df.copy()
    join_priority = [
        "series_name",
        "method_name",
        "model_name",
        "experiment",
        "config_slug",
        "chunk_size",
        "chunk_overlap",
        "include_title_chunk",
        "top_k",
    ]
    join_keys = [col for col in join_priority if col in summary_df.columns and col in results_df.columns]
    if join_keys and "RR@20" in results_df.columns:
        rr_by_run = (
            results_df.dropna(subset=["RR@20"])
            .groupby(join_keys, dropna=False)["RR@20"]
            .mean()
            .reset_index(name="rr20_mean")
        )
        viz_df = viz_df.merge(rr_by_run, on=join_keys, how="left")

    if "model_name" in viz_df.columns:
        viz_df["model_selector"] = (
            viz_df["series_name"].astype(str)
            + " | "
            + viz_df["model_name"].fillna("unknown").astype(str)
        )
    else:
        viz_df["model_selector"] = viz_df["series_name"].astype(str)
    return viz_df


def plot_model_differences(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_dir: Path,
    model_selector: Optional[str] = None,
    x_axis: Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> None:
    viz_df = build_viz_df(summary_df, results_df)
    if viz_df.empty:
        return

    model_options = sorted(viz_df["model_selector"].dropna().unique().tolist())
    if not model_options:
        return
    selected_model = model_selector or model_options[0]
    sub = viz_df[viz_df["model_selector"] == selected_model].copy()
    if sub.empty:
        return

    x_candidates = [
        col
        for col in ("chunk_size", "chunk_overlap", "top_k", "include_title_chunk", "config_slug", "evaluated_at")
        if col in sub.columns and sub[col].nunique(dropna=False) > 1
    ]
    selected_x = x_axis or (x_candidates[0] if x_candidates else "evaluated_at")
    metric_candidates = [
        col
        for col in ("average_score", "average_rank", "rr20_mean", "num_queries_total", "num_query_types")
        if col in sub.columns and pd.api.types.is_numeric_dtype(sub[col])
    ]
    selected_metrics = [metric for metric in (metrics or metric_candidates[:2]) if metric in metric_candidates]
    if not selected_metrics:
        return

    dedupe_cols = [selected_x] + selected_metrics + [
        col for col in ("chunk_size", "chunk_overlap", "top_k", "include_title_chunk", "config_slug") if col in sub.columns
    ]
    sub = sub.drop_duplicates(subset=[col for col in dedupe_cols if col in sub.columns])
    if selected_x == "evaluated_at" and "evaluated_at" in sub.columns:
        sub = sub.sort_values("evaluated_at")
        x_values = sub["evaluated_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("unknown_time")
    else:
        sub = sub.sort_values(selected_x, kind="stable")
        x_values = sub[selected_x].astype(str)

    preview_cols = [col for col in [selected_x, "chunk_size", "chunk_overlap", "top_k", "include_title_chunk", "config_slug"] if col in sub.columns]
    preview_cols += selected_metrics
    save_table(sub[preview_cols].reset_index(drop=True), output_dir / "model_differences.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    for metric in selected_metrics:
        ax.plot(x_values, sub[metric], marker="o", label=metric)
    ax.set_title(f"Run differences for {selected_model}")
    ax.set_xlabel(selected_x)
    ax.set_ylabel("metric value")
    ax.legend(title="field")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir / "model_differences.png")


def plot_query_type_between_runs(
    results_df: pd.DataFrame,
    output_dir: Path,
    model_selector: Optional[str] = None,
    run_axis: Optional[str] = None,
) -> None:
    if results_df.empty or "RR@20" not in results_df.columns:
        return
    qt_df = results_df.dropna(subset=["RR@20"]).copy()
    if qt_df.empty:
        return

    if "model_name" in qt_df.columns:
        qt_df["model_selector"] = (
            qt_df["series_name"].astype(str)
            + " | "
            + qt_df["model_name"].fillna("unknown").astype(str)
        )
    else:
        qt_df["model_selector"] = qt_df["series_name"].astype(str)

    model_options = sorted(qt_df["model_selector"].dropna().unique().tolist())
    if not model_options:
        return
    selected_model = model_selector or model_options[0]
    sub = qt_df[qt_df["model_selector"] == selected_model].copy()
    if sub.empty:
        return

    run_axis_candidates = [
        col
        for col in ("config_slug", "chunk_size", "chunk_overlap", "top_k", "include_title_chunk", "evaluated_at")
        if col in sub.columns and sub[col].nunique(dropna=False) > 1
    ]
    selected_axis = run_axis or (run_axis_candidates[0] if run_axis_candidates else "evaluated_at")

    if selected_axis == "evaluated_at" and "evaluated_at" in sub.columns:
        sub["run_label"] = sub["evaluated_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("unknown_time")
    else:
        sub["run_label"] = sub[selected_axis].astype(str)

    pivot = sub.pivot_table(
        index="query_type",
        columns="run_label",
        values="RR@20",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return

    save_table(pivot.reset_index(), output_dir / "query_type_between_runs.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"RR@20 by query type across runs for {selected_model}")
    ax.set_ylabel("RR@20")
    ax.set_xlabel("query_type")
    ax.legend(title=selected_axis, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir / "query_type_between_runs.png")


def plot_runs_in_scope(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_dir: Path,
    scope_field: Optional[str] = None,
    scope_value: Optional[str] = None,
    query_category: str = "__all__",
) -> None:
    comparison_summary_df = summary_df.copy()
    comparison_results_df = results_df.copy()
    comparison_summary_df["run_label"] = build_run_labels(comparison_summary_df)
    comparison_results_df["run_label"] = build_run_labels(comparison_results_df)

    scope_options: list[tuple[str, str]] = []
    if "source_name" in comparison_summary_df.columns:
        scope_options.extend([("source_name", value) for value in sorted(comparison_summary_df["source_name"].dropna().astype(str).unique().tolist())])
    if "model_name" in comparison_summary_df.columns:
        scope_options.extend([("model_name", value) for value in sorted(comparison_summary_df["model_name"].dropna().astype(str).unique().tolist())])
    if not scope_options:
        return

    selected_field, selected_value = (
        (scope_field, scope_value) if scope_field and scope_value else scope_options[0]
    )

    if query_category == "__all__":
        sub = comparison_summary_df[comparison_summary_df[selected_field].astype(str) == str(selected_value)].copy()
        metric_col = "average_score"
        title = f"All runs for {selected_field}={selected_value} (overall average_score)"
    else:
        sub = comparison_results_df[
            (comparison_results_df[selected_field].astype(str) == str(selected_value))
            & (comparison_results_df["query_type"].astype(str) == str(query_category))
        ].copy()
        metric_col = "RR@20"
        title = f"All runs for {selected_field}={selected_value} | category={query_category}"

    if sub.empty or metric_col not in sub.columns:
        return
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True)
    if sub.empty:
        return

    display_cols = [
        col
        for col in (
            "run_label",
            "method_name",
            "source_name",
            "model_name",
            "query_type",
            metric_col,
            "chunk_size",
            "chunk_overlap",
            "include_title_chunk",
            "text_source",
            "config_slug",
            "evaluated_at",
        )
        if col in sub.columns
    ]
    save_table(
        sub[display_cols].sort_values(metric_col, ascending=False).reset_index(drop=True),
        output_dir / "compare_runs_in_scope.csv",
    )

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(sub))))
    ax.barh(sub["run_label"], sub[metric_col], color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(metric_col)
    ax.set_ylabel("run")
    fig.tight_layout()
    save_figure(fig, output_dir / "compare_runs_in_scope.png")


def load_qrels_long_from_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    requested_sources = set()
    if "qrels_source" in run_df.columns:
        requested_sources.update(value for value in run_df["qrels_source"].dropna().astype(str).tolist() if value)
    if not requested_sources:
        requested_sources.add("google_sheet")

    parts = []
    for qrels_source in sorted(requested_sources):
        try:
            query_type_frames = load_qrels_dataset(qrels_source=qrels_source)
        except Exception:
            continue
        for query_type, frame in query_type_frames.items():
            qrels_part = frame["qrels"][["query_id", "doc_id", "relevance"]].copy()
            qrels_part["qrels_query_type"] = query_type
            qrels_part["source_name"] = qrels_source
            parts.append(qrels_part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_query_model_matrix(
    run_df: pd.DataFrame,
    output_dir: Path,
    source_name: Optional[str] = None,
    query_category: str = "__all__",
    selected_models: Optional[list[str]] = None,
) -> None:
    if run_df.empty:
        return
    qrels_long_df = load_qrels_long_from_runs(run_df)
    if qrels_long_df.empty:
        return

    run_work_df = run_df.copy()
    run_work_df["score"] = pd.to_numeric(run_work_df["score"], errors="coerce")
    run_work_df = run_work_df.dropna(subset=["score"]).copy()
    run_work_df["run_label"] = build_run_labels(run_work_df, include_method=False)
    run_work_df["model_selector"] = run_work_df["method_name"].fillna("unknown").astype(str)
    if "model_name" in run_work_df.columns:
        run_work_df["model_selector"] = (
            run_work_df["model_selector"].astype(str)
            + " | "
            + run_work_df["model_name"].fillna("unknown").astype(str)
        )

    run_id_cols = [
        col
        for col in (
            "method_name",
            "source_name",
            "model_name",
            "experiment",
            "config_slug",
            "chunk_size",
            "chunk_overlap",
            "include_title_chunk",
            "text_source",
            "evaluated_at",
            "file_source",
            "run_label",
            "model_selector",
        )
        if col in run_work_df.columns
    ]

    run_work_df = run_work_df.sort_values(
        run_id_cols + ["query_id", "score"],
        ascending=[True] * len(run_id_cols) + [True, False],
    )
    run_work_df["rank"] = run_work_df.groupby(run_id_cols + ["query_id"], dropna=False)["score"].rank(
        method="first",
        ascending=False,
    )
    run_join_cols = run_id_cols if "source_name" in run_id_cols else run_id_cols + ["source_name"]

    relevant_df = qrels_long_df.copy()
    relevant_df["relevance"] = pd.to_numeric(relevant_df["relevance"], errors="coerce").fillna(0)
    relevant_df = relevant_df[relevant_df["relevance"] > 0].copy()

    hit_df = run_work_df.merge(
        relevant_df[["query_id", "doc_id", "qrels_query_type", "source_name"]],
        on=["query_id", "doc_id", "source_name"],
        how="inner",
    )
    best_hit_df = (
        hit_df.groupby(run_join_cols + ["query_id", "qrels_query_type"], dropna=False)["rank"]
        .min()
        .reset_index()
    )
    best_hit_df["RR@20"] = best_hit_df["rank"].apply(
        lambda value: 1.0 / value if pd.notna(value) and value <= 20 else 0.0
    )

    query_catalog_df = relevant_df[["query_id", "qrels_query_type", "source_name"]].drop_duplicates()
    run_catalog_df = run_work_df[run_join_cols].drop_duplicates()
    all_pairs_df = run_catalog_df.merge(query_catalog_df, on="source_name", how="inner")
    query_rr_df = all_pairs_df.merge(
        best_hit_df[run_join_cols + ["query_id", "qrels_query_type", "RR@20"]],
        on=run_join_cols + ["query_id", "qrels_query_type"],
        how="left",
    )
    query_rr_df["RR@20"] = query_rr_df["RR@20"].fillna(0.0)

    source_options = sorted(query_rr_df["source_name"].dropna().astype(str).unique().tolist())
    if not source_options:
        return
    chosen_source = source_name or source_options[0]
    sub = query_rr_df[query_rr_df["source_name"].astype(str) == str(chosen_source)].copy()
    if query_category != "__all__":
        sub = sub[sub["qrels_query_type"].astype(str) == str(query_category)].copy()
    if selected_models:
        sub = sub[sub["model_selector"].isin(selected_models)].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="query_id",
        columns="model_selector",
        values="RR@20",
        aggfunc="mean",
        fill_value=0.0,
    )
    if pivot.empty:
        return

    save_table(pivot.reset_index(), output_dir / "query_model_matrix.csv")
    fig_width = max(8, 1.4 * len(pivot.columns))
    fig_height = max(6, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Per-query RR@20 matrix | source={chosen_source} | category={query_category}")
    ax.set_xlabel("model / method")
    ax.set_ylabel("query")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    color_bar = fig.colorbar(image, ax=ax)
    color_bar.set_label("RR@20")
    fig.tight_layout()
    save_figure(fig, output_dir / "query_model_matrix.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation plots without the notebook.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for generated plots/tables. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--compare-scope-field",
        choices=("source_name", "model_name"),
        help="Scope field for the compare-runs-in-scope plot.",
    )
    parser.add_argument(
        "--compare-scope-value",
        help="Specific source/model value for the compare-runs-in-scope plot.",
    )
    parser.add_argument(
        "--compare-category",
        default="__all__",
        help="Query category for compare-runs-in-scope. Use __all__ for overall scores.",
    )
    parser.add_argument(
        "--model-selector",
        help="Specific model selector for the model-differences and query-type-between-runs plots.",
    )
    parser.add_argument(
        "--model-x-axis",
        help="X-axis field for the model-differences plot.",
    )
    parser.add_argument(
        "--model-metrics",
        help="Comma-separated metrics for the model-differences plot.",
    )
    parser.add_argument(
        "--run-axis",
        help="Run axis for the query-type-between-runs plot.",
    )
    parser.add_argument(
        "--matrix-source",
        help="Source for the query-model matrix plot.",
    )
    parser.add_argument(
        "--matrix-category",
        default="__all__",
        help="Query category for the query-model matrix plot.",
    )
    parser.add_argument(
        "--matrix-models",
        help="Comma-separated model selectors to include in the query-model matrix plot.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, results_df, run_df = prepare_frames()

    plot_average_score_over_time(summary_df, output_dir)
    plot_best_run_per_method(summary_df, output_dir)
    plot_query_type_profile(results_df, output_dir)
    plot_model_differences(
        summary_df,
        results_df,
        output_dir,
        model_selector=args.model_selector,
        x_axis=args.model_x_axis,
        metrics=[value.strip() for value in args.model_metrics.split(",") if value.strip()]
        if args.model_metrics
        else None,
    )
    plot_query_type_between_runs(
        results_df,
        output_dir,
        model_selector=args.model_selector,
        run_axis=args.run_axis,
    )
    plot_runs_in_scope(
        summary_df,
        results_df,
        output_dir,
        scope_field=args.compare_scope_field,
        scope_value=args.compare_scope_value,
        query_category=args.compare_category,
    )
    plot_query_model_matrix(
        run_df,
        output_dir,
        source_name=args.matrix_source,
        query_category=args.matrix_category,
        selected_models=[value.strip() for value in args.matrix_models.split(",") if value.strip()]
        if args.matrix_models
        else None,
    )

    print(f"Saved plots and tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
