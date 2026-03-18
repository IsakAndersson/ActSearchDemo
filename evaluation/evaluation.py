import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ir_measures
import pandas as pd
from ir_measures import *

try:
    from .doc_id import normalize_doc_id
except ImportError:
    from doc_id import normalize_doc_id

EVALUATION_DIR = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)

""" 
Format för run:
run = pd.DataFrame([
    {'query_id': "Q0", 'doc_id': "D0", 'score': 1.2},
    {'query_id': "Q0", 'doc_id': "D1", 'score': 1.0},
    {'query_id': "Q1", 'doc_id': "D0", 'score': 2.4},
    {'query_id': "Q1", 'doc_id': "D3", 'score': 3.6},
])"""

def _validate_flat_metadata(metadata):
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a flat dictionary")

    validated = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"metadata['{key}'] must be a flat value, not nested")
        validated[key] = value
    return validated

def evaluate_system(
    search_function,
    k,
    metadata=None,
    output_dir: Optional[str] = None,
    aggregate_dir: Optional[str] = None,
    qrels_source: str = "google_sheet",
    qrels_path: Optional[str] = None,
):
    """
    Run evaluation and append outputs to CSV files.

    Returns:
        tuple[pd.DataFrame, float, float]:
            (results_by_query_type, average_rank, average_score)
    """
    print("[evaluation] Loading qrels data...", flush=True)
    query_type_frames = load_qrels_dataset(qrels_source=qrels_source, qrels_path=qrels_path)
    total_qrels_rows = sum(len(frame["qrels"]) for frame in query_type_frames.values())
    print(
        f"[evaluation] Loaded {total_qrels_rows} qrels rows across {len(query_type_frames)} query type(s).",
        flush=True,
    )

    #todo: lägg in for-loop som loopar igenom varje sökfunktion och kör evaluate
    print("[evaluation] Running retrieval evaluation...", flush=True)
    results_by_query, average_rank, average_score, run_df = evaluate(
        search_function,
        k,
        query_type_frames=query_type_frames,
        return_runs=True,
    )
    cet = timezone(timedelta(hours=1), name="CET")
    run_metadata = {
        "evaluated_at_cet": datetime.now(cet).isoformat(timespec="seconds"),
        "top_k": int(k),
        "num_queries_total": int(
            sum(frame["queries"]["query_id"].nunique() for frame in query_type_frames.values())
        ),
        "num_query_types": int(len(query_type_frames)),
        "qrels_source": qrels_source,
        "data_source": qrels_source,
    }
    if qrels_path:
        run_metadata["qrels_path"] = str(Path(qrels_path).resolve())
    run_metadata.update(_validate_flat_metadata(metadata))
    print_results(results_by_query, average_score, average_rank, run_metadata=run_metadata)
    save_results_to_csv(
        results_by_query,
        average_score,
        average_rank,
        run_metadata,
        run_df=run_df,
        output_dir=output_dir,
        aggregate_dir=aggregate_dir,
    )
    return results_by_query, average_rank, average_score

def save_results_to_csv(
    detailed_results,
    average_score,
    average_rank,
    run_metadata=None,
    run_df=None,
    output_dir: Optional[str] = None,
    aggregate_dir: Optional[str] = None,
):
    def append_df_to_csv(df: pd.DataFrame, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = True
        if output_path.exists() and output_path.stat().st_size > 0:
            with output_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().lstrip("\ufeff").strip()
            expected_header = ",".join(map(str, df.columns))
            needs_header = first_line != expected_header
        df.to_csv(output_path, mode="a", header=needs_header, index=False)

    run_output_dir = Path(output_dir).resolve() if output_dir else EVALUATION_DIR
    targets = [run_output_dir]
    if aggregate_dir:
        aggregate_path = Path(aggregate_dir).resolve()
        if aggregate_path not in targets:
            targets.append(aggregate_path)

    # 1) Detailed per-query-type metrics
    detailed_results_with_metadata = detailed_results.copy()
    if run_metadata:
        for key, value in run_metadata.items():
            detailed_results_with_metadata[key] = value
    for target in targets:
        append_df_to_csv(detailed_results_with_metadata, target / "evaluation_results.csv")

    # 2) Overall summary metrics
    summary_row = {
        "average_rank": average_rank,
        "average_score": average_score
    }
    if run_metadata:
        summary_row.update(run_metadata)
    for target in targets:
        append_df_to_csv(pd.DataFrame([summary_row]), target / "evaluation_summary.csv")

    # 3) Full run rows (query_id, doc_id, score) for reproducibility/debugging
    if run_df is not None and not run_df.empty:
        run_with_metadata = run_df.copy()
        if run_metadata:
            for key, value in run_metadata.items():
                run_with_metadata[key] = value
        for target in targets:
            append_df_to_csv(run_with_metadata, target / "evaluation_run.csv")

def evaluate(search_function, k, doc_ids=None, query_types_cols=None, return_runs=False, query_type_frames=None):
	rows = []
	all_runs = []
	if query_type_frames is None:
		if doc_ids is None or query_types_cols is None:
			raise ValueError("Either query_type_frames or both doc_ids and query_types_cols are required.")
		query_type_frames = _build_query_type_frames_from_sheet(doc_ids, query_types_cols)

	query_type_names = list(query_type_frames.keys())
	query_type_count = len(query_type_names)
	#loopar över varje query type och beräknar metrics 
	for query_type_index, query_type in enumerate(query_type_names, start=1):
		print(
			f"[evaluation] Query type {query_type_index}/{query_type_count}: {query_type}",
			flush=True,
		)
		query_frame = query_type_frames[query_type]
		queries = query_frame["queries"]["query_id"]
		qrels = query_frame["qrels"]
		if return_runs:
			rr20, average_rank, run_df = evaluate_query_type(
				queries=queries,
				search_function=search_function,
				k=k,
				qrels=qrels,
				return_run=True,
				progress_label=query_type,
			)
		else:
			rr20, average_rank = evaluate_query_type(
				queries=queries,
				search_function=search_function,
				k=k,
				qrels=qrels,
				progress_label=query_type,
			)
			run_df = None
		print(
			f"[evaluation] Done query type '{query_type}'. RR@20={rr20:.4f}, avg_rank={average_rank:.2f}",
			flush=True,
		)
		rows.append({"query_type": query_type, "RR@20": rr20, "average_rank": average_rank})
		if return_runs and run_df is not None and not run_df.empty:
			run_df = run_df.copy()
			run_df["query_type"] = query_type
			all_runs.append(run_df)
		
	results_by_query_type = pd.DataFrame(rows)
	average_score = results_by_query_type["RR@20"].mean()
	average_rank = results_by_query_type["average_rank"].mean()

	if return_runs:
		combined_run = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame(
			columns=["doc_id", "query_id", "score", "query_type"]
		)
		return (results_by_query_type, average_rank, average_score, combined_run)

	return (results_by_query_type, average_rank, average_score)

def evaluate_query_type(
	queries,
	search_function,
	k,
	qrels=None,
	doc_ids=None,
	return_run=False,
	progress_label: Optional[str] = None,
):
    if qrels is None:
        if doc_ids is None:
            raise ValueError("doc_ids is required when qrels is not provided.")
        qrels = create_qrels(doc_ids, queries)
    run = build_run_df(
		queries,
		search_function,
		k,
		progress_label=progress_label,
	)
    scores = calculate_metrics(qrels, run)
    rr20 = list(scores.values())[0]
    avg_rank = calculate_average_rank(qrels, run)
    if return_run:
        return rr20, avg_rank, run
    return rr20, avg_rank

def build_run_df(
    queries: pd.Series,
    search_fn: Callable[[str, int], List[Tuple[str, float]]],
    top_k: int = 20,
    progress_label: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    total_queries = len(queries)
    for index, query in enumerate(queries, start=1):
        if index == 1 or index % 5 == 0 or index == total_queries:
            prefix = f"{progress_label}: " if progress_label else ""
            print(
                f"[evaluation] {prefix}query {index}/{total_queries}",
                flush=True,
            )
        try:
            results = search_fn(query, top_k)
        except Exception as exc:
            LOG.warning("Search failed for query %r: %s", query, exc)
            results = []
        # results = [(doc_id, score), ...]
        for doc_id, score in results:
            normalized_doc_id = normalize_doc_id(doc_id)
            if not normalized_doc_id:
                continue
            rows.append({
                "doc_id": normalized_doc_id,
                "query_id": query,
                "score": float(score)
            })
    run = pd.DataFrame(rows, columns=["doc_id", "query_id", "score"])
    return run

def load_data():
    #Läser in data från Google Sheets: https://docs.google.com/spreadsheets/d/1_bjtkzZyc-59dAf8c0LlnNFqXl9rB01LGJF3HFoKo6I/edit?usp=sharing
	sheet_id = "1_bjtkzZyc-59dAf8c0LlnNFqXl9rB01LGJF3HFoKo6I"
	sheet_name = "Blad1"  # ändra om annan flik
	local_snapshot = Path(__file__).with_name("qrels.csv")

	url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
	try:
		df = pd.read_csv(url)
		#spara csv-filen i samma mapp som denna fil, för att kunna titta hur den ser ut
		df.to_csv(local_snapshot, index=False)
		return df
	except Exception:
		if local_snapshot.exists():
			return pd.read_csv(local_snapshot)
		raise


def _build_query_type_frames_from_sheet(doc_ids, query_types_cols) -> Dict[str, Dict[str, pd.DataFrame]]:
	frames = {}
	for query_type in query_types_cols.columns:
		queries = query_types_cols[query_type]
		frames[query_type] = {
			"queries": pd.DataFrame({"query_id": queries}),
			"qrels": create_qrels(doc_ids, queries),
		}
	return frames


def _load_google_sheet_query_type_frames() -> Dict[str, Dict[str, pd.DataFrame]]:
	df = load_data()

	#tar ut rätt kolumner för qrels 
	doc_ids = df["Titel på rätt dokument"]
	query_type_list = [
		"Case-beskrivning",
		"Titel på rätt dokument",
		"Titel utan stor bokstav",
		"2 nyckelord",
		"2 nyckelord med stavfel",
		"2 nyckelord med synonymer",
		"Längre sökterm",
	]
	query_types_cols = df[query_type_list]
	return _build_query_type_frames_from_sheet(doc_ids, query_types_cols)


def _load_form_submission_query_type_frames(qrels_path: str) -> Dict[str, Dict[str, pd.DataFrame]]:
	qrels_df = pd.read_csv(qrels_path)
	required_columns = {"query_id", "doc_id", "relevance"}
	missing = required_columns - set(qrels_df.columns)
	if missing:
		raise ValueError(
			f"Qrels CSV is missing required columns: {', '.join(sorted(missing))}"
		)

	qrels_df = qrels_df.copy()
	qrels_df["query_id"] = qrels_df["query_id"].astype(str)
	qrels_df["doc_id"] = qrels_df["doc_id"].astype(str).map(normalize_doc_id)
	qrels_df = qrels_df[qrels_df["doc_id"] != ""]

	query_type_series = (
		qrels_df["query_type"].astype(str)
		if "query_type" in qrels_df.columns
		else pd.Series(["form_submissions"] * len(qrels_df), index=qrels_df.index)
	)

	frames: Dict[str, Dict[str, pd.DataFrame]] = {}
	for query_type, group in qrels_df.groupby(query_type_series, sort=False):
		group = group[["query_id", "doc_id", "relevance"]].reset_index(drop=True)
		queries = pd.DataFrame({"query_id": group["query_id"].drop_duplicates().reset_index(drop=True)})
		frames[str(query_type)] = {
			"queries": queries,
			"qrels": group,
		}
	return frames


def load_qrels_dataset(
	qrels_source: str = "google_sheet",
	qrels_path: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
	if qrels_source == "google_sheet":
		return _load_google_sheet_query_type_frames()
	if qrels_source == "form_submissions":
		if not qrels_path:
			qrels_path = str(EVALUATION_DIR / "qrels_from_form_submissions.csv")
		return _load_form_submission_query_type_frames(qrels_path)
	raise ValueError(f"Unsupported qrels source: {qrels_source}")

""" Format för qrels:
qrels = pd.DataFrame([
    {'query_id': "Q0", 'doc_id': "D0", 'relevance': 0},
    {'query_id': "Q0", 'doc_id': "D1", 'relevance': 1},
    {'query_id': "Q1", 'doc_id': "D0", 'relevance': 0},
    {'query_id': "Q1", 'doc_id': "D3", 'relevance': 2},
])"""
def create_qrels(doc_ids, query_ids):
	normalized_doc_ids = [normalize_doc_id(doc_id) for doc_id in doc_ids]
	qrels = pd.DataFrame({
		"query_id": query_ids,
		"doc_id": normalized_doc_ids,
		"relevance": 1
	})
	qrels = qrels[qrels["doc_id"] != ""]
	return qrels

"""
Beräknar metrics, returnerar en python dictionary

MRR@20: För varje query beräknas: 1/positionen för det rätta dokumentet. 0 poäng om dokumentet hamnar utanför topp 20.
		Tar snittet över alla queries. 
"""
def calculate_metrics(qrels, run):
	metrics = [RR@20]
	results = ir_measures.calc_aggregate(metrics, qrels, run)
	return results

# Average rank for a query type
def calculate_average_rank(qrels, run):
	ranks = []
	results_gen = ir_measures.iter_calc([ir_measures.RR@20], qrels, run)
	results_list = list(results_gen)
 
	for score in results_list:
		if score.value > 0:
			ranks.append(1 / score.value)
   
	avg_rank = sum(ranks) / len(ranks) if ranks else 0
	return avg_rank

def print_results(results, average_score, average_rank, run_metadata=None):
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'
 
	print("\n---------------" + BOLD + " Utvärderingsresultat " + END +  "---------------")
	if run_metadata:
		print("\nRun metadata:")
		for key in sorted(run_metadata.keys()):
			print(f"  {key}: {run_metadata[key]}")
		print()
	print(results.to_string(index=False))
	print("\n Genomsnittlig RR@20: " + str(average_score))
	print(" Genomsnittlig average rank: " + str(average_rank) + "\n")
	print(UNDERLINE + "Förklaring av mått:" + END 
       + "\nRR@20 (Mean Reciprocal Rank): Mäter hur högt upp i resultatlistan det rätta svaret placeras i topp 20. Beräknas som 1/positionen för det rätta dokumentet. \n1.0 = Perfekt score (högst upp varje gång) \n0.0 = Utanför topp 20 varje gång\n"
       + "\nAverage rank: Genomsnittlig position av rätt dokument. Ignorerar resultat utanför top k.\n"
       )

def _parse_cli_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value

def _parse_meta_args(meta_args):
    metadata = {}
    for item in meta_args or []:
        if "=" not in item:
            raise ValueError(f"Invalid --meta '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --meta '{item}'. Key cannot be empty.")
        metadata[key] = _parse_cli_value(value.strip())
    return metadata

def _resolve_search_function(method: str):
    try:
        from .search_adapter import (
            bm25_search,
            docplus_live_search,
            dense_e5_search,
            dense_search,
            hybrid_e5_search,
            hybrid_search,
            sts_live_search,
        )
    except ImportError:
        from search_adapter import (
            bm25_search,
            docplus_live_search,
            dense_e5_search,
            dense_search,
            hybrid_e5_search,
            hybrid_search,
            sts_live_search,
        )

    search_functions = {
        "bm25": bm25_search,
        "dense": dense_search,
        "dense_e5": dense_e5_search,
        "hybrid": hybrid_search,
        "hybrid_e5": hybrid_e5_search,
        "docplus_live": docplus_live_search,
        "sts_live": sts_live_search,
    }
    return search_functions[method]

def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "dense_e5", "hybrid", "hybrid_e5", "docplus_live", "sts_live"],
        default="hybrid",
        help="Search method to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k documents retrieved per query.",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional run metadata. Repeat flag for multiple entries.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory where run CSV files are written.",
    )
    parser.add_argument(
        "--aggregate-dir",
        help="Optional directory that also receives appended aggregate CSV files.",
    )
    parser.add_argument(
        "--qrels-source",
        choices=["google_sheet", "form_submissions"],
        default="google_sheet",
        help="Where to load qrels from.",
    )
    parser.add_argument(
        "--qrels-path",
        help="Optional qrels CSV path when --qrels-source=form_submissions.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.qrels_source == "google_sheet" and args.qrels_path:
        raise ValueError("--qrels-path can only be used with --qrels-source=form_submissions")

    search_function = _resolve_search_function(args.method)
    metadata = _parse_meta_args(args.meta)
    metadata.setdefault("method", args.method)
    evaluate_system(
        search_function=search_function,
        k=args.top_k,
        metadata=metadata,
        output_dir=args.output_dir,
        aggregate_dir=args.aggregate_dir,
        qrels_source=args.qrels_source,
        qrels_path=args.qrels_path,
    )

if __name__ == "__main__":
    main()
