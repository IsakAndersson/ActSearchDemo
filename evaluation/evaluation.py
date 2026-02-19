import ir_measures
from ir_measures import *
import pandas as pd
import os
import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Tuple
from pathlib import Path

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

def evaluate_system(search_function, k, metadata=None):
    """
    Run evaluation and append outputs to CSV files.

    Returns:
        tuple[pd.DataFrame, float, float]:
            (results_by_query_type, average_rank, average_score)
    """
    df = load_data()

	#tar ut rätt kolumner för qrels 
    doc_ids = df["Titel på rätt dokument"]
    query_type_list = ["Case-beskrivning", "Titel på rätt dokument", "Titel utan stor bokstav", "2 nyckelord", 
                       "2 nyckelord med stavfel", "2 nyckelord med synonymer", "Längre sökterm"]
    query_types_cols = df[query_type_list]

    #todo: lägg in for-loop som loopar igenom varje sökfunktion och kör evaluate
    results_by_query, average_rank, average_score, run_df = evaluate(
        search_function, k, doc_ids, query_types_cols, return_runs=True
    )
    cet = timezone(timedelta(hours=1), name="CET")
    run_metadata = {
        "evaluated_at_cet": datetime.now(cet).isoformat(timespec="seconds"),
        "top_k": int(k),
        "num_queries_total": int(len(doc_ids)),
        "num_query_types": int(len(query_types_cols.columns)),
    }
    run_metadata.update(_validate_flat_metadata(metadata))
    print_results(results_by_query, average_score, average_rank, run_metadata=run_metadata)
    save_results_to_csv(results_by_query, average_score, average_rank, run_metadata, run_df=run_df)
    return results_by_query, average_rank, average_score

def save_results_to_csv(detailed_results, average_score, average_rank, run_metadata=None, run_df=None):
    def append_df_to_csv(df: pd.DataFrame, path: str):
        output_path = EVALUATION_DIR / path
        needs_header = True
        if output_path.exists() and output_path.stat().st_size > 0:
            with output_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().lstrip("\ufeff").strip()
            expected_header = ",".join(map(str, df.columns))
            needs_header = first_line != expected_header
        df.to_csv(output_path, mode="a", header=needs_header, index=False)

    # 1) Detailed per-query-type metrics
    detailed_results_with_metadata = detailed_results.copy()
    if run_metadata:
        for key, value in run_metadata.items():
            detailed_results_with_metadata[key] = value
    append_df_to_csv(detailed_results_with_metadata, "evaluation_results.csv")

    # 2) Overall summary metrics
    summary_row = {
        "average_rank": average_rank,
        "average_score": average_score
    }
    if run_metadata:
        summary_row.update(run_metadata)
    append_df_to_csv(pd.DataFrame([summary_row]), "evaluation_summary.csv")

    # 3) Full run rows (query_id, doc_id, score) for reproducibility/debugging
    if run_df is not None and not run_df.empty:
        run_with_metadata = run_df.copy()
        if run_metadata:
            for key, value in run_metadata.items():
                run_with_metadata[key] = value
        append_df_to_csv(run_with_metadata, "evaluation_run.csv")

def evaluate(search_function, k, doc_ids, query_types_cols, return_runs=False):
	rows = []
	all_runs = []
	#loopar över varje query type och beräknar metrics 
	for query_type in query_types_cols.columns:
		queries = query_types_cols[query_type]
		if return_runs:
			rr20, average_rank, run_df = evaluate_query_type(
				doc_ids, queries, search_function, k, return_run=True
			)
		else:
			rr20, average_rank = evaluate_query_type(doc_ids, queries, search_function, k)
			run_df = None
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

def evaluate_query_type(doc_ids, queries, search_function, k, return_run=False):
    qrels = create_qrels(doc_ids, queries)
    run = build_run_df(queries, search_function, k)
    scores = calculate_metrics(qrels, run)
    rr20 = list(scores.values())[0]
    avg_rank = calculate_average_rank(qrels, run)
    if return_run:
        return rr20, avg_rank, run
    return rr20, avg_rank

def build_run_df(
    queries: pd.Series,
    search_fn: Callable[[str, int], List[Tuple[str, float]]],
    top_k: int = 20
) -> pd.DataFrame:
    rows = []
    for query in queries:
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

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    search_function = _resolve_search_function(args.method)
    metadata = _parse_meta_args(args.meta)
    metadata.setdefault("method", args.method)
    evaluate_system(search_function=search_function, k=args.top_k, metadata=metadata)

if __name__ == "__main__":
    main()
