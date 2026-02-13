import argparse
from typing import Callable, List, Tuple

import ir_measures
import pandas as pd
from ir_measures import RR

from search_adapter import bm25_search, dense_search, hybrid_search

SearchFn = Callable[[str, int], List[Tuple[str, float]]]


QUERY_TYPE_LIST = [
    "Case-beskrivning",
    "Titel på rätt dokument",
    "Titel utan stor bokstav",
    "2 nyckelord",
    "2 nyckelord med stavfel",
    "2 nyckelord med synonymer",
    "Längre sökterm",
]


def load_data() -> pd.DataFrame:
    # Google Sheet: https://docs.google.com/spreadsheets/d/1_bjtkzZyc-59dAf8c0LlnNFqXl9rB01LGJF3HFoKo6I/edit?usp=sharing
    sheet_id = "1_bjtkzZyc-59dAf8c0LlnNFqXl9rB01LGJF3HFoKo6I"
    sheet_name = "Blad1"

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    df.to_csv("qrels.csv", index=False)
    return df


def create_qrels(doc_ids: pd.Series, query_ids: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": query_ids,
            "doc_id": doc_ids,
            "relevance": 1,
        }
    )


def create_run(query_ids: pd.Series, query_texts: pd.Series, search_fn: SearchFn, top_k: int) -> pd.DataFrame:
    rows = []
    for query_id, query_text in zip(query_ids.tolist(), query_texts.tolist()):
        if not isinstance(query_text, str) or not query_text.strip():
            continue

        results = search_fn(query_text, top_k)
        for doc_id, score in results:
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "score": float(score),
                }
            )

    run = pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])
    run.to_csv("run_test.csv", index=False)
    return run


def calculate_metrics(qrels: pd.DataFrame, run: pd.DataFrame):
    return ir_measures.calc_aggregate([RR @ 20], qrels, run)


def calculate_average_rank(qrels: pd.DataFrame, run: pd.DataFrame) -> float:
    ranks = []
    results_list = list(ir_measures.iter_calc([RR @ 20], qrels, run))

    for score in results_list:
        if score.value > 0:
            ranks.append(1 / score.value)

    return (sum(ranks) / len(ranks)) if ranks else 0.0


def print_results(results: pd.DataFrame, average_score: float, average_rank_df: float, model: str) -> None:
    bold = "\033[1m"
    underline = "\033[4m"
    end = "\033[0m"

    print("\n---------------" + bold + " Utvarderingsresultat " + end + "---------------")
    print("\nModell: " + model + "\n")
    print(results.to_string(index=False))
    print("\n Genomsnittlig RR@20: " + str(average_score))
    print(" Genomsnittlig position av ratt dokument (average rank): " + str(average_rank_df) + "\n")
    print(
        underline
        + "Forklaring av matt:"
        + end
        + "\nRR@20 (Mean Reciprocal Rank): Matar hur hogt upp i resultatlistan det ratta svaret placeras i topp 20."
        + " Beraknas som 1/positionen for det ratta dokumentet."
        + "\n1.0 = Perfekt score (hogst upp varje gang)"
        + "\n0.0 = Utanfor topp 20 varje gang\n"
    )


def resolve_search_fn(method: str) -> SearchFn:
    if method == "bm25":
        return bm25_search
    if method == "dense":
        return dense_search
    if method == "hybrid":
        return hybrid_search
    raise ValueError(f"Unknown method '{method}'. Use bm25, dense, or hybrid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IR evaluation over Docplus retrieval methods.")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    df = load_data()

    doc_ids = df["Titel på rätt dokument"]
    query_types_cols = df[QUERY_TYPE_LIST]

    aggregate_results_list = []
    average_rank_list = []

    search_fn = resolve_search_fn(args.method)

    for query_type in query_types_cols.columns:
        qrels = create_qrels(doc_ids=doc_ids, query_ids=query_types_cols[query_type])
        run = create_run(
            query_ids=query_types_cols[query_type],
            query_texts=query_types_cols[query_type],
            search_fn=search_fn,
            top_k=args.top_k,
        )

        scores = calculate_metrics(qrels, run)
        aggregate_results_list.append(list(scores.values())[0])

        average_rank = calculate_average_rank(qrels, run)
        average_rank_list.append(average_rank)

    rows = zip(QUERY_TYPE_LIST, aggregate_results_list, average_rank_list)
    results = pd.DataFrame(rows, columns=["query_type", "RR@20", "average_rank"])

    average_score_df = results.iloc[:, 1].mean()
    average_rank = results.iloc[:, 2].mean()

    print_results(results, average_score_df, average_rank, model=args.method)


if __name__ == "__main__":
    main()
