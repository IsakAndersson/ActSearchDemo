import ir_measures
from ir_measures import *
import pandas as pd
from typing import Callable, List, Tuple

""" 
Format för run:
run = pd.DataFrame([
    {'query_id': "Q0", 'doc_id': "D0", 'score': 1.2},
    {'query_id': "Q0", 'doc_id': "D1", 'score': 1.0},
    {'query_id': "Q1", 'doc_id': "D0", 'score': 2.4},
    {'query_id': "Q1", 'doc_id': "D3", 'score': 3.6},
])"""

def evaluate_system(search_function, k):
    df = load_data()

	#tar ut rätt kolumner för qrels 
    doc_ids = df["Titel på rätt dokument"]
    query_type_list = ["Case-beskrivning", "Titel på rätt dokument", "Titel utan stor bokstav", "2 nyckelord", 
                       "2 nyckelord med stavfel", "2 nyckelord med synonymer", "Längre sökterm"]
    query_types_cols = df[query_type_list]

	#todo: lägg in for-loop som loopar igenom varje sökfunktion och kör evaluate
    results_by_query, average_rank, average_score = evaluate(search_function, k, doc_ids, query_types_cols)
    print_results(results_by_query, average_score, average_rank)
    save_results_to_csv(results_by_query, average_score, average_rank)
    return results_by_query, average_rank, average_score

def save_results_to_csv(detailed_results, average_score, average_rank):
    # 1) Detailed per-query-type metrics
    detailed_results.to_csv("evaluation_results.csv", index=False)

    # 2) Overall summary metrics
    pd.DataFrame([{
        "average_rank": average_rank,
        "average_score": average_score
    }]).to_csv("evaluation_summary.csv", index=False)

def evaluate(search_function, k, doc_ids, query_types_cols):
	rows = []
	#loopar över varje query type och beräknar metrics 
	for query_type in query_types_cols.columns:
		queries = query_types_cols[query_type]
		rr20, average_rank = evaluate_query_type(doc_ids, queries, search_function, k)
		rows.append({"query_type": query_type, "RR@20": rr20, "average_rank": average_rank})
		
	results_by_query_type = pd.DataFrame(rows)
	average_score = results_by_query_type["RR@20"].mean()
	average_rank = results_by_query_type["average_rank"].mean()
 
	return (results_by_query_type, average_rank, average_score)

def evaluate_query_type(doc_ids, queries, search_function, k):
    qrels = create_qrels(doc_ids, queries)
    run = build_run_df(queries, search_function, k)
    scores = calculate_metrics(qrels, run)
    rr20 = list(scores.values())[0]
    avg_rank = calculate_average_rank(qrels, run)
    return rr20, avg_rank

def build_run_df(
    queries: pd.Series,
    search_fn: Callable[[str, int], List[Tuple[str, float]]],
    top_k: int = 20
) -> pd.DataFrame:
    rows = []
    for query in queries:
        results = search_fn(query, top_k)
        # results = [(doc_id, score), ...]
        for doc_id, score in results:
            rows.append({
                "doc_id": str(doc_id),   # titel
                "query_id": query,
                "score": float(score)
            })
    run = pd.DataFrame(rows)
    return run

def load_data():
    #Läser in data från Google Sheets: https://docs.google.com/spreadsheets/d/1_bjtkzZyc-59dAf8c0LlnNFqXl9rB01LGJF3HFoKo6I/edit?usp=sharing
	sheet_id = "1_bjtkzZyc-59dAf8c0LlnNFqXl9rB01LGJF3HFoKo6I"
	sheet_name = "Blad1"  # ändra om annan flik

	url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
	df = pd.read_csv(url)
 
	#spara csv-filen i samma mapp som denna fil, för att kunna titta hur den ser ut
	df.to_csv('qrels.csv', index=False)
 
	return df

""" Format för qrels:
qrels = pd.DataFrame([
    {'query_id': "Q0", 'doc_id': "D0", 'relevance': 0},
    {'query_id': "Q0", 'doc_id': "D1", 'relevance': 1},
    {'query_id': "Q1", 'doc_id': "D0", 'relevance': 0},
    {'query_id': "Q1", 'doc_id': "D3", 'relevance': 2},
])"""
def create_qrels(doc_ids, query_ids):

	qrels = pd.DataFrame({
		"query_id": query_ids,
		"doc_id": doc_ids,
		"relevance": 1
	})
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

def print_results(results, average_score, average_rank):
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'
 
	print("\n---------------" + BOLD + " Utvärderingsresultat " + END +  "---------------")
	print(results.to_string(index=False))
	print("\n Genomsnittlig RR@20: " + str(average_score))
	print(" Genomsnittlig average rank: " + str(average_rank) + "\n")
	print(UNDERLINE + "Förklaring av mått:" + END 
       + "\nRR@20 (Mean Reciprocal Rank): Mäter hur högt upp i resultatlistan det rätta svaret placeras i topp 20. Beräknas som 1/positionen för det rätta dokumentet. \n1.0 = Perfekt score (högst upp varje gång) \n0.0 = Utanför topp 20 varje gång\n"
       + "\nAverage rank: Genomsnittlig position av rätt dokument. Ignorerar resultat utanför top k.\n"
       )
