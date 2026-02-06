import ir_measures
from ir_measures import *
from qrels import qrels
from run import run

#metrics: 
#Judged@k: Percentage of results in the top k (cutoff) results that have relevance judgments.
#nDCG: The normalized Discounted Cumulative Gain (nDCG)
#P: Precision
#P(rel=2): Precision in the top k, considering only documents with relevance level 2 as relevant.

#returns a python dictionary 
results = ir_measures.calc_aggregate([nDCG@10, P@5, P(rel=2)@5, Judged@10], qrels, run)

print("--- Utvärderingsresultat ---")
print(results)