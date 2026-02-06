import pandas as pd
qrels = pd.DataFrame([
    {'query_id': "Q0", 'doc_id': "D0", 'relevance': 0},
    {'query_id': "Q0", 'doc_id': "D1", 'relevance': 1},
    {'query_id': "Q1", 'doc_id': "D0", 'relevance': 0},
    {'query_id': "Q1", 'doc_id': "D3", 'relevance': 2},
])