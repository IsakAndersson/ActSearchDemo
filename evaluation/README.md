# Evaluation Module

This module evaluates retrieval quality for Docplus search methods using labeled queries from a Google Sheet.

## Metrics

`evaluation.py` computes:

- `RR@20` per query type
- `average_rank` per query type
- global averages across query types

Note:

- `RR@20` includes misses as `0` (standard behavior).
- `average_rank` only averages ranks for hits within top `k` (misses are excluded).

## Expected Search Function Format

The evaluator expects a search function with this interface:

```python
search_function(query: str, top_k: int) -> List[Tuple[str, float]]
```

Each tuple must be:

```python
(doc_id, score)
```

where `doc_id` should match the values in the sheet column `Titel på rätt dokument`.

## Public API

Main entry point:

```python
evaluate_system(search_function, k, metadata=None)
```

Parameters:

- `search_function`: function with signature above
- `k`: top-k cutoff used in retrieval and metrics
- `metadata`: optional flat dictionary added to CSV outputs

Return value:

```python
(results_by_query_type, average_rank, average_score)
```

where:

- `results_by_query_type` is a DataFrame with columns `query_type`, `RR@20`, `average_rank`
- `average_rank` is mean of per-query-type average ranks
- `average_score` is mean of per-query-type `RR@20`

Concrete shape:

```python
Tuple[pd.DataFrame, float, float]
```

## Metadata Rules

`metadata` must be a flat dictionary:

- keys must be `str`
- values must be non-nested (no `dict`, `list`, `tuple`, `set`)

Example:

```python
metadata = {
    "method": "hybrid",
    "experiment": "baseline_v1",
    "model_name": "KBLab/bert-base-swedish-cased"
}
```

## How to Run

Run from repository root in Python:

```python
from evaluation.evaluation import evaluate_system
from evaluation.search_adapter import bm25_search, dense_search, hybrid_search

# BM25
evaluate_system(
    search_function=bm25_search,
    k=20,
    metadata={"method": "bm25", "experiment": "baseline"}
)

# Dense
evaluate_system(
    search_function=dense_search,
    k=20,
    metadata={"method": "dense", "experiment": "baseline"}
)

# Hybrid
evaluate_system(
    search_function=hybrid_search,
    k=20,
    metadata={"method": "hybrid", "experiment": "baseline"}
)
```

Run from terminal with flags:

```bash
./.venv/bin/python evaluation/evaluation.py --method hybrid --top-k 20 --meta experiment=baseline --meta run_tag=exp1
./.venv/bin/python evaluation/evaluation.py --method bm25 --top-k 20 --meta experiment=bm25_baseline
```

CLI flags:

- `--method`: `bm25`, `dense`, or `hybrid` (default: `hybrid`)
- `--top-k`: integer > 0 (default: `20`)
- `--meta KEY=VALUE`: optional metadata entry, repeat for multiple fields

`--meta` values are parsed as:

- `true`/`false` -> boolean
- integer text -> int
- float text -> float
- otherwise -> string

## Output Files

Evaluation appends rows to these files (it does not overwrite):

- `qrels.csv`: latest downloaded sheet snapshot
- `evaluation_results.csv`: per-query-type metrics
- `evaluation_summary.csv`: one summary row per evaluation run
- `evaluation_run.csv`: full run rows (`query_id`, `doc_id`, `score`, `query_type`)

Automatic run metadata written to outputs:

- `evaluated_at_cet` (CET timestamp)
- `top_k`
- `num_queries_total`
- `num_query_types`

Any user-provided `metadata` is also written to the CSV outputs above.
Run metadata is also printed in the terminal output for each evaluation run.

## Search Adapter

`search_adapter.py` provides ready-to-use adapters:

- `bm25_search`
- `dense_search`
- `hybrid_search`
- `search` (default = hybrid)

Optional environment variables used by `search_adapter.py`:

- `DOCPLUS_PARSED_DIR`
- `DOCPLUS_INDEX_PATH`
- `DOCPLUS_METADATA_PATH`
- `DOCPLUS_MODEL_NAME`
- `DOCPLUS_DEVICE`

## Prerequisites

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data/index prerequisites:

- Parsed docs in `flask/output/parsed`
- Dense index in `flask/output/vector_index/docplus.faiss`
- Dense metadata in `flask/output/vector_index/docplus_metadata.jsonl`
