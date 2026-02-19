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
from evaluation.search_adapter import (
    bm25_search,
    docplus_live_search,
    sts_live_search,
    dense_search,
    dense_e5_search,
    hybrid_search,
    hybrid_e5_search,
)

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

# Dense (E5 large instruct)
evaluate_system(
    search_function=dense_e5_search,
    k=20,
    metadata={"method": "dense_e5", "experiment": "baseline"}
)

# Hybrid (BM25 + E5 large instruct)
evaluate_system(
    search_function=hybrid_e5_search,
    k=20,
    metadata={"method": "hybrid_e5", "experiment": "baseline"}
)

# Live Docplus (queries publikdocplus.regionuppsala.se directly)
evaluate_system(
    search_function=docplus_live_search,
    k=20,
    metadata={"method": "docplus_live", "experiment": "baseline"}
)

# Live STS Docplus search (queries sts.search.datatovalue.se)
evaluate_system(
    search_function=sts_live_search,
    k=20,
    metadata={"method": "sts_live", "experiment": "baseline"}
)
```

Run from terminal with flags:

```bash
./.venv/bin/python evaluation/evaluation.py --method hybrid --top-k 20 --meta experiment=baseline --meta run_tag=exp1
./.venv/bin/python evaluation/evaluation.py --method bm25 --top-k 20 --meta experiment=bm25_baseline
./.venv/bin/python evaluation/evaluation.py --method docplus_live --top-k 20 --meta experiment=docplus_live
./.venv/bin/python evaluation/evaluation.py --method sts_live --top-k 20 --meta experiment=sts_live
```

CLI flags:

- `--method`: `bm25`, `dense`, `dense_e5`, `hybrid`, `hybrid_e5`, `docplus_live`, or `sts_live` (default: `hybrid`)
- `--top-k`: integer > 0 (default: `20`)
- `--meta KEY=VALUE`: optional metadata entry, repeat for multiple fields

`--meta` values are parsed as:

- `true`/`false` -> boolean
- integer text -> int
- float text -> float
- otherwise -> string

## Output Files

Evaluation output behavior:

- `qrels.csv`: latest downloaded sheet snapshot
- `evaluation_results.csv`: per-query-type metrics (appends)
- `evaluation_summary.csv`: one summary row per evaluation run (appends)
- `evaluation_run.csv`: full run rows (`query_id`, `doc_id`, `score`, `query_type`) for the latest run only (overwrites each run)
- `../flask/evaluation_visualization.ipynb`: notebook for plotting summary/results/run CSVs

Automatic run metadata written to outputs:

- `evaluated_at_cet` (CET timestamp)
- `top_k`
- `num_queries_total`
- `num_query_types`

Any user-provided `metadata` is also written to the CSV outputs above.
Run metadata is also printed in the terminal output for each evaluation run.

## Visualization Notebook

Open and run:

```bash
jupyter notebook flask/evaluation_visualization.ipynb
```

If you prefer to run Jupyter explicitly from the shared root env:

```bash
./.venv/bin/python -m jupyter notebook flask/evaluation_visualization.ipynb
```

The notebook reads:

- `evaluation_summary.csv`
- `evaluation_results.csv`
- `evaluation_run.csv`

and visualizes:

- average score over time
- best run per method
- RR@20 by query type
- retrieved score distributions per method

## Search Adapter

`search_adapter.py` provides ready-to-use adapters:

- `bm25_search`
- `dense_search`
- `dense_e5_search`
- `hybrid_search`
- `hybrid_e5_search`
- `docplus_live_search`
- `sts_live_search`
- `search` (default = hybrid)

Optional environment variables used by `search_adapter.py`:

- `DOCPLUS_PARSED_DIR`
- `DOCPLUS_INDEX_PATH`
- `DOCPLUS_METADATA_PATH`
- `DOCPLUS_MODEL_NAME`
- `DOCPLUS_E5_INDEX_PATH`
- `DOCPLUS_E5_METADATA_PATH`
- `DOCPLUS_E5_MODEL_NAME`
- `DOCPLUS_DEVICE`
- `DOCPLUS_LIVE_BASE_URL` (default: `https://publikdocplus.regionuppsala.se/`)
- `DOCPLUS_LIVE_SEARCH_PATH` (default: `/Home/Search`)
- `DOCPLUS_LIVE_TIMEOUT_SECONDS` (default: `20`)
- `DOCPLUS_LIVE_MAX_PAGES` (default: `1`, auto-expanded when `top_k > 20`)
- `DOCPLUS_LIVE_USER_AGENT`
- `DOCPLUS_STS_LIVE_BASE_URL` (default: `https://sts.search.datatovalue.se/`)
- `DOCPLUS_STS_LIVE_SEARCH_PATH` (default: `/`)
- `DOCPLUS_STS_LIVE_TIMEOUT_SECONDS` (default: `20`)
- `DOCPLUS_STS_LIVE_MAX_PAGES` (default: `1`, auto-expanded when `top_k > 20`)
- `DOCPLUS_STS_LIVE_USER_AGENT`

## Prerequisites

Use one shared environment at repository root (`./.venv`).

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
- E5 dense index in `flask/output/vector_index_e5/docplus.faiss`
- E5 dense metadata in `flask/output/vector_index_e5/docplus_metadata.jsonl`
