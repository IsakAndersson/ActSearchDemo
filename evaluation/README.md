# Evaluation

This folder contains the offline retrieval evaluation script for the Docplus search methods.

## What it evaluates

`evaluation.py` measures retrieval quality with:

- `RR@20` (reciprocal rank at 20)
- Average rank of the relevant document

It evaluates multiple query variants from the Google Sheet (for example case description, keywords with typos, etc.).

## Search methods

All methods use the common interface:

```python
search(query: str, top_k: int = 20) -> List[Tuple[str, float]]
```

where each tuple is `(doc_id, score)`.

## Prerequisites

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need scraped/parsed data and (for dense/hybrid) a built vector index:

- Parsed docs: `flask/output/parsed`
- Vector index: `flask/output/vector_index/docplus.faiss`
- Vector metadata: `flask/output/vector_index/docplus_metadata.jsonl`

See `flask/README.md` for scraping and index build commands.

## Run evaluation

Run from repository root:

```bash
python evaluation/evaluation.py --method bm25 --top-k 20
python evaluation/evaluation.py --method dense --top-k 20
python evaluation/evaluation.py --method hybrid --top-k 20
```

Arguments:

- `--method`: `bm25`, `dense`, or `hybrid` (default: `hybrid`)
- `--top-k`: number of retrieved documents per query (default: `20`)

## Outputs

During execution, the script writes:

- `qrels.csv`: downloaded sheet snapshot
- `run_test.csv`: generated run file (`query_id`, `doc_id`, `score`)

Then it prints a summary table with per-query-type metrics and global averages.

## Optional environment variables

`search_adapter.py` reads these optional variables:

- `DOCPLUS_PARSED_DIR`
- `DOCPLUS_INDEX_PATH`
- `DOCPLUS_METADATA_PATH`
- `DOCPLUS_MODEL_NAME`
- `DOCPLUS_DEVICE`

Use them if your data/index paths differ from defaults.
