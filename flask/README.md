# ThesisSearch

## Docplus Scraper

This repo contains a lightweight scraper that can crawl a Docplus instance, download documents, and extract text for downstream tasks such as fine-tuning or vector indexing.

### Setup

Use the shared project environment at repository root (`../.venv`), not `flask/.venv`.

```bash
# from repository root (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# or, if you are already in flask/
source ../.venv/bin/activate
```

The base install is enough for scraping, BM25 search, and the Flask API. Vector search
(`vector`, `vector_e5`, `hybrid_e5`) needs extra optional dependencies as described below.

### Usage for actsearch

The scraper now has Region Uppsala defaults baked in and runs as one operation
(scrape + parse + metadata enrichment + summary):

```bash
python -m scraper.docplus_scraper
```

To change how many pages are scraped, override `--page-end`:

```bash
python -m scraper.docplus_scraper --page-end 650
```

To scrape only search-result metadata and skip document downloads/parsing:

```bash
python -m scraper.docplus_scraper --metadata-only
```

Change the --page-end flag to match the number of pages available at "https://publikdocplus.regionuppsala.se/Home/Search?searchValue=&oldFilter=&facet=&facetVal=&page=1"

The scraper writes:

- `output/documents/` for downloaded binaries
- `output/metadata/` for metadata-only JSON payloads when `--metadata-only` is used
- `output/parsed/` for JSON payloads with `raw_text`, cleaned `text`, derived section chunks, and metadata
- `output/summary.json` for a crawl summary

For full scraping, parsed document metadata can include:

- `source_url`
- `downloaded_at`
- `content_type`
- `document_name`
- `title`
- `title_source`
- `page_count` for PDF files

When available on the Docplus search-results page, the scraper also captures metadata from
the information button (`file-metadata`) and stores fields such as:

- `document_collection`
- `process`
- `publish_date`
- `subject_area`
- `type_of_action`
- `valid_for_area`
- `version`
- `comment`
- `document_type`
- `metadata_url`
- `tax_keyword`

With `--metadata-only`, the scraper stores the search-result metadata without downloading
or parsing the underlying document files.

If you already have downloaded files in `output/documents/` and parsed JSON files in
`output/parsed/`, you can backfill `metadata.page_count` without re-running the full scraper:

```bash
python -m scraper.backfill_page_count --parsed-dir output/parsed --only-missing
```

This reads each parsed JSON file, looks up its `binary_path`, counts PDF pages, and writes
the result back to `metadata.page_count`.

To print summary statistics for the parsed dataset without writing any new files:

```bash
python -m scraper.parsed_stats --parsed-dir output/parsed --metadata-dir output/metadata
```

This prints:

- total number of parsed documents
- how many documents have `metadata.page_count`
- average, median, max, percentiles, and counts over 50/100 pages for page count
- counts per `content_type`
- Docplus metadata-field coverage from `output/metadata`: for fields such as `process`, `subject_area`, `version`, and `tax_keyword`, number and percent of documents with a non-empty value, plus number of unique values
- from `publish_date`, how many documents are older than 2 years at the time the script is run, plus average/median age, age percentiles, and the newest/oldest publish dates

It also writes one `.txt` file per Docplus metadata field to `output/metadata_field_values`
(or the directory passed via `--metadata-values-output-dir`), with values sorted
alphabetically and document counts per value.

To list all unique non-empty Docplus metadata values per field from `output/metadata`
with document counts per value:

```bash
python -m scraper.metadata_unique_values --metadata-dir output/metadata
```

## Vector indexing (BERT Swedish)

To build a vector index, first install the optional vector-search dependencies, then run
the indexer against your parsed output.

Available named profiles in `search.vector_index`:

- `swedish_bert` -> `KBLab/bert-base-swedish-cased`, `chunk_size=250`, `chunk_overlap=50`
- `e5_large_instruct` -> `intfloat/multilingual-e5-large-instruct`, `chunk_size=250`, `chunk_overlap=50`

Swedish BERT index:

```bash
# CPU-only vector setup
pip install faiss-cpu transformers<5 torch==2.2.2

python -m search.vector_index build \
  --parsed-dir output/parsed \
  --output-dir output/vector_index \
  --profile swedish_bert
```

E5 large instruct index:

```bash
python -m search.vector_index build \
  --parsed-dir output/parsed \
  --output-dir output/vector_index_e5 \
  --profile e5_large_instruct
```

Body embeddings use the cleaned parsed `text` field by default.

Both BM25 and vector indexing now prefer document-aware section chunks derived from
headings/sub-chapters in the parsed documents. Large sections are only split further
when they exceed the configured chunk size, and the full top-level document `text`
is not used as a fallback indexing body.

This creates:

- `output/vector_index/docplus.faiss` with normalized embeddings
- `output/vector_index/docplus_metadata.jsonl` with chunk text + metadata

Query the index with:

```bash
python -m search.vector_index query \
  --index-path output/vector_index/docplus.faiss \
  --metadata-path output/vector_index/docplus_metadata.jsonl \
  --query "I vilka fall behöver patienter adrenalin"
```

For both `swedish_bert` and `e5_large_instruct`, regular `build` includes
title chunks per document together with chapter-aware section chunks.

### Choosing CPU or GPU

The vector indexer accepts `--device auto`, `--device cpu`, or `--device cuda`.

- `auto` is the default. It uses CUDA when your installed PyTorch build supports it,
  otherwise it falls back to CPU.
- `cpu` forces CPU inference even on a machine with a GPU.
- `cuda` forces GPU inference. If CUDA is not available, the command raises an error.

CPU-only setup:

```bash
pip install faiss-cpu transformers<5 torch==2.2.2
```

GPU setup (NVIDIA CUDA example):

```bash
pip install --upgrade "torch==2.3.1+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install transformers<5 faiss-gpu
```

Examples:

```bash
# Explicit CPU build/query
python -m search.vector_index build --parsed-dir output/parsed --output-dir output/vector_index --profile swedish_bert --device cpu
python -m search.vector_index query --index-path output/vector_index/docplus.faiss --metadata-path output/vector_index/docplus_metadata.jsonl --query "adrenalin" --device cpu

# Explicit GPU build/query
python -m search.vector_index build --parsed-dir output/parsed --output-dir output/vector_index --profile swedish_bert --device cuda
python -m search.vector_index query --index-path output/vector_index/docplus.faiss --metadata-path output/vector_index/docplus_metadata.jsonl --query "adrenalin" --device cuda
```

Refer to the PyTorch and FAISS installation guides for exact versions that match your
driver, CUDA toolkit, and platform.

## BM25 search (lexical)

To run a BM25 search directly over the parsed JSON files, use the BM25 helper.
Default BM25 chunking now matches the E5 profile (`max_chars=250`, `overlap=50`)
and includes one title-only chunk per document when a title can be extracted.
Body indexing uses the stored section chunks so each hit is aligned to a chapter
or sub-category:

```bash
python -m search.bm25_search \
  --parsed-dir output/parsed \
  --query "I vilka fall behöver patienter adrenalin"
```

## Offline evaluation

Offline retrieval evaluation lives in `evaluation/` and supports `bm25`, `dense`,
`dense_e5`, `hybrid`, `hybrid_e5`, `docplus_live`, and `sts_live` methods with `RR@20` reporting.

From repository root:

```bash
python evaluation/evaluation.py --method hybrid --top-k 20
python evaluation/evaluation.py --method docplus_live --top-k 20
python evaluation/evaluation.py --method sts_live --top-k 20
```

See `evaluation/README.md` for full setup, methods, and outputs.

## Flask API server

Run the local API used by the Next.js frontend:

```bash
DOCPLUS_HOST=127.0.0.1 DOCPLUS_PORT=5000 python app.py
```

API endpoints:

- `GET /` health/info endpoint
- `POST /search` search endpoint (JSON body or form body). `method` supports `bm25`,
  `vector`, `vector_e5`, `hybrid_e5`, and `all` (returns `results_by_method`). The
  default method is `bm25`. Include `user_name` to attach a user identifier to the
  search logs; `participant_name` is also accepted as a fallback.
- `POST /search/click` click-tracking endpoint. Expects `search_id` and result metadata
  from the frontend when a user clicks a result link. Include `user_name` if you want it
  recorded in the click log.
- `POST /search/rating` result-rating endpoint. Expects `search_id`, query/result metadata,
  and `user_score` (1-5). Include `user_name` if you want it recorded in the rating log.

Search and click logs are written as CSV files:

- `output/logs/search_events.csv` (one row per hit, including `user_name`, method, rank, score, URL/title)
- `output/logs/click_events.csv` (one row per clicked result, including `user_name`)
- `output/logs/rating_events.csv` (one row per user rating, including `user_name`)

Each `/search` response includes a `search_id` that can be used to correlate click rows
to the exact query and result set.

Set defaults and CORS via environment variables if needed:

```bash
export DOCPLUS_PARSED_DIR=output/parsed
export DOCPLUS_INDEX_PATH=output/vector_index/docplus.faiss
export DOCPLUS_METADATA_PATH=output/vector_index/docplus_metadata.jsonl
export DOCPLUS_E5_INDEX_PATH=output/vector_index_e5/docplus.faiss
export DOCPLUS_E5_METADATA_PATH=output/vector_index_e5/docplus_metadata.jsonl
export DOCPLUS_MODEL_NAME=KBLab/bert-base-swedish-cased
export DOCPLUS_E5_MODEL_NAME=intfloat/multilingual-e5-large-instruct
export DOCPLUS_DEVICE=auto
export DOCPLUS_TOP_K=5
export DOCPLUS_ALLOWED_ORIGIN=https://your-vercel-app.vercel.app
export DOCPLUS_HOST=127.0.0.1
export DOCPLUS_PORT=5000
export DOCPLUS_ONLY_ALLOW_LOOPBACK=true
export DOCPLUS_SEARCH_LOG_PATH=output/logs/search_events.csv
export DOCPLUS_CLICK_LOG_PATH=output/logs/click_events.csv
export DOCPLUS_RATING_LOG_PATH=output/logs/rating_events.csv
```

### Vercel demo via ngrok

If your frontend is deployed on Vercel, the browser requires the API to be reachable over
HTTPS. The simplest temporary setup is to keep Flask local and expose it with ngrok.

1. Install ngrok and authenticate once:

```bash
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

2. Start Flask on loopback only so the server is reachable through ngrok on the same machine,
   but not directly from other hosts:

```bash
DOCPLUS_HOST=127.0.0.1 DOCPLUS_PORT=2200 DOCPLUS_ONLY_ALLOW_LOOPBACK=true \
DOCPLUS_ALLOWED_ORIGIN=https://act-search-demo-qjau.vercel.app \
python app.py
```

3. In another terminal, create the HTTPS tunnel:

```bash
ngrok http 2200
```

4. Copy the `https://...ngrok-free.app` URL from ngrok and set it in Vercel:

- `NEXT_PUBLIC_DOCPLUS_API_BASE_URL=https://<your-ngrok-domain>`

5. Redeploy the Vercel app so the new public env var is included in the client bundle.

Useful checks:

- ngrok request inspector: `http://127.0.0.1:4040`
- Flask health endpoint through tunnel: `https://<your-ngrok-domain>/`

Security note:

- When requests come through ngrok, Flask typically sees the immediate peer as `127.0.0.1`
  because the local ngrok agent forwards them into your app.
- `DOCPLUS_ONLY_ALLOW_LOOPBACK=true` rejects requests whose `REMOTE_ADDR` is not loopback.
  That allows ngrok-forwarded traffic while blocking direct connections to the Flask port.
- Treat `X-Forwarded-For` as logging data only. It is useful for recording the original client IP,
  but it should not be used by itself to authorize access.

### Notes

- The crawler only follows links on the same domain as `--base-url`.
- Parsed text extraction uses optional dependencies for PDF and DOCX. If you remove them from `requirements.txt`, extraction is skipped and empty text is stored.
- Be sure to respect Docplus usage policies and adjust crawl rate as needed.
- If your start paths include `&` characters, wrap the value in quotes (or escape `&`) so the shell does not treat them as background commands.
- Vector build/query now load embedding models with local-only Hugging Face cache access (`local_files_only=True`). No Hugging Face network requests are made at runtime; if a model is missing, pre-download it first.
- If `torch`, `transformers`, or FAISS are not installed, the app still runs for scraping
  and BM25 search. Only vector-search methods are unavailable.
