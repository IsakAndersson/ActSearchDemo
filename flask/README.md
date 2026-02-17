# ThesisSearch

## Docplus Scraper

This repo contains a lightweight scraper that can crawl a Docplus instance, download documents, and extract text for downstream tasks such as fine-tuning or vector indexing.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt #OBS for mac users change to torch in requirements 
```

### Usage for actsearch!

To crawl a contiguous range of search pages (e.g. pages 1 through 618), provide a single
search path and the page range:

```bash
python -m scraper.docplus_scraper \
  --base-url https://publikdocplus.regionuppsala.se/ \
  --start-paths "/Home/Search?searchValue=&oldFilter=&facet=&facetVal=&page=1" \
  --page-start 1 \
  --page-end 618 \
  --output-dir output \
  --delay 1.0
```

### Alternative usage

```bash
python -m scraper.docplus_scraper \
  --base-url https://publikdocplus.regionuppsala.se/ \
  --start-paths "/Home/Search?searchValue=&oldFilter=&facet=&facetVal=,/documents,/policies" \
  --output-dir output \
  --delay 1.0
```

The scraper writes:

- `output/documents/` for downloaded binaries
- `output/parsed/` for JSON payloads with extracted text + metadata
- `output/summary.json` for a crawl summary

## Vector indexing (BERT Swedish)

To build a vector index using the Swedish BERT model (`KBLab/bert-base-swedish-cased`),
install the extra dependencies and run the indexer against your parsed output.

```bash
pip install -r requirements.txt
python -m search.vector_index build \
  --parsed-dir output/parsed \
  --output-dir output/vector_index \
  --model-name KBLab/bert-base-swedish-cased
```

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

### Vector indexing with title chunks

To build a separate FAISS index that includes title-only chunks (derived from document
metadata/URL filenames) alongside body chunks:

```bash
python -m search.vector_index build-titles \
  --parsed-dir output/parsed \
  --output-dir output/vector_index_titles \
  --model-name KBLab/bert-base-swedish-cased
```

This creates:

- `output/vector_index_titles/docplus_titles.faiss` with normalized embeddings
- `output/vector_index_titles/docplus_titles_metadata.jsonl` with chunk text + metadata

### GPU acceleration

The vector indexer will use CUDA automatically if your PyTorch build supports it. To
force the GPU or CPU, pass `--device cuda` or `--device cpu` to the `build` and `query`
commands. If CUDA is not available and you request it, the indexer will raise an error.

For NVIDIA GPUs (like a GTX 1080 Ti), install CUDA-enabled builds of PyTorch and FAISS
before running the indexer, for example:

```bash
pip install --upgrade "torch==2.3.1+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

Refer to the PyTorch and FAISS installation guides for the exact versions that match
your driver and CUDA toolkit.

## BM25 search (lexical)

To run a BM25 search directly over the parsed JSON files (same chunks as the vector
index), use the BM25 helper:

```bash
python -m search.bm25_search \
  --parsed-dir output/parsed \
  --query "I vilka fall behöver patienter adrenalin"
```

## Offline evaluation

Offline retrieval evaluation lives in `evaluation/` and supports `bm25`, `dense`, and
`hybrid` methods with `RR@20` reporting.

From repository root:

```bash
python evaluation/evaluation.py --method hybrid --top-k 20
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
  `vector`, `vector_titles`, and `all` (returns `results_by_method`).
- `POST /search/click` click-tracking endpoint. Expects `search_id` and result metadata
  from the frontend when a user clicks a result link.
- `POST /search/rating` result-rating endpoint. Expects `search_id`, query/result metadata,
  and `user_score` (1-5).

Search and click logs are written as CSV files:

- `output/logs/search_events.csv` (one row per hit, with method, rank, score, URL/title)
- `output/logs/click_events.csv` (one row per clicked result)
- `output/logs/rating_events.csv` (one row per user rating)

Each `/search` response includes a `search_id` that can be used to correlate click rows
to the exact query and result set.

Set defaults and CORS via environment variables if needed:

```bash
export DOCPLUS_PARSED_DIR=output/parsed
export DOCPLUS_INDEX_PATH=output/vector_index/docplus.faiss
export DOCPLUS_METADATA_PATH=output/vector_index/docplus_metadata.jsonl
export DOCPLUS_TITLES_INDEX_PATH=output/vector_index_titles/docplus_titles.faiss
export DOCPLUS_TITLES_METADATA_PATH=output/vector_index_titles/docplus_titles_metadata.jsonl
export DOCPLUS_MODEL_NAME=KBLab/bert-base-swedish-cased
export DOCPLUS_DEVICE=auto
export DOCPLUS_TOP_K=5
export DOCPLUS_ALLOWED_ORIGIN=https://your-vercel-app.vercel.app
export DOCPLUS_HOST=127.0.0.1
export DOCPLUS_PORT=5000
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

2. Start Flask so it listens on all interfaces and your chosen demo port:

```bash
DOCPLUS_HOST=0.0.0.0 DOCPLUS_PORT=2200 \
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

### Notes

- The crawler only follows links on the same domain as `--base-url`.
- Parsed text extraction uses optional dependencies for PDF and DOCX. If you remove them from `requirements.txt`, extraction is skipped and empty text is stored.
- Be sure to respect Docplus usage policies and adjust crawl rate as needed.
- If your start paths include `&` characters, wrap the value in quotes (or escape `&`) so the shell does not treat them as background commands.
