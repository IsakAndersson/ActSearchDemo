# ThesisSearch

## Docplus Scraper

This repo contains a lightweight scraper that can crawl a Docplus instance, download documents, and extract text for downstream tasks such as fine-tuning or vector indexing.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt #OBS for mac users change to torch in requirements.txt
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

## Flask API server

Run the local API used by the Next.js frontend:

```bash
python app.py
```

API endpoints:

- `GET /` health/info endpoint
- `POST /search` search endpoint (JSON body or form body)

Set defaults and CORS via environment variables if needed:

```bash
export DOCPLUS_PARSED_DIR=output/parsed
export DOCPLUS_INDEX_PATH=output/vector_index/docplus.faiss
export DOCPLUS_METADATA_PATH=output/vector_index/docplus_metadata.jsonl
export DOCPLUS_MODEL_NAME=KBLab/bert-base-swedish-cased
export DOCPLUS_DEVICE=auto
export DOCPLUS_TOP_K=5
export DOCPLUS_ALLOWED_ORIGIN=https://your-vercel-app.vercel.app
```

### Notes

- The crawler only follows links on the same domain as `--base-url`.
- Parsed text extraction uses optional dependencies for PDF and DOCX. If you remove them from `requirements.txt`, extraction is skipped and empty text is stored.
- Be sure to respect Docplus usage policies and adjust crawl rate as needed.
- If your start paths include `&` characters, wrap the value in quotes (or escape `&`) so the shell does not treat them as background commands.
