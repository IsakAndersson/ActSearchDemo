# API Integration Guide

This guide contains the practical information needed to connect another application to the Docplus search API.

Use this as the implementation reference when building a client in another project. The shorter `API_SEARCH_USAGE.md` file is still useful as a quick-start, but this document includes request handling, response parsing, error behavior, and optional analytics events.

## Base URL

Current public tunnel URL:

```text
https://unclimbing-madelaine-unsavorily.ngrok-free.dev
```

Build requests relative to this URL:

```text
POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search
```

If the ngrok tunnel changes, only update the base URL in your application configuration.

Recommended client configuration:

```js
const DOCPLUS_API_BASE_URL = "https://unclimbing-madelaine-unsavorily.ngrok-free.dev";
```

## Required Headers

Send JSON requests with:

```http
Content-Type: application/json
```

The API supports browser clients through CORS. The server currently returns:

- `Access-Control-Allow-Origin: *` unless the server is configured differently
- `Access-Control-Allow-Methods: GET,POST,OPTIONS`
- `Access-Control-Allow-Headers: Content-Type`

## Health Check

Before wiring up search, verify that the API is reachable:

```bash
curl https://unclimbing-madelaine-unsavorily.ngrok-free.dev/
```

Expected response shape:

```json
{
  "ok": true,
  "message": "Docplus API is running.",
  "endpoints": {
    "search": "/search (POST)",
    "search_click": "/search/click (POST)",
    "search_rating": "/search/rating (POST)"
  }
}
```

## Search Endpoint

Use:

```text
POST /search
```

Full URL:

```text
https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search
```

## Supported Search Methods

Use one of these `method` values:

- `bm25`
- `sqlite_fts`
- `vector_e5`
- `hybrid_e5`
- `docplus_live`

Recommended default for most clients:

```text
hybrid_e5
```

`hybrid_e5` combines keyword search and E5 vector search, which usually makes it the best general-purpose option.

## Required User ID

All requests from this integration should include:

```json
{
  "user_name": "kandGrupp"
}
```

Use exactly `kandGrupp`. The server logs this value so searches from the group can be tracked consistently.

The server also accepts `participant_name`, `user`, or `name`, but for this integration use `user_name`.

## Search Request Body

Minimum request:

```json
{
  "query": "I vilka fall behöver patienter adrenalin",
  "method": "hybrid_e5",
  "user_name": "kandGrupp"
}
```

Recommended request:

```json
{
  "query": "I vilka fall behöver patienter adrenalin",
  "method": "hybrid_e5",
  "user_name": "kandGrupp",
  "top_k": 5
}
```

Request fields:

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `query` | Yes | string | The user search query. Must not be empty. |
| `method` | Yes | string | Search method. Use one of the supported values above. |
| `user_name` | Yes | string | Always send `kandGrupp`. |
| `top_k` | No | number | Number of results to retrieve. Defaults to `5` if omitted. |

Implementation note: the server also accepts an internal field named `chunk_fetch_k`. Do not use it in a normal external client. Use `top_k`.

## Search Response Body

Successful response shape:

```json
{
  "search_id": "e7c2f7d3-49d0-4b7f-9a15-2f7f4dfb1d1c",
  "query": "I vilka fall behöver patienter adrenalin",
  "method": "hybrid_e5",
  "defaults": {
    "parsed_dir": "output/parsed",
    "sqlite_fts_path": "output/sqlite_fts/docplus_fts.sqlite3",
    "index_path": "output/vector_index/docplus.faiss",
    "metadata_path": "output/vector_index/docplus_metadata.jsonl",
    "e5_index_path": "output/vector_index_e5/docplus.faiss",
    "e5_metadata_path": "output/vector_index_e5/docplus_metadata.jsonl",
    "model_name": "model-name",
    "e5_model_name": "intfloat/multilingual-e5-large-instruct",
    "device": "auto",
    "top_k": "5",
    "bm25_use_chunking": "true"
  },
  "results": [
    {
      "score": 0.031746031746031744,
      "chunk_id": 12,
      "text": "Matched document text...",
      "metadata": {
        "title": "Document title",
        "source_url": "https://example.com/document",
        "document_section_headings": [
          {
            "heading": "Behandling",
            "page": 3,
            "heuristic": false
          }
        ]
      },
      "source_path": "output/parsed/example.json",
      "chunk_type": "section",
      "preview_text": "Short result preview..."
    }
  ],
  "results_by_method": {},
  "errors": []
}
```

Fields your client should use:

| Field | Type | Description |
| --- | --- | --- |
| `search_id` | string | Unique ID for this search. Save it if you send click or rating events. |
| `query` | string | Echo of the request query. |
| `method` | string | Echo of the method used. |
| `results` | array | Main list of search hits. |
| `results_by_method` | object | Usually empty for normal supported methods. Used by internal comparison methods. |
| `errors` | array | Server-side errors or partial failures. Always inspect this field. |

Result fields your client should use:

| Field | Type | Description |
| --- | --- | --- |
| `score` | number | Ranking score. Higher means better within the selected method. Do not compare scores across different methods as absolute values. |
| `chunk_id` | number | Internal chunk identifier. Useful for logging/debugging. |
| `text` | string | Main matched text content. Use for snippets or expanded result views. |
| `metadata.title` | string | Best available document title. |
| `metadata.source_url` | string | Best available external document URL. Use this for opening the source document when present. |
| `metadata.document_section_headings` | array | Section headings detected in the document, when available. Useful for document navigation. |
| `source_path` | string | Server-side source path or generated live-search source ID. Mostly useful for logging. |
| `chunk_type` | string | Type of result chunk, for example `section` or `document`. |
| `preview_text` | string | Short preview when available. Fall back to `text` if missing. |

## HTTP Status and Error Handling

The API can return `200` even when `errors` is not empty. This can happen when a method partially succeeds or the server includes a non-fatal warning.

Client rule:

1. Check the HTTP status.
2. Parse the JSON body.
3. Always inspect `errors`.
4. Render `results` if present.
5. If `results` is empty, show an empty-state message and include/log `errors`.

Common status codes:

| Status | Meaning |
| --- | --- |
| `200` | Search completed fully or partially. Check `errors`. |
| `400` | Request failed, usually because the query is empty, the method is unknown, or all selected search methods failed. |
| `403` | Server is configured to block direct access. Use the configured tunnel/base URL. |

Common error messages:

| Error | Likely cause |
| --- | --- |
| `Query cannot be empty.` | `query` was missing or only whitespace. |
| `Unknown method '...'` | `method` was not one of the supported values. |
| `Chunk fetch count must be an integer; defaulted to 5.` | `top_k` or internal `chunk_fetch_k` was not parseable as an integer. |
| `BM25 search failed: ...` | Server-side BM25/index/data issue. |
| `SQLite FTS search failed: ...` | Server-side SQLite FTS issue. |
| `Vector E5 search failed: ...` | Server-side vector index/model issue. |
| `Docplus live search failed: ...` | Server-side live Docplus search issue. |

## Recommended JavaScript Client

```js
const DOCPLUS_API_BASE_URL = "https://unclimbing-madelaine-unsavorily.ngrok-free.dev";
const DOCPLUS_USER_NAME = "kandGrupp";

export async function searchDocplus(query, options = {}) {
  const method = options.method ?? "hybrid_e5";
  const topK = options.topK ?? 5;

  const response = await fetch(`${DOCPLUS_API_BASE_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      method,
      user_name: DOCPLUS_USER_NAME,
      top_k: topK
    })
  });

  let data;
  try {
    data = await response.json();
  } catch (error) {
    throw new Error(`Docplus API returned non-JSON response with status ${response.status}`);
  }

  if (!response.ok) {
    const message = Array.isArray(data.errors) && data.errors.length > 0
      ? data.errors.join("; ")
      : `Docplus API request failed with status ${response.status}`;
    throw new Error(message);
  }

  return {
    searchId: data.search_id,
    query: data.query,
    method: data.method,
    results: Array.isArray(data.results) ? data.results : [],
    errors: Array.isArray(data.errors) ? data.errors : []
  };
}
```

Example usage:

```js
try {
  const search = await searchDocplus("I vilka fall behöver patienter adrenalin", {
    method: "hybrid_e5",
    topK: 5
  });

  if (search.errors.length > 0) {
    console.warn("Docplus search completed with warnings:", search.errors);
  }

  for (const result of search.results) {
    const title = result.metadata?.title ?? "Untitled document";
    const url = result.metadata?.source_url;
    const preview = result.preview_text ?? result.text ?? "";

    console.log({ title, url, preview, score: result.score });
  }
} catch (error) {
  console.error(error);
}
```

## Rendering Search Results

Recommended UI mapping:

| UI element | Response field |
| --- | --- |
| Result title | `result.metadata.title` |
| Result link | `result.metadata.source_url` |
| Result preview/snippet | `result.preview_text`, fallback to `result.text` |
| Score/debug text | `result.score` |
| Section list | `result.metadata.document_section_headings` |

Recommended fallback logic:

```js
function normalizeResult(result) {
  const metadata = result.metadata ?? {};

  return {
    title: metadata.title || "Untitled document",
    url: metadata.source_url || null,
    preview: result.preview_text || result.text || "",
    score: typeof result.score === "number" ? result.score : null,
    sourcePath: result.source_path || "",
    chunkType: result.chunk_type || "",
    sectionHeadings: Array.isArray(metadata.document_section_headings)
      ? metadata.document_section_headings
      : []
  };
}
```

When `metadata.source_url` is missing, render the result without an external link.

## Terminal Test Requests

Health check:

```bash
curl https://unclimbing-madelaine-unsavorily.ngrok-free.dev/
```

Search with recommended method:

```bash
curl -X POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I vilka fall behöver patienter adrenalin",
    "method": "hybrid_e5",
    "user_name": "kandGrupp",
    "top_k": 5
  }'
```

Search with all supported external methods:

```bash
for method in bm25 sqlite_fts vector_e5 hybrid_e5 docplus_live; do
  echo "Testing $method"
  curl -X POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search \
    -H "Content-Type: application/json" \
    -d "{
      \"query\": \"I vilka fall behöver patienter adrenalin\",
      \"method\": \"$method\",
      \"user_name\": \"kandGrupp\",
      \"top_k\": 5
    }"
  echo
done
```

## Optional Click Logging

If your client shows clickable search results, you can report result clicks to:

```text
POST /search/click
```

This is optional, but useful for evaluating whether users open the returned documents.

Example request:

```json
{
  "search_id": "e7c2f7d3-49d0-4b7f-9a15-2f7f4dfb1d1c",
  "query": "I vilka fall behöver patienter adrenalin",
  "requested_method": "hybrid_e5",
  "user_name": "kandGrupp",
  "result_method": "hybrid_e5",
  "rank": 1,
  "score": 0.031746031746031744,
  "title": "Document title",
  "url": "https://example.com/document",
  "chunk_type": "section",
  "source_path": "output/parsed/example.json"
}
```

Required field:

- `search_id`

Recommended fields:

- `query`
- `requested_method`
- `user_name`
- `result_method`
- `rank`
- `score`
- `title`
- `url`
- `chunk_type`
- `source_path`

Successful response:

```json
{
  "ok": true
}
```

If `search_id` is missing, the server returns:

```json
{
  "ok": false,
  "errors": ["search_id is required."]
}
```

Example JavaScript:

```js
export async function logDocplusClick(search, result, rank) {
  await fetch(`${DOCPLUS_API_BASE_URL}/search/click`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      search_id: search.searchId,
      query: search.query,
      requested_method: search.method,
      user_name: DOCPLUS_USER_NAME,
      result_method: search.method,
      rank,
      score: result.score,
      title: result.metadata?.title ?? "",
      url: result.metadata?.source_url ?? "",
      chunk_type: result.chunk_type ?? "",
      source_path: result.source_path ?? ""
    })
  });
}
```

## Optional Rating Logging

If your client lets users rate result quality, send ratings to:

```text
POST /search/rating
```

Example request:

```json
{
  "search_id": "e7c2f7d3-49d0-4b7f-9a15-2f7f4dfb1d1c",
  "query": "I vilka fall behöver patienter adrenalin",
  "requested_method": "hybrid_e5",
  "user_name": "kandGrupp",
  "result_method": "hybrid_e5",
  "document": "Document title",
  "title": "Document title",
  "url": "https://example.com/document",
  "source_path": "output/parsed/example.json",
  "user_score": 5
}
```

Required fields:

- `search_id`
- `user_score`

`user_score` must be an integer from `1` to `5`.

Successful response:

```json
{
  "ok": true
}
```

Validation errors:

```json
{
  "ok": false,
  "errors": ["search_id is required."]
}
```

```json
{
  "ok": false,
  "errors": ["user_score must be an integer between 1 and 5."]
}
```

## Recommended Integration Checklist

1. Store the API base URL in configuration.
2. Send all searches to `POST /search`.
3. Use `method: "hybrid_e5"` unless you are intentionally comparing methods.
4. Always include `user_name: "kandGrupp"`.
5. Use `top_k: 5` to start.
6. Display `metadata.title`, `metadata.source_url`, and `preview_text`.
7. Fall back gracefully when `source_url` or `preview_text` is missing.
8. Check `errors` even when the HTTP response is `200`.
9. Save `search_id` with the rendered result list.
10. Optionally send `/search/click` when a user opens a result.
11. Optionally send `/search/rating` if the UI collects relevance feedback.

## Notes for Client Projects

- Treat the API as read-only unless you intentionally use the optional logging endpoints.
- Do not expose server-side defaults such as `parsed_dir`, index paths, or model names in the user interface. They are included for debugging.
- Do not rely on exact score ranges. Scores differ by method.
- Do not assume every result has a link. Some results may only have text and metadata.
- Avoid hard-coding the ngrok URL throughout the app. Put it in one config value.
- Use a request timeout in production clients so the UI does not wait indefinitely.
- For browser apps, handle network errors separately from API errors because tunnel availability can change.
