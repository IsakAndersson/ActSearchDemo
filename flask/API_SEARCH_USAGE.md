# `/search` API Usage

This document describes the supported `/search` API for external use.

## Supported Endpoint

- `POST /search`

## Supported Search Methods

Only these `method` values should be used:

- `bm25`
- `sqlite_fts`
- `vector_e5`
- `hybrid_e5`
- `docplus_live`

## Required User ID

All requests from your group should include:

```json
{
  "user_name": "kandGrupp"
}
```

Use exactly `kandGrupp` as the user identifier so requests can be tracked consistently in the server logs.

## Request Format

Send JSON with:

- `query`: the search text
- `method`: one of the supported methods above
- `user_name`: always `kandGrupp`

Optional fields:

- `top_k`: number of results to return

Example request body:

```json
{
  "query": "I vilka fall behöver patienter adrenalin",
  "method": "hybrid_e5",
  "user_name": "kandGrupp",
  "top_k": 5
}
```

## Response Format

The API returns JSON like this:

```json
{
  "search_id": "uuid",
  "query": "I vilka fall behöver patienter adrenalin",
  "method": "hybrid_e5",
  "defaults": {
    "top_k": "5"
  },
  "results": [
    {
      "score": 0.87,
      "chunk_id": 12,
      "text": "Result text...",
      "metadata": {
        "title": "Document title",
        "source_url": "https://example.com/document"
      },
      "source_path": "output/parsed/example.json",
      "chunk_type": "section",
      "preview_text": "Result preview..."
    }
  ],
  "results_by_method": {},
  "errors": []
}
```

Fields to use in the client:

- `search_id`: unique ID for the search request
- `results`: list of returned hits
- `results[].score`: ranking score
- `results[].text`: matched text content
- `results[].metadata.title`: document title, when available
- `results[].metadata.source_url`: source link, when available
- `results[].preview_text`: short preview text, when available
- `errors`: list of server-side errors

Check `errors` even if the HTTP status is `200`.

## JavaScript Example

```js
const response = await fetch("https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: "I vilka fall behöver patienter adrenalin",
    method: "hybrid_e5",
    user_name: "kandGrupp",
    top_k: 5
  })
});

const data = await response.json();
console.log(data);
```

## Terminal Testing

You can test the API directly from a terminal with `curl`.

### 1. Health Check

```bash
curl https://unclimbing-madelaine-unsavorily.ngrok-free.dev/
```

### 2. Test `bm25`

```bash
curl -X POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I vilka fall behöver patienter adrenalin",
    "method": "bm25",
    "user_name": "kandGrupp",
    "top_k": 5
  }'
```

### 3. Test `vector_e5`

```bash
curl -X POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I vilka fall behöver patienter adrenalin",
    "method": "vector_e5",
    "user_name": "kandGrupp",
    "top_k": 5
  }'
```

### 4. Test `hybrid_e5`

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

### 5. Test `docplus_live`

```bash
curl -X POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I vilka fall behöver patienter adrenalin",
    "method": "docplus_live",
    "user_name": "kandGrupp",
    "top_k": 5
  }'
```

### 6. Test `sqlite_fts`

```bash
curl -X POST https://unclimbing-madelaine-unsavorily.ngrok-free.dev/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I vilka fall behöver patienter adrenalin",
    "method": "sqlite_fts",
    "user_name": "kandGrupp",
    "top_k": 5
  }'
```

## Notes

- Only use the `/search` endpoint described here.
- Always send `user_name` as `kandGrupp`.
- Only use these methods: `bm25`, `sqlite_fts`, `vector_e5`, `hybrid_e5`, `docplus_live`.
- If `results` is empty, also inspect the `errors` field in the response.
