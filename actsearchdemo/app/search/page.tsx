"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "actsearch-authenticated";
const DEFAULT_API_BASE_URL = "http://127.0.0.1:5000";

type SearchMethod = "bm25" | "vector" | "vector_titles" | "all";

type SearchResult = {
  score?: number;
  text?: string;
  chunk_text?: string;
  chunk_type?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
};

type SearchResultsByMethod = Partial<Record<SearchMethod, SearchResult[]>>;

const getStringValue = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const getMetadataValue = (
  metadata: Record<string, unknown> | undefined,
  keys: string[],
): string | undefined => {
  if (!metadata) {
    return undefined;
  }
  for (const key of keys) {
    const value = getStringValue(metadata[key]);
    if (value) {
      return value;
    }
  }
  return undefined;
};

const getNameFromPath = (value: string): string | undefined => {
  try {
    const url = new URL(value);
    const last = url.pathname.split("/").filter(Boolean).pop();
    return last ? decodeURIComponent(last) : undefined;
  } catch {
    const last = value.split("/").filter(Boolean).pop();
    return last || undefined;
  }
};

const getTitleFromUrl = (value: string): string | undefined => {
  try {
    const url = new URL(value);
    const filename =
      url.searchParams.get("filename") ||
      url.searchParams.get("file") ||
      url.searchParams.get("name");
    if (filename) {
      return decodeURIComponent(filename);
    }
    const last = url.pathname.split("/").filter(Boolean).pop();
    if (last && last.toLowerCase() !== "getdocument") {
      return decodeURIComponent(last);
    }
  } catch {
    const last = value.split("/").filter(Boolean).pop();
    if (last && last.toLowerCase() !== "getdocument") {
      return last;
    }
  }
  return undefined;
};

const getResultUrl = (result: SearchResult): string | undefined => {
  const url = getMetadataValue(result.metadata, [
    "source_url",
    "url",
    "link",
    "href",
    "source",
  ]);
  if (!url) {
    return undefined;
  }
  return /^https?:\/\//i.test(url) ? url : undefined;
};

const getResultTitle = (result: SearchResult): string => {
  const fromMetadata = getMetadataValue(result.metadata, [
    "title",
    "document_title",
    "doc_title",
    "name",
    "filename",
    "file_name",
  ]);
  if (fromMetadata) {
    return fromMetadata;
  }
  const url = getResultUrl(result);
  const urlName = url ? getTitleFromUrl(url) : undefined;
  if (urlName) {
    return urlName;
  }
  const sourcePath = getStringValue(result.source_path);
  const pathName = sourcePath ? getNameFromPath(sourcePath) : undefined;
  if (pathName) {
    return pathName;
  }
  return "Untitled document";
};

const RESULT_WIDTH_CLASS = "w-full max-w-4xl mx-auto";

export default function SearchPage() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [resultsByMethod, setResultsByMethod] = useState<SearchResultsByMethod>({});
  const [method, setMethod] = useState<SearchMethod>("bm25");
  const [query, setQuery] = useState("");
  const [parsedDir, setParsedDir] = useState("output/parsed");
  const [indexPath, setIndexPath] = useState("output/vector_index/docplus.faiss");
  const [metadataPath, setMetadataPath] = useState(
    "output/vector_index/docplus_metadata.jsonl",
  );
  const [titlesIndexPath, setTitlesIndexPath] = useState(
    "output/vector_index_titles/docplus_titles.faiss",
  );
  const [titlesMetadataPath, setTitlesMetadataPath] = useState(
    "output/vector_index_titles/docplus_titles_metadata.jsonl",
  );
  const [modelName, setModelName] = useState("KBLab/bert-base-swedish-cased");
  const [device, setDevice] = useState("auto");
  const [topK, setTopK] = useState("5");
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);

  useEffect(() => {
    const isAuthenticated = localStorage.getItem(SESSION_KEY) === "true";
    if (!isAuthenticated) {
      router.replace("/");
      return;
    }
    setIsReady(true);
  }, [router]);

  const canSubmit = useMemo(() => query.trim().length > 0 && !isLoading, [isLoading, query]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setErrors([]);
    setResults([]);
    setResultsByMethod({});

    try {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method,
          query,
          parsed_dir: parsedDir,
          index_path: indexPath,
          metadata_path: metadataPath,
          titles_index_path: titlesIndexPath,
          titles_metadata_path: titlesMetadataPath,
          model_name: modelName,
          device,
          top_k: topK,
        }),
      });

      const payload = (await response.json()) as {
        errors?: string[];
        results?: SearchResult[];
        results_by_method?: SearchResultsByMethod;
      };

      if (!response.ok) {
        setErrors(payload.errors && payload.errors.length ? payload.errors : ["Search failed."]);
        return;
      }

      setErrors(payload.errors ?? []);
      setResults(payload.results ?? []);
      setResultsByMethod(payload.results_by_method ?? {});
    } catch (error) {
      setErrors([
        error instanceof Error
          ? `Request failed: ${error.message}`
          : "Request failed with an unknown error.",
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isReady) {
    return null;
  }

  return (
    <div className="min-h-screen px-4 py-10 md:px-8">
      <main className="mx-auto w-full max-w-5xl">
        <div className={`card border border-base-300 bg-base-100 shadow-xl ${RESULT_WIDTH_CLASS}`}>
          <div className="card-body gap-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
                  Docplus Search
                </h1>
                <p className="mt-1 text-sm text-base-content/70">
                  Clean test UI with Flask API as backend.
                </p>
              </div>
              <span className="badge badge-outline badge-primary">Next.js + Flask</span>
            </div>

            <form className="space-y-4" onSubmit={onSubmit}>
              <label className="form-control w-full">
                <div className="label">
                  <span className="label-text text-base">Query</span>
                </div>
                <input
                  className="input input-bordered input-primary w-full"
                  type="text"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search in documents..."
                  required
                />
              </label>

              <label className="form-control w-full">
                <div className="label">
                  <span className="label-text text-base">Search type</span>
                </div>
                <select
                  className="select select-bordered w-full"
                  value={method}
                  onChange={(event) => setMethod(event.target.value as SearchMethod)}
                >
                  <option value="bm25">BM25</option>
                  <option value="vector">Vector (FAISS)</option>
                  <option value="vector_titles">Vector (FAISS + titles)</option>
                  <option value="all">All (side-by-side)</option>
                </select>
              </label>

              <details className="dropdown dropdown-bottom w-full">
                <summary className="btn btn-ghost btn-sm">Advanced settings</summary>
                <div className="card card-sm z-10 mt-3 w-full border border-base-300 bg-base-100 shadow-lg">
                  <div className="card-body grid gap-4 md:grid-cols-2">
                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">API base URL</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={apiBaseUrl}
                        onChange={(event) => setApiBaseUrl(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Model name</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={modelName}
                        onChange={(event) => setModelName(event.target.value)}
                        list="model-options"
                      />
                    </label>
                    <datalist id="model-options">
                      <option value="KBLab/bert-base-swedish-cased" />
                      <option value="sentence-transformers/paraphrase-multilingual-mpnet-base-v2" />
                      <option value="intfloat/multilingual-e5-base" />
                    </datalist>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Parsed dir (BM25)</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={parsedDir}
                        onChange={(event) => setParsedDir(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Top-k</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="number"
                        min={1}
                        value={topK}
                        onChange={(event) => setTopK(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">FAISS index path</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={indexPath}
                        onChange={(event) => setIndexPath(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Metadata path</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={metadataPath}
                        onChange={(event) => setMetadataPath(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Titles FAISS index path</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={titlesIndexPath}
                        onChange={(event) => setTitlesIndexPath(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Titles metadata path</span>
                      </div>
                      <input
                        className="input input-bordered"
                        type="text"
                        value={titlesMetadataPath}
                        onChange={(event) => setTitlesMetadataPath(event.target.value)}
                      />
                    </label>

                    <label className="form-control">
                      <div className="label">
                        <span className="label-text">Device</span>
                      </div>
                      <select
                        className="select select-bordered"
                        value={device}
                        onChange={(event) => setDevice(event.target.value)}
                      >
                        <option value="auto">auto</option>
                        <option value="cpu">cpu</option>
                        <option value="cuda">cuda</option>
                      </select>
                    </label>
                  </div>
                </div>
              </details>

              <div className="pt-1">
                <button className="btn btn-primary btn-wide" type="submit" disabled={!canSubmit}>
                  {isLoading ? (
                    <>
                      <span className="loading loading-spinner loading-sm" />
                      Searching
                    </>
                  ) : (
                    "Search"
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {errors.length > 0 ? (
          <section className="mt-5 space-y-2">
            {errors.map((error) => (
              <div className="alert alert-error" key={error}>
                <span>{error}</span>
              </div>
            ))}
          </section>
        ) : null}

        {method === "all" && Object.keys(resultsByMethod).length > 0 ? (
          <section className={`mt-5 grid gap-4 ${RESULT_WIDTH_CLASS}`}>
            <div className="grid gap-4 lg:grid-cols-3">
              {[
                { key: "bm25", label: "BM25" },
                { key: "vector", label: "Vector (FAISS)" },
                { key: "vector_titles", label: "Vector (FAISS + titles)" },
              ].map(({ key, label }) => {
                const methodResults = resultsByMethod[key as SearchMethod] ?? [];
                return (
                  <div
                    className="card border border-base-300 bg-base-100 shadow-md"
                    key={key}
                  >
                    <div className="card-body gap-3">
                      <div className="flex items-center justify-between">
                        <h2 className="text-lg font-semibold">{label}</h2>
                        <span className="badge badge-outline">{methodResults.length}</span>
                      </div>
                      {methodResults.length === 0 ? (
                        <p className="text-sm text-base-content/60">No results.</p>
                      ) : (
                        <div className="space-y-3">
                          {methodResults.map((result, index) => {
                            const url = getResultUrl(result);
                            const title = getResultTitle(result);
                            return (
                              <article
                                className="rounded-lg border border-base-200 p-3"
                                key={`${key}-${index}-${String(result.score ?? "")}`}
                              >
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="badge badge-primary badge-outline whitespace-nowrap text-xs">
                                    Rank {index + 1}
                                  </span>
                                  <span className="text-xs text-base-content/70 whitespace-nowrap">
                                    Score:{" "}
                                    {typeof result.score === "number"
                                      ? result.score.toFixed(4)
                                      : String(result.score ?? "n/a")}
                                  </span>
                                  {result.chunk_type === "title" ? (
                                    <span className="badge badge-secondary badge-outline whitespace-nowrap text-xs">
                                      Title match
                                    </span>
                                  ) : null}
                                </div>
                                <div className="mt-2 space-y-1">
                                  <p className="text-sm font-semibold">{title}</p>
                                  {url ? (
                                    <a
                                      className="link link-primary break-all text-xs"
                                      href={url}
                                      target="_blank"
                                      rel="noreferrer"
                                    >
                                      {url}
                                    </a>
                                  ) : (
                                    <p className="text-xs text-base-content/60">
                                      No source URL available.
                                    </p>
                                  )}
                                </div>
                                <p className="mt-2 line-clamp-4 whitespace-pre-wrap text-xs leading-5">
                                  {String(result.chunk_text ?? result.text ?? "")}
                                </p>
                              </article>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        ) : null}

        {method !== "all" && results.length > 0 ? (
          <section className={`mt-5 grid gap-3 ${RESULT_WIDTH_CLASS}`}>
            {results.map((result, index) => {
              const url = getResultUrl(result);
              const title = getResultTitle(result);
              return (
                <article
                  className="card w-full border border-base-300 bg-base-100 shadow-md"
                  key={`${index}-${String(result.score ?? "")}`}
                >
                  <div className="card-body min-w-0 gap-3">
                    <div className="flex items-center gap-2">
                      <span className="badge badge-primary badge-outline">Rank {index + 1}</span>
                      <span className="text-sm text-base-content/70">
                        Score:{" "}
                        {typeof result.score === "number"
                          ? result.score.toFixed(4)
                          : String(result.score ?? "n/a")}
                      </span>
                      {result.chunk_type === "title" ? (
                        <span className="badge badge-secondary badge-outline">Title match</span>
                      ) : null}
                    </div>
                    <div className="space-y-1">
                      <p className="text-base font-semibold">{title}</p>
                      {url ? (
                        <a
                          className="link link-primary break-all text-sm"
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {url}
                        </a>
                      ) : (
                        <p className="text-sm text-base-content/60">
                          No source URL available.
                        </p>
                      )}
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-6">
                      {String(result.chunk_text ?? result.text ?? "")}
                    </p>
                  </div>
                </article>
              );
            })}
          </section>
        ) : null}
      </main>
    </div>
  );
}
