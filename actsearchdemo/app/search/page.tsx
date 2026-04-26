"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "actsearch-authenticated";
const DEFAULT_API_BASE_URL =
  process.env.NEXT_PUBLIC_DOCPLUS_API_BASE_URL ?? "http://127.0.0.1:5000";

type SearchMethod =
  | "bm25"
  | "sqlite_fts"
  | "vector"
  | "vector_e5"
  | "hybrid_e5"
  | "docplus_live"
  | "all";

type SearchResult = {
  score?: number;
  text?: string;
  preview_text?: string;
  chunk_text?: string;
  chunk_type?: string;
  section_heading?: string;
  section_index?: number;
  section_level?: number;
  section_text?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
};

type SearchResultsByMethod = Partial<Record<SearchMethod, SearchResult[]>>;
type ClickTrackPayload = {
  search_id: string;
  query: string;
  requested_method: SearchMethod;
  result_method: SearchMethod;
  rank: number;
  score: string;
  title: string;
  url: string;
  chunk_type: string;
  source_path: string;
};
type RatingTrackPayload = {
  search_id: string;
  query: string;
  requested_method: SearchMethod;
  result_method: SearchMethod;
  document: string;
  title: string;
  url: string;
  source_path: string;
  user_score: number;
};

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
const MATCHED_CHUNK_PREVIEW_LIMIT = 500;

const getResultSectionHeading = (result: SearchResult): string | undefined => {
  const directHeading = getStringValue(result.section_heading);
  if (directHeading) {
    return directHeading;
  }
  return getMetadataValue(result.metadata, ["section_heading"]);
};

const getResultPreviewText = (result: SearchResult): string => {
  const preview = getStringValue(result.preview_text);
  if (preview) {
    return preview;
  }
  const chunk = getStringValue(result.chunk_text);
  if (chunk) {
    return chunk;
  }
  return getStringValue(result.text) ?? "";
};

const getResultChunkText = (result: SearchResult): string => {
  const sectionText = getStringValue(result.section_text);
  if (sectionText) {
    return sectionText;
  }
  const metadataSectionText = getMetadataValue(result.metadata, ["section_text"]);
  if (metadataSectionText) {
    return metadataSectionText;
  }
  const chunk = getStringValue(result.chunk_text);
  if (chunk) {
    return chunk;
  }
  const fullText = getStringValue(result.text);
  if (fullText) {
    return fullText;
  }
  return getResultPreviewText(result);
};

export default function SearchPage() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [resultsByMethod, setResultsByMethod] = useState<SearchResultsByMethod>({});
  const [searchId, setSearchId] = useState<string>("");
  const [lastRequestedMethod, setLastRequestedMethod] = useState<SearchMethod>("hybrid_e5");
  const [lastSearchQuery, setLastSearchQuery] = useState<string>("");
  const [resultRatings, setResultRatings] = useState<Record<string, number>>({});
  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});
  const [method, setMethod] = useState<SearchMethod>("hybrid_e5");
  const [query, setQuery] = useState("");
  const [parsedDir] = useState("output/parsed");
  const [sqliteFtsPath] = useState("output/sqlite_fts/docplus_fts.sqlite3");
  const [indexPath] = useState("output/vector_index/docplus.faiss");
  const [metadataPath] = useState(
    "output/vector_index/docplus_metadata.jsonl",
  );
  const [e5IndexPath] = useState("output/vector_index_e5/docplus.faiss");
  const [e5MetadataPath] = useState(
    "output/vector_index_e5/docplus_metadata.jsonl",
  );
  const [device] = useState("auto");
  const [topK, setTopK] = useState("5");
  const [apiBaseUrl] = useState(DEFAULT_API_BASE_URL);

  useEffect(() => {
    const isAuthenticated = localStorage.getItem(SESSION_KEY) === "true";
    if (!isAuthenticated) {
      router.replace("/");
      return;
    }
    setIsReady(true);
  }, [router]);

  const canSubmit = useMemo(() => query.trim().length > 0 && !isLoading, [isLoading, query]);

  const scoreToText = (value: unknown): string => {
    if (typeof value === "number") {
      return Number.isFinite(value) ? value.toString() : "";
    }
    return getStringValue(value) ?? "";
  };

  const trackResultClick = (payload: ClickTrackPayload): void => {
    const endpoint = `${apiBaseUrl.replace(/\/$/, "")}/search/click`;
    const body = JSON.stringify(payload);

    if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
      const queued = navigator.sendBeacon(endpoint, new Blob([body], { type: "application/json" }));
      if (queued) {
        return;
      }
    }

    void fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => undefined);
  };

  const onResultClick = (
    result: SearchResult,
    rank: number,
    resultMethod: SearchMethod,
    title: string,
    url: string,
  ): void => {
    if (!searchId) {
      return;
    }
    trackResultClick({
      search_id: searchId,
      query: lastSearchQuery,
      requested_method: lastRequestedMethod,
      result_method: resultMethod,
      rank,
      score: scoreToText(result.score),
      title,
      url,
      chunk_type: getStringValue(result.chunk_type) ?? "",
      source_path: getStringValue(result.source_path) ?? "",
    });
  };

  const trackResultRating = async (payload: RatingTrackPayload): Promise<void> => {
    const endpoint = `${apiBaseUrl.replace(/\/$/, "")}/search/rating`;
    await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  };

  const onRateResult = async (
    result: SearchResult,
    rank: number,
    resultMethod: SearchMethod,
    title: string,
    url: string,
    userScore: number,
  ): Promise<void> => {
    if (!searchId) {
      return;
    }
    const resultKey = `${searchId}:${resultMethod}:${rank}`;
    setResultRatings((previous) => ({ ...previous, [resultKey]: userScore }));
    try {
      await trackResultRating({
        search_id: searchId,
        query: lastSearchQuery,
        requested_method: lastRequestedMethod,
        result_method: resultMethod,
        document: title || getStringValue(result.source_path) || url,
        title,
        url,
        source_path: getStringValue(result.source_path) ?? "",
        user_score: userScore,
      });
    } catch {
      // Ratings are best-effort telemetry and should not block usage.
    }
  };

  const toggleExpandedChunk = (chunkKey: string): void => {
    setExpandedChunks((previous) => ({
      ...previous,
      [chunkKey]: !previous[chunkKey],
    }));
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setErrors([]);
    setResults([]);
    setResultsByMethod({});
    setSearchId("");
    setResultRatings({});
    setExpandedChunks({});

    const submittedMethod = method;

    try {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method,
          query,
          parsed_dir: parsedDir,
          sqlite_fts_path: sqliteFtsPath,
          index_path: indexPath,
          metadata_path: metadataPath,
          e5_index_path: e5IndexPath,
          e5_metadata_path: e5MetadataPath,
          device,
          top_k: topK,
        }),
      });

      const payload = (await response.json()) as {
        search_id?: string;
        errors?: string[];
        results?: SearchResult[];
        results_by_method?: SearchResultsByMethod;
      };

      if (!response.ok) {
        setErrors(payload.errors && payload.errors.length ? payload.errors : ["Search failed."]);
        return;
      }

      setSearchId(payload.search_id ?? "");
      setLastRequestedMethod(submittedMethod);
      setLastSearchQuery(query.trim());
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
                  ActSearch Demo
                </h1>
                <p className="mt-1 text-sm text-base-content/70">
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  className="btn btn-outline btn-sm"
                  type="button"
                  onClick={() => router.push("/evaluation-form")}
                >
                  Gå till insamling av utvärderingsdata
                </button>
              </div>
            </div>

            <form className="flex flex-col gap-4" onSubmit={onSubmit}>
              <label className="form-control w-full">
                <input
                  className="input input-bordered input-primary w-full"
                  type="text"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Sök efter dokument..."
                  required
                />
              </label>

              <div className="flex flex-row gap-4">
                <label className="form-control w-full">
                  <div className="label">
                    <span className="label-text text-base">Söktyp</span>
                  </div>
                  <select
                    className="select select-bordered w-full"
                    value={method}
                    onChange={(event) => setMethod(event.target.value as SearchMethod)}
                  >
                    <option value="bm25">BM25</option>
                    <option value="sqlite_fts">SQLite FTS (STS-gruppens version)</option>
                    <option value="vector">Vector (FAISS)</option>
                    <option value="vector_e5">Vector (E5 large instruct)</option>
                    <option value="hybrid_e5">Hybrid (BM25 + E5)</option>
                    <option value="docplus_live">Docplus Live (web)</option>
                    <option value="all">All (side-by-side)</option>
                  </select>
                </label>

                <label className="form-control w-full">
                  <div className="label">
                    <span className="label-text text-base">Visa antal sökträffar</span>
                  </div>
                  <input
                    className="input input-bordered w-full"
                    type="number"
                    min={1}
                    value={topK}
                    onChange={(event) => setTopK(event.target.value)}
                  />
                </label>
              </div>

              <div className="pt-1">
                <button className="btn btn-primary btn-wide" type="submit" disabled={!canSubmit}>
                  {isLoading ? (
                    <>
                      <span className="loading loading-spinner loading-sm" />
                      Söker
                    </>
                  ) : (
                    "Sök"
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
                { key: "sqlite_fts", label: "SQLite FTS" },
                { key: "vector", label: "Vector (FAISS)" },
                { key: "vector_e5", label: "Vector (E5 large instruct)" },
                { key: "hybrid_e5", label: "Hybrid (BM25 + E5)" },
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
                            const sectionHeading = getResultSectionHeading(result);
                            const chunkText = getResultChunkText(result);
                            const resultMethod = key as SearchMethod;
                            const resultKey = `${searchId}:${resultMethod}:${index + 1}`;
                            const isExpanded = expandedChunks[resultKey] ?? false;
                            const isChunkLong =
                              chunkText.length > MATCHED_CHUNK_PREVIEW_LIMIT;
                            const visibleChunkText =
                              isChunkLong && !isExpanded
                                ? `${chunkText.slice(0, MATCHED_CHUNK_PREVIEW_LIMIT)}...`
                                : chunkText;
                            const selectedRating = resultRatings[resultKey] ?? 0;
                            return (
                              <article
                                className="rounded-lg border border-base-200 p-3"
                                key={`${key}-${index}-${String(result.score ?? "")}`}
                              >
                                <div className="flex flex-wrap items-center gap-2">
                                  {result.chunk_type === "title" ? (
                                    <span className="badge badge-secondary badge-outline whitespace-nowrap text-xs">
                                      Title match
                                    </span>
                                  ) : null}
                                  {sectionHeading ? (
                                    <span className="badge badge-accent badge-outline text-xs">
                                      {sectionHeading}
                                    </span>
                                  ) : null}
                                </div>
                                <div className="mt-2 space-y-1">
                                  {url ? (
                                    <a
                                      className="link link-primary break-all text-sm font-semibold"
                                      href={url}
                                      target="_blank"
                                      rel="noreferrer"
                                      onClick={() =>
                                        onResultClick(
                                          result,
                                          index + 1,
                                          key as SearchMethod,
                                          title,
                                          url,
                                        )
                                      }
                                    >
                                      {title}
                                    </a>
                                  ) : (
                                    <p className="text-sm font-semibold">{title}</p>
                                  )}
                                </div>
                                <div className="mt-3 rounded-lg border border-base-200 bg-base-200/50 p-3">
                                  <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.14em] text-base-content/55">
                                    Matched chunk
                                  </p>
                                  <p className="whitespace-pre-wrap break-words text-xs leading-5 text-base-content/85">
                                    {visibleChunkText}
                                  </p>
                                  {isChunkLong ? (
                                    <button
                                      className="btn btn-link btn-xs mt-2 h-auto min-h-0 px-0"
                                      type="button"
                                      onClick={() => toggleExpandedChunk(resultKey)}
                                    >
                                      {isExpanded ? "Visa mindre" : "Visa hela"}
                                    </button>
                                  ) : null}
                                </div>
                                <div className="mt-2 flex items-center gap-1">
                                  {[1, 2, 3, 4, 5].map((star) => (
                                    <button
                                      key={`${resultKey}-star-${star}`}
                                      type="button"
                                      className={`btn btn-ghost btn-xs min-h-0 h-7 px-1 ${
                                        selectedRating >= star ? "text-warning" : "text-base-content/40"
                                      }`}
                                      onClick={() =>
                                        onRateResult(result, index + 1, resultMethod, title, url ?? "", star)
                                      }
                                      aria-label={`Rate ${star} star${star > 1 ? "s" : ""}`}
                                      title={`Rate ${star} star${star > 1 ? "s" : ""}`}
                                    >
                                      ★
                                    </button>
                                  ))}
                                  <span className="ml-1 text-xs text-base-content/60">
                                    {selectedRating > 0 ? `${selectedRating}/5` : "Rate result"}
                                  </span>
                                </div>
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
              const sectionHeading = getResultSectionHeading(result);
              const chunkText = getResultChunkText(result);
              const resultMethod = lastRequestedMethod;
              const resultKey = `${searchId}:${resultMethod}:${index + 1}`;
              const isExpanded = expandedChunks[resultKey] ?? false;
              const isChunkLong = chunkText.length > MATCHED_CHUNK_PREVIEW_LIMIT;
              const visibleChunkText =
                isChunkLong && !isExpanded
                  ? `${chunkText.slice(0, MATCHED_CHUNK_PREVIEW_LIMIT)}...`
                  : chunkText;
              const selectedRating = resultRatings[resultKey] ?? 0;
              return (
                <article
                  className="card w-full border border-base-300 bg-base-100 shadow-md"
                  key={`${index}-${String(result.score ?? "")}`}
                >
                  <div className="card-body min-w-0 gap-3">
                    <div className="flex items-center gap-2">
                      {result.chunk_type === "title" ? (
                        <span className="badge badge-secondary badge-outline">Title match</span>
                      ) : null}
                      {sectionHeading ? (
                        <span className="badge badge-accent badge-outline">{sectionHeading}</span>
                      ) : null}
                    </div>
                    <div className="space-y-1">
                      {url ? (
                        <a
                          className="link link-primary break-all text-base font-semibold"
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                          onClick={() => onResultClick(result, index + 1, lastRequestedMethod, title, url)}
                        >
                          {title}
                        </a>
                      ) : (
                        <p className="text-base font-semibold">{title}</p>
                      )}
                    </div>
                    <div className="rounded-xl border border-base-200 bg-base-200/50 p-4">
                      <p className="mb-2 text-xs font-medium uppercase tracking-[0.16em] text-base-content/55">
                        Matched chunk
                      </p>
                      <p className="whitespace-pre-wrap break-words text-sm leading-6 text-base-content/85">
                        {visibleChunkText}
                      </p>
                      {isChunkLong ? (
                        <button
                          className="btn btn-link btn-sm mt-2 h-auto min-h-0 px-0"
                          type="button"
                          onClick={() => toggleExpandedChunk(resultKey)}
                        >
                          {isExpanded ? "Visa mindre" : "Visa hela"}
                        </button>
                      ) : null}
                    </div>
                    <div className="flex items-center gap-1">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <button
                          key={`${resultKey}-star-${star}`}
                          type="button"
                          className={`btn btn-ghost btn-sm min-h-0 h-8 px-1 ${
                            selectedRating >= star ? "text-warning" : "text-base-content/40"
                          }`}
                          onClick={() =>
                            onRateResult(result, index + 1, resultMethod, title, url ?? "", star)
                          }
                          aria-label={`Rate ${star} star${star > 1 ? "s" : ""}`}
                          title={`Rate ${star} star${star > 1 ? "s" : ""}`}
                        >
                          ★
                        </button>
                      ))}
                      <span className="ml-1 text-sm text-base-content/60">
                        {selectedRating > 0 ? `${selectedRating}/5` : "Rate result"}
                      </span>
                    </div>
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
