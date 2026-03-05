"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "actsearch-authenticated";
const USER_NAME_KEY = "actsearch-user-name";
const DEMO_API_BASE_URL =
  process.env.NEXT_PUBLIC_DOCPLUS_API_BASE_URL ?? "http://127.0.0.1:5000";
const METHODS = ["bm25", "vector", "vector_e5", "hybrid_e5"] as const;
const DEFAULT_TOP_K = 10;

type SearchMethod = (typeof METHODS)[number];
type SearchApiMethod = SearchMethod | "all";
type RelevanceRating = "relevant" | "not_relevant";
type RelevantScope = "whole_document" | "part_of_document";

type SearchResult = {
  score?: number;
  text?: string;
  chunk_text?: string;
  chunk_type?: string;
  metadata?: Record<string, unknown>;
  source_path?: string;
  result_method?: SearchMethod;
};

type SearchResultsByMethod = Partial<Record<SearchMethod, SearchResult[]>>;

type SearchPipeline = {
  byMethod: Record<SearchMethod, SearchResult[]>;
  pooledBeforeDedup: SearchResult[];
  pooledAfterDedup: SearchResult[];
  finalResults: SearchResult[];
};

const getResultTitle = (result: SearchResult): string => {
  const title = result.metadata?.title;
  return typeof title === "string" ? title : "Untitled document";
};

const getResultUrl = (result: SearchResult): string => {
  const sourceUrl = result.metadata?.source_url;
  return typeof sourceUrl === "string" ? sourceUrl : "#";
};

const EMPTY_BY_METHOD: Record<SearchMethod, SearchResult[]> = {
  bm25: [],
  vector: [],
  vector_e5: [],
  hybrid_e5: [],
};

const EMPTY_PIPELINE: SearchPipeline = {
  byMethod: EMPTY_BY_METHOD,
  pooledBeforeDedup: [],
  pooledAfterDedup: [],
  finalResults: [],
};

const withMethodMetadata = (method: SearchMethod, result: SearchResult, rank: number): SearchResult => {
  const metadata = result.metadata ?? {};
  const existingScores =
    typeof metadata.score_by_method === "object" &&
    metadata.score_by_method &&
    !Array.isArray(metadata.score_by_method)
      ? metadata.score_by_method
      : {};
  const existingRanks =
    typeof metadata.rank_by_method === "object" &&
    metadata.rank_by_method &&
    !Array.isArray(metadata.rank_by_method)
      ? metadata.rank_by_method
      : {};

  return {
    ...result,
    result_method: method,
    metadata: {
      ...metadata,
      pooled_from: [method],
      score_by_method: {
        ...existingScores,
        [method]: typeof result.score === "number" ? result.score : null,
      },
      rank_by_method: {
        ...existingRanks,
        [method]: rank,
      },
    },
  };
};

const normalizeByMethodResults = (resultsByMethod: SearchResultsByMethod | undefined) =>
  METHODS.reduce<Record<SearchMethod, SearchResult[]>>((accumulator, method) => {
    const rawResults = resultsByMethod?.[method];
    const safeResults = Array.isArray(rawResults) ? rawResults : [];
    accumulator[method] = safeResults.map((result, index) =>
      withMethodMetadata(method, result, index + 1),
    );
    return accumulator;
  }, { ...EMPTY_BY_METHOD });

const dedupeResults = (results: SearchResult[]): SearchResult[] => {
  const seen = new Map<string, SearchResult>();

  for (const result of results) {
    const key =
      typeof result.source_path === "string" && result.source_path.length > 0
        ? result.source_path
        : getResultTitle(result);

    const existing = seen.get(key);
    if (!existing) {
      seen.set(key, {
        ...result,
        score: undefined,
        metadata: {
          ...result.metadata,
          pooled_from: result.result_method ? [result.result_method] : [],
        },
      });
      continue;
    }

    const pooledFrom = Array.isArray(existing.metadata?.pooled_from)
      ? existing.metadata?.pooled_from.filter(
          (method): method is SearchMethod =>
            typeof method === "string" && METHODS.includes(method as SearchMethod),
        )
      : [];

    const existingScores =
      existing.metadata?.score_by_method &&
      typeof existing.metadata.score_by_method === "object" &&
      !Array.isArray(existing.metadata.score_by_method)
        ? existing.metadata.score_by_method
        : {};

    const incomingScores =
      result.metadata?.score_by_method &&
      typeof result.metadata.score_by_method === "object" &&
      !Array.isArray(result.metadata.score_by_method)
        ? result.metadata.score_by_method
        : {};

    const existingRanks =
      existing.metadata?.rank_by_method &&
      typeof existing.metadata.rank_by_method === "object" &&
      !Array.isArray(existing.metadata.rank_by_method)
        ? existing.metadata.rank_by_method
        : {};

    const incomingRanks =
      result.metadata?.rank_by_method &&
      typeof result.metadata.rank_by_method === "object" &&
      !Array.isArray(result.metadata.rank_by_method)
        ? result.metadata.rank_by_method
        : {};

    seen.set(key, {
      ...existing,
      score: undefined,
      metadata: {
        ...existing.metadata,
        pooled_from: Array.from(
          new Set(result.result_method ? [...pooledFrom, result.result_method] : pooledFrom),
        ),
        score_by_method: {
          ...existingScores,
          ...incomingScores,
        },
        rank_by_method: {
          ...existingRanks,
          ...incomingRanks,
        },
      },
    });
  }

  return Array.from(seen.values());
};

const seededShuffle = <T,>(items: T[], seedInput: string): T[] => {
  const output = [...items];
  let seed = 0;

  for (const char of seedInput) {
    seed = (seed * 31 + char.charCodeAt(0)) >>> 0;
  }

  for (let index = output.length - 1; index > 0; index -= 1) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const swapIndex = seed % (index + 1);
    [output[index], output[swapIndex]] = [output[swapIndex], output[index]];
  }

  return output;
};

const buildPipeline = (
  query: string,
  runId: number,
  resultsByMethod: SearchResultsByMethod | undefined,
): SearchPipeline => {
  const byMethod = normalizeByMethodResults(resultsByMethod);

  const pooledBeforeDedup = METHODS.flatMap((method) => byMethod[method]);
  const pooledAfterDedup = dedupeResults(pooledBeforeDedup);
  const finalResults = seededShuffle(pooledAfterDedup, `${query}:${runId}`);

  return {
    byMethod,
    pooledBeforeDedup,
    pooledAfterDedup,
    finalResults,
  };
};

export default function DemoSearchPage() {
  const router = useRouter();
  const stepTwoRef = useRef<HTMLElement | null>(null);
  const [informationNeed, setInformationNeed] = useState("");
  const [query, setQuery] = useState("");
  const [comment, setComment] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [runId, setRunId] = useState(0);
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [searchErrors, setSearchErrors] = useState<string[]>([]);
  const [pipeline, setPipeline] = useState<SearchPipeline>(EMPTY_PIPELINE);
  const [debugMode, setDebugMode] = useState(false);
  const [ratings, setRatings] = useState<Record<string, RelevanceRating>>({});
  const [relevantScopes, setRelevantScopes] = useState<Record<string, RelevantScope>>({});
  const [relevantSections, setRelevantSections] = useState<Record<string, string>>({});
  const [resultComments, setResultComments] = useState<Record<string, string>>({});
  const [hasSubmittedRatings, setHasSubmittedRatings] = useState(false);
  const [isSubmittingToBackend, setIsSubmittingToBackend] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const canSubmit = query.trim().length > 0;
  const hasSubmittedQuery = submittedQuery.trim().length > 0;
  const isAuthenticated =
    typeof window !== "undefined" && localStorage.getItem(SESSION_KEY) === "true";

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace("/");
    }
  }, [isAuthenticated, router]);
  const allResultsRated =
    hasSubmittedQuery &&
    pipeline.finalResults.length > 0 &&
    pipeline.finalResults.every((result, index) => {
      const resultKey = result.source_path ?? `${getResultTitle(result)}-${index}`;
      return ratings[resultKey] === "relevant" || ratings[resultKey] === "not_relevant";
    });

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }
    setIsLoadingSearch(true);
    setSearchErrors([]);
    setSubmitError(null);
    setHasSubmittedRatings(false);
    setRatings({});
    setRelevantScopes({});
    setRelevantSections({});
    setResultComments({});

    try {
      const response = await fetch(`${DEMO_API_BASE_URL.replace(/\/$/, "")}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          method: "all" as SearchApiMethod,
          query: trimmedQuery,
          top_k: DEFAULT_TOP_K,
        }),
      });

      const payload = (await response.json()) as {
        errors?: string[];
        results_by_method?: SearchResultsByMethod;
      };

      if (!response.ok) {
        setPipeline(EMPTY_PIPELINE);
        setSearchErrors(payload.errors && payload.errors.length > 0 ? payload.errors : ["Search failed."]);
        return;
      }

      const nextRunId = runId + 1;
      setRunId(nextRunId);
      setSubmittedQuery(trimmedQuery);
      setPipeline(buildPipeline(trimmedQuery, nextRunId, payload.results_by_method));
      setSearchErrors(payload.errors ?? []);
      requestAnimationFrame(() => {
        stepTwoRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (error) {
      setPipeline(EMPTY_PIPELINE);
      setSearchErrors([
        error instanceof Error ? `Request failed: ${error.message}` : "Request failed with an unknown error.",
      ]);
    } finally {
      setIsLoadingSearch(false);
    }
  };

  const onFinalSubmit = async () => {
    if (!allResultsRated) {
      return;
    }

    const participantName =
      typeof window !== "undefined" ? localStorage.getItem(USER_NAME_KEY)?.trim() ?? "" : "";

    const results = pipeline.finalResults.map((result, index) => {
      const resultKey = result.source_path ?? `${getResultTitle(result)}-${index}`;
      const selectedRating = ratings[resultKey] ?? null;
      const selectedScope = relevantScopes[resultKey] ?? null;
      const sectionLabel = relevantSections[resultKey] ?? "";
      const resultComment = resultComments[resultKey] ?? "";

      return {
        ...result,
        assessment: {
          rating: selectedRating,
          relevant_scope: selectedRating === "relevant" ? selectedScope : null,
          relevant_section:
            selectedRating === "relevant" && selectedScope === "part_of_document"
              ? sectionLabel
              : "",
          comment: resultComment,
        },
      };
    });

    const payload = {
      participant_name: participantName,
      information_need: informationNeed.trim(),
      query: submittedQuery.trim(),
      general_comment: comment.trim(),
      results,
      pipeline_snapshot: {
        by_method: pipeline.byMethod,
        pooled_before_dedup: pipeline.pooledBeforeDedup,
        pooled_after_dedup: pipeline.pooledAfterDedup,
        final_results: pipeline.finalResults,
      },
    };

    setSubmitError(null);
    setIsSubmittingToBackend(true);

    try {
      const response = await fetch(`${DEMO_API_BASE_URL}/demo/submit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = (await response.json().catch(() => null)) as
        | { ok?: boolean; errors?: string[] }
        | null;

      if (!response.ok || !data?.ok) {
        const errorMessage = Array.isArray(data?.errors) && data.errors.length > 0
          ? data.errors.join(" ")
          : "Kunde inte spara formuläret.";
        throw new Error(errorMessage);
      }

      setHasSubmittedRatings(true);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Kunde inte spara formuläret.");
    } finally {
      setIsSubmittingToBackend(false);
    }
  };

  const onLogout = () => {
    localStorage.removeItem(SESSION_KEY);
    router.push("/");
  };

  const getScoreByMethod = (result: SearchResult): Partial<Record<SearchMethod, number>> => {
    const value = result.metadata?.score_by_method;
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return {};
    }

    const entries = Object.entries(value).filter(
      (entry): entry is [SearchMethod, number] =>
        METHODS.includes(entry[0] as SearchMethod) && typeof entry[1] === "number",
    );

    return Object.fromEntries(entries) as Partial<Record<SearchMethod, number>>;
  };

  const getRankByMethod = (result: SearchResult): Partial<Record<SearchMethod, number>> => {
    const value = result.metadata?.rank_by_method;
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return {};
    }

    const entries = Object.entries(value).filter(
      (entry): entry is [SearchMethod, number] =>
        METHODS.includes(entry[0] as SearchMethod) && typeof entry[1] === "number",
    );

    return Object.fromEntries(entries) as Partial<Record<SearchMethod, number>>;
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#fff8eb,#f3efe6_45%,#eaf4ff)] px-6 py-10 text-[#1e241f]">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <label
            className={`flex items-center justify-between gap-4 rounded-2xl border px-4 py-3 text-sm font-medium transition md:min-w-[22rem] ${
              debugMode
                ? "border-[#9bc7c7] bg-[#eef6f3] text-[#1f4f4f]"
                : "border-[#d8ddd3] bg-white text-[#556055]"
            }`}
          >
            <div className="flex flex-col">
              <span>Debug-läge</span>
              <span className="text-xs font-normal opacity-80">
                {debugMode ? "På: visar intern pipeline och metoddata" : "Av: visar bara resultatlistan"}
              </span>
            </div>
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] ${
                debugMode ? "bg-[#1f6e6e] text-white" : "bg-[#eef1ec] text-[#5f685f]"
              }`}
            >
              {debugMode ? "På" : "Av"}
            </span>
            <input
              className="toggle border-[#cfd4c9] bg-white text-[#1f6e6e] [--tglbg:#ffffff]"
              type="checkbox"
              checked={debugMode}
              onChange={(event) => setDebugMode(event.target.checked)}
            />
          </label>

          <div className="flex flex-wrap justify-end gap-2">
            <button
              className="rounded-full border border-[#c8cfbf] bg-white/80 px-4 py-2 text-sm text-[#425043] transition hover:border-[#1f6e6e] hover:text-[#1f6e6e]"
              type="button"
              onClick={() => router.push("/search")}
            >
              Gå till riktig söksida
            </button>
            <button
              className="rounded-full border border-[#c8cfbf] bg-white/80 px-4 py-2 text-sm text-[#425043] transition hover:border-[#1f6e6e] hover:text-[#1f6e6e]"
              type="button"
              onClick={onLogout}
            >
              Logga ut
            </button>
          </div>
        </div>

        {debugMode ? null : (
          <section className="rounded-[2rem] border border-[#d6d8cf] bg-[#fffdf8]/95 p-8 shadow-[0_24px_80px_rgba(34,42,28,0.08)]">
            <div className="max-w-4xl space-y-5 text-sm leading-7 text-[#4f5850]">
              <p className="font-medium text-[#253229]">
                Det här är en del av ett masterarbete som görs i samarbete med DataToValue.
              </p>

              <div>
                <p className="font-medium text-[#253229]">Så här går det till:</p>
                  <p>1. Skapa en sökterm. Tryck på &quot;nästa&quot;.</p>
                  <p>2. För varje dokument i listan nedan, bedöm hur relevant resultatet är utifrån din sökterm.</p>
              </div>

              <p>
                Upprepa detta så många gånger som du har tid med/orkar. Du kan komma tillbaka när
                du vill och lägga till fler söktermer. Ju mer data vi har desto bättre blir vårt
                arbete!
              </p>

              <div>
                <p>Notera att dina söktermer kommer sparas. Skriv inte in känslig data såsom patientdata.</p>
              </div>

              <p>
                Den här datan kommer användas för att utvärdera och jämföra olika söksystem och för att förbättra vårt söksystem.
              </p>
              <p>
                Vid frågor kontakta:{" "}
                <a
                  className="font-medium text-[#1f6e6e] underline decoration-[#9bc7c7] underline-offset-4"
                  href="mailto:tragardh.anna@gmail.com"
                >
                  tragardh.anna@gmail.com
                </a>
              </p>

              <p className="font-medium text-[#253229]">
                Tack för din tid! Ditt deltagande är värdefullt för oss :)
              </p>
            </div>
          </section>
        )}

        <section
          className={`rounded-[2rem] border p-8 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            hasSubmittedQuery
              ? "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-75"
              : "border-[#d6d8cf] bg-[#fffdf8]/95"
          }`}
        >
          {debugMode ? (
            <div className="mb-8">
              <p className="max-w-3xl text-sm leading-6 text-[#5e655e]">
                Varje sökning anropar `bm25`, `vector`, `vector_e5` och `hybrid_e5`, samlar deras
                topp 10-resultat från backend, tar bort dubletter och
                randomiserar ordningen innan visning.
              </p>
            </div>
          ) : null}

          <form className="flex flex-col gap-3" onSubmit={onSubmit}>
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
              1. Beskriv informationsbehov
            </p>
            <div className="space-y-2">
              <p className="text-sm leading-6 text-[#4f5850]">
                Beskriv ett exempel på ett informationsbehov som kan uppstå i ditt dagliga arbete,
                där du behöver söka i DocPlus. Exempel: &quot;Familjen vill sova med sitt spädbarn
                mellan sig. Vilken information behöver jag förmedla till föräldrarna?&quot; Eller &quot;Vilka arbetsuppgifter har undersköterskan vid assistering under en vakuumextraktion?&quot;.
              </p>
              <input
                className={`w-full rounded-2xl border px-5 py-4 text-base outline-none transition ${
                  hasSubmittedQuery
                    ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                    : "border-[#cfd4c9] bg-white text-[#1e241f] focus:border-[#1f6e6e]"
                }`}
                type="text"
                value={informationNeed}
                disabled={hasSubmittedQuery}
                onChange={(event) => setInformationNeed(event.target.value)}
              />
            </div>

            <p className="pt-2 text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
              2. Skapa en sökterm
            </p>
            <p className="text-sm leading-6 text-[#4f5850]">
              Vad skulle du skriva in i sökrutan utifrån ovanstående informationsbehov?
            </p>
            <div className="flex flex-col gap-3">
              <input
                className={`min-w-0 flex-1 rounded-2xl border px-5 py-4 text-base outline-none transition ${
                  hasSubmittedQuery
                    ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                    : "border-[#cfd4c9] bg-white text-[#1e241f] focus:border-[#1f6e6e]"
                }`}
                type="text"
                value={query}
                disabled={hasSubmittedQuery}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>

            <label className="flex flex-col gap-2">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                Valfri kommentar
              </span>
              <textarea
                className={`min-h-28 rounded-2xl border px-5 py-4 text-base outline-none transition ${
                  hasSubmittedQuery
                    ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                    : "border-[#cfd4c9] bg-white text-[#1e241f] focus:border-[#1f6e6e]"
                }`}
                value={comment}
                disabled={hasSubmittedQuery}
                onChange={(event) => setComment(event.target.value)}
              />
            </label>

            <p className="text-sm leading-6 text-[#4f5850]">
              När du är nöjd, tryck på &quot;Nästa steg&quot;. Du kan inte gå tillbaka och ändra i din information.
            </p>

            <button
              className={`self-start rounded-2xl px-6 py-4 font-semibold text-white transition ${
                canSubmit && !hasSubmittedQuery && !isLoadingSearch
                  ? "bg-[#1f6e6e] hover:bg-[#184f4f]"
                  : "cursor-not-allowed bg-[#9fb8b8]"
              }`}
              type="submit"
              disabled={!canSubmit || hasSubmittedQuery || isLoadingSearch}
            >
              {isLoadingSearch ? "Söker..." : "Nästa steg"}
            </button>

          </form>
        </section>

        {searchErrors.length > 0 ? (
          <section className="rounded-[1.5rem] border border-[#f0b79f] bg-[#ffe8dc] p-4 text-sm text-[#7a2e0d]">
            {searchErrors.map((error) => (
              <p key={error}>{error}</p>
            ))}
          </section>
        ) : null}

        <section
          ref={stepTwoRef}
          className={`rounded-[2rem] border p-8 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            hasSubmittedRatings
              ? "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-75"
              : hasSubmittedQuery
              ? "border-[#d6d8cf] bg-[#fffdf8]/95"
              : "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-60"
          }`}
        >
          <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
            2. Bedöm relevans
          </h2>
          <div className="mt-3 space-y-1 text-sm leading-6 text-[#4f5850]">
            <p>För varje sökträff, klicka på länken för att få upp dokumentet och bedöm hur relevant dokumentet är utifrån söktermen.</p>
            <p>Observera att rangordningen är slumpartad.</p>
          </div>
          <div className={`mt-4 grid gap-3 text-sm text-[#6b7468] ${debugMode ? "md:grid-cols-4" : "md:grid-cols-1"}`}>
            <div className="rounded-2xl bg-[#f8f5ee] px-4 py-3">
              <span className="block text-xs uppercase tracking-[0.16em] text-[#7d7568]">
                Din sökterm
              </span>
              <span>{submittedQuery.trim() || "Ingen query än"}</span>
            </div>
            {debugMode ? (
              <>
                <div className="rounded-2xl bg-[#f8f5ee] px-4 py-3">
                  <span className="block text-xs uppercase tracking-[0.16em] text-[#7d7568]">
                    Pool före dedupe
                  </span>
                  <span>{pipeline.pooledBeforeDedup.length} resultat</span>
                </div>
                <div className="rounded-2xl bg-[#f8f5ee] px-4 py-3">
                  <span className="block text-xs uppercase tracking-[0.16em] text-[#7d7568]">
                    Efter dedupe
                  </span>
                  <span>{pipeline.pooledAfterDedup.length} unika resultat</span>
                </div>
                <div className="rounded-2xl bg-[#f8f5ee] px-4 py-3">
                  <span className="block text-xs uppercase tracking-[0.16em] text-[#7d7568]">
                    Visas nu
                  </span>
                  <span>{pipeline.finalResults.length} randomiserade resultat</span>
                </div>
              </>
            ) : null}
          </div>

          {hasSubmittedQuery ? (
            <div className="mt-6 grid gap-4">
              {pipeline.finalResults.map((result, index) => {
              const pooledFrom = Array.isArray(result.metadata?.pooled_from)
                ? result.metadata.pooled_from.filter((value): value is string => typeof value === "string")
                : [];
              const scoreByMethod = getScoreByMethod(result);
              const rankByMethod = getRankByMethod(result);
              const resultKey = result.source_path ?? `${getResultTitle(result)}-${index}`;
              const selectedRating = ratings[resultKey];
              const selectedScope = relevantScopes[resultKey];
              const sectionLabel = relevantSections[resultKey] ?? "";
              const resultComment = resultComments[resultKey] ?? "";

                return (
                  <article
                    className="rounded-[1.5rem] border border-[#d6d8cf] bg-white/90 p-6 shadow-[0_16px_50px_rgba(35,44,32,0.06)]"
                    key={`${resultKey}-${index}`}
                  >
                    {!debugMode ? (
                      <div
                        className={`grid gap-4 md:items-start ${
                          selectedRating === "relevant"
                            ? "xl:grid-cols-[minmax(0,1.1fr)_12rem_18rem_minmax(0,1fr)]"
                            : "lg:grid-cols-[minmax(0,1.1fr)_12rem_minmax(0,1fr)]"
                        }`}
                      >
                        <div className="min-w-0">
                          <h2 className="font-serif text-xl text-[#203327]">{getResultTitle(result)}</h2>
                          <a
                            className="mt-4 inline-flex break-all text-sm font-medium text-[#1f6e6e] underline decoration-[#9bc7c7] underline-offset-4"
                            href={getResultUrl(result)}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Öppna exempellänk
                          </a>
                        </div>

                        <fieldset>
                          <legend className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                            Relevans
                          </legend>
                          <div className="mt-3 flex flex-col gap-2">
                            <label className="flex items-center gap-2 rounded-full border border-[#d9ddd4] px-4 py-2 text-sm text-[#465048]">
                            <input
                              checked={selectedRating === "relevant"}
                              className="h-4 w-4 accent-[#1f6e6e]"
                              disabled={hasSubmittedRatings}
                              name={`rating-${resultKey}`}
                              type="radio"
                              onChange={() =>
                                  setRatings((current) => ({ ...current, [resultKey]: "relevant" }))
                                }
                              />
                              Relevant
                            </label>
                            <label className="flex items-center gap-2 rounded-full border border-[#d9ddd4] px-4 py-2 text-sm text-[#465048]">
                            <input
                              checked={selectedRating === "not_relevant"}
                              className="h-4 w-4 accent-[#1f6e6e]"
                              disabled={hasSubmittedRatings}
                              name={`rating-${resultKey}`}
                              type="radio"
                              onChange={() =>
                                  setRatings((current) => ({
                                    ...current,
                                    [resultKey]: "not_relevant",
                                  }))
                                }
                              />
                              Inte relevant
                            </label>
                          </div>
                        </fieldset>

                        {selectedRating === "relevant" ? (
                          <div className="space-y-4 rounded-[1.25rem] border border-[#dfe4db] bg-[#f8fbf8] p-4">
                            <div className="space-y-2">
                              <p className="text-sm font-medium text-[#2f3a31]">
                                Var i dokumentet finns den relevanta informationen?
                              </p>
                              <div className="flex flex-col gap-2">
                                <label className="flex items-center gap-2 text-sm text-[#465048]">
                                  <input
                                    checked={selectedScope === "whole_document"}
                                    className="h-4 w-4 accent-[#1f6e6e]"
                                    disabled={hasSubmittedRatings}
                                    name={`scope-${resultKey}`}
                                    type="radio"
                                    onChange={() =>
                                      setRelevantScopes((current) => ({
                                        ...current,
                                        [resultKey]: "whole_document",
                                      }))
                                    }
                                  />
                                  Hela dokumentet
                                </label>
                                <label className="flex items-center gap-2 text-sm text-[#465048]">
                                  <input
                                    checked={selectedScope === "part_of_document"}
                                    className="h-4 w-4 accent-[#1f6e6e]"
                                    disabled={hasSubmittedRatings}
                                    name={`scope-${resultKey}`}
                                    type="radio"
                                    onChange={() =>
                                      setRelevantScopes((current) => ({
                                        ...current,
                                        [resultKey]: "part_of_document",
                                      }))
                                    }
                                  />
                                  En del/delar av dokumentet
                                </label>
                              </div>
                            </div>

                            {selectedScope === "part_of_document" ? (
                              <label className="flex flex-col gap-2">
                                <span className="text-sm text-[#2f3a31]">
                                  Ange i vilken/vilka delar/kapitel
                                </span>
                                <input
                                  className={`rounded-2xl border px-4 py-3 text-sm outline-none transition ${
                                    hasSubmittedRatings
                                      ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                                      : "border-[#cfd4c9] bg-white focus:border-[#1f6e6e]"
                                  }`}
                                  disabled={hasSubmittedRatings}
                                  type="text"
                                  value={sectionLabel}
                                  onChange={(event) =>
                                    setRelevantSections((current) => ({
                                      ...current,
                                      [resultKey]: event.target.value,
                                    }))
                                  }
                                />
                              </label>
                            ) : null}
                          </div>
                        ) : null}

                        <label className={selectedRating === "relevant" ? "" : "lg:col-start-3"}>
                          <span className="text-sm text-[#2f3a31]">Valfri kommentar</span>
                          <textarea
                            className={`mt-2 min-h-24 w-full rounded-2xl border px-4 py-3 text-sm outline-none transition ${
                              hasSubmittedRatings
                                ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                                : "border-[#cfd4c9] bg-white focus:border-[#1f6e6e]"
                            }`}
                            disabled={hasSubmittedRatings}
                            value={resultComment}
                            onChange={(event) =>
                              setResultComments((current) => ({
                                ...current,
                                [resultKey]: event.target.value,
                              }))
                            }
                          />
                        </label>
                      </div>
                    ) : (
                      <>
                        <h2 className="font-serif text-xl text-[#203327]">{getResultTitle(result)}</h2>
                        <a
                          className="mt-4 inline-flex break-all text-sm font-medium text-[#1f6e6e] underline decoration-[#9bc7c7] underline-offset-4"
                          href={getResultUrl(result)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Öppna exempellänk
                        </a>
                      </>
                    )}

                  {debugMode ? (
                    <div className="mt-4 space-y-3 rounded-2xl bg-[#f8f5ee] p-4 text-sm text-[#4c564f]">
                      <div className="flex flex-wrap gap-2 text-xs">
                        <span className="rounded-full bg-[#f3efe2] px-3 py-1 text-[#6d624e]">
                          {String(result.metadata?.category ?? "Okänd kategori")}
                        </span>
                        <span className="rounded-full bg-[#eef1ff] px-3 py-1 text-[#4f5d8a]">
                          Chunk {String(result.chunk_type ?? "body")}
                        </span>
                        {pooledFrom.map((method) => (
                          <span
                            className="rounded-full border border-[#d9ddd4] px-3 py-1 uppercase tracking-[0.14em] text-[#556055]"
                            key={`${result.source_path}-${method}`}
                          >
                            {method}
                          </span>
                        ))}
                      </div>

                      <div className="rounded-xl bg-white/80 p-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                          Score per metod
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs">
                          {METHODS.map((method) => (
                            <span
                              className="rounded-full border border-[#d9ddd4] px-3 py-1 text-[#465048]"
                              key={`${result.source_path}-score-${method}`}
                            >
                              {method}:{" "}
                              {typeof scoreByMethod[method] === "number"
                                ? scoreByMethod[method]?.toFixed(2)
                                : "-"}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl bg-white/80 p-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                          Rank per metod
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs">
                          {METHODS.map((method) => (
                            <span
                              className="rounded-full border border-[#d9ddd4] px-3 py-1 text-[#465048]"
                              key={`${result.source_path}-rank-${method}`}
                            >
                              {method}:{" "}
                              {typeof rankByMethod[method] === "number" ? `#${rankByMethod[method]}` : "-"}
                            </span>
                          ))}
                        </div>
                      </div>

                      <p className="leading-6">{String(result.chunk_text ?? result.text ?? "")}</p>

                      <pre className="overflow-x-auto whitespace-pre-wrap text-xs leading-5 text-[#465048]">
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    </div>
                  ) : null}
                  </article>
                );
              })}
            </div>
          ) : (
            <div className="mt-6 rounded-[1.5rem] border border-dashed border-[#d4d7cf] bg-white/60 p-6 text-sm text-[#6a7169]">
              Skriv in en sökterm och tryck på `Nästa` för att visa sökträffar.
            </div>
          )}
        </section>

        <section
          className={`rounded-[2rem] border p-8 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            allResultsRated
              ? "border-[#d6d8cf] bg-[#fffdf8]/95"
              : "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-60"
          }`}
        >
          <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
            3. Skicka in
          </h2>
          <p className="mt-3 text-sm leading-6 text-[#4f5850]">
            När alla sökträffar är bedömda kan du skicka in dina svar.
          </p>
          {submitError ? (
            <p className="mt-4 rounded-2xl border border-[#f0b79f] bg-[#ffe8dc] px-4 py-3 text-sm text-[#7a2e0d]">
              {submitError}
            </p>
          ) : null}
          <button
            className={`mt-5 rounded-2xl px-6 py-4 font-semibold text-white transition ${
              allResultsRated && !hasSubmittedRatings && !isSubmittingToBackend
                ? "bg-[#1f6e6e] hover:bg-[#184f4f]"
                : hasSubmittedRatings || isSubmittingToBackend
                  ? "cursor-default bg-[#7fa2a2]"
                  : "cursor-not-allowed bg-[#9fb8b8]"
            }`}
            type="button"
            disabled={!allResultsRated || hasSubmittedRatings || isSubmittingToBackend}
            onClick={onFinalSubmit}
          >
            {hasSubmittedRatings ? "Inskickat" : isSubmittingToBackend ? "Skickar..." : "Skicka in"}
          </button>
        </section>

        {debugMode ? (
          <section className="grid gap-4">
            <div className="rounded-[1.5rem] border border-[#d6d8cf] bg-[#fcfbf8] p-6">
              <h2 className="font-serif text-2xl text-[#203327]">Före pooling</h2>
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                {METHODS.map((method) => (
                  <div className="rounded-2xl bg-white p-4" key={method}>
                    <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                      {method}
                    </p>
                    <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-5 text-[#465048]">
                      {JSON.stringify(pipeline.byMethod[method], null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[1.5rem] border border-[#d6d8cf] bg-[#fcfbf8] p-6">
              <h2 className="font-serif text-2xl text-[#203327]">Efter pooling, före dedupe</h2>
              <pre className="mt-4 overflow-x-auto whitespace-pre-wrap text-xs leading-5 text-[#465048]">
                {JSON.stringify(pipeline.pooledBeforeDedup, null, 2)}
              </pre>
            </div>

            <div className="rounded-[1.5rem] border border-[#d6d8cf] bg-[#fcfbf8] p-6">
              <h2 className="font-serif text-2xl text-[#203327]">Efter dedupe</h2>
              <pre className="mt-4 overflow-x-auto whitespace-pre-wrap text-xs leading-5 text-[#465048]">
                {JSON.stringify(pipeline.pooledAfterDedup, null, 2)}
              </pre>
            </div>

            <div className="rounded-[1.5rem] border border-[#d6d8cf] bg-[#fcfbf8] p-6">
              <h2 className="font-serif text-2xl text-[#203327]">Slutligt randomiserat resultat</h2>
              <pre className="mt-4 overflow-x-auto whitespace-pre-wrap text-xs leading-5 text-[#465048]">
                {JSON.stringify(pipeline.finalResults, null, 2)}
              </pre>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
