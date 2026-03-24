"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "actsearch-authenticated";
const USER_NAME_KEY = "actsearch-user-name";
const DEMO_API_BASE_URL =
  process.env.NEXT_PUBLIC_DOCPLUS_API_BASE_URL ?? "http://127.0.0.1:5000";
const METHODS = ["bm25", "dense_e5", "hybrid_e5"] as const;
const DEFAULT_TOP_K = 5;

type SearchMethod = (typeof METHODS)[number];
type SearchApiMethod = SearchMethod | "evaluation_form_search";
type RelevanceRating = "relevant" | "not_relevant";
type RelevantScope = "whole_document" | "part_of_document";

type SearchResult = {
  score?: number;
  chunk_id?: number;
  text?: string;
  preview_text?: string;
  chunk_text?: string;
  chunk_type?: string;
  section_heading?: string;
  section_index?: number;
  section_level?: number;
  section_text?: string;
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

type DummyDoc = {
  id: string;
  title: string;
  url: string;
  category: string;
  text: string;
};

const getResultTitle = (result: SearchResult): string => {
  const title = result.metadata?.title;
  return typeof title === "string" ? title : "Untitled document";
};

const getResultUrl = (result: SearchResult): string => {
  const sourceUrl = result.metadata?.source_url;
  return typeof sourceUrl === "string" ? sourceUrl : "#";
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

const getResultSectionHeading = (result: SearchResult): string | undefined => {
  const directHeading = getStringValue(result.section_heading);
  if (directHeading) {
    return directHeading;
  }
  return getMetadataValue(result.metadata, ["section_heading"]);
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
  const preview = getStringValue(result.preview_text);
  if (preview) {
    return preview;
  }
  return getStringValue(result.text) ?? "";
};

const EMPTY_BY_METHOD: Record<SearchMethod, SearchResult[]> = {
  bm25: [],
  dense_e5: [],
  hybrid_e5: [],
};

const EMPTY_PIPELINE: SearchPipeline = {
  byMethod: EMPTY_BY_METHOD,
  pooledBeforeDedup: [],
  pooledAfterDedup: [],
  finalResults: [],
};

const DUMMY_DOCS: DummyDoc[] = [
  {
    id: "dummy-1",
    title: "Handläggning av feber hos vuxna",
    url: "https://dummy.local/doc/feber-vuxna",
    category: "Akutmedicin",
    text: "Översikt av initial bedömning, vitalparametrar och uppföljning vid feber.",
  },
  {
    id: "dummy-2",
    title: "Säker läkemedelshantering på avdelning",
    url: "https://dummy.local/doc/lakemedelshantering",
    category: "Patientsäkerhet",
    text: "Rutiner för dubbelkontroll, signering och avvikelsehantering.",
  },
  {
    id: "dummy-3",
    title: "Postoperativ smärtbehandling",
    url: "https://dummy.local/doc/postop-smarta",
    category: "Anestesi",
    text: "Rekommendationer för multimodal smärtbehandling och monitorering.",
  },
  {
    id: "dummy-4",
    title: "Vård av nyfött barn första dygnet",
    url: "https://dummy.local/doc/nyfott-forsta-dygnet",
    category: "Neonatalvård",
    text: "Basala kontroller och stödjande vård första dygnet efter förlossning.",
  },
  {
    id: "dummy-5",
    title: "Trycksårsprevention i slutenvård",
    url: "https://dummy.local/doc/trycksarsprevention",
    category: "Omvårdnad",
    text: "Riskbedömning, lägesändring och hudinspektion enligt lokala riktlinjer.",
  },
  {
    id: "dummy-6",
    title: "Fallriskbedömning och åtgärder",
    url: "https://dummy.local/doc/fallrisk-atgarder",
    category: "Omvårdnad",
    text: "Strukturerad fallriskbedömning och förebyggande åtgärder på vårdavdelning.",
  },
  {
    id: "dummy-7",
    title: "Dokumentation i patientjournal",
    url: "https://dummy.local/doc/journaldokumentation",
    category: "Administration",
    text: "Principer för saklig, tydlig och spårbar dokumentation i journal.",
  },
  {
    id: "dummy-8",
    title: "Basal hygien i vård och omsorg",
    url: "https://dummy.local/doc/basal-hygien",
    category: "Vårdhygien",
    text: "Handhygien, skyddsutrustning och rutiner för att förebygga smittspridning.",
  },
  {
    id: "dummy-9",
    title: "Handläggning av dehydrering hos äldre",
    url: "https://dummy.local/doc/dehydrering-aldre",
    category: "Geriatrik",
    text: "Bedömning av vätskebrist och rekommenderad initial behandling.",
  },
  {
    id: "dummy-10",
    title: "Säker överrapportering mellan team",
    url: "https://dummy.local/doc/overrapportering",
    category: "Kommunikation",
    text: "Standardiserad överrapportering med fokus på patientsäkerhet och ansvar.",
  },
  {
    id: "dummy-11",
    title: "Provtagning och provhantering",
    url: "https://dummy.local/doc/provtagning-hantering",
    category: "Laboratoriemedicin",
    text: "Steg för korrekt provtagning, märkning och transport till lab.",
  },
  {
    id: "dummy-12",
    title: "Triagering på akutmottagning",
    url: "https://dummy.local/doc/triagering-akut",
    category: "Akutmedicin",
    text: "Prioriteringsprinciper och flöde vid triagering av inkommande patienter.",
  },
];

const DUMMY_DOC_ORDER_BY_METHOD: Record<SearchMethod, string[]> = {
  bm25: ["dummy-1", "dummy-2", "dummy-3", "dummy-4", "dummy-5", "dummy-6", "dummy-7", "dummy-8", "dummy-9", "dummy-10", "dummy-11", "dummy-12"],
  dense_e5: ["dummy-8", "dummy-3", "dummy-1", "dummy-10", "dummy-2", "dummy-12", "dummy-4", "dummy-5", "dummy-9", "dummy-11", "dummy-6", "dummy-7"],
  hybrid_e5: ["dummy-3", "dummy-8", "dummy-1", "dummy-10", "dummy-2", "dummy-4", "dummy-12", "dummy-5", "dummy-6", "dummy-9", "dummy-7", "dummy-11"],
};

const hashString = (value: string): number => {
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return hash;
};

const buildDummyMethodResults = (
  method: SearchMethod,
  query: string,
  topK: number,
): SearchResult[] => {
  const queryHash = hashString(`${method}:${query.toLowerCase()}`);
  const baseOrder = DUMMY_DOC_ORDER_BY_METHOD[method];

  return baseOrder.slice(0, topK).map((docId, index) => {
    const doc = DUMMY_DOCS.find((item) => item.id === docId);
    if (!doc) {
      return {
        score: 0,
        text: "Saknar dummydata.",
        metadata: { title: "Okänd dummyträff", source_url: "#" },
        source_path: "dummy/missing.json",
      };
    }

    const score = Number((1 - index * 0.04 + (queryHash % 17) * 0.001).toFixed(4));
    return {
      score,
      chunk_id: index,
      text: `${doc.text} Matchad mot sökfrågan: "${query}".`,
      metadata: {
        title: doc.title,
        source_url: doc.url,
        category: doc.category,
        source: "dummydata",
      },
      source_path: `dummy/${doc.id}.json`,
      chunk_type: index % 4 === 0 ? "title" : "body",
    };
  });
};

const runDummySearch = async (query: string, topK: number): Promise<SearchResultsByMethod> => {
  await new Promise((resolve) => setTimeout(resolve, 180));

  return {
    bm25: buildDummyMethodResults("bm25", query, topK),
    dense_e5: buildDummyMethodResults("dense_e5", query, topK),
    hybrid_e5: buildDummyMethodResults("hybrid_e5", query, topK),
  };
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

const isRelevantLikeRating = (
  rating: RelevanceRating | null | undefined,
): rating is "relevant" =>
  rating === "relevant";

const isAssessmentComplete = (
  rating: RelevanceRating | null | undefined,
  scope: RelevantScope | undefined,
  sectionLabel: string | undefined,
): boolean => {
  if (rating === "not_relevant") {
    return true;
  }

  if (!isRelevantLikeRating(rating)) {
    return false;
  }

  if (scope === "whole_document") {
    return true;
  }

  if (scope === "part_of_document") {
    return (sectionLabel ?? "").trim().length > 0;
  }

  return false;
};

export default function DemoSearchPage() {
  const router = useRouter();
  const stepOneRef = useRef<HTMLElement | null>(null);
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
  const [useDummyData, setUseDummyData] = useState(false);
  const [ratings, setRatings] = useState<Record<string, RelevanceRating>>({});
  const [relevantScopes, setRelevantScopes] = useState<Record<string, RelevantScope>>({});
  const [relevantSections, setRelevantSections] = useState<Record<string, string>>({});
  const [resultComments, setResultComments] = useState<Record<string, string>>({});
  const [hasSubmittedRatings, setHasSubmittedRatings] = useState(false);
  const [isSubmittingToBackend, setIsSubmittingToBackend] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isHydrated, setIsHydrated] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [participantName, setParticipantName] = useState("");
  const canSubmit = query.trim().length > 0;
  const hasSubmittedQuery = submittedQuery.trim().length > 0;
  const isAdminUser = participantName === "admin" || participantName === "Admin";

  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsAuthenticated(localStorage.getItem(SESSION_KEY) === "true");
      setParticipantName(localStorage.getItem(USER_NAME_KEY)?.trim() ?? "");
    }
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    if (isHydrated && !isAuthenticated) {
      router.replace("/");
    }
  }, [isHydrated, isAuthenticated, router]);

  useEffect(() => {
    if (!isAdminUser) {
      setDebugMode(false);
      setUseDummyData(false);
    }
  }, [isAdminUser]);
  const allResultsRated =
    hasSubmittedQuery &&
    pipeline.finalResults.length > 0 &&
    pipeline.finalResults.every((result, index) => {
      const resultKey = result.source_path ?? `${getResultTitle(result)}-${index}`;
      return isAssessmentComplete(
        ratings[resultKey],
        relevantScopes[resultKey],
        relevantSections[resultKey],
      );
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
      let payload: {
        errors?: string[];
        results_by_method?: SearchResultsByMethod;
      };

      if (useDummyData) {
        const resultsByMethod = await runDummySearch(trimmedQuery, DEFAULT_TOP_K);
        payload = {
          errors: [],
          results_by_method: resultsByMethod,
        };
      } else {
        const response = await fetch(`${DEMO_API_BASE_URL.replace(/\/$/, "")}/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            method: "evaluation_form_search" as SearchApiMethod,
            query: trimmedQuery,
            information_need: informationNeed.trim(),
            top_k: DEFAULT_TOP_K,
          }),
        });

        payload = (await response.json()) as {
          errors?: string[];
          results_by_method?: SearchResultsByMethod;
        };

        if (!response.ok) {
          setPipeline(EMPTY_PIPELINE);
          setSearchErrors(payload.errors && payload.errors.length > 0 ? payload.errors : ["Search failed."]);
          return;
        }
      }

      if (payload.errors && payload.errors.length > 0) {
        setPipeline(EMPTY_PIPELINE);
        setSubmittedQuery("");
        setSearchErrors(payload.errors);
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
          relevant_scope: isRelevantLikeRating(selectedRating) ? selectedScope : null,
          relevant_section:
            isRelevantLikeRating(selectedRating) && selectedScope === "part_of_document"
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

      if (typeof window !== "undefined") {
        window.alert(
          "Ditt bidrag är inskickat! Nu kan du upprepa processen igen. Ju mer data vi får desto bättre blir vårt arbete :)",
        );
      }

      setHasSubmittedRatings(false);
      setInformationNeed("");
      setQuery("");
      setComment("");
      setSubmittedQuery("");
      setRunId(0);
      setSearchErrors([]);
      setPipeline(EMPTY_PIPELINE);
      setRatings({});
      setRelevantScopes({});
      setRelevantSections({});
      setResultComments({});
      setSubmitError(null);

      stepOneRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Kunde inte spara formuläret.");
    } finally {
      setIsSubmittingToBackend(false);
    }
  };

  const onCancelRelevanceStep = () => {
    setHasSubmittedRatings(false);
    setSubmitError(null);
    setSearchErrors([]);
    setSubmittedQuery("");
    setRunId(0);
    setPipeline(EMPTY_PIPELINE);
    setRatings({});
    setRelevantScopes({});
    setRelevantSections({});
    setResultComments({});
    requestAnimationFrame(() => {
      stepOneRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const onLogout = () => {
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(USER_NAME_KEY);
    setIsAuthenticated(false);
    setParticipantName("");
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

  if (!isHydrated || !isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#fff8eb,#f3efe6_45%,#eaf4ff)] px-4 py-10 text-[#1e241f] md:px-5">
      <main className="mx-auto flex w-full max-w-7xl flex-col gap-6">
                  <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
            Insamling av utvärderingsdata
          </h1>
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          {isAdminUser ? (
            <>
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

              <label
                className={`flex items-center justify-between gap-4 rounded-2xl border px-4 py-3 text-sm font-medium transition md:min-w-[22rem] ${
                  useDummyData
                    ? "border-[#9bc7c7] bg-[#eef6f3] text-[#1f4f4f]"
                    : "border-[#d8ddd3] bg-white text-[#556055]"
                }`}
              >
                <div className="flex flex-col">
                  <span>Sök med dummydata</span>
                  <span className="text-xs font-normal opacity-80">
                    {useDummyData
                      ? "På: använder lokala dummyfunktioner"
                      : "Av: använder Flask-backend"}
                  </span>
                </div>
                <span
                  className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] ${
                    useDummyData ? "bg-[#1f6e6e] text-white" : "bg-[#eef1ec] text-[#5f685f]"
                  }`}
                >
                  {useDummyData ? "På" : "Av"}
                </span>
                <input
                  className="toggle border-[#cfd4c9] bg-white text-[#1f6e6e] [--tglbg:#ffffff]"
                  type="checkbox"
                  checked={useDummyData}
                  onChange={(event) => setUseDummyData(event.target.checked)}
                />
              </label>
            </>
          ) : null}

          <div className="flex flex-wrap justify-end gap-2">
            <button
              className="rounded-full border border-[#c8cfbf] bg-white/80 px-4 py-2 text-sm text-[#425043] transition hover:border-[#1f6e6e] hover:text-[#1f6e6e]"
              type="button"
              onClick={() => router.push("/search")}
            >
              Gå till demosida
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
              <p>Hej!</p>
              <p>
                Syftet med det här formuläret är att du ska hjälpa oss skapa verklighetstrogna sökningar samt att relevansbedöma sökträffar kopplade till dessa sökningar. När du har gjort klart stegen nedan kommer du kunna göra om dem på nytt. Hur många gånger du vill göra det är helt upp till dig. Ju mer data desto bättre för vår del :). Du kan komma tillbaka till den här sidan när du vill och fortsätta.<br></br>
              </p>
              <p>
                Datan kommer användas för att utvärdera vår sökalgoritm och göra den bättre. <br></br>
                Notera att allt du skriver kommer sparas, så skriv inte in känslig information såsom patientdata.</p>
              <p></p>
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
          ref={stepOneRef}
          className={`rounded-[2rem] border p-8 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            hasSubmittedQuery
              ? "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-75"
              : "border-[#d6d8cf] bg-[#fffdf8]/95"
          }`}
        >
          {debugMode ? (
            <div className="mb-8">
              <p className="max-w-3xl text-sm leading-6 text-[#5e655e]">
                {useDummyData
                  ? "Varje sökning använder lokala dummyfunktioner för `bm25`, `dense_e5` och `hybrid_e5`, tar bort dubletter och randomiserar ordningen innan visning."
                  : "Varje sökning anropar `bm25`, `dense_e5` och `hybrid_e5` för både sökfrasen och informationsbehovet, samlar deras topp 5-resultat från backend, tar bort dubletter och randomiserar ordningen innan visning."}
              </p>
            </div>
          ) : null}

          <form className="flex flex-col gap-3" onSubmit={onSubmit}>
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
              1. Beskriv informationsbehov
            </p>
            <div className="space-y-2">
              <p className="text-sm leading-6 text-[#4f5850]">
                Tänk på ett informationsbehov som kan uppstå i ditt arbete där du behöver söka i DocPlus. <br></br>

                Beskriv kort vilken information du behöver få fram i DocPlus.
                Formulera det som en fråga eller ett kort informationsbehov. 1–2 meningar räcker.<br></br>

                Skriv inte hur du skulle söka ännu, det gör du i nästa steg.<br></br>

                 <br></br>
                
                Exempel: <br></br>

                &quot;Vilka arbetsuppgifter har undersköterskan vid assistering under en vakuumextraktion?<br></br>

                &quot;Vilka åtgärder ska vidtas om ett nyfött barn har 36,0 i temperatur?
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
              2. Skapa en sökning
            </p>
            <p className="text-sm leading-6 text-[#4f5850]">
              Hur skulle du skriva din sökning i DocPlus för att hitta denna information? Exempel: &quot;undersköterska assistering sugklocka&quot;
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
              När du är nöjd, tryck på &quot;Nästa steg&quot; för att generera sökträffar.
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

        <div className="mt-1">
          <button
            className={`rounded-xl border px-4 py-2 text-sm font-medium transition ${
              hasSubmittedQuery && !isLoadingSearch && !isSubmittingToBackend
                ? "border-[#c8cfbf] bg-white/80 text-[#425043] hover:border-[#aeb7a5] hover:bg-white"
                : "cursor-not-allowed border-[#d6dcd1] bg-[#f0f2ec] text-[#8a9486]"
            }`}
            type="button"
            disabled={!hasSubmittedQuery || isLoadingSearch || isSubmittingToBackend}
            onClick={onCancelRelevanceStep}
          >
            <span aria-hidden="true">↑</span>{" "}Avbryt steg 3 och gå tillbaka
          </button>
          <p className="mt-2 text-sm text-[#4f5850]">
            Notera att ditt arbete i steg 3 nollställs.
          </p>
        </div>

        {searchErrors.length > 0 ? (
          <section className="rounded-[1.5rem] border border-[#f0b79f] bg-[#ffe8dc] p-4 text-sm text-[#7a2e0d]">
            {searchErrors.map((error) => (
              <p key={error}>{error}</p>
            ))}
          </section>
        ) : null}

        {debugMode ? (
          <section className="rounded-[1.5rem] border border-[#d6d8cf] bg-[#fcfbf8] p-6">
            <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
              Debug: Pool per sökfunktion (före dedupe)
            </h2>
            <p className="mt-2 text-sm text-[#4f5850]">
              Lista över dokumenttitlar som respektive sökfunktion bidrar med till poolen.
            </p>
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              {METHODS.map((method) => (
                <div className="rounded-2xl border border-[#d9ddd4] bg-white p-4" key={`pool-list-${method}`}>
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                    {method}
                  </p>
                  {pipeline.byMethod[method].length > 0 ? (
                    <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm text-[#465048]">
                      {pipeline.byMethod[method].map((result, index) => (
                        <li key={`pool-title-${method}-${result.source_path ?? index}`}>
                          {getResultTitle(result)}
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="mt-3 text-sm text-[#6a7169]">
                      Inga resultat ännu.
                    </p>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-2xl border border-[#d9ddd4] bg-white p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                Pool efter dedupe
              </p>
              <p className="mt-2 text-sm text-[#4f5850]">
                Unik lista efter att dubletter tagits bort.
              </p>
              {pipeline.pooledAfterDedup.length > 0 ? (
                <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm text-[#465048]">
                  {pipeline.pooledAfterDedup.map((result, index) => (
                    <li key={`deduped-title-${result.source_path ?? index}`}>
                      {getResultTitle(result)}
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="mt-3 text-sm text-[#6a7169]">
                  Inga resultat ännu.
                </p>
              )}
            </div>
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
            3. Bedöm relevans
          </h2>
          <div className="mt-3 space-y-1 text-sm leading-6 text-[#4f5850]">
            <p style={{ marginBottom: '20px' }}>För varje sökträff i listan nedan, klicka på länken för att få upp dokumentet och bedöm hur relevant det är utifrån informationsbehovet du definierade i steget ovan.</p>
            <p>Relevant = Dokumentet innehåller information som hjälper till att besvara informationsbehovet.</p>
            <p style={{ marginBottom: '20px' }}>Inte relevant = Dokumentet innehåller inte information som hjälper till att besvara informationsbehovet.</p>
            <p style={{ marginBottom: '20px' }}>
              Dokumentet behöver inte vara det bästa eller mest kompletta svaret för att räknas som relevant. <br></br>
              Om flera  dokument innehåller relevant information kan alla markeras som relevanta.<br></br>
            </p>
            <p style={{ fontWeight: '600' }} >Observera att dokumentens rangordning är slumpmässig.</p>
          </div>
          <div className={`mt-4 grid gap-3 text-sm text-[#6b7468] ${debugMode ? "md:grid-cols-4" : "md:grid-cols-1"}`}>
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
              const isRelevantLike = isRelevantLikeRating(selectedRating);
              const selectedScope = relevantScopes[resultKey];
              const sectionLabel = relevantSections[resultKey] ?? "";
              const isRatingComplete = isAssessmentComplete(
                selectedRating,
                selectedScope,
                sectionLabel,
              );
              const resultComment = resultComments[resultKey] ?? "";

                return (
                  <article
                    className={`rounded-[1.5rem] border p-6 shadow-[0_16px_50px_rgba(35,44,32,0.06)] transition ${
                      isRatingComplete
                        ? "border-[#79b8a3] bg-[#f4fbf7]"
                        : "border-[#d6d8cf] bg-white/90"
                    }`}
                    key={`${resultKey}-${index}`}
                  >
                    {!debugMode ? (
                      <div
                        className="grid gap-4 md:items-start lg:grid-cols-[minmax(0,1.1fr)_12rem_minmax(0,1fr)] xl:grid-cols-[minmax(0,1.1fr)_12rem_22rem_minmax(0,0.8fr)]"
                      >
                        <div className="min-w-0 lg:col-start-1">
                          <h2 className="font-serif text-lg text-[#203327]">{getResultTitle(result)}</h2>
                          {getResultSectionHeading(result) ? (
                            <p className="mt-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#6f7b72]">
                              {getResultSectionHeading(result)}
                            </p>
                          ) : null}
                          <a
                            className="mt-4 inline-flex break-all text-xs font-medium text-[#1f6e6e] underline decoration-[#9bc7c7] underline-offset-4"
                            href={getResultUrl(result)}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Öppna länk{" "}<span aria-hidden="true">{"\u2197"}</span>
                          </a>
                          <div className="mt-5 rounded-[1.25rem] border border-[#dfe4db] bg-[#f8fbf8] p-4">
                            <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.14em] text-[#667166]">
                              Förhandsvisning av underkapitel
                            </p>
                            <p className="whitespace-pre-wrap break-words text-sm leading-6 text-[#314135]">
                              {getResultChunkText(result)}
                            </p>
                          </div>
                        </div>

                        <fieldset className="w-full max-w-[10.5rem] self-start lg:col-start-2">
                          <legend className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                            Relevans
                          </legend>
                          <div className="mt-3 overflow-hidden rounded-xl border border-[#d9ddd4] bg-white">
                            <label className="flex items-center gap-2 px-3 py-2 text-xs text-[#465048]">
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
                            <label className="flex items-center gap-2 border-t border-[#d9ddd4] px-3 py-2 text-xs text-[#465048]">
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

                        {isRelevantLike ? (
                          <div className="space-y-4 rounded-[1.25rem] border border-[#dfe4db] bg-[#f8fbf8] p-4 lg:col-start-3 xl:col-start-3 xl:ml-[-0.75rem]">
                            <div className="space-y-2">
                              <p className="text-xs font-medium text-[#2f3a31]">
                                Var i dokumentet finns den relevanta informationen?
                              </p>
                              <div className="flex flex-col gap-2">
                                <label className="flex items-center gap-2 text-xs text-[#465048]">
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
                                <label className="flex items-center gap-2 text-xs text-[#465048]">
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
                                <span className="text-xs text-[#2f3a31]">
                                  Ange i vilken/vilka kapitel och/eller sidor
                                </span>
                                <textarea
                                  className={`min-h-20 rounded-2xl border px-4 py-3 text-xs outline-none transition ${
                                    hasSubmittedRatings
                                      ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                                      : "border-[#cfd4c9] bg-white focus:border-[#1f6e6e]"
                                  }`}
                                  disabled={hasSubmittedRatings}
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

                        <label className="lg:col-start-3 xl:col-start-4">
                          <span className="text-xs text-[#2f3a31]">Valfri kommentar</span>
                          <textarea
                            className={`mt-2 min-h-24 w-full rounded-2xl border px-4 py-3 text-xs outline-none transition ${
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
                          Öppna exempellänk{" "}<span aria-hidden="true">{"\u2197"}</span>
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

                      <p className="leading-6">{getResultChunkText(result)}</p>

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
            4. Skicka in
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
