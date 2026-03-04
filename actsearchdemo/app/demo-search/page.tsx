"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "actsearch-authenticated";
const USER_NAME_KEY = "actsearch-user-name";
const METHODS = ["bm25", "dense", "hybrid", "docplus"] as const;

type SearchMethod = (typeof METHODS)[number];
type RelevanceRating = "relevant" | "not_relevant";
type RelevantScope = "whole_document" | "part_of_document";

type SearchResult = {
  score?: number;
  text?: string;
  chunk_text?: string;
  chunk_type?: string;
  metadata?: Record<string, unknown>;
  source_path?: string;
  demo_method?: SearchMethod;
};

type DemoDocument = {
  id: string;
  title: string;
  summary: string;
  body: string;
  category: string;
  tags: string[];
  url: string;
  chunkType: "body" | "title";
};

type SearchPipeline = {
  byMethod: Record<SearchMethod, SearchResult[]>;
  pooledBeforeDedup: SearchResult[];
  pooledAfterDedup: SearchResult[];
  finalResults: SearchResult[];
};

const DEMO_DOCUMENTS: DemoDocument[] = [
  {
    id: "anafylaxi-vuxna",
    title: "Anafylaxi hos vuxna",
    summary: "Initial handläggning och adrenalinbehandling vid misstänkt anafylaxi.",
    body: "Rutin för bedömning, adrenalin i.m., observation, luftväg och vidare uppföljning i akutmottagning.",
    category: "Akutvård",
    tags: ["adrenalin", "anafylaxi", "allergi", "akut"],
    url: "https://example.org/documents/anafylaxi-vuxna",
    chunkType: "title",
  },
  {
    id: "feber-barn",
    title: "Feber hos barn i primärvård",
    summary: "Översikt över alarmsymtom och när barnet ska vidare till akut bedömning.",
    body: "Egenvårdsråd, statusfynd, CRP-överväganden och remisskriterier vid feber hos barn.",
    category: "Primärvård",
    tags: ["barn", "feber", "infektion", "primärvård"],
    url: "https://example.org/documents/feber-barn",
    chunkType: "body",
  },
  {
    id: "sepsis-tidig",
    title: "Tidig identifiering av sepsis",
    summary: "Checklista för NEWS, odlingar och antibiotikastart vid misstänkt sepsis.",
    body: "Sepsislarm, vitalparametrar, provtagning och antibiotika inom en timme vid klinisk misstanke.",
    category: "Slutenvård",
    tags: ["sepsis", "news", "antibiotika", "akut"],
    url: "https://example.org/documents/sepsis-tidig",
    chunkType: "body",
  },
  {
    id: "diabetesfot",
    title: "Diabetesfot och sårbehandling",
    summary: "Riskklassning, fotstatus och remissvägar till specialistteam.",
    body: "Sårbehandling, avlastning, infektionstecken och uppföljning av diabetesfot i öppenvård.",
    category: "Endokrinologi",
    tags: ["diabetes", "fot", "sår", "remiss"],
    url: "https://example.org/documents/diabetesfot",
    chunkType: "body",
  },
  {
    id: "hjartsvikt",
    title: "Uppfoljning vid hjartsvikt",
    summary: "Läkemedelsupptrappning och planerad uppföljning efter utskrivning.",
    body: "Kontroll av vikt, blodtryck, kreatinin och symtom efter inledd behandling vid hjartsvikt.",
    category: "Kardiologi",
    tags: ["hjärtsvikt", "uppföljning", "kardiologi", "läkemedel"],
    url: "https://example.org/documents/hjartsvikt",
    chunkType: "body",
  },
  {
    id: "stroke-tia",
    title: "TIA och stroke akut handläggning",
    summary: "Akut neurologstatus, tidsfönster och kontakt med strokeenhet.",
    body: "Initial utredning, trombolysbedömning, NIHSS och radiologi vid akut stroke eller TIA.",
    category: "Neurologi",
    tags: ["stroke", "tia", "neurologi", "trombolys"],
    url: "https://example.org/documents/stroke-tia",
    chunkType: "title",
  },
  {
    id: "astma-barn",
    title: "Akut astma hos barn",
    summary: "Bedömning av andningsarbete och inhalationsbehandling i akuta lägen.",
    body: "Saturation, inhalationer, steroider och observationstid vid akut astma hos barn.",
    category: "Barnmedicin",
    tags: ["astma", "barn", "luftväg", "obstruktivitet"],
    url: "https://example.org/documents/astma-barn",
    chunkType: "body",
  },
  {
    id: "kol-exacerbation",
    title: "KOL-exacerbation i akutmottagning",
    summary: "Syrgas, inhalationer och antibiotikaindikationer vid KOL-exacerbation.",
    body: "Handläggning med blodgas, saturation, bronkdilaterare och uppföljning efter stabilisering.",
    category: "Lungmedicin",
    tags: ["kol", "andning", "akut", "syrgas"],
    url: "https://example.org/documents/kol-exacerbation",
    chunkType: "body",
  },
  {
    id: "suicidrisk",
    title: "Initial bedömning av suicidrisk",
    summary: "Riskfaktorer, skyddsfaktorer och dokumentation vid akut psykiatrisk bedömning.",
    body: "Strukturerad anamnes, akut skyddsbedömning och remiss till psykiatrisk specialistvård.",
    category: "Psykiatri",
    tags: ["psykiatri", "suicidrisk", "bedömning", "akut"],
    url: "https://example.org/documents/suicidrisk",
    chunkType: "body",
  },
  {
    id: "urinvagsinfektion",
    title: "Urinvägsinfektion hos kvinnor",
    summary: "Diagnostik, differentialdiagnoser och förstahandsbehandling i primärvård.",
    body: "Symtombild, odling, antibiotikaval och handläggning vid recidiverande UVI.",
    category: "Infektion",
    tags: ["uvi", "antibiotika", "primärvård", "urinväg"],
    url: "https://example.org/documents/urinvagsinfektion",
    chunkType: "body",
  },
];

const getResultTitle = (result: SearchResult): string => {
  const title = result.metadata?.title;
  return typeof title === "string" ? title : "Untitled document";
};

const getResultUrl = (result: SearchResult): string => {
  const sourceUrl = result.metadata?.source_url;
  return typeof sourceUrl === "string" ? sourceUrl : "#";
};

const normalizeText = (value: string): string =>
  value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");

const getQueryTerms = (query: string): string[] =>
  normalizeText(query)
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean);

const countMatches = (document: DemoDocument, terms: string[]): number => {
  if (terms.length === 0) {
    return 1;
  }

  const haystack = normalizeText(
    [
      document.title,
      document.summary,
      document.body,
      document.category,
      document.tags.join(" "),
    ].join(" "),
  );
  const normalizedTitle = normalizeText(document.title);
  const normalizedTags = document.tags.map((tag) => normalizeText(tag));

  let score = 0;
  for (const term of terms) {
    if (normalizedTitle.includes(term)) {
      score += 6;
    }
    if (normalizedTags.some((tag) => tag.includes(term))) {
      score += 4;
    }
    if (haystack.includes(term)) {
      score += 2;
    }
  }
  return score;
};

const methodBoost = (method: SearchMethod, index: number): number => {
  const rankBias = Math.max(0, 10 - index) * 0.015;

  switch (method) {
    case "bm25":
      return 0.24 + rankBias;
    case "dense":
      return 0.28 + (index % 3) * 0.012;
    case "hybrid":
      return 0.32 + rankBias / 2;
    case "docplus":
      return 0.26 + ((index + 1) % 4) * 0.014;
  }
};

const rotateDocuments = (documents: DemoDocument[], offset: number): DemoDocument[] => {
  if (documents.length === 0) {
    return [];
  }
  const normalizedOffset = offset % documents.length;
  return documents.slice(normalizedOffset).concat(documents.slice(0, normalizedOffset));
};

const assignRanksByScore = (
  items: Array<{ document: DemoDocument; score: number }>,
): Array<{ document: DemoDocument; score: number; rank: number }> =>
  items
    .sort((left, right) => right.score - left.score)
    .map((item, index) => ({
      ...item,
      rank: index + 1,
    }));

const toSearchResult = (
  document: DemoDocument,
  method: SearchMethod,
  score: number,
  matchedTerms: string[],
  rank: number,
): SearchResult => ({
  score: Math.min(0.99, score),
  text: document.body,
  chunk_text: `${document.summary} ${document.body}`,
  chunk_type: document.chunkType,
  source_path: `demo/${document.id}.json`,
  demo_method: method,
  metadata: {
    title: document.title,
    source_url: document.url,
    category: document.category,
    tags: document.tags,
    demo_method: method,
    matched_terms: matchedTerms,
    score_by_method: {
      [method]: Math.min(0.99, score),
    },
    rank_by_method: {
      [method]: rank,
    },
  },
});

const runMethod = (method: SearchMethod, query: string): SearchResult[] => {
  const terms = getQueryTerms(query);
  const orderedDocuments = rotateDocuments(DEMO_DOCUMENTS, METHODS.indexOf(method) * 2);

  const scoredDocuments = orderedDocuments.map((document, index) => {
    const matchScore = countMatches(document, terms);
    const score = methodBoost(method, index) + matchScore * 0.045;
    return { document, score };
  });

  return assignRanksByScore(scoredDocuments)
    .slice(0, 10)
    .map((item) => toSearchResult(item.document, method, item.score, terms, item.rank));
};

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
          pooled_from: [result.demo_method],
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
        pooled_from: Array.from(new Set([...pooledFrom, result.demo_method])),
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

const buildPipeline = (query: string, runId: number): SearchPipeline => {
  const byMethod = {
    bm25: runMethod("bm25", query),
    dense: runMethod("dense", query),
    hybrid: runMethod("hybrid", query),
    docplus: runMethod("docplus", query),
  };

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
  const [debugMode, setDebugMode] = useState(false);
  const [ratings, setRatings] = useState<Record<string, RelevanceRating>>({});
  const [relevantScopes, setRelevantScopes] = useState<Record<string, RelevantScope>>({});
  const [relevantSections, setRelevantSections] = useState<Record<string, string>>({});
  const [resultComments, setResultComments] = useState<Record<string, string>>({});
  const [hasSubmittedRatings, setHasSubmittedRatings] = useState(false);
  const canSubmit = query.trim().length > 0;
  const hasSubmittedQuery = submittedQuery.trim().length > 0;
  const isAuthenticated =
    typeof window !== "undefined" && localStorage.getItem(SESSION_KEY) === "true";

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace("/");
    }
  }, [isAuthenticated, router]);

  const pipeline = useMemo(
    () => buildPipeline(submittedQuery.trim(), runId),
    [runId, submittedQuery],
  );

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmittedQuery(query);
    setRunId((current) => current + 1);
    setHasSubmittedRatings(false);
    requestAnimationFrame(() => {
      stepTwoRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const onLogout = () => {
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(USER_NAME_KEY);
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

  const allResultsRated =
    hasSubmittedQuery &&
    pipeline.finalResults.length > 0 &&
    pipeline.finalResults.every((result, index) => {
      const resultKey = result.source_path ?? `${getResultTitle(result)}-${index}`;
      return ratings[resultKey] === "relevant" || ratings[resultKey] === "not_relevant";
    });

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#fff8eb,#f3efe6_45%,#eaf4ff)] px-5 py-7 text-[#1e241f]">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-4">
        <h1 className="font-serif text-3xl text-[#1d3529]">Insamling av utvärderingsdata</h1>

        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <label
            className={`flex items-center justify-between gap-3 rounded-2xl border px-4 py-2.5 text-sm font-medium transition md:min-w-[22rem] ${
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
          <section className="rounded-[2rem] border border-[#d6d8cf] bg-[#fffdf8]/95 p-6 shadow-[0_24px_80px_rgba(34,42,28,0.08)]">
            <div className="max-w-4xl space-y-3.5 text-sm leading-6 text-[#4f5850]">
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
          className={`rounded-[2rem] border p-6 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            hasSubmittedQuery
              ? "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-75"
              : "border-[#d6d8cf] bg-[#fffdf8]/95"
          }`}
        >
          {debugMode ? (
            <div className="mb-5">
              <p className="max-w-3xl text-sm leading-6 text-[#5e655e]">
                Varje sökning anropar `bm25`, `dense`, `hybrid` och `docplus`, samlar deras
                topp 10-resultat i samma format som riktiga svar, tar bort dubletter och
                randomiserar ordningen innan visning.
              </p>
            </div>
          ) : null}

          <form className="flex flex-col gap-2.5" onSubmit={onSubmit}>
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
              1. Beskriv informationsbehov
            </p>
            <div className="space-y-1.5">
              <p className="text-sm leading-5 text-[#4f5850]">
                &quot;Beskriv en situation där du behövde söka information i DocPlus.&quot;
                Beskriv ett exempel på ett informationsbehov som kan uppstå i ditt dagliga arbete,
                där du behöver söka i DocPlus. Exempel: &quot;Familjen vill sova med sitt spädbarn
                mellan sig. Vilken information behöver jag förmedla till föräldrarna?&quot; Eller &quot;Vilka arbetsuppgifter har undersköterskan vid assistering under en vakuumextraktion?&quot;.
                Det kan vara både situationer där du letar efter ett specifikt dokument och situationer där du vill få en överblick.
              </p>
              <input
                className={`w-full rounded-2xl border px-4 py-3 text-base outline-none transition ${
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
            <p className="text-sm leading-5 text-[#4f5850]">
              Vad skulle du skriva in i sökrutan (I docplus?) utifrån ovanstående informationsbehov?
            </p>
            <div className="flex flex-col gap-3">
              <input
                className={`min-w-0 flex-1 rounded-2xl border px-4 py-3 text-base outline-none transition ${
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

            <label className="flex flex-col gap-1.5">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                Valfri kommentar
              </span>
              <textarea
                className={`min-h-20 rounded-2xl border px-4 py-3 text-base outline-none transition ${
                  hasSubmittedQuery
                    ? "border-[#d7dbd2] bg-[#f3f3ef] text-[#8a8f86]"
                    : "border-[#cfd4c9] bg-white text-[#1e241f] focus:border-[#1f6e6e]"
                }`}
                value={comment}
                disabled={hasSubmittedQuery}
                onChange={(event) => setComment(event.target.value)}
              />
            </label>

            <p className="text-sm leading-5 text-[#4f5850]">
              När du är nöjd, tryck på &quot;Nästa steg&quot;. Du kan inte gå tillbaka och ändra i din information. 
            </p>

            <button
              className={`self-start rounded-2xl px-5 py-3 font-semibold text-white transition ${
                canSubmit && !hasSubmittedQuery
                  ? "bg-[#1f6e6e] hover:bg-[#184f4f]"
                  : "cursor-not-allowed bg-[#9fb8b8]"
              }`}
              type="submit"
              disabled={!canSubmit || hasSubmittedQuery}
            >
              Nästa steg
            </button>

          </form>
        </section>

        <section
          ref={stepTwoRef}
          className={`rounded-[2rem] border p-6 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            hasSubmittedQuery
              ? hasSubmittedRatings
                ? "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-60"
                : "border-[#d6d8cf] bg-[#fffdf8]/95"
              : "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-60"
          }`}
        >
          <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
            2. Bedöm relevans
          </h2>
          <div className="mt-2 space-y-1 text-sm leading-5 text-[#4f5850]">
            <p>För varje sökträff, klicka på länken för att få upp dokumentet och bedöm hur relevant dokumentet är utifrån söktermen.</p>
            <p>Observera att rangordningen är slumpartad.</p>
          </div>
          <div className={`mt-3 grid gap-2 text-sm text-[#6b7468] ${debugMode ? "md:grid-cols-4" : "md:grid-cols-1"}`}>
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
            <div className="mt-4 grid gap-3">
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
                    className="rounded-[1.5rem] border border-[#d6d8cf] bg-white/90 p-4 shadow-[0_16px_50px_rgba(35,44,32,0.06)]"
                    key={`${resultKey}-${index}`}
                  >
                    <h2 className="font-serif text-xl text-[#203327]">{getResultTitle(result)}</h2>

                  <a
                    className="mt-2 inline-flex break-all text-sm font-medium text-[#1f6e6e] underline decoration-[#9bc7c7] underline-offset-4"
                    href={getResultUrl(result)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Öppna exempellänk
                  </a>

                  {!debugMode ? (
                    <>
                      <fieldset className="mt-3">
                        <legend className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                          Relevans
                        </legend>
                        <div className="mt-2 flex flex-col gap-2 md:flex-row md:flex-wrap">
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
                        <div className="mt-3 space-y-3 rounded-[1.25rem] border border-[#dfe4db] bg-[#f8fbf8] p-3">
                          <div className="space-y-1.5">
                            <p className="text-sm font-medium text-[#2f3a31]">
                              Var i dokumentet finns den relevanta informationen för din söksituation?
                            </p>
                            <div className="flex flex-col gap-1.5">
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
                                Hela dokumentet/större delar av dokumentet är relevant
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
                                En del av dokumentet är relevant
                              </label>
                            </div>
                          </div>

                          {selectedScope === "part_of_document" ? (
                            <label className="flex flex-col gap-1.5">
                              <span className="text-sm text-[#2f3a31]">
                                Ange i vilken del/kapitel som den relevanta informationen finns
                                (ange rubrik/underrubrik)
                              </span>
                              <input
                                className={`rounded-2xl border px-4 py-2.5 text-sm outline-none transition ${
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

                      <label className="mt-3 flex flex-col gap-1.5">
                        <span className="text-sm text-[#2f3a31]">Valfri kommentar</span>
                        <textarea
                          className={`min-h-20 rounded-2xl border px-4 py-2.5 text-sm outline-none transition ${
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
                    </>
                  ) : null}

                  {debugMode ? (
                    <div className="mt-3 space-y-2 rounded-2xl bg-[#f8f5ee] p-3 text-sm text-[#4c564f]">
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

                      <div className="rounded-xl bg-white/80 p-2.5">
                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                          Score per metod
                        </p>
                        <div className="mt-1.5 flex flex-wrap gap-2 text-xs">
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

                      <div className="rounded-xl bg-white/80 p-2.5">
                        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#58635b]">
                          Rank per metod
                        </p>
                        <div className="mt-1.5 flex flex-wrap gap-2 text-xs">
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

                      <p className="leading-5">{String(result.chunk_text ?? result.text ?? "")}</p>

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
            <div className="mt-4 rounded-[1.5rem] border border-dashed border-[#d4d7cf] bg-white/60 p-4 text-sm text-[#6a7169]">
              Skriv in en sökterm och tryck på `Nästa` för att visa sökträffar.
            </div>
          )}
        </section>

        <section
          className={`rounded-[2rem] border p-6 shadow-[0_24px_80px_rgba(34,42,28,0.08)] transition ${
            allResultsRated && !debugMode
              ? "border-[#d6d8cf] bg-[#fffdf8]/95"
              : "border-[#e1e4dc] bg-[#f5f4ef]/90 opacity-60"
          }`}
        >
          <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#58635b]">
            3. Skicka in
          </h2>
          <p className="mt-2 text-sm leading-5 text-[#4f5850]">
            När du har bedömt alla sökträffar kan du skicka in dina svar.
          </p>
          <button
            className={`mt-4 rounded-2xl px-5 py-3 font-semibold text-white transition ${
              allResultsRated && !debugMode
                ? "bg-[#1f6e6e] hover:bg-[#184f4f]"
                : "cursor-not-allowed bg-[#9fb8b8]"
            }`}
            type="button"
            disabled={!allResultsRated || debugMode}
            onClick={() => setHasSubmittedRatings(true)}
          >
            Skicka in
          </button>
          {hasSubmittedRatings ? (
            <p className="mt-4 text-sm text-[#1f6e6e]">
              Tack. Dina bedömningar är markerade som inskickade i demon.
            </p>
          ) : null}
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
