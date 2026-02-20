"""LLM-assisted result summaries with source references for search hits."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[0-9A-Za-zÅÄÖåäö]+", flags=re.UNICODE)


@dataclass(frozen=True)
class SummaryConfig:
    enabled: bool
    api_url: str
    api_key: str
    model: str
    timeout_seconds: float


def load_summary_config() -> SummaryConfig:
    enabled_value = os.getenv("DOCPLUS_LLM_SUMMARY_ENABLED", "1").strip().lower()
    enabled = enabled_value not in {"0", "false", "no", "off"}
    return SummaryConfig(
        enabled=enabled,
        api_url=os.getenv(
            "DOCPLUS_LLM_API_URL",
            "https://api.openai.com/v1/chat/completions",
        ).strip(),
        api_key=os.getenv("DOCPLUS_LLM_API_KEY", "").strip(),
        model=os.getenv("DOCPLUS_LLM_MODEL", "gpt-4o-mini").strip(),
        timeout_seconds=float(os.getenv("DOCPLUS_LLM_TIMEOUT_SECONDS", "12").strip()),
    )


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _result_reference(result: Dict[str, Any], chunk_text: str, query: str) -> str:
    merged_hits_raw = result.get("merged_hits")
    merged_hits = 0
    if isinstance(merged_hits_raw, int):
        merged_hits = merged_hits_raw
    else:
        merged_hits_text = _to_text(merged_hits_raw)
        if merged_hits_text:
            try:
                merged_hits = int(merged_hits_text)
            except ValueError:
                merged_hits = 0

    merged_chunk_ids_raw = result.get("merged_chunk_ids")
    chunk_ids: list[str] = []
    if isinstance(merged_chunk_ids_raw, list):
        for value in merged_chunk_ids_raw:
            parsed = _to_text(value)
            if parsed:
                chunk_ids.append(parsed)

    chunk_id_raw = result.get("chunk_id")
    chunk_id = _to_text(chunk_id_raw) if chunk_id_raw is not None else "okänd"
    chunk_type = _to_text(result.get("chunk_type")) or "body"
    if merged_hits > 1:
        chunks_label = ", ".join(chunk_ids) if chunk_ids else chunk_id
        reference = f"{merged_hits} träffar, chunks [{chunks_label}] ({chunk_type})"
    else:
        reference = f"chunk {chunk_id} ({chunk_type})"

    query_tokens = set(_tokenize(query))
    if query_tokens and chunk_text:
        chunk_tokens = _tokenize(chunk_text)
        for idx, token in enumerate(chunk_tokens):
            if token in query_tokens:
                reference += f", första termträff ungefär ord #{idx + 1}"
                break

    return reference


def _first_sentences(text: str, max_sentences: int = 2, max_chars: int = 260) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""

    sentences = _SENTENCE_SPLIT_RE.split(normalized)
    selected = " ".join(sentences[:max_sentences]).strip()
    if not selected:
        selected = normalized[:max_chars]
    if len(selected) > max_chars:
        selected = selected[: max_chars - 1].rstrip() + "…"
    return selected


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        candidate = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, dict) else None


def _fallback_summary(query: str, chunk_text: str, result: Dict[str, Any]) -> Dict[str, Any]:
    snippet = _first_sentences(chunk_text)
    if snippet:
        summary = snippet
    else:
        summary = "Ingen chunk-text tillgänglig för summering."
    return {
        "summary": summary,
        "reference": _result_reference(result=result, chunk_text=chunk_text, query=query),
        "provider": "fallback",
        "model": "",
    }


def _llm_summary(query: str, chunk_text: str, result: Dict[str, Any], config: SummaryConfig) -> Dict[str, Any]:
    system_prompt = (
        "Du är en medicinsk sökassistent. Summera kort vad chunken säger i relation till frågan. "
        "Hallucinera inte."
    )
    user_prompt = (
        "Fråga:\n"
        f"{query}\n\n"
        "Chunk:\n"
        f"{chunk_text}\n\n"
        "Svara ENDAST med JSON-objekt enligt exakt schema:\n"
        '{"summary":"kort svensk summering, max 2 meningar","evidence":"kort citat eller parafras ur chunken, max 25 ord"}'
    )

    response = requests.post(
        config.api_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        json={
            "model": config.model,
            "temperature": 0.1,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    decoded = _extract_json_object(_to_text(content))
    if not decoded:
        raise ValueError("LLM response did not contain a valid JSON object.")

    summary = _to_text(decoded.get("summary"))
    evidence = _to_text(decoded.get("evidence"))
    if not summary:
        raise ValueError("LLM response missing summary.")

    reference = _result_reference(result=result, chunk_text=chunk_text, query=query)
    if evidence:
        reference = f"{reference} | evidens: {evidence}"

    return {
        "summary": summary,
        "reference": reference,
        "provider": "openai_compatible",
        "model": config.model,
    }


def summarize_result(query: str, result: Dict[str, Any], config: Optional[SummaryConfig] = None) -> Dict[str, Any]:
    summary_config = config or load_summary_config()
    chunk_text = _to_text(result.get("chunk_text") or result.get("text"))
    if not summary_config.enabled:
        return _fallback_summary(query=query, chunk_text=chunk_text, result=result)

    if not summary_config.api_key:
        return _fallback_summary(query=query, chunk_text=chunk_text, result=result)

    try:
        return _llm_summary(
            query=query,
            chunk_text=chunk_text,
            result=result,
            config=summary_config,
        )
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return _fallback_summary(query=query, chunk_text=chunk_text, result=result)
