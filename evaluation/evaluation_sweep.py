"""Run evaluation sweeps across chunking and preprocessing configurations."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from evaluation import evaluate_system
from search_adapter import (
    SearchConfig,
    bm25_search,
    dense_e5_search,
    dense_search,
    hybrid_e5_search,
    hybrid_search,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from search.vector_index import (
        DEFAULT_MODEL,
        build_index,
        resolve_model_name,
        resolve_text_source,
    )
except ModuleNotFoundError:
    DEFAULT_MODEL = "KBLab/bert-base-swedish-cased"

    def resolve_model_name(profile_key, model_name):
        if model_name:
            return model_name
        return DEFAULT_MODEL

    def resolve_text_source(text_source):
        return text_source or "text"


DEFAULT_TOP_K = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


METHODS_WITH_VECTOR = {"dense", "dense_e5", "hybrid", "hybrid_e5"}


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "run"


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        values.append(int(candidate))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_bool(token: str) -> bool:
    lowered = token.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {token}")


def _parse_bool_list(raw: str) -> List[bool]:
    values: List[bool] = []
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        values.append(_parse_bool(candidate))
    if not values:
        raise ValueError("Expected at least one boolean value.")
    return values


def _bool_label(value: bool) -> str:
    return "title_on" if value else "title_off"


def _vector_model_default(method: str) -> str:
    if method in {"dense_e5", "hybrid_e5"}:
        return "intfloat/multilingual-e5-large-instruct"
    return DEFAULT_MODEL


def _resolve_vector_model(method: str, profile: str | None, model_name: str | None) -> str:
    if model_name:
        return model_name
    if method in {"dense_e5", "hybrid_e5"} and not profile:
        return _vector_model_default(method)
    return resolve_model_name(profile_key=profile, model_name=model_name)


def _vector_paths(index_dir: Path) -> tuple[str, str]:
    return (str(index_dir / "docplus.faiss"), str(index_dir / "docplus_metadata.jsonl"))


def _text_source_label(text_source: str) -> str:
    return resolve_text_source(text_source)


def _build_vector_index_for_config(
    parsed_dir: str,
    index_dir: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
    include_title_chunk: bool,
    batch_size: int,
    device: str,
    text_source: str,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    build_index(
        parsed_dir=parsed_dir,
        output_dir=str(index_dir),
        model_name=model_name,
        max_chars=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        device_preference=device,
        include_title_chunk=include_title_chunk,
        text_source=text_source,
    )


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_path(path_value: str, *, must_exist: bool = False, prefer_repo: bool = False) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        resolved = candidate
    else:
        cwd_relative = (Path.cwd() / candidate).resolve()
        repo_relative = (PROJECT_ROOT / candidate).resolve()
        if prefer_repo:
            resolved = repo_relative
        elif cwd_relative.exists() or not repo_relative.exists():
            resolved = cwd_relative
        else:
            resolved = repo_relative
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def _iter_grid(chunk_sizes: Sequence[int], overlaps: Sequence[int], include_title_chunk: Sequence[bool]):
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            for include_title in include_title_chunk:
                yield {
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(overlap),
                    "include_title_chunk": bool(include_title),
                }


def _search_fn_for_method(method: str, config: SearchConfig) -> Callable[[str, int], List[tuple[str, float]]]:
    if method == "bm25":
        return lambda query, top_k: bm25_search(query=query, top_k=top_k, config=config)
    if method == "dense":
        return lambda query, top_k: dense_search(query=query, top_k=top_k, config=config)
    if method == "dense_e5":
        return lambda query, top_k: dense_e5_search(query=query, top_k=top_k, config=config)
    if method == "hybrid":
        return lambda query, top_k: hybrid_search(query=query, top_k=top_k, config=config)
    if method == "hybrid_e5":
        return lambda query, top_k: hybrid_e5_search(query=query, top_k=top_k, config=config)
    raise ValueError(f"Unsupported method: {method}")


def run_sweep(
    method: str,
    parsed_dir: str,
    experiment_dir: Path,
    top_k: int,
    chunk_sizes: Sequence[int],
    overlaps: Sequence[int],
    include_title_chunk_values: Sequence[bool],
    batch_size: int,
    device: str,
    model_name: str | None,
    profile: str | None,
    text_source: str,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    model = _resolve_vector_model(method=method, profile=profile, model_name=model_name)
    resolved_text_source = resolve_text_source(text_source)
    model_slug = _slugify(model)
    aggregate_dir = experiment_dir / "results" / "aggregate"
    runs_root = experiment_dir / "results" / "runs"
    indexes_root = experiment_dir / "indexes"

    experiment_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "parsed_dir": str(Path(parsed_dir).resolve()),
        "top_k": int(top_k),
        "chunk_sizes": [int(v) for v in chunk_sizes],
        "overlaps": [int(v) for v in overlaps],
        "include_title_chunk": [bool(v) for v in include_title_chunk_values],
        "batch_size": int(batch_size),
        "device": device,
        "model_name": model if method in METHODS_WITH_VECTOR else None,
        "text_source": resolved_text_source,
    }
    _write_json(experiment_dir / "experiment_manifest.json", experiment_manifest)

    for params in _iter_grid(chunk_sizes, overlaps, include_title_chunk_values):
        chunk_size = params["chunk_size"]
        overlap = params["chunk_overlap"]
        include_title_chunk = params["include_title_chunk"]
        config_slug = (
            f"chunk_{chunk_size}__overlap_{overlap}__"
            f"{_bool_label(include_title_chunk)}__{_text_source_label(resolved_text_source)}"
        )

        if method in METHODS_WITH_VECTOR:
            index_dir = indexes_root / method / model_slug / config_slug
        else:
            index_dir = indexes_root / method / config_slug

        if method in METHODS_WITH_VECTOR:
            _build_vector_index_for_config(
                parsed_dir=parsed_dir,
                index_dir=index_dir,
                model_name=model,
                chunk_size=chunk_size,
                overlap=overlap,
                include_title_chunk=include_title_chunk,
                batch_size=batch_size,
                device=device,
                text_source=resolved_text_source,
            )

        index_manifest = {
            "method": method,
            "model_name": model if method in METHODS_WITH_VECTOR else None,
            "parsed_dir": str(Path(parsed_dir).resolve()),
            "text_source": resolved_text_source,
            **params,
        }
        if method in METHODS_WITH_VECTOR:
            index_path, metadata_path = _vector_paths(index_dir)
            index_manifest.update(
                {
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                }
            )
        _write_json(index_dir / "config.json", index_manifest)

        if method in METHODS_WITH_VECTOR:
            index_path, metadata_path = _vector_paths(index_dir)
            if method in {"dense", "hybrid"}:
                config = SearchConfig(
                    parsed_dir=parsed_dir,
                    index_path=index_path,
                    metadata_path=metadata_path,
                    model_name=model,
                    device=device,
                    bm25_max_chars=chunk_size,
                    bm25_overlap=overlap,
                    bm25_include_title_chunk=include_title_chunk,
                )
            else:
                config = SearchConfig(
                    parsed_dir=parsed_dir,
                    e5_index_path=index_path,
                    e5_metadata_path=metadata_path,
                    e5_model_name=model,
                    device=device,
                    bm25_max_chars=chunk_size,
                    bm25_overlap=overlap,
                    bm25_include_title_chunk=include_title_chunk,
                )
        else:
            config = SearchConfig(
                parsed_dir=parsed_dir,
                device=device,
                bm25_max_chars=chunk_size,
                bm25_overlap=overlap,
                bm25_include_title_chunk=include_title_chunk,
            )

        run_dir = runs_root / method / (model_slug if method in METHODS_WITH_VECTOR else "") / config_slug
        search_function = _search_fn_for_method(method=method, config=config)
        run_metadata = {
            "method": method,
            "model_name": model if method in METHODS_WITH_VECTOR else "bm25",
            "experiment": experiment_dir.name,
            "config_slug": config_slug,
            "text_source": resolved_text_source,
            **params,
            "index_dir": str(index_dir),
        }
        evaluate_system(
            search_function=search_function,
            k=top_k,
            metadata=run_metadata,
            output_dir=str(run_dir),
            aggregate_dir=str(aggregate_dir),
        )

        _write_json(
            run_dir / "run_config.json",
            {
                "search_config": asdict(config),
                "run_metadata": run_metadata,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chunking/preprocessing evaluation sweeps.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "dense_e5", "hybrid", "hybrid_e5"],
        required=True,
        help="Search method to evaluate in a parameter sweep.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="flask/output/parsed",
        help="Directory with parsed Docplus JSON files.",
    )
    parser.add_argument(
        "--experiment-name",
        default=f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Folder name under evaluation/experiments/.",
    )
    parser.add_argument(
        "--experiments-root",
        default="evaluation/experiments",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k for retrieval/evaluation.",
    )
    parser.add_argument(
        "--chunk-sizes",
        default=str(DEFAULT_CHUNK_SIZE),
        help="Comma-separated chunk size values (characters).",
    )
    parser.add_argument(
        "--overlaps",
        default=str(DEFAULT_OVERLAP),
        help="Comma-separated overlap values (characters).",
    )
    parser.add_argument(
        "--include-title-chunk",
        default="true,false",
        help="Comma-separated booleans controlling title-only chunks (e.g. true,false).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size for vector index builds.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for vector embedding/query.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional vector profile key used when --model-name is not set.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model override for vector methods.",
    )
    parser.add_argument(
        "--text-source",
        choices=["text", "cleaned_text"],
        default="text",
        help="Parsed JSON field to use when building vector indexes.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    chunk_sizes = _parse_int_list(args.chunk_sizes)
    overlaps = _parse_int_list(args.overlaps)
    include_title_chunk_values = _parse_bool_list(args.include_title_chunk)

    parsed_dir = _resolve_path(args.parsed_dir, must_exist=True, prefer_repo=True)
    experiments_root = _resolve_path(args.experiments_root, must_exist=False, prefer_repo=True)
    experiment_dir = experiments_root / _slugify(args.experiment_name)
    run_sweep(
        method=args.method,
        parsed_dir=str(parsed_dir),
        experiment_dir=experiment_dir,
        top_k=args.top_k,
        chunk_sizes=chunk_sizes,
        overlaps=overlaps,
        include_title_chunk_values=include_title_chunk_values,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model_name,
        profile=args.profile if args.method in METHODS_WITH_VECTOR else None,
        text_source=args.text_source,
    )


if __name__ == "__main__":
    main()
