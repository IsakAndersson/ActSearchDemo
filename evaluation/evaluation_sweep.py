"""Run evaluation sweeps across chunking and preprocessing configurations."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

try:
    from .evaluation import evaluate_system
    from .form_submissions_to_qrels import (
        DEFAULT_SUBMISSIONS_DB,
        build_submission_audit_df,
        export_qrels_from_db,
    )
    from .search_adapter import (
        SearchConfig,
        bm25_search,
        dense_e5_search,
        dense_search,
        hybrid_e5_search,
        hybrid_search,
        sqlite_fts_search,
    )
except ImportError:
    from evaluation import evaluate_system
    from form_submissions_to_qrels import (
        DEFAULT_SUBMISSIONS_DB,
        build_submission_audit_df,
        export_qrels_from_db,
    )
    from search_adapter import (
        SearchConfig,
        bm25_search,
        dense_e5_search,
        dense_search,
        hybrid_e5_search,
        hybrid_search,
        sqlite_fts_search,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from search.vector_index import (
        DEFAULT_MODEL,
        build_index,
        resolve_model_name,
        resolve_text_source,
    )
    from search.sqlite_fts_search import build_sqlite_fts_index
except ModuleNotFoundError:
    DEFAULT_MODEL = "KBLab/bert-base-swedish-cased"

    def resolve_model_name(profile_key, model_name):
        if model_name:
            return model_name
        return DEFAULT_MODEL

    def resolve_text_source(text_source):
        return text_source or "text"

    def build_sqlite_fts_index(**kwargs):
        raise RuntimeError("SQLite FTS search module is unavailable.")


DEFAULT_TOP_K = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


METHODS_WITH_VECTOR = {"dense", "dense_e5", "hybrid", "hybrid_e5"}
METHODS_WITH_SQLITE_FTS = {"sqlite_fts"}
VECTOR_INDEX_COMPATIBLE_METHODS = {
    "dense": {"dense", "hybrid"},
    "hybrid": {"dense", "hybrid"},
    "dense_e5": {"dense_e5", "hybrid_e5"},
    "hybrid_e5": {"dense_e5", "hybrid_e5"},
}


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


def _parse_int_set(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return set(_parse_int_list(raw))


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


def _parse_path_list(raw: str | None) -> List[Path]:
    if not raw:
        return []
    values: List[Path] = []
    for item in raw.split(","):
        candidate = item.strip()
        if candidate:
            values.append(_resolve_path(candidate, must_exist=True, prefer_repo=True))
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


def _sqlite_fts_path(index_dir: Path) -> str:
    return str(index_dir / "docplus_fts.sqlite3")


def _text_source_label(text_source: str) -> str:
    return resolve_text_source(text_source)


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _config_path_value(config_path: Path, value: object, fallback_name: str) -> Path:
    if isinstance(value, str) and value.strip():
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (config_path.parent / candidate).resolve()
    return (config_path.parent / fallback_name).resolve()


def _vector_index_paths_from_config(config_path: Path, config: Dict[str, object]) -> tuple[Path, Path]:
    return (
        _config_path_value(config_path, config.get("index_path"), "docplus.faiss"),
        _config_path_value(config_path, config.get("metadata_path"), "docplus_metadata.jsonl"),
    )


def _vector_index_config_matches(
    config_path: Path,
    config: Dict[str, object],
    requested_method: str,
    parsed_dir: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    include_title_chunk: bool,
    text_source: str,
) -> bool:
    config_method = str(config.get("method") or "")
    if config_method not in VECTOR_INDEX_COMPATIBLE_METHODS.get(requested_method, set()):
        return False
    if str(config.get("model_name") or "") != model_name:
        return False
    if str(config.get("text_source") or "text") != text_source:
        return False
    if int(config.get("chunk_size", -1)) != int(chunk_size):
        return False
    if int(config.get("chunk_overlap", -1)) != int(overlap):
        return False
    if bool(config.get("include_title_chunk")) != bool(include_title_chunk):
        return False

    config_parsed_dir = config.get("parsed_dir")
    if isinstance(config_parsed_dir, str) and config_parsed_dir.strip():
        if not _same_resolved_path(config_parsed_dir, parsed_dir):
            return False

    index_path, metadata_path = _vector_index_paths_from_config(config_path, config)
    return index_path.exists() and metadata_path.exists()


def _find_existing_vector_index(
    search_roots: Sequence[Path],
    requested_method: str,
    parsed_dir: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    include_title_chunk: bool,
    text_source: str,
) -> tuple[Path, Path, Path] | None:
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for config_path in sorted(search_root.rglob("config.json")):
            try:
                config = _read_json(config_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not _vector_index_config_matches(
                config_path=config_path,
                config=config,
                requested_method=requested_method,
                parsed_dir=parsed_dir,
                model_name=model_name,
                chunk_size=chunk_size,
                overlap=overlap,
                include_title_chunk=include_title_chunk,
                text_source=text_source,
            ):
                continue
            index_path, metadata_path = _vector_index_paths_from_config(config_path, config)
            return config_path.parent.resolve(), index_path, metadata_path
    return None


def _prepare_form_submission_qrels(
    experiment_dir: Path,
    db_path: Path,
    exclude_submission_ids: set[int],
    include_non_relevant: bool,
    include_metadata: bool,
    query_type: str | None,
    qrels_output_path: Path | None,
    audit_output_path: Path | None,
) -> Path:
    qrels_path = qrels_output_path or experiment_dir / "qrels" / "qrels_from_form_submissions.csv"
    audit_path = audit_output_path or experiment_dir / "qrels" / "form_submissions_audit.csv"

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission_audit_df(db_path).to_csv(audit_path, index=False)
    export_qrels_from_db(
        db_path=db_path,
        output_csv=qrels_path,
        include_non_relevant=include_non_relevant,
        exclude_submission_ids=exclude_submission_ids,
        include_metadata=include_metadata,
        query_type=query_type,
    )
    return qrels_path


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
    if method == "sqlite_fts":
        return lambda query, top_k: sqlite_fts_search(query=query, top_k=top_k, config=config)
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
    qrels_source: str = "google_sheet",
    qrels_path: str | None = None,
    form_db_path: str | None = None,
    exclude_submission_ids: set[int] | None = None,
    include_non_relevant: bool = False,
    include_qrels_metadata: bool = False,
    form_query_type: str | None = None,
    form_audit_output: str | None = None,
    reuse_vector_indexes: bool = True,
    vector_index_search_roots: Sequence[Path] | None = None,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if qrels_source == "google_sheet" and qrels_path:
        raise ValueError("qrels_path can only be used with qrels_source='form_submissions'")
    if qrels_source == "google_sheet" and (
        form_db_path
        or exclude_submission_ids
        or include_non_relevant
        or include_qrels_metadata
        or form_query_type
        or form_audit_output
    ):
        raise ValueError("Form-submission options require qrels_source='form_submissions'")

    resolved_qrels_path = qrels_path
    if qrels_source == "form_submissions" and (
        form_db_path
        or exclude_submission_ids
        or include_non_relevant
        or include_qrels_metadata
        or form_query_type
        or form_audit_output
    ):
        resolved_form_db_path = _resolve_path(
            form_db_path or str(DEFAULT_SUBMISSIONS_DB),
            must_exist=True,
            prefer_repo=True,
        )
        resolved_qrels_output_path = (
            _resolve_path(qrels_path, must_exist=False, prefer_repo=True) if qrels_path else None
        )
        resolved_audit_output_path = (
            _resolve_path(form_audit_output, must_exist=False, prefer_repo=True)
            if form_audit_output
            else None
        )
        resolved_qrels_path = str(
            _prepare_form_submission_qrels(
                experiment_dir=experiment_dir,
                db_path=resolved_form_db_path,
                exclude_submission_ids=exclude_submission_ids or set(),
                include_non_relevant=include_non_relevant,
                include_metadata=include_qrels_metadata,
                query_type=form_query_type,
                qrels_output_path=resolved_qrels_output_path,
                audit_output_path=resolved_audit_output_path,
            )
        )
    elif qrels_path:
        resolved_qrels_path = str(_resolve_path(qrels_path, must_exist=True, prefer_repo=True))

    model = _resolve_vector_model(method=method, profile=profile, model_name=model_name)
    resolved_text_source = resolve_text_source(text_source)
    model_slug = _slugify(model)
    aggregate_dir = experiment_dir / "results" / "aggregate"
    runs_root = experiment_dir / "results" / "runs"
    indexes_root = experiment_dir / "indexes"
    default_vector_index_search_roots = [experiment_dir.parent]
    resolved_vector_index_search_roots = (
        [Path(path).resolve() for path in vector_index_search_roots]
        if vector_index_search_roots
        else default_vector_index_search_roots
    )

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
        "qrels_source": qrels_source,
        "qrels_path": str(Path(resolved_qrels_path).resolve()) if resolved_qrels_path else None,
        "reuse_vector_indexes": bool(reuse_vector_indexes),
        "vector_index_search_roots": [
            str(path.resolve()) for path in resolved_vector_index_search_roots
        ],
    }
    if qrels_source == "form_submissions":
        experiment_manifest.update(
            {
                "form_db_path": str(Path(form_db_path or DEFAULT_SUBMISSIONS_DB).resolve()),
                "exclude_submission_ids": sorted(exclude_submission_ids or set()),
                "include_non_relevant": bool(include_non_relevant),
                "include_qrels_metadata": bool(include_qrels_metadata),
                "form_query_type": form_query_type,
                "form_audit_output": (
                    str(Path(form_audit_output).resolve()) if form_audit_output else None
                ),
            }
        )
    _write_json(experiment_dir / "experiment_manifest.json", experiment_manifest)

    for params in _iter_grid(chunk_sizes, overlaps, include_title_chunk_values):
        chunk_size = params["chunk_size"]
        overlap = params["chunk_overlap"]
        include_title_chunk = params["include_title_chunk"]
        config_slug = (
            f"chunk_{chunk_size}__overlap_{overlap}__"
            f"{_bool_label(include_title_chunk)}__{_text_source_label(resolved_text_source)}"
        )

        index_reused = False
        reused_from_config_dir: str | None = None
        index_path: str | None = None
        metadata_path: str | None = None

        if method in METHODS_WITH_VECTOR:
            index_dir = indexes_root / method / model_slug / config_slug
            existing_index = None
            if reuse_vector_indexes:
                existing_index = _find_existing_vector_index(
                    search_roots=resolved_vector_index_search_roots,
                    requested_method=method,
                    parsed_dir=parsed_dir,
                    model_name=model,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    include_title_chunk=include_title_chunk,
                    text_source=resolved_text_source,
                )
            if existing_index:
                source_index_dir, source_index_path, source_metadata_path = existing_index
                index_dir = source_index_dir
                index_path = str(source_index_path)
                metadata_path = str(source_metadata_path)
                index_reused = True
                reused_from_config_dir = str(source_index_dir)
                print(
                    f"[sweep] Reusing vector index for {config_slug}: {source_index_dir}",
                    flush=True,
                )
            else:
                print(
                    f"[sweep] Building vector index for {config_slug}: {index_dir}",
                    flush=True,
                )
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
                index_path, metadata_path = _vector_paths(index_dir)
        else:
            index_dir = indexes_root / method / config_slug

        if method in METHODS_WITH_SQLITE_FTS:
            print(
                f"[sweep] Building SQLite FTS index for {config_slug}: {index_dir}",
                flush=True,
            )
            index_dir.mkdir(parents=True, exist_ok=True)
            build_sqlite_fts_index(
                parsed_dir=parsed_dir,
                db_path=_sqlite_fts_path(index_dir),
                max_chars=chunk_size,
                overlap=overlap,
                include_title_chunk=include_title_chunk,
                use_chunking=True,
            )
        elif method not in METHODS_WITH_VECTOR:
            index_dir.mkdir(parents=True, exist_ok=True)

        index_manifest = {
            "method": method,
            "model_name": model if method in METHODS_WITH_VECTOR else None,
            "parsed_dir": str(Path(parsed_dir).resolve()),
            "text_source": resolved_text_source,
            "index_reused": index_reused,
            "reused_from_config_dir": reused_from_config_dir,
            **params,
        }
        if method in METHODS_WITH_VECTOR:
            index_manifest.update(
                {
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                }
            )
        elif method in METHODS_WITH_SQLITE_FTS:
            index_manifest["sqlite_fts_path"] = _sqlite_fts_path(index_dir)
        if not index_reused:
            _write_json(index_dir / "config.json", index_manifest)

        if method in METHODS_WITH_VECTOR:
            if not index_path or not metadata_path:
                raise RuntimeError(f"Vector index paths were not resolved for {config_slug}")
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
        elif method in METHODS_WITH_SQLITE_FTS:
            config = SearchConfig(
                parsed_dir=parsed_dir,
                sqlite_fts_path=_sqlite_fts_path(index_dir),
                device=device,
                bm25_max_chars=chunk_size,
                bm25_overlap=overlap,
                bm25_include_title_chunk=include_title_chunk,
                bm25_use_chunking=True,
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
            "model_name": model if method in METHODS_WITH_VECTOR else method,
            "experiment": experiment_dir.name,
            "config_slug": config_slug,
            "text_source": resolved_text_source,
            "qrels_source": qrels_source,
            "index_reused": index_reused,
            "reused_from_config_dir": reused_from_config_dir,
            **params,
            "index_dir": str(index_dir),
        }
        if resolved_qrels_path:
            run_metadata["qrels_path"] = str(Path(resolved_qrels_path).resolve())
        evaluate_system(
            search_function=search_function,
            k=top_k,
            metadata=run_metadata,
            output_dir=str(run_dir),
            aggregate_dir=str(aggregate_dir),
            qrels_source=qrels_source,
            qrels_path=resolved_qrels_path,
        )

        _write_json(
            run_dir / "run_config.json",
            {
                "search_config": asdict(config),
                "run_metadata": run_metadata,
                "index_manifest": index_manifest,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chunking/preprocessing evaluation sweeps.")
    parser.add_argument(
        "--method",
        choices=["bm25", "sqlite_fts", "dense", "dense_e5", "hybrid", "hybrid_e5"],
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
    parser.add_argument(
        "--no-reuse-vector-indexes",
        action="store_true",
        help="Always rebuild vector indexes instead of reusing matching existing indexes.",
    )
    parser.add_argument(
        "--vector-index-search-roots",
        help=(
            "Comma-separated directories scanned for reusable vector indexes. "
            "Defaults to the experiments root."
        ),
    )
    parser.add_argument(
        "--qrels-source",
        choices=["google_sheet", "form_submissions"],
        default="google_sheet",
        help="Where to load qrels from.",
    )
    parser.add_argument(
        "--qrels-path",
        help=(
            "Optional qrels CSV path when --qrels-source=form_submissions. "
            "If form export flags are used, this is also the export destination."
        ),
    )
    parser.add_argument(
        "--form-db-path",
        default=None,
        help=(
            "Path to form_submissions.sqlite3. When set with --qrels-source=form_submissions, "
            "fresh qrels are exported before the sweep."
        ),
    )
    parser.add_argument(
        "--exclude-submission-ids",
        help="Comma-separated form submission ids to exclude when exporting fresh form qrels.",
    )
    parser.add_argument(
        "--include-non-relevant",
        action="store_true",
        help="Include form-submission rows with relevance=0 when exporting fresh form qrels.",
    )
    parser.add_argument(
        "--include-qrels-metadata",
        action="store_true",
        help="Include submission provenance columns when exporting fresh form qrels.",
    )
    parser.add_argument(
        "--form-query-type",
        help="Optional query_type value written to every freshly exported form qrels row.",
    )
    parser.add_argument(
        "--form-audit-output",
        help="Optional path for the form-submission audit CSV when exporting fresh form qrels.",
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
        qrels_source=args.qrels_source,
        qrels_path=args.qrels_path,
        form_db_path=args.form_db_path,
        exclude_submission_ids=_parse_int_set(args.exclude_submission_ids),
        include_non_relevant=args.include_non_relevant,
        include_qrels_metadata=args.include_qrels_metadata,
        form_query_type=args.form_query_type,
        form_audit_output=args.form_audit_output,
        reuse_vector_indexes=not args.no_reuse_vector_indexes,
        vector_index_search_roots=_parse_path_list(args.vector_index_search_roots),
    )


if __name__ == "__main__":
    main()
