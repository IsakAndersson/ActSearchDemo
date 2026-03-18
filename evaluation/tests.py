import pandas as pd
import evaluation
import pytest
from pathlib import Path
try:
    from .search_adapter import (
        SearchConfig,
        _extract_live_docplus_links,
        _extract_live_sts_links,
        docplus_live_search,
        sts_live_search,
    )
    from .form_submissions_to_qrels import (
        export_qrels_from_db,
        submission_to_qrels_rows,
    )
except ImportError:
    from search_adapter import (
        SearchConfig,
        _extract_live_docplus_links,
        _extract_live_sts_links,
        docplus_live_search,
        sts_live_search,
    )
    from form_submissions_to_qrels import (
        export_qrels_from_db,
        submission_to_qrels_rows,
    )
 
def run_tests():
    
    evaluation.evaluate_system(make_search_hit_at_rank(1, "Grundläggande omhändertagande av nyfött barn"), 20, 
        metadata={
        "istest": "YES",
        "search_method": "hybrid",
        "model_name": "KBLab/bert-base-swedish-cased",
        "run_tag": "baseline_v1"
    })
    
    #evaluate_query_type
    test_evaluate_query_type_no_hits_returns_zero()
    test_evaluate_query_type_hit_outside_top_k_returns_zero()
    test_evaluate_query_type_one_hit_one_miss_average_rank_uses_only_hits()
    
    #Tests for calculate average rank
    test_calculate_average_rank_basic()
    test_calculate_average_rank_no_hits_returns_zero()
    
    #Integration test
    test_evaluate_integration_with_real_pipeline()
    
# UTILS

def search_no_hits(query: str, top_k: int = 20):
    # Returnerar alltid irrelevanta dokument
    return [("D_x", 2.0), ("D_y", 1.0)][:top_k]

def make_search_hit_at_rank(rank: int, rel_doc: str):
    """
    Skapar en search-funktion där rel_doc hamnar på exakt given rank (1-index).
    rank=1 => rel_doc först, rank=25 => rel_doc utanför top_k=20 om k=20 osv.
    """
    def _search(query: str, top_k: int = 20):
        results = []
        # Lägg in rank-1 irrelevanta docs med högre score
        for i in range(rank - 1):
            results.append((f"D_irrel_{i}", float(100 - i)))
        # Lägg in relevanta doc med lägre score än de ovan
        results.append((rel_doc, 0.1))
        # Fyll på lite mer
        for j in range(50):
            results.append((f"D_tail_{j}", float(-j)))
        return results[:top_k]
    return _search

#Tests for function evaluate_query_type

def test_evaluate_query_type_no_hits_returns_zero():
    doc_ids = pd.Series(["D_rel_0", "D_rel_1"])
    queries = pd.Series(["query_0", "query_1"])

    rr20, avg_rank = evaluation.evaluate_query_type(
        doc_ids=doc_ids,
        queries=queries,
        search_function=search_no_hits,
        k=20
    )

    assert rr20 == 0
    assert avg_rank == 0

def test_evaluate_query_type_hit_outside_top_k_returns_zero():
    # Relevant doc hamnar på rank 25, men k=20 => RR@20 = 0
    doc_ids = pd.Series(["D_rel_0"])
    queries = pd.Series(["query_0"])

    search_fn = make_search_hit_at_rank(rank=25, rel_doc="D_rel_0")

    rr20, avg_rank = evaluation.evaluate_query_type(
        doc_ids=doc_ids,
        queries=queries,
        search_function=search_fn,
        k=20
    )

    assert rr20 == 0
    assert avg_rank == 0

def test_evaluate_query_type_one_hit_one_miss_average_rank_uses_only_hits():
    # Q0: hit på rank 4 => RR=1/4, rank=4
    # Q1: miss => RR=0, ignoreras i avg_rank (enligt din calculate_average_rank)
    doc_ids = pd.Series(["D_rel_0", "D_rel_1"])
    queries = pd.Series(["query_0", "query_1"])

    def search_mixed(query: str, top_k: int = 20):
        if query == "query_0":
            # relevant på rank 4
            return [
                ("D_a", 3.0),
                ("D_b", 2.0),
                ("D_c", 1.5),
                ("D_rel_0", 1.0),
                ("D_tail", 0.1),
            ][:top_k]
        else:
            # ingen relevant
            return [("D_x", 2.0), ("D_y", 1.0)][:top_k]

    rr20, avg_rank = evaluation.evaluate_query_type(
        doc_ids=doc_ids,
        queries=queries,
        search_function=search_mixed,
        k=20
    )

    # RR@20 aggregate: (1/4 + 0) / 2 = 0.125
    assert rr20 == pytest.approx((1/4) / 2, abs=1e-6)

    # avg_rank räknas bara på de queries som hade score>0 => bara rank 4
    assert avg_rank == pytest.approx(4.0, abs=1e-6)


#Tests for calculate average rank


def test_calculate_average_rank_basic():
    # Två queries: relevanta dokument på rank 1 respektive rank 5 => avg rank = (1 + 5) / 2 = 3.0
    qrels = pd.DataFrame([
        {"query_id": "Q0", "doc_id": "D_rel_0", "relevance": 1},
        {"query_id": "Q1", "doc_id": "D_rel_1", "relevance": 1},
    ])

    run = pd.DataFrame([
        # Q0: relevant på rank 1
        {"query_id": "Q0", "doc_id": "D_rel_0", "score": 100.0},
        {"query_id": "Q0", "doc_id": "D_x",     "score":  90.0},

        # Q1: relevant på rank 5 (fyra docs med högre score)
        {"query_id": "Q1", "doc_id": "D_a",     "score": 100.0},
        {"query_id": "Q1", "doc_id": "D_b",     "score":  90.0},
        {"query_id": "Q1", "doc_id": "D_c",     "score":  80.0},
        {"query_id": "Q1", "doc_id": "D_d",     "score":  70.0},
        {"query_id": "Q1", "doc_id": "D_rel_1", "score":  60.0},
    ])

    avg_rank = evaluation.calculate_average_rank(qrels, run)
    assert avg_rank == pytest.approx(3.0, abs=1e-9)

def test_calculate_average_rank_no_hits_returns_zero():
    # Ett query där relevant dokument inte finns i run => RR@20=0 => ranks blir tom => return 0
    qrels = pd.DataFrame([
        {"query_id": "Q0", "doc_id": "D_rel", "relevance": 1},
    ])

    run = pd.DataFrame([
        {"query_id": "Q0", "doc_id": "D_a", "score": 2.0},
        {"query_id": "Q0", "doc_id": "D_b", "score": 1.0},
    ])

    avg_rank = evaluation.calculate_average_rank(qrels, run)
    assert avg_rank == 0


def test_build_run_df_empty_has_expected_columns():
    run = evaluation.build_run_df(
        queries=pd.Series(["q0"]),
        search_fn=lambda query, top_k: [],
        top_k=20,
    )
    assert list(run.columns) == ["doc_id", "query_id", "score"]
    assert run.empty


def test_submission_to_qrels_rows_includes_only_relevant_docs_by_default():
    payload = {
        "query": "feber barn",
        "results": [
            {
                "title": "Rutin A.pdf",
                "assessment": {"rating": "relevant"},
            },
            {
                "title": "Rutin B.pdf",
                "assessment": {"rating": "not_relevant"},
            },
        ],
    }

    rows = submission_to_qrels_rows(payload)

    assert rows == [
        {
            "query_id": "feber barn",
            "doc_id": "rutin a",
            "relevance": 1,
        }
    ]


def test_submission_to_qrels_rows_can_include_non_relevant_docs():
    payload = {
        "query": "feber barn",
        "results": [
            {
                "metadata": {"title": "Rutin A.pdf"},
                "assessment": {"rating": "relevant"},
            },
            {
                "source_path": "docs/Rutin B.pdf",
                "assessment": {"rating": "not_relevant"},
            },
        ],
    }

    rows = submission_to_qrels_rows(payload, include_non_relevant=True)

    assert rows == [
        {
            "query_id": "feber barn",
            "doc_id": "rutin a",
            "relevance": 1,
        },
        {
            "query_id": "feber barn",
            "doc_id": "docs/rutin b",
            "relevance": 0,
        },
    ]


def test_export_qrels_from_db_writes_csv(tmp_path: Path):
    db_path = tmp_path / "form_submissions.sqlite3"
    output_path = tmp_path / "qrels.csv"

    import sqlite3

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                participant_name TEXT NOT NULL,
                information_need TEXT NOT NULL,
                query_text TEXT NOT NULL,
                general_comment TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO submissions (
                created_at,
                participant_name,
                information_need,
                query_text,
                general_comment,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2025-01-01T00:00:00+00:00",
                "Test User",
                "Need A",
                "feber barn",
                "",
                """
                {
                  "query": "feber barn",
                  "results": [
                    {
                      "title": "Rutin A.pdf",
                      "assessment": {"rating": "relevant"}
                    },
                    {
                      "title": "Rutin B.pdf",
                      "assessment": {"rating": "not_relevant"}
                    }
                  ]
                }
                """,
            ),
        )
        connection.commit()

    qrels_df = export_qrels_from_db(db_path, output_path, include_non_relevant=True)

    assert list(qrels_df.columns) == ["query_id", "doc_id", "relevance"]
    assert qrels_df.to_dict("records") == [
        {"query_id": "feber barn", "doc_id": "rutin a", "relevance": 1},
        {"query_id": "feber barn", "doc_id": "rutin b", "relevance": 0},
    ]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "query_id,doc_id,relevance",
        "feber barn,rutin a,1",
        "feber barn,rutin b,0",
    ]


def test_load_qrels_dataset_from_form_submissions_defaults_to_single_query_type(tmp_path: Path):
    qrels_path = tmp_path / "qrels.csv"
    qrels_path.write_text(
        "\n".join(
            [
                "query_id,doc_id,relevance",
                "feber barn,Rutin A.pdf,1",
                "feber barn,Rutin B.pdf,0",
                "hosta vuxen,Rutin C.pdf,1",
            ]
        ),
        encoding="utf-8",
    )

    query_type_frames = evaluation.load_qrels_dataset(
        qrels_source="form_submissions",
        qrels_path=str(qrels_path),
    )

    assert list(query_type_frames.keys()) == ["form_submissions"]
    frame = query_type_frames["form_submissions"]
    assert frame["queries"]["query_id"].tolist() == ["feber barn", "hosta vuxen"]
    assert frame["qrels"].to_dict("records") == [
        {"query_id": "feber barn", "doc_id": "rutin a", "relevance": 1},
        {"query_id": "feber barn", "doc_id": "rutin b", "relevance": 0},
        {"query_id": "hosta vuxen", "doc_id": "rutin c", "relevance": 1},
    ]


def test_load_qrels_dataset_from_form_submissions_supports_query_type_column(tmp_path: Path):
    qrels_path = tmp_path / "qrels.csv"
    qrels_path.write_text(
        "\n".join(
            [
                "query_id,doc_id,relevance,query_type",
                "feber barn,Rutin A.pdf,1,short_query",
                "hosta vuxen,Rutin C.pdf,1,long_query",
            ]
        ),
        encoding="utf-8",
    )

    query_type_frames = evaluation.load_qrels_dataset(
        qrels_source="form_submissions",
        qrels_path=str(qrels_path),
    )

    assert list(query_type_frames.keys()) == ["short_query", "long_query"]
    assert query_type_frames["short_query"]["queries"]["query_id"].tolist() == ["feber barn"]
    assert query_type_frames["long_query"]["qrels"].to_dict("records") == [
        {"query_id": "hosta vuxen", "doc_id": "rutin c", "relevance": 1},
    ]


#Integration test

def test_evaluate_integration_with_real_pipeline():
    # ---- Input till evaluate() ----
    # Gold docs (rätt dokument per rad/query)
    doc_ids = pd.Series(["DocA", "DocB"])

    # Två query-typer med två queries vardera (rad 0 hör ihop med DocA, rad 1 med DocB)
    query_types_cols = pd.DataFrame({
        "TypeA": ["qA0", "qA1"],
        "TypeB": ["qB0", "qB1"],
    })

    # ---- Dummy search som ger kontrollerade ranker beroende på query ----
    # Vi vill få:
    # TypeA:
    #   qA0 -> DocA rank 1 => RR=1
    #   qA1 -> DocB rank 3 => RR=1/3
    #   MRR = (1 + 1/3)/2
    #   avg_rank = (1 + 3)/2
    #
    # TypeB:
    #   qB0 -> DocA rank 5 => RR=1/5
    #   qB1 -> DocB miss (ej i top_k=20) => RR=0
    #   MRR = (1/5 + 0)/2
    #   avg_rank ignorerar miss => avg_rank = 5

    def dummy_search(query: str, top_k: int = 20):
        if query == "qA0":
            return [("DocA", 10.0), ("X", 9.0), ("Y", 8.0)][:top_k]
        if query == "qA1":
            return [("X", 10.0), ("Y", 9.0), ("DocB", 8.0)][:top_k]
        if query == "qB0":
            return [("X1", 10.0), ("X2", 9.0), ("X3", 8.0), ("X4", 7.0), ("DocA", 6.0)][:top_k]
        if query == "qB1":
            # ingen DocB i top_k => miss
            return [("X", 10.0), ("Y", 9.0), ("Z", 8.0)][:top_k]
        return []

    # ---- Kör evaluate ----
    results_df, avg_rank, avg_score = evaluation.evaluate(
        search_function=dummy_search,
        k=20,
        doc_ids=doc_ids,
        query_types_cols=query_types_cols
    )

    # ---- Förväntade per query-type ----
    expected_typeA_rr = (1 + 1/3) / 2
    expected_typeA_rank = (1 + 3) / 2

    expected_typeB_rr = ((1/5) + 0) / 2
    expected_typeB_rank = 5.0  # miss ignoreras av calculate_average_rank

    # ---- Kolla att DF har rätt format ----
    assert set(results_df.columns) == {"query_type", "RR@20", "average_rank"}
    assert len(results_df) == 2

    # ---- Plocka rader ----
    rowA = results_df[results_df["query_type"] == "TypeA"].iloc[0]
    rowB = results_df[results_df["query_type"] == "TypeB"].iloc[0]

    assert rowA["RR@20"] == pytest.approx(expected_typeA_rr, abs=1e-6)
    assert rowA["average_rank"] == pytest.approx(expected_typeA_rank, abs=1e-6)

    assert rowB["RR@20"] == pytest.approx(expected_typeB_rr, abs=1e-6)
    assert rowB["average_rank"] == pytest.approx(expected_typeB_rank, abs=1e-6)

    # ---- Kolla att totalsnitten stämmer ----
    expected_avg_score = (expected_typeA_rr + expected_typeB_rr) / 2
    expected_avg_rank = (expected_typeA_rank + expected_typeB_rank) / 2

    assert avg_score == pytest.approx(expected_avg_score, abs=1e-6)
    assert avg_rank == pytest.approx(expected_avg_rank, abs=1e-6)


def test_extract_live_docplus_links_filters_non_documents():
    html = """
    <html><body>
      <a href="/Home/GetDocument?filename=Alpha%20Plan.pdf">Alpha Plan</a>
      <a href="/Home/Details/123">Not a document</a>
      <a href="https://publikdocplus.regionuppsala.se/Home/GetDocument?file=Beta.docx"></a>
    </body></html>
    """
    links = _extract_live_docplus_links(
        html=html,
        page_url="https://publikdocplus.regionuppsala.se/Home/Search",
    )

    assert len(links) == 2
    assert links[0]["source_url"].endswith("filename=Alpha%20Plan.pdf")
    assert links[0]["title"] == "Alpha Plan"
    assert "Beta.docx" in links[1]["source_url"]
    assert links[1]["title"] == "Beta.docx"


def test_docplus_live_search_returns_normalized_ranked_doc_ids():
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self):
            return None

    class _FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def get(self, url, params, timeout):
            self.calls.append((url, params, timeout))
            return _FakeResponse(
                """
                <html><body>
                  <a href="/Home/GetDocument?filename=Vårdprogram%20A.pdf">Vårdprogram A.pdf</a>
                  <a href="/Home/GetDocument?filename=Vårdprogram%20B.pdf">Vårdprogram B.pdf</a>
                </body></html>
                """
            )

    fake_session = _FakeSession()
    config = SearchConfig(live_max_pages=1)
    results = docplus_live_search(
        query="vårdprogram",
        top_k=2,
        config=config,
        session=fake_session,  # type: ignore[arg-type]
    )

    assert results == [("vårdprogram a", 2.0), ("vårdprogram b", 1.0)]
    assert len(fake_session.calls) == 1


def test_extract_live_sts_links_filters_non_documents():
    html = """
    <html><body>
      <a href="/redirect?to=https://publikdocplus.regionuppsala.se/Home/GetDocument?filename=Gamma%20Doc.pdf">Gamma</a>
      <a href="https://publikdocplus.regionuppsala.se/Home/GetDocument?file=Delta.docx">Delta</a>
      <a href="/about">About</a>
    </body></html>
    """
    links = _extract_live_sts_links(
        html=html,
        page_url="https://sts.search.datatovalue.se/",
    )

    assert len(links) == 2
    assert "Gamma%20Doc.pdf" in links[0]["source_url"]
    assert links[0]["title"] == "Gamma"
    assert "Delta.docx" in links[1]["source_url"]
    assert links[1]["title"] == "Delta"


def test_sts_live_search_returns_normalized_ranked_doc_ids():
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self):
            return None

    class _FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def get(self, url, params, timeout):
            self.calls.append((url, params, timeout))
            return _FakeResponse(
                """
                <html><body>
                  <a href="https://publikdocplus.regionuppsala.se/Home/GetDocument?filename=Rutin%20A.pdf">Rutin A</a>
                  <a href="https://publikdocplus.regionuppsala.se/Home/GetDocument?filename=Rutin%20B.pdf">Rutin B</a>
                </body></html>
                """
            )

    fake_session = _FakeSession()
    config = SearchConfig(sts_live_max_pages=1)
    results = sts_live_search(
        query="rutin",
        top_k=2,
        config=config,
        session=fake_session,  # type: ignore[arg-type]
    )

    assert results == [("rutin a", 2.0), ("rutin b", 1.0)]
    assert len(fake_session.calls) == 1

if __name__ == "__main__":
    run_tests()
