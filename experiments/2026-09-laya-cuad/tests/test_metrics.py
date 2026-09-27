from jev_bench.context import chunk_text
from jev_bench.metrics import summarize
from jev_bench.models.base import parse_yes_no


def test_summarize_perfect():
    recs = [
        {"category": "A", "label": True, "pred": True, "latency_s": 1.0},
        {"category": "A", "label": False, "pred": False, "latency_s": 1.0},
    ]
    s = summarize(recs)
    assert s["micro"]["f1"] == 1.0
    assert s["micro"]["accuracy"] == 1.0


def test_unparseable_counts_as_wrong():
    recs = [{"category": "A", "label": True, "pred": None, "latency_s": 0.0}]
    s = summarize(recs)
    assert s["unparseable"] == 1
    assert s["micro"]["accuracy"] == 0.0


def test_parse_yes_no():
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no('{"answer": "No"}') is False
    assert parse_yes_no("maybe") is None


def test_chunk_text_overlap():
    chunks = chunk_text("a" * 25, max_chars=10, overlap=2)
    assert all(len(c) <= 10 for c in chunks)
    assert sum(len(c) for c in chunks) >= 25
