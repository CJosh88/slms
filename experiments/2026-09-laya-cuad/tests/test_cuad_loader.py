import pytest

from jev_bench.data.cuad import RAW_JSON, N_CATEGORIES, N_CONTRACTS, categories, parse_cuad_json, validate

pytestmark = pytest.mark.skipif(not RAW_JSON.exists(), reason="run scripts/download_cuad.py first")


@pytest.fixture(scope="module")
def examples():
    return parse_cuad_json()


def test_shape(examples):
    validate(examples)
    assert len(categories(examples)) == N_CATEGORIES
    assert len({e.contract_id for e in examples}) == N_CONTRACTS


def test_labels_match_spans(examples):
    for e in examples:
        assert e.label == bool(e.spans)


def test_each_category_has_positives(examples):
    # Not every category has negatives: every contract has a Document Name.
    for c in categories(examples):
        assert any(e.label for e in examples if e.category == c), c


def test_question_is_yes_no(examples):
    assert all("Answer Yes or No" in e.question for e in examples)
