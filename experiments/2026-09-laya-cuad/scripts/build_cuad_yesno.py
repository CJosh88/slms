"""Convert raw CUAD_v1.json into data/processed/cuad_yesno.jsonl (41 yes/no per contract)."""

import json
from collections import Counter

from jev_bench.data.cuad import PROCESSED_JSONL, categories, parse_cuad_json, validate, write_jsonl


def main() -> None:
    examples = parse_cuad_json()
    validate(examples)
    write_jsonl(examples)

    cats = categories(examples)
    (PROCESSED_JSONL.parent / "categories.json").write_text(json.dumps(cats, indent=2), encoding="utf-8")

    contracts = {ex.contract_id for ex in examples}
    yes = Counter(ex.category for ex in examples if ex.label)
    print(f"contracts={len(contracts)} categories={len(cats)} examples={len(examples)}")
    print(f"overall yes rate: {sum(yes.values()) / len(examples):.1%}\n")
    print(f"{'category':<40} {'yes':>5} {'no':>5}")
    for c in cats:
        print(f"{c:<40} {yes[c]:>5} {len(contracts) - yes[c]:>5}")
    print(f"\nwrote {PROCESSED_JSONL}")


if __name__ == "__main__":
    main()
