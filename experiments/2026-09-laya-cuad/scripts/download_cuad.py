"""Download the raw CUAD v1 files from Hugging Face (PDFs skipped).

Note: datasets.load_dataset("theatticusproject/cuad") is NOT usable - the
auto-converted parquet is a single `text` column. We pull the raw files instead.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "cuad"

ALLOW_PATTERNS = [
    "CUAD_v1/CUAD_v1.json",
    "CUAD_v1/master_clauses.csv",
    "CUAD_v1/full_contract_txt/**",
    "CUAD_v1/CUAD v1 ReadMe _ Datasheet/*",
]


def main() -> None:
    path = snapshot_download(
        repo_id="theatticusproject/cuad",
        repo_type="dataset",
        local_dir=TARGET,
        allow_patterns=ALLOW_PATTERNS,
    )
    base = Path(path) / "CUAD_v1"
    n_txt = len(list((base / "full_contract_txt").rglob("*.txt")))
    print(f"Downloaded to {base}")
    print(f"  CUAD_v1.json:       {(base / 'CUAD_v1.json').exists()}")
    print(f"  master_clauses.csv: {(base / 'master_clauses.csv').exists()}")
    print(f"  contract txt files: {n_txt}")


if __name__ == "__main__":
    main()
