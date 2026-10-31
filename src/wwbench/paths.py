from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
METADATA = DATA / "metadata"
RESULTS = DATA / "results"
DOCS = ROOT / "docs"


def ensure_dirs() -> None:
    for path in (PROCESSED, METADATA, RESULTS, RESULTS / "models", DOCS):
        path.mkdir(parents=True, exist_ok=True)
