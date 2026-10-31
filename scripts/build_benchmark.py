import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wwbench.build import build_all
from wwbench.docsgen import write_docs
from wwbench.experiments import run_experiments


if __name__ == "__main__":
    summary = build_all()
    print("BUILD", summary["summary"])
    results = run_experiments()
    print("SYNTHETIC_F1", results["synthetic_f1"])
    print("BSM1_MACRO_F1", results["bsm1_scenario_macro_f1"])
    print("SCREEN", {key: {"z": value["flagged_z"], "if": value["flagged_if"], "rows": value["rows"]} for key, value in results["screening"].items()})
    write_docs()
    print("DOCS")
