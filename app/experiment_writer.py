from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # app/ → query_executer → DB_Performance
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def write_experiment(
    query_id: str,
    database: str,
    parameters: dict,
    runtime_ms: float,
    filters: dict,
    plan_json: dict,
    ):
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("EXPERIMENTS_DIR =", EXPERIMENTS_DIR)
    base = EXPERIMENTS_DIR / database / query_id
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir()

    (run_dir / "plan.json").write_text(json.dumps(plan_json, indent=2))

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "runtime_ms": runtime_ms,
                "filters": filters,
                "parameters": parameters,
            },
            indent=2,
        )
    )

    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "database": database,
                "query_id": query_id,
                "timestamp": ts,
            },
            indent=2,
        )
    )