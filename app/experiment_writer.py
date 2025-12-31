from pathlib import Path
import json
from datetime import datetime

def write_experiment(
    query_id,
    database,
    parameters,
    runtime_ms,
    filters,
    plan_json,
):
    base = Path("experiments") / database / query_id
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir()

    (run_dir / "plan.json").write_text(json.dumps(plan_json, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({
        "runtime_ms": runtime_ms,
        "filters": filters,
        "parameters": parameters,
    }, indent=2)