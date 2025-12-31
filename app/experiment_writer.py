from pathlib import Path
import json
from datetime import datetime
import uuid


def write_experiment(
    query_id: str,
    database: str,
    parameters: dict,
    runtime_ms: float,
    filters: dict,
    plan_json,
):
    """
    Persist one experiment run to disk.

    experiments/
      └── postgres/
          └── q1/
              └── <run_id>/
                  ├── plan.json
                  ├── metrics.json
                  └── meta.json
    """

    db = database.lower()
    base = Path("experiments") / db / query_id
    base.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True)

    # Normalize Postgres JSON format (usually [ { ... } ])
    if isinstance(plan_json, list) and len(plan_json) == 1:
        plan_json = plan_json[0]

    (run_dir / "plan.json").write_text(
        json.dumps(plan_json, indent=2)
    )

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
                "query_id": query_id,
                "database": db,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        )
    )