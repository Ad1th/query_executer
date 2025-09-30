import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input_csvs = [
    'tpcds_10_q21.csv',
    'tpch_10_q7.csv',
]
engines = ['postgres', 'duckdb', 'mysql']
time_suffix = '_runtime_ms'
sep = ';'

# === QERROR FUNCTION ===
def compute_qerr(a, b):
    return max(a / b, b / a) if a > 0 and b > 0 else np.nan

# === STORAGE ===
qerrs_by_engine = {engine: [] for engine in engines}
metrics_rows = []  # optional: per-file/per-engine summary

for input_csv in input_csvs:
    if not os.path.exists(input_csv):
        print(f"⚠️ File not found, skipping: {input_csv}")
        continue

    # Load & normalize
    df = pd.read_csv(input_csv, sep=sep)
    df.columns = df.columns.str.strip()

    if 'row_value' not in df.columns:
        print(f"⚠️ 'row_value' missing in {input_csv}; skipping this file.")
        continue

    df['row_value'] = pd.to_numeric(df['row_value'], errors='coerce')
    df = df.sort_values('row_value')
    dataset_name = os.path.splitext(os.path.basename(input_csv))[0]

    for engine in engines:
        col = engine + time_suffix
        if col not in df.columns:
            print(f"⚠️ [{dataset_name}] Missing column: {col}; skipping.")
            continue

        runtime = (
            df[col].astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
        row_count = df['row_value']

        # Drop NaNs
        valid = (~runtime.isna()) & (~row_count.isna())
        runtime = runtime[valid].reset_index(drop=True)
        row_count = row_count[valid].reset_index(drop=True)

        if len(runtime) < 2:
            print(f"⚠️ [{dataset_name}:{engine}] Not enough points; skipping.")
            continue

        # Consecutive Q-Errors (computed within this dataset only)
        q = np.array([compute_qerr(a, b) for a, b in zip(runtime[:-1], runtime[1:])], dtype=float)
        q = q[np.isfinite(q)]
        if q.size == 0:
            print(f"⚠️ [{dataset_name}:{engine}] No valid Q-Errors; skipping.")
            continue

        # Accumulate for this engine across datasets
        qerrs_by_engine[engine].extend(q.tolist())

        # Optional per-file metrics
        avg_q = float(np.nanmean(q))
        metrics_rows.append({
            'dataset': dataset_name,
            'engine': engine,
            'n_pairs': int(q.size),
            'avg_consecutive_qerr': avg_q,
        })

# Optional: print per-file metrics
if metrics_rows:
    results = pd.DataFrame(metrics_rows).sort_values(['dataset', 'engine'])
    print(results.to_string(index=False))
else:
    print("No per-file metrics computed (check inputs/columns).")

# === GLOBAL CDF PLOT: one curve per engine ===
has_any = False
plt.figure(figsize=(9, 6))

for engine in engines:
    q_all = np.array(qerrs_by_engine[engine], dtype=float)
    q_all = q_all[np.isfinite(q_all)]
    if q_all.size == 0:
        print(f"ℹ️ No Q-Errors aggregated for engine: {engine}")
        continue

    q_sorted = np.sort(q_all)
    cdf = np.arange(1, q_sorted.size + 1) / q_sorted.size
    plt.plot(q_sorted, cdf, label=engine)
    has_any = True

if has_any:
    plt.xlabel('Q-Error Threshold')
    plt.ylabel('Fraction of Points ≤ Threshold')
    plt.title('CDF of Consecutive Q-Errors (aggregated across query templates)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('qerror_cdf_aggregated_by_engine.png')
    print("✅ Saved: qerror_cdf_aggregated_by_engine.png")
    plt.close()
else:
    print("No Q-Error curves to plot.")
