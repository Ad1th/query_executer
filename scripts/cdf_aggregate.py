#!/usr/bin/env python3
# qerror_cdf_agg_by_strategy.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# =========================
# Utilities
# =========================
def qerror(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = (y_true > 0) & (y_pred > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    out = np.full_like(y_true, np.nan, dtype=float)
    out[ok] = np.maximum(y_true[ok] / y_pred[ok], y_pred[ok] / y_true[ok])
    return out

def qerr_stats(arr):
    import numpy as np
    v = np.asarray(arr, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(v)),
        "p95":  float(np.percentile(v, 95)),
        "p99":  float(np.percentile(v, 99)),
        "max":  float(np.max(v)),
    }

def ccdf(values, t_grid):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(t_grid, dtype=float)
    v_sorted = np.sort(v)
    idx = np.searchsorted(v_sorted, t_grid, side="left")
    return (v_sorted.size - idx) / v_sorted.size

def sample_indices_uniform(n_points, target_n):
    target_n = max(2, min(target_n, n_points))
    return np.unique(np.linspace(0, n_points - 1, num=target_n, dtype=int))

def sample_indices_random(n_points, target_n):
    target_n = max(2, min(target_n, n_points))
    if n_points == 2 and target_n == 2:
        return np.array([0, 1], dtype=int)

    rng = np.random.default_rng()
    keep = target_n - 2
    pool = np.arange(1, n_points - 1, dtype=int) if n_points > 2 else np.array([], dtype=int)
    rest = rng.choice(pool, size=keep, replace=False) if keep > 0 and pool.size > 0 else np.array([], dtype=int)
    idx = np.sort(np.unique(np.concatenate(([0, n_points - 1], rest))))
    if idx.size < target_n:
        extra = np.setdiff1d(np.linspace(0, n_points - 1, target_n, dtype=int), idx)[:(target_n - idx.size)]
        extra = np.setdiff1d(np.linspace(0, n_points - 1, target_n, dtype=int), idx)[:(target_n - idx.size)]
        idx = np.sort(np.concatenate((idx, extra)))
    return idx

def fit_predict_all(x_all, y_all, idx_sample, degree):
    x_all = np.asarray(x_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)

    x_s = x_all[idx_sample].reshape(-1, 1)
    y_s = y_all[idx_sample]

    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    def scale(z):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        if xmax == xmin:
            return np.zeros_like(z)
        return (z - xmin) / (xmax - xmin)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xs = poly.fit_transform(scale(x_s))
    model = LinearRegression().fit(Xs, y_s)

    X_all = poly.transform(scale(x_all))
    return model.predict(X_all)

# ===== stratified + adaptive samplers =====
def _hamilton_quotas(strata_labels, K, N, n_target):
    counts = np.bincount(strata_labels, minlength=K).astype(float)
    shares = counts / max(1, N) * n_target
    base = np.floor(shares).astype(int)
    rem = shares - base
    need = n_target - base.sum()
    if need > 0:
        order = np.argsort(-rem)
        base[order[:need]] += 1
    base = np.minimum(base, counts.astype(int))
    while base.sum() < n_target:
        leftover = n_target - base.sum()
        candidates = np.where(counts > base)[0]
        if candidates.size == 0:
            break
        take = min(leftover, candidates.size)
        base[candidates[:take]] += 1
    return base

def stratified_time_buckets(N, K):
    bins = np.linspace(0, N, K + 1, dtype=int)
    return np.digitize(np.arange(N), bins[1:], right=False)

def sample_stratified_indices(N, n_target, K=12):
    n_target = max(2, min(n_target, N))
    rng = np.random.default_rng()

    if N == 2 and n_target == 2:
        return np.array([0, 1], dtype=int)
    if N <= 2:
        return np.arange(min(N, n_target), dtype=int)

    bins = np.linspace(0, N, K + 1, dtype=int)
    strata = np.digitize(np.arange(N), bins[1:], right=False)

    forced = {0, N - 1}
    remaining = max(0, n_target - len(forced))
    quota = _hamilton_quotas(strata, K, N, remaining)

    for e in list(forced):
        if remaining <= 0:
            break
        b = strata[e]
        if quota[b] > 0:
            quota[b] -= 1
            remaining -= 1
        else:
            donor = np.argmax(quota)
            if quota[donor] > 0:
                quota[donor] -= 1
                remaining -= 1

    chosen = [np.array(sorted(forced), dtype=int)] if forced else []
    forced_arr = chosen[0] if chosen else np.array([], dtype=int)
    for k in range(K):
        qk = int(quota[k])
        if qk <= 0:
            continue
        idx = np.where(strata == k)[0]
        idx = idx[~np.isin(idx, forced_arr)]
        if idx.size == 0:
            continue
        take = min(qk, idx.size)
        sel = rng.choice(idx, size=take, replace=False)
        chosen.append(np.asarray(sel, dtype=int))

    out = np.sort(np.unique(np.concatenate(chosen))) if chosen else np.arange(min(N, n_target))
    if out.size > n_target:
        out = out[:n_target]
    elif out.size < n_target:
        need = n_target - out.size
        extras = np.setdiff1d(np.linspace(0, N - 1, out.size + need, dtype=int), out)[:need]
        out = np.sort(np.concatenate([out, extras.astype(int)]))
    return out.astype(int)

def sample_adaptive_balanced_indices(x, y, n_target, batches=None):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    N = len(x)
    n_target = max(2, min(n_target, N))
    rng = np.random.default_rng()

    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    dy = np.gradient(y_s, edge_order=2)
    d2 = np.gradient(dy, edge_order=2)
    curv = np.abs(d2)

    win = max(3, int(round(N / 20)))
    pad = win // 2
    y_pad = np.pad(y_s, (pad, pad), mode="edge")
    rolling = np.sqrt(pd.Series(y_pad).rolling(window=win, center=True).var().to_numpy())
    rolling = rolling[pad:-pad]

    def _robust_norm(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.zeros_like(curv)
        med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-12
        return (v - med) / (1.4826 * mad)

    z_curv = np.abs(_robust_norm(curv))
    z_vol = np.abs(_robust_norm(rolling))
    score = np.nan_to_num(z_curv + z_vol, nan=0.0)

    if batches is None:
        batches = max(2, int(round(np.sqrt(n_target))))
    strata = stratified_time_buckets(N, batches)
    score[0] += 1e6; score[-1] += 1e6
    quota = _hamilton_quotas(strata, batches, N, n_target)

    chosen_sorted_idx = []
    for b in range(batches):
        idx = np.where(strata == b)[0]
        if idx.size == 0 or quota[b] == 0:
            continue
        jitter = rng.standard_normal(idx.size) * 1e-12
        key = score[idx] + jitter
        sel_local = idx[np.argsort(key)[:quota[b]]]
        chosen_sorted_idx.append(sel_local)

    chosen_sorted_idx = np.sort(np.concatenate(chosen_sorted_idx)) if chosen_sorted_idx else np.arange(min(N, n_target))
    chosen_orig_idx = order[chosen_sorted_idx]

    if chosen_orig_idx.size > n_target:
        chosen_orig_idx = np.sort(chosen_orig_idx)[:n_target]
    elif chosen_orig_idx.size < n_target:
        need = n_target - chosen_orig_idx.size
        extra = np.setdiff1d(sample_indices_uniform(N, chosen_orig_idx.size + need), chosen_orig_idx)[:need]
        chosen_orig_idx = np.sort(np.concatenate([chosen_orig_idx, extra]))
    return chosen_orig_idx

# =========================
# Main Program
# =========================
def main():
    p = argparse.ArgumentParser(
        description="Aggregate Q-Error CDF across multiple CSVs, grouped by sampling strategy."
    )
    p.add_argument("--csv", nargs="+", required=True,
                   help="One or more ';'-separated CSVs with columns row_value;*_runtime_ms")
    p.add_argument("--engine", choices=["mysql", "duckdb", "postgres"], required=True,
                   help="Which runtime column to model")
    p.add_argument("--strategy", nargs="+", default=["uniform", "random", "adapt", "str"],
                   choices=["uniform", "random", "adapt", "str"],
                   help="Sampling strategies to plot (one curve per strategy, aggregated across files)")
    p.add_argument("--n", type=int, default=20, help="Number of sampled points per file")
    p.add_argument("--degree", type=int, default=2, help="Polynomial degree for regression")
    p.add_argument("--str_k", type=int, default=12, help="K buckets for 'str' strategy")
    p.add_argument("--tmax", type=float, default=2.0, help="Max Q-Error on the X axis")
    p.add_argument("--points", type=int, default=200, help="Number of thresholds along X axis")
    p.add_argument("--out", default=None, help="Output PNG filename")
    args = p.parse_args()

    engine_col = {
        "mysql": "mysql_runtime_ms",
        "duckdb": "duckdb_runtime_ms",
        "postgres": "postgres_runtime_ms",
    }[args.engine]

    # --- accumulate Q-Errors per STRATEGY across all files ---
    qerrs_by_strategy = {s: [] for s in args.strategy}
    per_file_logs = []

    for path in args.csv:
        if not os.path.exists(path):
            print(f"⚠️ File not found, skipping: {path}")
            continue

        df = pd.read_csv(path, sep=";")
        missing = {"row_value", engine_col} - set(df.columns)
        if missing:
            print(f"⚠️ {os.path.basename(path)} missing columns: {missing}; skipping.")
            continue

        df = df.sort_values("row_value").reset_index(drop=True)
        df["row_value"] = pd.to_numeric(df["row_value"], errors="coerce")
        for col in ["postgres_runtime_ms", "duckdb_runtime_ms", "mysql_runtime_ms"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["row_value", engine_col])
        df = df[df[engine_col] > 0]
        if len(df) < 2:
            print(f"⚠️ {os.path.basename(path)} has <2 positive rows after cleaning; skipping.")
            continue

        x_all = df["row_value"].to_numpy()
        y_all = df[engine_col].to_numpy()
        N = len(df)

        n_target = int(round(N * args.n / 100.0))
        n_target = max(2, min(n_target, N))

        for strat in args.strategy:
            if strat == "uniform":
                idx = sample_indices_uniform(N, n_target)
            elif strat == "random":
                idx = sample_indices_random(N, n_target)
            elif strat == "str":
                idx = sample_stratified_indices(N, n_target, K=args.str_k)
            elif strat == "adapt":
                idx = sample_adaptive_balanced_indices(x_all, y_all, n_target=n_target)
            else:
                continue

            y_pred = fit_predict_all(x_all, y_all, idx, degree=args.degree)
            qe = qerror(y_all, y_pred)
            finite = qe[np.isfinite(qe)]
            if finite.size == 0:
                continue

            # accumulate under the strategy
            qerrs_by_strategy[strat].extend(finite.tolist())

            per_file_logs.append({
                "file": os.path.basename(path),
                "strategy": strat,
                "n_sample": int(len(idx)),
                "mean_qe": float(np.mean(finite)),
                "p95": float(np.percentile(finite, 95)),
                "p99": float(np.percentile(finite, 99)),
                "max": float(np.max(finite)),
                "count_pairs": int(finite.size),
            })

    if per_file_logs:
        print(pd.DataFrame(per_file_logs).sort_values(["strategy", "file"]).to_string(index=False))
    else:
        print("No per-file metrics computed (check inputs/columns).")

    # ---- plot one curve PER STRATEGY (aggregated across files) ----
    t_grid = np.linspace(1.0, args.tmax, args.points)
    has_any = False
    plt.figure(figsize=(9, 6))
    for strat in args.strategy:
        vals = np.asarray(qerrs_by_strategy.get(strat, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"ℹ️ No Q-Errors aggregated for strategy: {strat}")
            continue
        curve = ccdf(vals, t_grid)
        stats = qerr_stats(vals)
        label = f"{strat} (p95={stats['p95']:.2f}, p99={stats['p99']:.2f})"
        plt.plot(t_grid, curve, label=label)
        has_any = True

    if has_any:
        plt.xlabel("Q-Error")
        plt.ylabel("Fraction of predictions ≥ Q-Error")
        plt.title(f"Cumulative Q-Error Distribution — aggregated across query templates ({args.engine})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        out = args.out or f"qerror_cdf_{args.engine}_by_strategy.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        print(f"✅ Saved plot to {out}")
    else:
        print("No Q-Error curves to plot.")

if __name__ == "__main__":
    main()
