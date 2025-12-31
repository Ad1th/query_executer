# TODO: IMPORTANT These are mostly AI generated so they might be parsing wrong
#            Make sure that these parsers are correct

import json
import re
from typing import Dict, Any, List


def parse_analyze_mysql(plan_text: str, filter_vars: list):
    # Updated regex to also match filters with (never executed)
    pattern = re.compile(
        r"Filter: \((.*?)\)\s*\((?:(?:actual time=(.*?) rows=([^\s]+) loops=([^\)]+))|never executed)\)",
        re.DOTALL
    )

    results = []
    for match in pattern.findall(plan_text):
        condition = match[0]
        time_range = match[1]
        rows = match[2]
        loops = match[3]

        condition_lower = condition.lower()
        for var in filter_vars:
            if var.lower() in condition_lower:
                if time_range:  # Means it was executed
                    total_rows = int(float(rows) * float(loops))
                else:
                    total_rows = 0

                results.append({
                    "variable": var,
                    "total_rows": total_rows,
                })
                break

    return results


def extract_total_runtime(plan_text: str) -> float:
    """
    Extracts the total runtime (in milliseconds) from the top-level node
    in MySQL EXPLAIN ANALYZE output by taking the right-hand value
    """
    match = re.search(r"actual time=\d+\.?\d*\.\.(\d+\.?\d*)", plan_text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_runtime_and_filter_scans_duckdb(profile_json: str, filters: List[str]) -> Dict[str, Any]:
    """
    profile_json: JSON string of the DuckDB profile (what EXPLAIN ANALYZE JSON emits)
    filters: list of substrings to look for inside extra_info["Filters"], e.g. ["o_orderdate"]

    Returns:
        {
          "total_runtime": <float>,   # from profile["latency"] if present, else 0.0
          "filters": [
            {"variable": "<filter>", "total_rows": <int>}, ...
          ]
        }
    """
    # Parse JSON string → Python dict (jsonb-like structure)
    try:
        profile: Dict[str, Any] = json.loads(profile_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid profile JSON: {e}") from e

    # total runtime
    total_runtime = float(profile.get("latency", 0.0))

    # prepare accumulator
    scans: Dict[str, int] = {f: 0 for f in filters}
    norm_filters = [f.lower() for f in filters]

    def walk(node: Dict[str, Any]):
        # check Filters in extra_info
        extra = node.get("extra_info") or {}
        filt = extra.get("Filters")
        if isinstance(filt, str):
            filt_l = filt.lower()
            for f_in, f_raw in zip(norm_filters, filters):
                if f_in in filt_l:
                    scans[f_raw] += int(node.get("operator_rows_scanned", 0))

        # recurse
        for child in (node.get("children") or []):
            walk(child)

    walk(profile)

    # dict → list of objects
    filters_list = [{"variable": k, "total_rows": v} for k, v in scans.items()]

    return {"total_runtime": total_runtime, "filters": filters_list}


def extract_runtime_and_filter_scans_postgres(plan_json: dict, filters: List[str]) -> Dict[str, Any]:
    """
    plan_json: the JSON output of EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) from PostgreSQL
    filters: list of substrings to search for inside Filter expressions
             e.g., ["o_orderdate", "r_name"]

    Returns:
        {
          "total_runtime": <float>,   # in seconds
          "filters": [
            {"variable": "<filter>", "total_rows": <int>, "rows_removed": <int>}, ...
          ]
        }
    """
    # Handle case where plan_json is a string (legacy text format)
    if isinstance(plan_json, str):
        return _extract_runtime_and_filter_scans_postgres_text(plan_json, filters)

    # --- total runtime (seconds) ---
    total_runtime = 0.0
    if "Execution Time" in plan_json:
        total_runtime = float(plan_json["Execution Time"]) / 1000.0

    # --- prepare accumulators ---
    scans: Dict[str, Dict[str, int]] = {f: {"total_rows": 0, "rows_removed": 0} for f in filters}
    norm_filters = [f.lower() for f in filters]

    def walk_node(node: Dict[str, Any]):
        """Recursively walk the plan tree and extract filter information."""
        if not isinstance(node, dict):
            return

        # Check for Filter in this node
        filter_expr = node.get("Filter", "")
        if filter_expr:
            filter_expr_lower = filter_expr.lower()
            
            # Get actual rows and loops for this node
            actual_rows = node.get("Actual Rows", 0)
            actual_loops = node.get("Actual Loops", 1)
            rows_removed = node.get("Rows Removed by Filter", 0)
            
            # Calculate total rows (output + removed) * loops
            rows_output = actual_rows * max(1, actual_loops)
            total_rows_removed = rows_removed * max(1, actual_loops)
            total_rows_scanned = rows_output + total_rows_removed

            for f_norm, f_raw in zip(norm_filters, filters):
                if f_norm in filter_expr_lower:
                    scans[f_raw]["total_rows"] += total_rows_scanned
                    scans[f_raw]["rows_removed"] += total_rows_removed

        # Recurse into child plans
        for child in node.get("Plans", []):
            walk_node(child)

    # Get the root plan node
    root = plan_json.get("Plan", plan_json)
    walk_node(root)

    filters_list = [
        {"variable": k, "total_rows": v["total_rows"], "rows_removed": v["rows_removed"]}
        for k, v in scans.items()
    ]
    return {"total_runtime": total_runtime, "filters": filters_list}


def _extract_runtime_and_filter_scans_postgres_text(explain_text: str, filters: List[str]) -> Dict[str, Any]:
    """
    Legacy function for parsing text format EXPLAIN ANALYZE output.
    explain_text: the raw text output of EXPLAIN ANALYZE (BUFFERS) from PostgreSQL
    filters: list of substrings to search for inside lines like: 'Filter: (<expr>)'

    Returns:
        {
          "total_runtime": <float>,   # in seconds
          "filters": [
            {"variable": "<filter>", "total_rows": <int>, "rows_removed": <int>}, ...
          ]
        }
    """

    # --- total runtime (seconds) ---
    total_runtime = 0.0
    m_rt = re.search(r"Execution Time:\s*([\d.]+)\s*ms", explain_text)
    if m_rt:
        total_runtime = float(m_rt.group(1)) / 1000.0

    # --- prepare accumulators ---
    scans: Dict[str, Dict[str, int]] = {f: {"total_rows": 0, "rows_removed": 0} for f in filters}
    norm_filters = [f.lower() for f in filters]

    # Regexes for parsing key lines
    re_node_header = re.compile(r"\(actual time=\s*[\d.]+\s*\.\.\s*[\d.]+\s*rows=(\d+)\s+loops=(\d+)\)")
    re_filter_line = re.compile(r"\bFilter:\s*(.+)")
    re_rows_removed = re.compile(r"\bRows Removed by Filter:\s*(\d+)")

    # State while streaming lines
    current_rows = 0
    current_loops = 1
    last_matched_filters: List[str] = []
    last_rows_output = 0

    lines = explain_text.splitlines()

    def flush_pending_without_removed():
        nonlocal last_matched_filters, last_rows_output
        if last_matched_filters:
            for f in last_matched_filters:
                scans[f]["total_rows"] += last_rows_output
            last_matched_filters = []
            last_rows_output = 0

    for line in lines:
        m_head = re_node_header.search(line)
        if m_head:
            flush_pending_without_removed()
            current_rows = int(m_head.group(1))
            current_loops = int(m_head.group(2))
            continue

        m_filt = re_filter_line.search(line)
        if m_filt:
            flush_pending_without_removed()
            filt_expr_l = m_filt.group(1).lower()

            matched: List[str] = []
            for f_in, f_raw in zip(norm_filters, filters):
                if f_in in filt_expr_l:
                    matched.append(f_raw)

            if matched:
                last_matched_filters = matched
                last_rows_output = current_rows * max(1, current_loops)
            continue

        m_rr = re_rows_removed.search(line)
        if m_rr and last_matched_filters:
            removed = int(m_rr.group(1)) * max(1, current_loops)
            for f in last_matched_filters:
                scans[f]["total_rows"] += last_rows_output + removed
                scans[f]["rows_removed"] += removed
            last_matched_filters = []
            last_rows_output = 0
            continue

    if last_matched_filters:
        for f in last_matched_filters:
            scans[f]["total_rows"] += last_rows_output

    filters_list = [
        {"variable": k, "total_rows": v["total_rows"], "rows_removed": v["rows_removed"]}
        for k, v in scans.items()
    ]
    return {"total_runtime": total_runtime, "filters": filters_list}
