"""
DataFlow360 — Battery Data Validation
======================================
Purpose : Runs quality-gate checks on the processed parquet before it is
          loaded into the database. Produces a validation_report.json that
          records every check result (pass / warning / error).

Validation Layers:
  1. Schema      — correct columns and data types
  2. Range       — physically possible values
  3. Completeness— no critical nulls, all batteries present, monotonic cycles
  4. Domain Logic— SOH trend, flag consistency, capacity fade direction

Run with:
    python -m src.validation.battery_validation
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

PROCESSED_DATA_DIR  = Path("data/processed")
PROCESSED_FILE      = PROCESSED_DATA_DIR / "battery_cycles_processed.parquet"
REPORT_FILE         = PROCESSED_DATA_DIR / "validation_report.json"

EXPECTED_BATTERIES  = ["B0005", "B0006", "B0007", "B0018"]
MIN_CYCLES_PER_BATT = 50
EOL_THRESHOLD       = 80.0
NEAR_EOL_THRESHOLD  = 85.0

REQUIRED_COLUMNS = [
    "battery_id", "raw_cycle_index", "discharge_cycle",
    "capacity_ah", "rated_capacity_ah", "soh_pct",
    "end_voltage_v", "avg_temperature_c", "ambient_temperature_c",
    "is_end_of_life", "capacity_fade_pct", "soh_capped_pct",
    "rolling_soh_7", "soh_delta", "is_near_eol",
    "life_fraction", "cycles_to_eol",
]

EXPECTED_DTYPES = {
    "battery_id":            "object",
    "discharge_cycle":       "int",
    "capacity_ah":           "float",
    "soh_pct":               "float",
    "end_voltage_v":         "float",
    "avg_temperature_c":     "float",
    "life_fraction":         "float",
    "is_end_of_life":        "bool",
    "is_near_eol":           "bool",
}


# ==============================================================================
# Result container
# ==============================================================================

@dataclass
class CheckResult:
    """
    Represents the outcome of a single validation check.

    Attributes:
        layer   : One of 'schema', 'range', 'completeness', 'domain'
        name    : Short identifier for the check
        status  : 'PASS', 'WARN', or 'ERROR'
        message : Human-readable explanation
    """
    layer:   str
    name:    str
    status:  str   # 'PASS' | 'WARN' | 'ERROR'
    message: str


# ==============================================================================
# Layer 1 — Schema
# ==============================================================================

def check_schema(df: pd.DataFrame) -> List[CheckResult]:
    """
    Verifies that all required columns are present with the expected dtypes.
    """
    results = []
    logger.info("  [Schema] Checking columns and types...")

    # 1a — required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        results.append(CheckResult(
            layer="schema", name="required_columns",
            status="ERROR",
            message=f"Missing columns: {missing}"
        ))
    else:
        results.append(CheckResult(
            layer="schema", name="required_columns",
            status="PASS",
            message=f"All {len(REQUIRED_COLUMNS)} required columns present"
        ))

    # 1b — data type spot-checks
    type_errors = []
    for col, expected_kind in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if expected_kind == "int" and not pd.api.types.is_integer_dtype(df[col]):
            type_errors.append(f"{col}: expected int, got {actual}")
        elif expected_kind == "float" and not pd.api.types.is_float_dtype(df[col]):
            type_errors.append(f"{col}: expected float, got {actual}")
        elif expected_kind == "bool" and not pd.api.types.is_bool_dtype(df[col]):
            type_errors.append(f"{col}: expected bool, got {actual}")
        elif expected_kind == "object" and not pd.api.types.is_object_dtype(df[col]):
            type_errors.append(f"{col}: expected object/str, got {actual}")

    if type_errors:
        results.append(CheckResult(
            layer="schema", name="column_dtypes",
            status="WARN",
            message=f"Type mismatches: {type_errors}"
        ))
    else:
        results.append(CheckResult(
            layer="schema", name="column_dtypes",
            status="PASS",
            message="All column dtypes match expectations"
        ))

    return results


# ==============================================================================
# Layer 2 — Range Checks
# ==============================================================================

def check_ranges(df: pd.DataFrame) -> List[CheckResult]:
    """
    Confirms all values fall within physically possible ranges.
    """
    results = []
    logger.info("  [Range] Checking value bounds...")

    checks = [
        # (column, min_val, max_val, severity)
        ("soh_pct",              0.0,  100.0,  "ERROR"),
        ("capacity_ah",          0.0,    None,  "ERROR"),
        ("end_voltage_v",        0.0,    None,  "ERROR"),
        ("avg_temperature_c",  -40.0,  100.0,  "WARN"),
        ("life_fraction",        0.0,    1.0,   "ERROR"),
        ("capacity_fade_pct",    0.0,   None,   "WARN"),
    ]

    for col, lo, hi, severity in checks:
        if col not in df.columns:
            continue
        violations = pd.Series(False, index=df.index)
        if lo is not None:
            violations |= df[col] < lo
        if hi is not None:
            violations |= df[col] > hi

        count = violations.sum()
        if count > 0:
            results.append(CheckResult(
                layer="range", name=f"{col}_bounds",
                status=severity,
                message=(
                    f"{count} rows violate {col} bounds "
                    f"[{lo}, {hi}]. "
                    f"Actual range: [{df[col].min():.4f}, {df[col].max():.4f}]"
                )
            ))
        else:
            results.append(CheckResult(
                layer="range", name=f"{col}_bounds",
                status="PASS",
                message=(
                    f"{col} in bounds [{df[col].min():.4f}, {df[col].max():.4f}]"
                )
            ))

    return results


# ==============================================================================
# Layer 3 — Completeness / Integrity
# ==============================================================================

def check_completeness(df: pd.DataFrame) -> List[CheckResult]:
    """
    Ensures no critical nulls, all expected batteries present,
    minimum cycle count per battery, and monotonic cycle ordering.
    """
    results = []
    logger.info("  [Completeness] Checking nulls, batteries, cycles...")

    # 3a — nulls in critical columns
    critical = ["battery_id", "discharge_cycle", "soh_pct", "capacity_ah"]
    for col in critical:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            results.append(CheckResult(
                layer="completeness", name=f"{col}_nulls",
                status="ERROR",
                message=f"{null_count} null values in critical column '{col}'"
            ))
        else:
            results.append(CheckResult(
                layer="completeness", name=f"{col}_nulls",
                status="PASS",
                message=f"No nulls in '{col}'"
            ))

    # 3b — all expected batteries present
    found = set(df["battery_id"].unique()) if "battery_id" in df.columns else set()
    missing_batts = set(EXPECTED_BATTERIES) - found
    if missing_batts:
        results.append(CheckResult(
            layer="completeness", name="battery_presence",
            status="ERROR",
            message=f"Missing batteries: {sorted(missing_batts)}"
        ))
    else:
        results.append(CheckResult(
            layer="completeness", name="battery_presence",
            status="PASS",
            message=f"All {len(EXPECTED_BATTERIES)} expected batteries present"
        ))

    # 3c — minimum cycle count per battery
    if "battery_id" in df.columns and "discharge_cycle" in df.columns:
        cycle_counts = df.groupby("battery_id")["discharge_cycle"].count()
        too_few = cycle_counts[cycle_counts < MIN_CYCLES_PER_BATT]
        if not too_few.empty:
            results.append(CheckResult(
                layer="completeness", name="min_cycle_count",
                status="WARN",
                message=f"Batteries with < {MIN_CYCLES_PER_BATT} cycles: {too_few.to_dict()}"
            ))
        else:
            results.append(CheckResult(
                layer="completeness", name="min_cycle_count",
                status="PASS",
                message=f"All batteries have >= {MIN_CYCLES_PER_BATT} cycles. "
                        f"Counts: {cycle_counts.to_dict()}"
            ))

    # 3d — cycle sequences are monotonically increasing
    if "battery_id" in df.columns and "discharge_cycle" in df.columns:
        non_mono = []
        for batt_id, group in df.groupby("battery_id"):
            cycles = group.sort_values("discharge_cycle")["discharge_cycle"]
            if not cycles.is_monotonic_increasing:
                non_mono.append(batt_id)
        if non_mono:
            results.append(CheckResult(
                layer="completeness", name="cycle_monotonic",
                status="WARN",
                message=f"Non-monotonic cycle sequences in: {non_mono}"
            ))
        else:
            results.append(CheckResult(
                layer="completeness", name="cycle_monotonic",
                status="PASS",
                message="All batteries have monotonically increasing cycle numbers"
            ))

    return results


# ==============================================================================
# Layer 4 — Domain Logic
# ==============================================================================

def check_domain_logic(df: pd.DataFrame) -> List[CheckResult]:
    """
    Validates battery-engineering rules:
      - SOH generally trends downward over cycles
      - is_end_of_life and is_near_eol flags are consistent with SOH values
      - Capacity fade is non-negative (batteries only lose capacity)
    """
    results = []
    logger.info("  [Domain] Checking engineering rules...")

    # 4a — SOH trend is generally decreasing (first half avg > second half avg)
    if "battery_id" in df.columns and "soh_pct" in df.columns:
        trend_violations = []
        for batt_id, group in df.groupby("battery_id"):
            g = group.sort_values("discharge_cycle")["soh_pct"]
            midpoint = len(g) // 2
            first_half_avg = g.iloc[:midpoint].mean()
            second_half_avg = g.iloc[midpoint:].mean()
            if second_half_avg >= first_half_avg:
                trend_violations.append(
                    f"{batt_id} (first_half={first_half_avg:.1f}%, second_half={second_half_avg:.1f}%)"
                )
        if trend_violations:
            results.append(CheckResult(
                layer="domain", name="soh_degradation_trend",
                status="WARN",
                message=f"SOH not decreasing as expected: {trend_violations}"
            ))
        else:
            results.append(CheckResult(
                layer="domain", name="soh_degradation_trend",
                status="PASS",
                message="SOH is decreasing over time for all batteries"
            ))

    # 4b — is_end_of_life flag must match soh_pct < 80%
    if "is_end_of_life" in df.columns and "soh_pct" in df.columns:
        flag_true_but_soh_ok  = ((df["is_end_of_life"] == True)  & (df["soh_pct"] >= EOL_THRESHOLD)).sum()
        flag_false_but_soh_eol = ((df["is_end_of_life"] == False) & (df["soh_pct"] <  EOL_THRESHOLD)).sum()
        inconsistent = flag_true_but_soh_ok + flag_false_but_soh_eol
        if inconsistent > 0:
            results.append(CheckResult(
                layer="domain", name="eol_flag_consistency",
                status="ERROR",
                message=(
                    f"{inconsistent} rows have inconsistent is_end_of_life flag. "
                    f"(flag=True but SOH>={EOL_THRESHOLD}%: {flag_true_but_soh_ok}, "
                    f"flag=False but SOH<{EOL_THRESHOLD}%: {flag_false_but_soh_eol})"
                )
            ))
        else:
            results.append(CheckResult(
                layer="domain", name="eol_flag_consistency",
                status="PASS",
                message="is_end_of_life flag is consistent with soh_pct for all rows"
            ))

    # 4c — is_near_eol flag must match soh_pct < 85%
    if "is_near_eol" in df.columns and "soh_pct" in df.columns:
        flag_true_but_ok   = ((df["is_near_eol"] == True)  & (df["soh_pct"] >= NEAR_EOL_THRESHOLD)).sum()
        flag_false_but_near = ((df["is_near_eol"] == False) & (df["soh_pct"] <  NEAR_EOL_THRESHOLD)).sum()
        inconsistent = flag_true_but_ok + flag_false_but_near
        if inconsistent > 0:
            results.append(CheckResult(
                layer="domain", name="near_eol_flag_consistency",
                status="ERROR",
                message=(
                    f"{inconsistent} rows have inconsistent is_near_eol flag. "
                    f"(flag=True but SOH>={NEAR_EOL_THRESHOLD}%: {flag_true_but_ok}, "
                    f"flag=False but SOH<{NEAR_EOL_THRESHOLD}%: {flag_false_but_near})"
                )
            ))
        else:
            results.append(CheckResult(
                layer="domain", name="near_eol_flag_consistency",
                status="PASS",
                message="is_near_eol flag is consistent with soh_pct for all rows"
            ))

    # 4d — capacity_fade_pct must be >= 0 (batteries only lose capacity)
    if "capacity_fade_pct" in df.columns:
        neg_fade = (df["capacity_fade_pct"] < -0.01).sum()   # allow tiny float rounding
        if neg_fade > 0:
            results.append(CheckResult(
                layer="domain", name="capacity_fade_direction",
                status="WARN",
                message=f"{neg_fade} rows show negative capacity fade (capacity gain) — check first-cycle reference"
            ))
        else:
            results.append(CheckResult(
                layer="domain", name="capacity_fade_direction",
                status="PASS",
                message="Capacity fade is non-negative for all rows"
            ))

    return results


# ==============================================================================
# Report Builder
# ==============================================================================

def build_report(all_results: List[CheckResult], df: pd.DataFrame) -> dict:
    """
    Assembles all check results into a structured JSON-serialisable report.
    """
    errors   = [r for r in all_results if r.status == "ERROR"]
    warnings = [r for r in all_results if r.status == "WARN"]
    passes   = [r for r in all_results if r.status == "PASS"]

    overall = "PASS" if not errors else "FAIL"

    return {
        "generated_at":    datetime.now().isoformat(),
        "input_file":      str(PROCESSED_FILE),
        "total_rows":      len(df),
        "total_checks":    len(all_results),
        "overall_status":  overall,
        "summary": {
            "errors":   len(errors),
            "warnings": len(warnings),
            "passes":   len(passes),
        },
        "checks": [asdict(r) for r in all_results],
    }


def save_report(report: dict) -> Path:
    """Writes the validation report as a formatted JSON file."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved: {REPORT_FILE}")
    return REPORT_FILE


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def run_validation() -> bool:
    """
    Orchestrates all 4 validation layers against the processed parquet.

    Returns:
        True  — all checks passed (or only warnings); safe to proceed
        False — at least one ERROR check failed; pipeline should stop
    """
    logger.info("=" * 60)
    logger.info("DataFlow360 — Validation Pipeline Started")
    logger.info("=" * 60)

    if not PROCESSED_FILE.exists():
        logger.error(
            f"Processed file not found: {PROCESSED_FILE}\n"
            "Run transformation first: python -m src.transformation.battery_transformation"
        )
        return False

    df = pd.read_parquet(PROCESSED_FILE)
    logger.info(f"Loaded processed data: {len(df)} rows, {len(df.columns)} columns")

    all_results: List[CheckResult] = []
    all_results += check_schema(df)
    all_results += check_ranges(df)
    all_results += check_completeness(df)
    all_results += check_domain_logic(df)

    report = build_report(all_results, df)
    save_report(report)

    # --- Print summary ---
    errors   = report["summary"]["errors"]
    warnings = report["summary"]["warnings"]
    passes   = report["summary"]["passes"]

    logger.info("=" * 60)
    logger.info("Validation Complete")
    logger.info(f"  Total checks : {report['total_checks']}")
    logger.info(f"  ✅ PASS      : {passes}")
    logger.info(f"  ⚠️  WARN      : {warnings}")
    logger.info(f"  ❌ ERROR     : {errors}")
    logger.info(f"  Overall      : {report['overall_status']}")
    logger.info("=" * 60)

    if errors > 0:
        logger.error("Validation FAILED — fix errors before proceeding to database load.")
        for r in all_results:
            if r.status == "ERROR":
                logger.error(f"  [{r.layer}] {r.name}: {r.message}")
        return False

    if warnings > 0:
        for r in all_results:
            if r.status == "WARN":
                logger.warning(f"  [{r.layer}] {r.name}: {r.message}")

    logger.info("Data cleared for database loading.")
    return True


if __name__ == "__main__":
    run_validation()
