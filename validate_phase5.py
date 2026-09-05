"""Phase 5 — Full Pipeline Validation.

Runs the entire pipeline end-to-end and validates:
  1. Deterministic reproducibility (Tier 1+2 identical across runs)
  2. Full-pipeline accuracy against answer_key.csv
  3. Per-record correctness with detailed reporting
  4. Edge case coverage (all 8 mismatch types addressed)
  5. Data integrity checks (no duplicate results, no missing records)
  6. Value reconciliation correctness
  7. Timing/performance

Usage:
    conda activate razorpay
    python validate_phase5.py
"""

import logging
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import pandas as pd

from engine.pipeline import load_data, run_pipeline, validate_against_answer_key

DATA_DIR = Path("data")


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str):
    print(f"\n  --- {title} ---")


def check(label: str, passed: bool, detail: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    msg = f"  {status} {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return passed


# =====================================================================
# Load data
# =====================================================================
section("PHASE 5: FULL PIPELINE VALIDATION")
print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

settlements, orders = load_data()
answer_key = pd.read_csv(DATA_DIR / "answer_key.csv")

check("Data loaded", True,
      f"{len(settlements)} settlements, {len(orders)} orders, "
      f"{len(answer_key)} answer key records")

all_passed = True


# =====================================================================
# TEST 1: Deterministic Reproducibility (Tier 1+2 only)
# =====================================================================
section("TEST 1: DETERMINISTIC REPRODUCIBILITY (Tier 1+2)")
print("  Running Tier 1+2 three times to confirm identical output...\n")

det_runs = []
for i in range(3):
    t0 = time.time()
    results, rem_stl, rem_ord = run_pipeline(settlements, orders, use_ai=False)
    elapsed = time.time() - t0
    det_runs.append(results)
    print(f"    Run {i+1}: {len(results)} results in {elapsed:.2f}s")

# Compare all runs
for i in range(1, len(det_runs)):
    # Compare row counts
    same_len = len(det_runs[i]) == len(det_runs[0])
    # Compare content (sort by settlement_id for deterministic comparison)
    cols = ["order_id", "settlement_id", "tier", "category", "confidence"]
    df0 = det_runs[0][cols].sort_values("settlement_id").reset_index(drop=True)
    dfi = det_runs[i][cols].sort_values("settlement_id").reset_index(drop=True)
    same_content = df0.equals(dfi)

    p = check(f"Run {i+1} identical to Run 1",
              same_len and same_content,
              f"rows={len(det_runs[i])}")
    if not p:
        all_passed = False

# Verify tier counts
det = det_runs[0]
t1_count = len(det[det["tier"] == 1])
t2_count = len(det[det["tier"] == 2])
p = check(f"Tier 1 count = 40", t1_count == 40, f"got {t1_count}")
all_passed = all_passed and p
p = check(f"Tier 2 count = 25", t2_count == 25, f"got {t2_count}")
all_passed = all_passed and p
p = check(f"Deterministic total = 65", t1_count + t2_count == 65,
          f"got {t1_count + t2_count}")
all_passed = all_passed and p


# =====================================================================
# TEST 2: Tier 1+2 Precision (zero false positives)
# =====================================================================
section("TEST 2: DETERMINISTIC PRECISION (zero false positives)")

t1_results = det[det["tier"] == 1]
t2_results = det[det["tier"] == 2]

t1_fps = 0
for _, row in t1_results.iterrows():
    key_match = answer_key[answer_key["settlement_id"] == row["settlement_id"]]
    if key_match.empty:
        t1_fps += 1
        print(f"    [FP] T1: {row['settlement_id']} not in answer key!")
    elif key_match.iloc[0]["mismatch_type"] != "perfect_match":
        t1_fps += 1
        print(f"    [FP] T1: {row['settlement_id']} expected "
              f"{key_match.iloc[0]['mismatch_type']}, got exact")

p = check("Tier 1 false positives = 0", t1_fps == 0, f"got {t1_fps}")
all_passed = all_passed and p

t2_fps = 0
t2_valid_types = {"fee_adjusted", "date_drift"}
for _, row in t2_results.iterrows():
    key_match = answer_key[answer_key["settlement_id"] == row["settlement_id"]]
    if key_match.empty:
        t2_fps += 1
        print(f"    [FP] T2: {row['settlement_id']} not in answer key!")
    elif key_match.iloc[0]["mismatch_type"] not in t2_valid_types:
        t2_fps += 1
        print(f"    [FP] T2: {row['settlement_id']} expected "
              f"{key_match.iloc[0]['mismatch_type']}, got {row['category']}")

p = check("Tier 2 false positives = 0", t2_fps == 0, f"got {t2_fps}")
all_passed = all_passed and p


# =====================================================================
# TEST 3: Full Pipeline (with AI) — single run
# =====================================================================
section("TEST 3: FULL PIPELINE RUN (with AI)")
print("  Running full pipeline with all 4 tiers...\n")

t0 = time.time()
full_results, full_rem_stl, full_rem_ord = run_pipeline(
    settlements, orders, use_ai=True)
full_elapsed = time.time() - t0

print(f"\n  Full pipeline completed in {full_elapsed:.1f}s")
print(f"  Total results: {len(full_results)}")
print(f"  Remaining: {len(full_rem_stl)} settlements, {len(full_rem_ord)} orders")

p = check("All records processed (0 remaining)",
          len(full_rem_stl) == 0 and len(full_rem_ord) == 0,
          f"rem_stl={len(full_rem_stl)}, rem_ord={len(full_rem_ord)}")
all_passed = all_passed and p


# =====================================================================
# TEST 4: Answer Key Validation (per-record)
# =====================================================================
section("TEST 4: ANSWER KEY VALIDATION (per-record)")

metrics = validate_against_answer_key(full_results, full_rem_stl, full_rem_ord)

subsection("Per-Tier Results")
for tier_num in [1, 2, 3, 4]:
    s = metrics["tier_stats"][tier_num]
    label = {1: "Exact Match", 2: "Rule-Based",
             3: "AI Fuzzy", 4: "AI Categorize"}[tier_num]
    print(f"  Tier {tier_num} ({label}): {s['matched']} matched, "
          f"{s['correct']} correct, {s['wrong']} wrong, "
          f"{s['not_in_key']} supplementary")

subsection("Overall Metrics")
print(f"  Accuracy:    {metrics['total_correct']}/{metrics['total_scorable']} "
      f"({metrics['overall_accuracy_pct']}%)")
print(f"  Coverage:    {metrics['coverage_pct']}% of answer key")
print(f"  Wrong:       {metrics['total_wrong']}")

p = check("Zero wrong classifications", metrics["total_wrong"] == 0,
          f"got {metrics['total_wrong']}")
all_passed = all_passed and p

p = check("Accuracy >= 85%", metrics["overall_accuracy_pct"] >= 85.0,
          f"got {metrics['overall_accuracy_pct']}%")
all_passed = all_passed and p

p = check("Coverage >= 80%", metrics["coverage_pct"] >= 80.0,
          f"got {metrics['coverage_pct']}%")
all_passed = all_passed and p


# =====================================================================
# TEST 5: Mismatch Type Coverage (all 8 types addressed)
# =====================================================================
section("TEST 5: MISMATCH TYPE COVERAGE")

expected_types = {
    "perfect_match", "fee_adjusted", "date_drift", "partial_refund",
    "duplicate_order", "orphan_settlement", "orphan_order", "split_settlement"
}

# Check which mismatch types from the answer key are covered by results
covered_types = set()
for _, ak_row in answer_key.iterrows():
    stl_id = ak_row["settlement_id"]
    order_id = ak_row["order_id"]
    mtype = ak_row["mismatch_type"]

    found = False

    # For orphan_order: settlement_id is NaN, match by order_id
    if pd.isna(stl_id):
        match_in_results = full_results[full_results["order_id"] == order_id]
        found = not match_in_results.empty
    # For split_settlement: settlement_id is comma-separated, match any
    elif isinstance(stl_id, str) and "," in stl_id:
        for sid in stl_id.split(","):
            match_in_results = full_results[full_results["settlement_id"] == sid.strip()]
            if not match_in_results.empty:
                found = True
                break
    else:
        match_in_results = full_results[full_results["settlement_id"] == stl_id]
        found = not match_in_results.empty

    if found:
        covered_types.add(mtype)

for mtype in sorted(expected_types):
    count = len(answer_key[answer_key["mismatch_type"] == mtype])
    is_covered = mtype in covered_types
    p = check(f"{mtype} ({count} records)", is_covered,
              "covered" if is_covered else "NOT COVERED")
    all_passed = all_passed and p


# =====================================================================
# TEST 6: Data Integrity Checks
# =====================================================================
section("TEST 6: DATA INTEGRITY")

# No duplicate settlement IDs in results (except NaN for orphan orders)
non_null_stl = full_results[full_results["settlement_id"].notna()]["settlement_id"]
dup_stl = non_null_stl[non_null_stl.duplicated()]
p = check("No duplicate settlement_ids in results",
          len(dup_stl) == 0,
          f"{len(dup_stl)} duplicates" if len(dup_stl) > 0 else "clean")
all_passed = all_passed and p

# Every result has an explanation
has_explanation = full_results["explanation"].notna() & (full_results["explanation"] != "")
p = check("Every record has an explanation",
          has_explanation.all(),
          f"{has_explanation.sum()}/{len(full_results)} have explanations")
all_passed = all_passed and p

# Every result has a valid tier (1-4)
valid_tiers = full_results["tier"].isin([1, 2, 3, 4])
p = check("Every record has valid tier (1-4)",
          valid_tiers.all(),
          f"{valid_tiers.sum()}/{len(full_results)} valid")
all_passed = all_passed and p

# Confidence in valid range [0, 1]
conf_valid = (full_results["confidence"] >= 0) & (full_results["confidence"] <= 1)
p = check("All confidence scores in [0, 1]",
          conf_valid.all(),
          f"{conf_valid.sum()}/{len(full_results)} valid")
all_passed = all_passed and p

# Tier 1 confidence always 1.0
t1_conf = full_results[full_results["tier"] == 1]["confidence"]
p = check("Tier 1 confidence always 1.0",
          (t1_conf == 1.0).all(),
          f"min={t1_conf.min()}, max={t1_conf.max()}")
all_passed = all_passed and p

# Tier 2 confidence always 0.98
t2_conf = full_results[full_results["tier"] == 2]["confidence"]
p = check("Tier 2 confidence always 0.98",
          (t2_conf == 0.98).all(),
          f"min={t2_conf.min()}, max={t2_conf.max()}")
all_passed = all_passed and p

# No Tier 4 with match_status = "matched"
t4_matched = full_results[(full_results["tier"] == 4) & 
                           (full_results["match_status"] == "matched")]
p = check("Tier 4 never claims 'matched'",
          len(t4_matched) == 0,
          f"{len(t4_matched)} incorrectly marked matched")
all_passed = all_passed and p

# Every T4 record has a suggested_action
t4_results = full_results[full_results["tier"] == 4]
t4_actions = t4_results["suggested_action"].notna() & (t4_results["suggested_action"] != "")
p = check("Every Tier 4 has a suggested_action",
          t4_actions.all(),
          f"{t4_actions.sum()}/{len(t4_results)} have actions")
all_passed = all_passed and p


# =====================================================================
# TEST 7: Value Reconciliation
# =====================================================================
section("TEST 7: VALUE RECONCILIATION")

stl_data = pd.read_csv(DATA_DIR / "settlement_report.csv")
ord_data = pd.read_csv(DATA_DIR / "order_ledger.csv")

total_settlement_value = stl_data["amount_settled"].sum()
total_order_value = ord_data["order_amount"].sum()

# Calculate matched value (Tier 1-3)
matched_results = full_results[full_results["tier"].isin([1, 2, 3])]
matched_stl_ids = matched_results["settlement_id"].dropna().unique()
matched_value = stl_data[stl_data["settlement_id"].isin(matched_stl_ids)]["amount_settled"].sum()

match_value_pct = (matched_value / total_settlement_value * 100) if total_settlement_value > 0 else 0

print(f"  Total settlement value:  Rs {total_settlement_value:,.2f}")
print(f"  Total order value:       Rs {total_order_value:,.2f}")
print(f"  Matched value:           Rs {matched_value:,.2f} ({match_value_pct:.1f}%)")
print(f"  Unreconciled value:      Rs {total_settlement_value - matched_value:,.2f}")

p = check("Value reconciliation > 70%", match_value_pct > 70,
          f"{match_value_pct:.1f}%")
all_passed = all_passed and p


# =====================================================================
# TEST 8: Category Distribution Sanity
# =====================================================================
section("TEST 8: CATEGORY DISTRIBUTION")

cat_counts = full_results["category"].value_counts()
print("  Category distribution in results:")
for cat, count in cat_counts.items():
    print(f"    {cat}: {count}")

# Ensure we have at least 5 distinct categories
p = check("At least 5 distinct categories",
          len(cat_counts) >= 5,
          f"got {len(cat_counts)}")
all_passed = all_passed and p


# =====================================================================
# TEST 9: Tier Distribution Matches Expected
# =====================================================================
section("TEST 9: TIER DISTRIBUTION")

tier_counts = full_results["tier"].value_counts().sort_index()
print("  Tier distribution in results:")
for tier, count in tier_counts.items():
    pct = count / len(full_results) * 100
    print(f"    Tier {tier}: {count} ({pct:.1f}%)")

# Answer key expects: T1=40, T2=25, T3=21, T4=14
# Results include supplementary records (orphan orders), so total > 100
p = check("Tier 1 count matches expected (40)",
          tier_counts.get(1, 0) == 40,
          f"got {tier_counts.get(1, 0)}")
all_passed = all_passed and p

p = check("Tier 2 count matches expected (25)",
          tier_counts.get(2, 0) == 25,
          f"got {tier_counts.get(2, 0)}")
all_passed = all_passed and p


# =====================================================================
# FINAL VERDICT
# =====================================================================
section("FINAL VERDICT")

total_checks_text = "all checks above"
if all_passed:
    print(f"\n  *** ALL CHECKS PASSED ***")
    print(f"  Phase 5 validation: COMPLETE")
else:
    print(f"\n  *** SOME CHECKS FAILED ***")
    print(f"  Review failures above before proceeding to Phase 6+")

print(f"\n  Key numbers for PITCH_NOTES.md and README:")
print(f"    Records processed:       {len(answer_key)}")
print(f"    Total results:           {len(full_results)}")
print(f"    Deterministic (T1+2):    {metrics['deterministic_resolved']} "
      f"({metrics['deterministic_rate_pct']}%)")
print(f"    AI-resolved (T3):        {metrics['tier_stats'][3]['matched']}")
print(f"    AI-categorized (T4):     {metrics['tier_stats'][4]['matched']}")
print(f"    Accuracy (scorable):     {metrics['total_correct']}/{metrics['total_scorable']} "
      f"({metrics['overall_accuracy_pct']}%)")
print(f"    Coverage:                {metrics['coverage_pct']}%")
print(f"    False positives:         {metrics['total_wrong']}")
print(f"    Pipeline time:           {full_elapsed:.1f}s")

# Save full results
full_results.to_csv("data/pipeline_results.csv", index=False)
print(f"\n  Results saved to data/pipeline_results.csv ({len(full_results)} rows)")

print(f"\n  Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
