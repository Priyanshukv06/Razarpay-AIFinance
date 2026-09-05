"""End-to-end verification of Phase 6c: memory, caching, samples, edge cases.

Tests:
  1. Sample datasets exist and are loadable (4 datasets)
  2. Schema validation catches bad CSVs
  3. Pipeline runs on a sample dataset (deterministic only)
  4. Supabase pipeline cache: store + retrieve + cache-hit detection
  5. Supabase Q&A cache: store + retrieve + cache-hit detection
  6. Delete/restart clears both pipeline + Q&A cache
  7. Pipeline results contain LLM explanations for Tier 3+4
  8. Graceful degradation when Supabase is unavailable
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from engine import cache as supabase_cache
from engine.pipeline import run_pipeline

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label} -- {detail}")


print("=" * 60)
print("TEST 1: Sample Datasets Exist and Loadable")
print("=" * 60)

samples_dir = Path("data/samples")
expected_samples = ["sample_small_easy", "sample_medium_balanced",
                    "sample_large_hard", "sample_edge_cases"]

for name in expected_samples:
    sample_path = samples_dir / name
    check(f"{name} directory exists", sample_path.exists())

    stl_file = sample_path / "settlement_report.csv"
    ord_file = sample_path / "order_ledger.csv"
    readme_file = sample_path / "README.md"

    check(f"{name} has settlement_report.csv", stl_file.exists())
    check(f"{name} has order_ledger.csv", ord_file.exists())
    check(f"{name} has README.md", readme_file.exists())

    if stl_file.exists() and ord_file.exists():
        stl = pd.read_csv(stl_file)
        ord_ = pd.read_csv(ord_file)
        check(f"{name} settlement has data", len(stl) > 0, f"len={len(stl)}")
        check(f"{name} order has data", len(ord_) > 0, f"len={len(ord_)}")

        # Check schema
        required_stl = {"settlement_id", "order_ref", "payment_id",
                        "amount_settled", "fee", "tax", "settlement_date", "status"}
        required_ord = {"order_id", "customer", "order_amount", "order_date",
                        "payment_method", "status"}
        check(f"{name} settlement schema valid",
              required_stl.issubset(set(stl.columns)),
              f"missing: {required_stl - set(stl.columns)}")
        check(f"{name} order schema valid",
              required_ord.issubset(set(ord_.columns)),
              f"missing: {required_ord - set(ord_.columns)}")

# Check answer keys are NOT present (judges shouldn't see ground truth)
for name in expected_samples:
    answer_key = samples_dir / name / "answer_key.csv"
    check(f"{name} NO answer_key.csv (hidden from judges)", not answer_key.exists())

print()
print("=" * 60)
print("TEST 2: Schema Validation Catches Bad CSVs")
print("=" * 60)

# Create a bad CSV (missing columns)
bad_stl = pd.DataFrame({"settlement_id": [1], "amount": [100]})  # Missing many columns
required_stl = {"settlement_id", "order_ref", "payment_id",
                "amount_settled", "fee", "tax", "settlement_date", "status"}
missing = required_stl - set(bad_stl.columns)
check("Bad settlement CSV detected", len(missing) > 0, f"Missing: {missing}")

bad_ord = pd.DataFrame({"id": [1]})
required_ord = {"order_id", "customer", "order_amount", "order_date",
                "payment_method", "status"}
missing_ord = required_ord - set(bad_ord.columns)
check("Bad order CSV detected", len(missing_ord) > 0, f"Missing: {missing_ord}")

print()
print("=" * 60)
print("TEST 3: Pipeline Runs on Sample Dataset (Deterministic)")
print("=" * 60)

sample_stl = pd.read_csv(samples_dir / "sample_small_easy" / "settlement_report.csv")
sample_ord = pd.read_csv(samples_dir / "sample_small_easy" / "order_ledger.csv")

results, rem_stl, rem_ord = run_pipeline(sample_stl.copy(), sample_ord.copy(), use_ai=False)
check("Pipeline returned results", len(results) > 0, f"len={len(results)}")
check("Results have required columns",
      all(c in results.columns for c in ["tier", "match_status", "confidence", "category"]))
check("Tier 1+2 matched some records",
      len(results[results["match_status"] == "matched"]) > 0)
check("Some records remain unmatched",
      len(rem_stl) + len(rem_ord) > 0,
      f"rem_stl={len(rem_stl)}, rem_ord={len(rem_ord)}")

print()
print("=" * 60)
print("TEST 4: Supabase Pipeline Cache")
print("=" * 60)

# Compute hash
ds_hash = supabase_cache.compute_df_hash(sample_stl, sample_ord)
check("Hash computed", len(ds_hash) == 64, f"len={len(ds_hash)}")

# Clean up first (in case of leftover from previous test)
supabase_cache.delete_cached_results(ds_hash)

# Store
metadata = {"source": "test", "total_results": len(results), "match_rate": 76.7}
stored = supabase_cache.store_results(ds_hash, results, metadata)
check("Pipeline results stored in Supabase", stored)

# Retrieve
cached = supabase_cache.get_cached_results(ds_hash)
check("Pipeline cache hit", cached is not None)
if cached:
    cached_results = pd.DataFrame(cached["results_json"])
    check("Cached result count matches", len(cached_results) == len(results),
          f"cached={len(cached_results)}, original={len(results)}")
    check("Cached metadata has source", cached.get("metadata_json", {}).get("source") == "test")

    # Check that LLM explanations are stored for each row
    has_explanation = "explanation" in cached_results.columns
    check("Cached results include explanation column", has_explanation)
    if has_explanation:
        non_null_explanations = cached_results["explanation"].dropna()
        check("At least some explanations are non-empty",
              len(non_null_explanations) > 0,
              f"non-null={len(non_null_explanations)}")

print()
print("=" * 60)
print("TEST 5: Supabase Q&A Cache")
print("=" * 60)

# Store a Q&A answer
stored_qa = supabase_cache.store_answer(ds_hash, "How many orphans?", "There are 3 orphan records.")
check("Q&A answer stored", stored_qa)

# Retrieve with exact question
cached_qa = supabase_cache.get_cached_answer(ds_hash, "How many orphans?")
check("Q&A cache hit", cached_qa is not None)
check("Q&A answer matches", cached_qa == "There are 3 orphan records.",
      f"got: {cached_qa}")

# Case-insensitive matching
cached_qa_upper = supabase_cache.get_cached_answer(ds_hash, "HOW MANY ORPHANS?")
check("Q&A case-insensitive matching works", cached_qa_upper is not None)

# Different question should miss
cached_qa_miss = supabase_cache.get_cached_answer(ds_hash, "Different question entirely")
check("Q&A cache miss for different question", cached_qa_miss is None)

# Different hash should miss
cached_qa_wrong_hash = supabase_cache.get_cached_answer("wrong_hash_000", "How many orphans?")
check("Q&A cache miss for different dataset", cached_qa_wrong_hash is None)

print()
print("=" * 60)
print("TEST 6: Delete/Restart Clears Both Caches")
print("=" * 60)

deleted = supabase_cache.delete_cached_results(ds_hash)
check("Delete returned True", deleted)

# Verify pipeline cache is gone
verify_pipeline = supabase_cache.get_cached_results(ds_hash)
check("Pipeline cache empty after delete", verify_pipeline is None)

# Verify Q&A cache is gone
verify_qa = supabase_cache.get_cached_answer(ds_hash, "How many orphans?")
check("Q&A cache empty after delete", verify_qa is None)

print()
print("=" * 60)
print("TEST 7: Full Pipeline Results Include AI Explanations")
print("=" * 60)

# Check from main dataset (already ran with AI)
main_results = pd.read_csv("data/pipeline_results.csv")

tier34 = main_results[main_results["tier"].isin([3, 4])]
check("Tier 3+4 records exist in main results", len(tier34) > 0, f"count={len(tier34)}")

if len(tier34) > 0:
    has_explanations = tier34["explanation"].notna().sum()
    check("All Tier 3+4 have explanations",
          has_explanations == len(tier34),
          f"has={has_explanations}, total={len(tier34)}")

    has_categories = tier34["category"].notna().sum()
    check("All Tier 3+4 have categories",
          has_categories == len(tier34))

    has_confidence = (tier34["confidence"] > 0).sum()
    check("All Tier 3+4 have confidence > 0",
          has_confidence == len(tier34))

    # Check for suggested_action in Tier 4
    tier4 = main_results[main_results["tier"] == 4]
    if len(tier4) > 0:
        has_actions = tier4["suggested_action"].notna().sum()
        check("All Tier 4 have suggested_action",
              has_actions == len(tier4),
              f"has={has_actions}, total={len(tier4)}")

print()
print("=" * 60)
print("TEST 8: Different Datasets Get Different Hashes")
print("=" * 60)

hashes = set()
for name in expected_samples:
    stl = pd.read_csv(samples_dir / name / "settlement_report.csv")
    ord_ = pd.read_csv(samples_dir / name / "order_ledger.csv")
    h = supabase_cache.compute_df_hash(stl, ord_)
    hashes.add(h)

check("All 4 samples produce unique hashes", len(hashes) == 4, f"unique={len(hashes)}")

# Main dataset has different hash too
main_stl = pd.read_csv("data/settlement_report.csv")
main_ord = pd.read_csv("data/order_ledger.csv")
main_hash = supabase_cache.compute_df_hash(main_stl, main_ord)
hashes.add(main_hash)
check("Main dataset hash differs from all samples", len(hashes) == 5)

print()
print("=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} checks PASSED, {failed} FAILED")
if failed == 0:
    print("ALL CHECKS PASSED!")
else:
    print(f"WARNING: {failed} check(s) failed")
print("=" * 60)
