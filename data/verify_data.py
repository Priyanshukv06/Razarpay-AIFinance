"""Data quality verification — run after generate_data.py to validate output.

Usage:
    conda activate razorpay
    python data/verify_data.py
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")


def main():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} -- {detail}")
            failed += 1

    # ── Load data ──
    stl = pd.read_csv(DATA_DIR / "settlement_report.csv")
    ord_ = pd.read_csv(DATA_DIR / "order_ledger.csv")
    key = pd.read_csv(DATA_DIR / "answer_key.csv")

    print("=" * 64)
    print("DATA QUALITY VERIFICATION")
    print("=" * 64)

    # ── 1. File integrity ──
    print("\n[1] File Integrity")
    check("settlement_report.csv exists and non-empty", len(stl) > 0)
    check("order_ledger.csv exists and non-empty", len(ord_) > 0)
    check("answer_key.csv exists and non-empty", len(key) > 0)

    # ── 2. Schema validation ──
    print("\n[2] Schema Validation")
    stl_expected = {"settlement_id", "order_ref", "payment_id", "amount_settled",
                    "fee", "tax", "settlement_date", "status"}
    ord_expected = {"order_id", "customer", "order_amount", "order_date",
                    "payment_method", "status"}
    key_expected = {"order_id", "settlement_id", "expected_tier", "mismatch_type",
                    "expected_match_status", "notes"}

    check("Settlement schema correct",
          set(stl.columns) == stl_expected,
          f"got {set(stl.columns)}")
    check("Order schema correct",
          set(ord_.columns) == ord_expected,
          f"got {set(ord_.columns)}")
    check("Answer key schema correct",
          set(key.columns) == key_expected,
          f"got {set(key.columns)}")

    # ── 3. Data types and ranges ──
    print("\n[3] Data Types & Ranges")
    check("amount_settled > 0 for all settlements",
          (stl["amount_settled"] > 0).all(),
          f"{(stl['amount_settled'] <= 0).sum()} non-positive amounts")
    check("fee >= 0 for all settlements",
          (stl["fee"] >= 0).all())
    check("tax >= 0 for all settlements",
          (stl["tax"] >= 0).all())
    check("order_amount > 0 for all orders",
          (ord_["order_amount"] > 0).all())
    check("settlement_date is valid date format",
          pd.to_datetime(stl["settlement_date"], errors="coerce").notna().all())
    check("order_date is valid date format",
          pd.to_datetime(ord_["order_date"], errors="coerce").notna().all())
    check("settlement status in {settled, pending, reversed}",
          stl["status"].isin(["settled", "pending", "reversed"]).all(),
          f"unexpected: {set(stl['status'].unique()) - {'settled', 'pending', 'reversed'}}")
    check("order status in {paid, refunded, partially_refunded, pending}",
          ord_["status"].isin(["paid", "refunded", "partially_refunded", "pending"]).all(),
          f"unexpected: {set(ord_['status'].unique()) - {'paid', 'refunded', 'partially_refunded', 'pending'}}")
    check("payment_method in {UPI, card, netbanking, wallet}",
          ord_["payment_method"].isin(["UPI", "card", "netbanking", "wallet"]).all())

    # ── 4. Record counts ──
    print("\n[4] Record Counts")
    print(f"     Settlements: {len(stl)}")
    print(f"     Orders:      {len(ord_)}")
    print(f"     Answer keys: {len(key)}")
    check("Settlement count is 100", len(stl) == 100, f"got {len(stl)}")
    check("Order count is 99 (93 base + 6 duplicates)", len(ord_) == 99, f"got {len(ord_)}")
    check("Answer key count is 100", len(key) == 100, f"got {len(key)}")

    # ── 5. Answer key distribution ──
    print("\n[5] Mismatch Type Distribution")
    type_counts = key["mismatch_type"].value_counts()
    for mtype, expected in [("perfect_match", 40), ("fee_adjusted", 15),
                            ("date_drift", 10), ("partial_refund", 8),
                            ("duplicate_order", 6), ("orphan_settlement", 7),
                            ("orphan_order", 7), ("split_settlement", 7)]:
        actual = type_counts.get(mtype, 0)
        check(f"{mtype}: {actual} (expected {expected})", actual == expected)

    print("\n[6] Tier Distribution")
    tier_counts = key["expected_tier"].value_counts().sort_index()
    for tier, expected in [(1, 40), (2, 25), (3, 21), (4, 14)]:
        actual = tier_counts.get(tier, 0)
        check(f"Tier {tier}: {actual} (expected {expected})", actual == expected)

    # ── 6. Cross-referencing checks ──
    print("\n[7] Cross-Reference Integrity")

    # Perfect matches: order_ref in settlement should exist in orders
    perfect_keys = key[key["mismatch_type"] == "perfect_match"]
    for _, row in perfect_keys.iterrows():
        oid = row["order_id"]
        sid = row["settlement_id"]
        stl_row = stl[stl["settlement_id"] == sid]
        ord_row = ord_[ord_["order_id"] == oid]
        if not stl_row.empty and not ord_row.empty:
            # Check amounts match (UPI, zero fee)
            if abs(stl_row.iloc[0]["amount_settled"] - ord_row.iloc[0]["order_amount"]) > 0.01:
                check(f"Perfect match {oid}: amounts equal", False,
                      f"stl={stl_row.iloc[0]['amount_settled']} ord={ord_row.iloc[0]['order_amount']}")
                break
    else:
        check("Perfect matches: amount_settled == order_amount (spot check)", True)

    # Fee-adjusted: check that amount_settled ≈ order_amount - fee - tax
    fee_keys = key[key["mismatch_type"] == "fee_adjusted"]
    fee_ok = True
    for _, row in fee_keys.iterrows():
        oid = row["order_id"]
        sid = row["settlement_id"]
        stl_row = stl[stl["settlement_id"] == sid]
        ord_row = ord_[ord_["order_id"] == oid]
        if not stl_row.empty and not ord_row.empty:
            s = stl_row.iloc[0]
            o = ord_row.iloc[0]
            expected_settled = round(o["order_amount"] - s["fee"] - s["tax"], 2)
            if abs(s["amount_settled"] - expected_settled) > 0.01:
                check(f"Fee-adjusted {oid}: settled = amount - fee - tax", False,
                      f"settled={s['amount_settled']} expected={expected_settled}")
                fee_ok = False
                break
    if fee_ok:
        check("Fee-adjusted: amount_settled == order_amount - fee - tax (all 15)", True)

    # Orphan settlements: order_id should be NaN in answer key
    orphan_stl = key[key["mismatch_type"] == "orphan_settlement"]
    check("Orphan settlements: all have NaN order_id in answer key",
          orphan_stl["order_id"].isna().all(),
          f"{orphan_stl['order_id'].notna().sum()} have order_id set")

    # Orphan settlements: their order_ref should NOT exist in order ledger
    orphan_stl_ids = orphan_stl["settlement_id"].tolist()
    orphan_refs = stl[stl["settlement_id"].isin(orphan_stl_ids)]["order_ref"].tolist()
    phantom_in_orders = ord_[ord_["order_id"].isin(orphan_refs)]
    check("Orphan settlement order_refs don't exist in order ledger",
          len(phantom_in_orders) == 0,
          f"{len(phantom_in_orders)} phantom refs found in orders")

    # Orphan orders: settlement_id should be NaN in answer key
    orphan_ord = key[key["mismatch_type"] == "orphan_order"]
    check("Orphan orders: all have NaN settlement_id in answer key",
          orphan_ord["settlement_id"].isna().all(),
          f"{orphan_ord['settlement_id'].notna().sum()} have settlement_id set")

    # Duplicates: should have exactly 2 rows per order_id in ledger
    dup_keys = key[key["mismatch_type"] == "duplicate_order"]
    dup_ok = True
    for _, row in dup_keys.iterrows():
        oid = row["order_id"]
        count = (ord_["order_id"] == oid).sum()
        if count != 2:
            check(f"Duplicate {oid}: has 2 rows in ledger", False, f"got {count}")
            dup_ok = False
            break
    if dup_ok:
        check("Duplicate orders: all have exactly 2 rows in ledger (all 6)", True)

    # Split settlements: should have exactly 2 settlement rows per answer key entry
    split_keys = key[key["mismatch_type"] == "split_settlement"]
    split_ok = True
    for _, row in split_keys.iterrows():
        sids = str(row["settlement_id"]).split(",")
        if len(sids) != 2:
            check(f"Split {row['order_id']}: has 2 settlement_ids", False, f"got {len(sids)}")
            split_ok = False
            break
        for sid in sids:
            if stl[stl["settlement_id"] == sid.strip()].empty:
                check(f"Split {row['order_id']}: {sid} exists in settlements", False)
                split_ok = False
                break
    if split_ok:
        check("Split settlements: all have 2 valid settlement_ids (all 7)", True)

    # ── 7. API Keys ──
    print("\n[8] API Key Configuration")
    providers = {
        "GROQ_API_KEYS": "Groq",
        "GEMINI_API_KEYS": "Gemini",
        "OPENROUTER_API_KEYS": "OpenRouter",
        "MISTRAL_API_KEYS": "Mistral",
        "NVIDIA_API_KEYS": "NVIDIA NIM",
    }
    total_keys = 0
    for env_var, name in providers.items():
        raw = os.getenv(env_var, "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        total_keys += len(keys)
        check(f"{name} ({env_var}): {len(keys)} key(s) configured", len(keys) > 0)

    print(f"\n     Total API keys: {total_keys}")

    # ── Summary ──
    print("\n" + "=" * 64)
    print(f"RESULT: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL CHECKS PASSED — Phase 1 data is verified.")
    else:
        print("SOME CHECKS FAILED — review issues above.")
    print("=" * 64)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
