"""Synthetic data generator for the AI Finance Controller.

Produces three CSV files:
  - settlement_report.csv  (Source A — what the gateway settled)
  - order_ledger.csv       (Source B — merchant's internal records)
  - answer_key.csv         (private ground truth for accuracy scoring)

Deliberately seeds all 8 mismatch types from PRD.md §9.4 in documented
proportions, so the accuracy claim against the answer key is reproducible.

Usage:
    conda activate razorpay
    python data/generate_data.py              # default 100 records
    python data/generate_data.py --records 75 # custom count
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42  # Reproducibility — same seed = same data every run
DEFAULT_RECORD_COUNT = 100

# Mismatch type distribution (must sum to 1.0)
# PRD §9.4: "realistic proportion of hard cases (~15–25% requiring Tier 3/4)"
# We seed ~22% hard cases (types 5-8), ~18% rule-matchable (types 2-4),
# and ~60% exact matches (type 1).
DISTRIBUTION = {
    "perfect_match":      0.40,  # Type 1: exact match on ID + amount
    "fee_adjusted":       0.15,  # Type 2: amount differs by fee+tax
    "date_drift":         0.10,  # Type 3: settlement T+1/T+2 (ID matches, amount matches after fee)
    "partial_refund":     0.08,  # Type 4: partial refund changes order status + amount
    "duplicate_order":    0.06,  # Type 5: same order_id appears twice in ledger
    "orphan_settlement":  0.07,  # Type 6: settlement with no matching order
    "orphan_order":       0.07,  # Type 7: order with no matching settlement
    "split_settlement":   0.07,  # Type 8: one order settled across two UTRs
}

# Payment methods and their typical fee rates
PAYMENT_METHODS = {
    "UPI":        {"fee_rate": 0.00, "tax_rate": 0.18},   # Zero MDR on UPI in India
    "card":       {"fee_rate": 0.02, "tax_rate": 0.18},   # ~2% MDR
    "netbanking": {"fee_rate": 0.018, "tax_rate": 0.18},  # ~1.8% MDR
    "wallet":     {"fee_rate": 0.015, "tax_rate": 0.18},  # ~1.5% MDR
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
fake = Faker("en_IN")  # Indian locale for realistic merchant data


def _round_money(val: float) -> float:
    """Round to 2 decimal places — financial precision."""
    return round(val, 2)


def _compute_fee(order_amount: float, method: str) -> tuple[float, float]:
    """Compute gateway fee and tax on fee for a payment method."""
    rates = PAYMENT_METHODS[method]
    fee = _round_money(order_amount * rates["fee_rate"])
    tax = _round_money(fee * rates["tax_rate"])
    return fee, tax


def _settlement_date(order_date: datetime, drift_days: int = None) -> datetime:
    """Settlement is typically T+1 or T+2 after the order date."""
    if drift_days is None:
        drift_days = random.choice([1, 2])
    return order_date + timedelta(days=drift_days)


def _generate_order_id(index: int) -> str:
    return f"ORD{1000 + index}"


def _generate_settlement_id(index: int) -> str:
    return f"STL{5000 + index}"


def _generate_payment_id() -> str:
    return f"pay_{fake.bothify(text='??####??', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')}"


# ---------------------------------------------------------------------------
# Record generators — one per mismatch type
# ---------------------------------------------------------------------------
def _gen_perfect_match(order_idx: int, stl_idx: int, base_date: datetime):
    """Type 1: order_ref == order_id, amount_settled == order_amount (UPI, zero fee)."""
    order_id = _generate_order_id(order_idx)
    amount = _round_money(random.uniform(200, 15000))
    method = "UPI"  # Zero MDR means amount_settled == order_amount
    fee, tax = _compute_fee(amount, method)
    order_date = base_date - timedelta(days=random.randint(0, 30))
    stl_date = _settlement_date(order_date)

    settlement = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": amount,  # Equal to order_amount since UPI has 0% fee
        "fee": fee,               # 0.00
        "tax": tax,               # 0.00
        "settlement_date": stl_date.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    order = {
        "order_id": order_id,
        "customer": fake.name(),
        "order_amount": amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "paid",
    }
    answer = {
        "order_id": order_id,
        "settlement_id": settlement["settlement_id"],
        "expected_tier": 1,
        "mismatch_type": "perfect_match",
        "expected_match_status": "matched",
        "notes": "Exact ID + amount match (UPI, zero fee)",
    }
    return settlement, order, answer


def _gen_fee_adjusted(order_idx: int, stl_idx: int, base_date: datetime):
    """Type 2: order_ref == order_id, amount differs by fee+tax deduction."""
    order_id = _generate_order_id(order_idx)
    method = random.choice(["card", "netbanking", "wallet"])
    amount = _round_money(random.uniform(500, 20000))
    fee, tax = _compute_fee(amount, method)
    settled = _round_money(amount - fee - tax)
    order_date = base_date - timedelta(days=random.randint(0, 30))
    stl_date = _settlement_date(order_date)

    settlement = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": settled,
        "fee": fee,
        "tax": tax,
        "settlement_date": stl_date.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    order = {
        "order_id": order_id,
        "customer": fake.name(),
        "order_amount": amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "paid",
    }
    answer = {
        "order_id": order_id,
        "settlement_id": settlement["settlement_id"],
        "expected_tier": 2,
        "mismatch_type": "fee_adjusted",
        "expected_match_status": "matched",
        "notes": f"Amount differs by fee({fee})+tax({tax}). Method: {method}",
    }
    return settlement, order, answer


def _gen_date_drift(order_idx: int, stl_idx: int, base_date: datetime):
    """Type 3: ID matches, fee-adjusted amount matches, but settlement date
    is T+3 or T+4 (outside normal T+1/T+2 window)."""
    order_id = _generate_order_id(order_idx)
    method = random.choice(["card", "netbanking", "wallet", "UPI"])
    amount = _round_money(random.uniform(300, 12000))
    fee, tax = _compute_fee(amount, method)
    settled = _round_money(amount - fee - tax)
    order_date = base_date - timedelta(days=random.randint(5, 35))
    # Deliberately wider drift than T+1/T+2
    drift = random.choice([3, 4, 5])
    stl_date = _settlement_date(order_date, drift_days=drift)

    settlement = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": settled,
        "fee": fee,
        "tax": tax,
        "settlement_date": stl_date.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    order = {
        "order_id": order_id,
        "customer": fake.name(),
        "order_amount": amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "paid",
    }
    answer = {
        "order_id": order_id,
        "settlement_id": settlement["settlement_id"],
        "expected_tier": 2,
        "mismatch_type": "date_drift",
        "expected_match_status": "matched",
        "notes": f"Fee-adjusted match with T+{drift} settlement date drift",
    }
    return settlement, order, answer


def _gen_partial_refund(order_idx: int, stl_idx: int, base_date: datetime):
    """Type 4: order was partially refunded, so settled amount < order_amount - fee - tax."""
    order_id = _generate_order_id(order_idx)
    method = random.choice(["card", "netbanking", "wallet"])
    original_amount = _round_money(random.uniform(1000, 20000))
    # Refund 20-60% of the original amount
    refund_pct = random.uniform(0.20, 0.60)
    refund_amount = _round_money(original_amount * refund_pct)
    effective_amount = _round_money(original_amount - refund_amount)
    fee, tax = _compute_fee(effective_amount, method)
    settled = _round_money(effective_amount - fee - tax)
    order_date = base_date - timedelta(days=random.randint(5, 30))
    stl_date = _settlement_date(order_date)

    settlement = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": settled,
        "fee": fee,
        "tax": tax,
        "settlement_date": stl_date.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    order = {
        "order_id": order_id,
        "customer": fake.name(),
        "order_amount": original_amount,  # Ledger still shows original amount
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "partially_refunded",
    }
    answer = {
        "order_id": order_id,
        "settlement_id": settlement["settlement_id"],
        "expected_tier": 3,
        "mismatch_type": "partial_refund",
        "expected_match_status": "ai_resolved",
        "notes": f"Original ₹{original_amount}, refund ₹{refund_amount} "
                 f"({refund_pct:.0%}), settled ₹{settled}",
    }
    return settlement, order, answer


def _gen_duplicate_order(order_idx: int, stl_idx: int, base_date: datetime):
    """Type 5: same order_id appears twice in the ledger (data entry error).
    Only one settlement exists for it."""
    order_id = _generate_order_id(order_idx)
    amount = _round_money(random.uniform(500, 10000))
    method = "UPI"
    fee, tax = _compute_fee(amount, method)
    order_date = base_date - timedelta(days=random.randint(0, 30))
    stl_date = _settlement_date(order_date)

    settlement = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": amount,
        "fee": fee,
        "tax": tax,
        "settlement_date": stl_date.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    # Two ledger entries for the same order
    customer = fake.name()
    order_1 = {
        "order_id": order_id,
        "customer": customer,
        "order_amount": amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "paid",
    }
    # Duplicate — slightly different timestamp simulating a double-entry
    order_2 = {
        "order_id": order_id,
        "customer": customer,
        "order_amount": amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "paid",
    }
    answer = {
        "order_id": order_id,
        "settlement_id": settlement["settlement_id"],
        "expected_tier": 3,
        "mismatch_type": "duplicate_order",
        "expected_match_status": "ai_resolved",
        "notes": "Duplicate order_id in ledger — one settlement, two order entries",
    }
    return settlement, [order_1, order_2], answer


def _gen_orphan_settlement(stl_idx: int, base_date: datetime):
    """Type 6: settlement exists but no matching order in the ledger.
    Returns (settlement, None, answer) — no order to add."""
    amount = _round_money(random.uniform(100, 8000))
    method = random.choice(list(PAYMENT_METHODS.keys()))
    fee, tax = _compute_fee(amount, method)
    settled = _round_money(amount - fee - tax)
    stl_date = base_date - timedelta(days=random.randint(0, 20))
    # order_ref points to an order that doesn't exist in our ledger
    phantom_order_id = f"ORD{random.randint(9000, 9999)}"

    settlement = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": phantom_order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": settled,
        "fee": fee,
        "tax": tax,
        "settlement_date": stl_date.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    answer = {
        "order_id": None,
        "settlement_id": settlement["settlement_id"],
        "expected_tier": 4,
        "mismatch_type": "orphan_settlement",
        "expected_match_status": "unresolved",
        "notes": f"No order {phantom_order_id} exists in the ledger",
    }
    return settlement, None, answer


def _gen_orphan_order(order_idx: int, base_date: datetime):
    """Type 7: order exists but no settlement arrived for it.
    Returns (None, order, answer) — no settlement to add."""
    order_id = _generate_order_id(order_idx)
    amount = _round_money(random.uniform(200, 12000))
    method = random.choice(list(PAYMENT_METHODS.keys()))
    order_date = base_date - timedelta(days=random.randint(0, 30))
    # Status is pending — payment was never completed
    status = random.choice(["pending", "paid"])

    order = {
        "order_id": order_id,
        "customer": fake.name(),
        "order_amount": amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": status,
    }
    answer = {
        "order_id": order_id,
        "settlement_id": None,
        "expected_tier": 4,
        "mismatch_type": "orphan_order",
        "expected_match_status": "unresolved",
        "notes": f"Order {order_id} has no matching settlement. Status: {status}",
    }
    return None, order, answer


def _gen_split_settlement(order_idx: int, stl_idx: int, base_date: datetime):
    """Type 8: one order settled across two UTRs (split settlement).
    Returns (list[settlement], order, answer)."""
    order_id = _generate_order_id(order_idx)
    method = random.choice(["card", "netbanking"])
    total_amount = _round_money(random.uniform(2000, 25000))
    fee, tax = _compute_fee(total_amount, method)
    net_amount = _round_money(total_amount - fee - tax)

    # Split into two unequal parts
    split_ratio = random.uniform(0.3, 0.7)
    part1 = _round_money(net_amount * split_ratio)
    part2 = _round_money(net_amount - part1)

    order_date = base_date - timedelta(days=random.randint(5, 30))
    stl_date_1 = _settlement_date(order_date, drift_days=1)
    stl_date_2 = _settlement_date(order_date, drift_days=2)

    # Fee/tax are split proportionally too
    fee1 = _round_money(fee * split_ratio)
    fee2 = _round_money(fee - fee1)
    tax1 = _round_money(tax * split_ratio)
    tax2 = _round_money(tax - tax1)

    settlement_1 = {
        "settlement_id": _generate_settlement_id(stl_idx),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": part1,
        "fee": fee1,
        "tax": tax1,
        "settlement_date": stl_date_1.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    settlement_2 = {
        "settlement_id": _generate_settlement_id(stl_idx + 1),
        "order_ref": order_id,
        "payment_id": _generate_payment_id(),
        "amount_settled": part2,
        "fee": fee2,
        "tax": tax2,
        "settlement_date": stl_date_2.strftime("%Y-%m-%d"),
        "status": "settled",
    }
    order = {
        "order_id": order_id,
        "customer": fake.name(),
        "order_amount": total_amount,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "payment_method": method,
        "status": "paid",
    }
    answer = {
        "order_id": order_id,
        "settlement_id": f"{settlement_1['settlement_id']},{settlement_2['settlement_id']}",
        "expected_tier": 3,
        "mismatch_type": "split_settlement",
        "expected_match_status": "ai_resolved",
        "notes": f"Total ₹{total_amount}, settled as ₹{part1} + ₹{part2} across 2 UTRs",
    }
    return [settlement_1, settlement_2], order, answer


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate(record_count: int = DEFAULT_RECORD_COUNT, output_dir: str = None):
    """Generate the full dataset and write to CSV files.

    Returns (settlements_df, orders_df, answer_key_df) — or None on error.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    Faker.seed(SEED)

    base_date = datetime(2026, 9, 3)  # Recent date for realistic data

    # Allocate records per type based on distribution
    type_counts = {}
    remaining = record_count
    types_list = list(DISTRIBUTION.keys())
    for i, mtype in enumerate(types_list):
        if i == len(types_list) - 1:
            # Last type gets whatever is remaining to avoid rounding issues
            type_counts[mtype] = remaining
        else:
            count = round(record_count * DISTRIBUTION[mtype])
            type_counts[mtype] = count
            remaining -= count

    settlements = []
    orders = []
    answer_keys = []

    order_idx = 0
    stl_idx = 0

    # --- Type 1: Perfect matches ---
    for _ in range(type_counts["perfect_match"]):
        stl, order, answer = _gen_perfect_match(order_idx, stl_idx, base_date)
        settlements.append(stl)
        orders.append(order)
        answer_keys.append(answer)
        order_idx += 1
        stl_idx += 1

    # --- Type 2: Fee-adjusted matches ---
    for _ in range(type_counts["fee_adjusted"]):
        stl, order, answer = _gen_fee_adjusted(order_idx, stl_idx, base_date)
        settlements.append(stl)
        orders.append(order)
        answer_keys.append(answer)
        order_idx += 1
        stl_idx += 1

    # --- Type 3: Date drift ---
    for _ in range(type_counts["date_drift"]):
        stl, order, answer = _gen_date_drift(order_idx, stl_idx, base_date)
        settlements.append(stl)
        orders.append(order)
        answer_keys.append(answer)
        order_idx += 1
        stl_idx += 1

    # --- Type 4: Partial refund ---
    for _ in range(type_counts["partial_refund"]):
        stl, order, answer = _gen_partial_refund(order_idx, stl_idx, base_date)
        settlements.append(stl)
        orders.append(order)
        answer_keys.append(answer)
        order_idx += 1
        stl_idx += 1

    # --- Type 5: Duplicate order ---
    for _ in range(type_counts["duplicate_order"]):
        stl, order_pair, answer = _gen_duplicate_order(order_idx, stl_idx, base_date)
        settlements.append(stl)
        # order_pair is a list of two orders with the same order_id
        orders.extend(order_pair)
        answer_keys.append(answer)
        order_idx += 1
        stl_idx += 1

    # --- Type 6: Orphan settlement ---
    for _ in range(type_counts["orphan_settlement"]):
        stl, _, answer = _gen_orphan_settlement(stl_idx, base_date)
        settlements.append(stl)
        # No order to add
        answer_keys.append(answer)
        stl_idx += 1

    # --- Type 7: Orphan order ---
    for _ in range(type_counts["orphan_order"]):
        _, order, answer = _gen_orphan_order(order_idx, base_date)
        # No settlement to add
        orders.append(order)
        answer_keys.append(answer)
        order_idx += 1

    # --- Type 8: Split settlement ---
    for _ in range(type_counts["split_settlement"]):
        stl_pair, order, answer = _gen_split_settlement(order_idx, stl_idx, base_date)
        settlements.extend(stl_pair)  # Two settlement rows for one order
        orders.append(order)
        answer_keys.append(answer)
        order_idx += 1
        stl_idx += 2  # Two settlement IDs consumed

    # Shuffle to avoid clustering by type — real data is never sorted by
    # mismatch category, and a pipeline that accidentally relies on order
    # would pass testing but fail on real data.
    random.shuffle(settlements)
    random.shuffle(orders)

    # --- Write CSV files ---
    stl_path = output_dir / "settlement_report.csv"
    ord_path = output_dir / "order_ledger.csv"
    key_path = output_dir / "answer_key.csv"

    stl_fields = ["settlement_id", "order_ref", "payment_id", "amount_settled",
                   "fee", "tax", "settlement_date", "status"]
    ord_fields = ["order_id", "customer", "order_amount", "order_date",
                   "payment_method", "status"]
    key_fields = ["order_id", "settlement_id", "expected_tier", "mismatch_type",
                   "expected_match_status", "notes"]

    _write_csv(stl_path, stl_fields, settlements)
    _write_csv(ord_path, ord_fields, orders)
    _write_csv(key_path, key_fields, answer_keys)

    # --- Print summary ---
    print("=" * 60)
    print("SYNTHETIC DATA GENERATED")
    print("=" * 60)
    print(f"  Settlement records:  {len(settlements)}")
    print(f"  Order records:       {len(orders)}")
    print(f"  Answer key entries:  {len(answer_keys)}")
    print()
    print("  Type distribution:")
    for mtype, count in type_counts.items():
        pct = count / record_count * 100
        print(f"    {mtype:<22} {count:>3}  ({pct:.0f}%)")
    print()
    print(f"  Files written to: {output_dir}")
    print(f"    {stl_path.name}")
    print(f"    {ord_path.name}")
    print(f"    {key_path.name}")
    print("=" * 60)

    return type_counts


def _write_csv(path: Path, fields: list[str], rows: list[dict]):
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic reconciliation data")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORD_COUNT,
                        help=f"Number of base records (default: {DEFAULT_RECORD_COUNT})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    generate(record_count=args.records, output_dir=args.output_dir)
