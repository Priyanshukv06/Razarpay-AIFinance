"""Tier 2 -- Rule-Based Match Engine.

Matches settlement records to order records by:
  - order_ref == order_id  (ID linkage, same as Tier 1)
  - amount_settled ~= order_amount - fee - tax  (fee/tax-adjusted, within 0.01)

This catches the common case where the gateway deducted a fee + tax
before settling. The fee and tax fields are right there in the
settlement report, so this is a fully deterministic, explainable match.

Also handles date drift -- settlement dates that are T+3/T+4/T+5
instead of the usual T+1/T+2. The date is noted in the explanation
but does NOT affect whether the match succeeds (the amounts and IDs
are what matter for correctness; the date is informational context).

Design note: Tier 2 ONLY runs on records that Tier 1 did not match.
It never re-examines Tier 1 results. This is enforced by the pipeline.
"""

import pandas as pd


def run(settlements: pd.DataFrame, orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Tier 2 rule-based matching.

    Args:
        settlements: Unmatched settlements from Tier 1.
        orders: Unmatched orders from Tier 1.

    Returns:
        (results, unmatched_settlements, unmatched_orders):
        - results: DataFrame of matched records in unified result schema.
        - unmatched_settlements: Settlements not matched at this tier.
        - unmatched_orders: Orders not matched at this tier.
    """
    results = []
    matched_stl_ids = set()
    matched_ord_ids = set()

    # Build order lookup (same pattern as Tier 1 -- handles duplicate IDs)
    order_lookup = {}
    for idx, order in orders.iterrows():
        oid = order["order_id"]
        if oid not in order_lookup:
            order_lookup[oid] = []
        order_lookup[oid].append((idx, order))

    for stl_idx, stl_row in settlements.iterrows():
        order_ref = stl_row["order_ref"]
        amount_settled = stl_row["amount_settled"]
        fee = stl_row["fee"]
        tax = stl_row["tax"]
        stl_date = pd.to_datetime(stl_row["settlement_date"])

        if order_ref not in order_lookup:
            continue

        candidates = order_lookup[order_ref]

        # Same guard as Tier 1: if multiple orders share this order_id,
        # the match is ambiguous. Leave for Tier 3.
        unmatched_candidates = [(i, o) for i, o in candidates if i not in matched_ord_ids]
        if len(unmatched_candidates) != 1 and len(candidates) > 1:
            continue

        for ord_idx, ord_row in candidates:
            if ord_idx in matched_ord_ids:
                continue

            order_amount = ord_row["order_amount"]
            ord_date = pd.to_datetime(ord_row["order_date"])

            # Fee-adjusted match: order_amount - fee - tax should equal amount_settled
            # This is the core Tier 2 logic. The fee and tax are FROM the
            # settlement report itself, so this is purely deterministic.
            expected_settled = round(order_amount - fee - tax, 2)

            if abs(amount_settled - expected_settled) < 0.01:
                # Calculate date drift for the explanation
                date_diff = (stl_date - ord_date).days

                # Determine the category based on date drift
                # Normal settlement is T+1 or T+2; anything beyond is notable
                if date_diff > 2:
                    category = "date_drift"
                    date_note = f" Settlement arrived T+{date_diff} (normal is T+1/T+2)."
                else:
                    category = "fee_adjusted"
                    date_note = ""

                fee_total = round(fee + tax, 2)
                explanation = (
                    f"Fee-adjusted match: order {order_ref} gross amount "
                    f"{order_amount:.2f} minus gateway fee {fee:.2f} + tax "
                    f"{tax:.2f} (total deduction {fee_total:.2f}) = "
                    f"{expected_settled:.2f}, matching settlement "
                    f"{amount_settled:.2f}.{date_note}"
                )

                results.append({
                    "order_id": order_ref,
                    "settlement_id": stl_row["settlement_id"],
                    "tier": 2,
                    "match_status": "matched",
                    "confidence": 0.98,  # High but not 1.0 -- fee formula is deterministic
                                         # but conceptually one step below "identical values"
                    "category": category,
                    "explanation": explanation,
                    "suggested_action": None,
                })
                matched_stl_ids.add(stl_idx)
                matched_ord_ids.add(ord_idx)
                break  # Settlement matched, move on

    results_df = pd.DataFrame(results)

    # Filter out matched records for downstream tiers
    unmatched_stl = settlements[~settlements.index.isin(matched_stl_ids)].copy()
    unmatched_ord = orders[~orders.index.isin(matched_ord_ids)].copy()

    return results_df, unmatched_stl, unmatched_ord
