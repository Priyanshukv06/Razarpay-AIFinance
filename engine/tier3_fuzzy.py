"""Tier 3 -- Fuzzy Candidate Matching with AI Confirmation.

Finds candidate matches using amount/date tolerance windows, then asks
the LLM to confirm or reject each pairing with an explanation.

This tier handles:
  - Partial refunds (amount differs significantly, same or similar ID)
  - Duplicate orders (same order_id with multiple entries)
  - Split settlements (one order, two settlement UTRs)

Design note: candidates are generated DETERMINISTICALLY using fuzzy logic.
The LLM only CONFIRMS or REJECTS -- it never invents matches the fuzzy
logic didn't find. This keeps the AI's role narrow and auditable.
"""

import json
import logging

import pandas as pd
from rapidfuzz import fuzz

from engine.llm.prompts import build_tier3_prompt
from engine.llm.router import RoutedLLMBackend, RouteResult

log = logging.getLogger(__name__)

# Tolerance windows for fuzzy matching. These are generous enough to catch
# real-world drift without being so wide they match unrelated records.
# NOT tuned to our synthetic data -- these are reasonable defaults for
# any Indian payment reconciliation scenario.
AMOUNT_TOLERANCE_PCT = 0.70    # Allow up to 70% amount difference (catches partial refunds)
DATE_TOLERANCE_DAYS = 7        # Allow up to 7 days between order and settlement


def _parse_json_response(text: str) -> dict | None:
    """Try to extract valid JSON from an LLM response.

    Models sometimes wrap JSON in markdown code blocks or add preamble.
    This handles the common cases without being brittle.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _find_candidates(settlement: pd.Series,
                     orders: pd.DataFrame) -> list[tuple[int, pd.Series, str]]:
    """Find order candidates for a settlement using fuzzy logic.

    Returns list of (order_index, order_row, match_reason).
    """
    candidates = []
    stl_amount = settlement["amount_settled"]
    stl_date = pd.to_datetime(settlement["settlement_date"])
    stl_ref = str(settlement["order_ref"])

    for idx, order in orders.iterrows():
        reasons = []
        order_amount = order["order_amount"]
        ord_date = pd.to_datetime(order["order_date"])

        # Check 1: ID similarity (exact or fuzzy)
        order_id = str(order["order_id"])
        id_match = stl_ref == order_id
        id_fuzzy = fuzz.ratio(stl_ref, order_id) > 80  # High threshold for ID similarity

        # Check 2: Amount within tolerance
        if order_amount > 0:
            amount_diff_pct = abs(stl_amount - order_amount) / order_amount
        else:
            amount_diff_pct = 1.0
        amount_close = amount_diff_pct <= AMOUNT_TOLERANCE_PCT

        # Check 3: Date within tolerance
        date_diff = abs((stl_date - ord_date).days)
        date_close = date_diff <= DATE_TOLERANCE_DAYS

        # A candidate needs at least ID match/similarity AND one other signal
        if id_match and amount_close:
            reasons.append(f"ID match, amount within {amount_diff_pct:.0%}")
            if date_close:
                reasons.append(f"date within {date_diff}d")
        elif id_match and date_close:
            reasons.append(f"ID match, date within {date_diff}d")
        elif id_fuzzy and amount_close and date_close:
            reasons.append(f"fuzzy ID ({fuzz.ratio(stl_ref, order_id)}%), "
                           f"amount within {amount_diff_pct:.0%}, "
                           f"date within {date_diff}d")

        if reasons:
            candidates.append((idx, order, "; ".join(reasons)))

    return candidates


def run(settlements: pd.DataFrame, orders: pd.DataFrame,
        router: RoutedLLMBackend) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Tier 3 fuzzy matching with AI confirmation.

    Args:
        settlements: Unmatched settlements from Tier 2.
        orders: Unmatched orders from Tier 2.
        router: Initialized LLM router for AI calls.

    Returns:
        (results, unmatched_settlements, unmatched_orders)
    """
    results = []
    matched_stl_ids = set()
    matched_ord_ids = set()
    ai_calls = 0

    for stl_idx, stl_row in settlements.iterrows():
        if stl_idx in matched_stl_ids:
            continue

        candidates = _find_candidates(stl_row, orders)

        # Filter out already-matched orders
        candidates = [(i, o, r) for i, o, r in candidates if i not in matched_ord_ids]

        if not candidates:
            continue  # No fuzzy match found -- will go to Tier 4

        # For each candidate, ask the LLM to confirm or reject
        for ord_idx, ord_row, match_reason in candidates:
            messages = build_tier3_prompt(stl_row.to_dict(), ord_row.to_dict())
            ai_calls += 1

            result = router.route(messages)
            if result is None:
                # All providers failed -- log and skip to next candidate
                log.warning("Tier 3: all providers failed for %s / %s",
                            stl_row["settlement_id"], ord_row["order_id"])
                continue

            parsed = _parse_json_response(result.text)
            if parsed is None:
                log.warning("Tier 3: invalid JSON from %s for %s / %s: %s",
                            result.model.name, stl_row["settlement_id"],
                            ord_row["order_id"], result.text[:200])
                continue

            # Check if the AI confirmed the match
            match_confirmed = parsed.get("match_confirmed", False)
            if match_confirmed:
                category = parsed.get("category", "other")
                confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
                explanation = parsed.get("explanation", "AI-confirmed fuzzy match")
                suggested_action = parsed.get("suggested_action")

                results.append({
                    "order_id": ord_row["order_id"],
                    "settlement_id": stl_row["settlement_id"],
                    "tier": 3,
                    "match_status": "ai_resolved",
                    "confidence": confidence,
                    "category": category,
                    "explanation": (
                        f"[AI/{result.model.provider}] {explanation} "
                        f"(fuzzy: {match_reason})"
                    ),
                    "suggested_action": suggested_action,
                })
                matched_stl_ids.add(stl_idx)
                matched_ord_ids.add(ord_idx)
                break  # Settlement matched, move to next one

    results_df = pd.DataFrame(results)
    unmatched_stl = settlements[~settlements.index.isin(matched_stl_ids)].copy()
    unmatched_ord = orders[~orders.index.isin(matched_ord_ids)].copy()

    log.info("Tier 3: %d AI calls, %d matches confirmed", ai_calls, len(results))
    print(f"  Tier 3: {ai_calls} AI calls, {len(results)} matches confirmed")

    return results_df, unmatched_stl, unmatched_ord
