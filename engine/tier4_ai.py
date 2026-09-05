"""Tier 4 -- AI Categorization for Unmatched Records.

Records that reach Tier 4 have NO candidate match at all. These are:
  - Orphan settlements (settlement with no matching order)
  - Orphan orders (order with no matching settlement)

The LLM categorizes each one and suggests what a human should do.
These are ALWAYS reported as 'unresolved' -- never auto-matched.
This is the "honest exception list" the track brief asks for.

Design note: Tier 4 results never have match_status='matched'.
They are explanations of WHY something is unmatched, not matches.
"""

import json
import logging
import re

import pandas as pd

from engine.llm.prompts import build_tier4_prompt
from engine.llm.router import RoutedLLMBackend

log = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict | None:
    """Extract valid JSON from an LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def run(settlements: pd.DataFrame, orders: pd.DataFrame,
        router: RoutedLLMBackend,
        total_settlements: int, total_orders: int,
        already_matched: int) -> pd.DataFrame:
    """Run Tier 4 AI categorization on all remaining unmatched records.

    Args:
        settlements: Unmatched settlements from Tier 3.
        orders: Unmatched orders from Tier 3.
        router: Initialized LLM router.
        total_settlements: Total settlements in the full dataset (for context).
        total_orders: Total orders in the full dataset (for context).
        already_matched: How many records were already matched by Tier 1-3.

    Returns:
        DataFrame of categorized unmatched records in unified result schema.
    """
    results = []
    remaining_count = len(settlements) + len(orders)
    ai_calls = 0
    ai_failures = 0

    # --- Categorize unmatched settlements ---
    for _, stl_row in settlements.iterrows():
        messages = build_tier4_prompt(
            record=stl_row.to_dict(),
            record_type="settlement",
            total_settlements=total_settlements,
            total_orders=total_orders,
            already_matched=already_matched,
            remaining_count=remaining_count,
        )
        ai_calls += 1

        result = router.route(messages)
        if result is not None:
            parsed = _parse_json_response(result.text)
        else:
            parsed = None

        if parsed is not None:
            category = parsed.get("category", "orphan_settlement")
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
            explanation = parsed.get("explanation", "AI-categorized unmatched settlement")
            suggested_action = parsed.get("suggested_action",
                                          "Investigate this settlement in the gateway dashboard")
        else:
            # Graceful degradation -- all providers failed or returned invalid JSON
            ai_failures += 1
            category = "orphan_settlement"
            confidence = 0.0
            explanation = "Flagged for manual review (AI classification unavailable)"
            suggested_action = ("All AI providers failed for this record. "
                                "Check settlement in gateway dashboard manually.")
            if result is not None:
                log.warning("Tier 4: invalid JSON from %s for %s: %s",
                            result.model.name, stl_row["settlement_id"],
                            result.text[:200])
            else:
                log.warning("Tier 4: all providers failed for %s",
                            stl_row["settlement_id"])

        results.append({
            "order_id": stl_row.get("order_ref"),  # Best guess at the order ID
            "settlement_id": stl_row["settlement_id"],
            "tier": 4,
            "match_status": "unresolved",
            "confidence": confidence,
            "category": category,
            "explanation": (
                f"[AI/{result.model.provider if result else 'none'}] {explanation}"
            ),
            "suggested_action": suggested_action,
        })

    # --- Categorize unmatched orders ---
    for _, ord_row in orders.iterrows():
        messages = build_tier4_prompt(
            record=ord_row.to_dict(),
            record_type="order",
            total_settlements=total_settlements,
            total_orders=total_orders,
            already_matched=already_matched,
            remaining_count=remaining_count,
        )
        ai_calls += 1

        result = router.route(messages)
        if result is not None:
            parsed = _parse_json_response(result.text)
        else:
            parsed = None

        if parsed is not None:
            category = parsed.get("category", "orphan_order")
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
            explanation = parsed.get("explanation", "AI-categorized unmatched order")
            suggested_action = parsed.get("suggested_action",
                                          "Check if this order was paid through another channel")
        else:
            ai_failures += 1
            category = "orphan_order"
            confidence = 0.0
            explanation = "Flagged for manual review (AI classification unavailable)"
            suggested_action = ("All AI providers failed for this record. "
                                "Check order payment status manually.")
            if result is not None:
                log.warning("Tier 4: invalid JSON from %s for %s: %s",
                            result.model.name, ord_row["order_id"],
                            result.text[:200])
            else:
                log.warning("Tier 4: all providers failed for %s",
                            ord_row["order_id"])

        results.append({
            "order_id": ord_row["order_id"],
            "settlement_id": None,
            "tier": 4,
            "match_status": "unresolved",
            "confidence": confidence,
            "category": category,
            "explanation": (
                f"[AI/{result.model.provider if result else 'none'}] {explanation}"
            ),
            "suggested_action": suggested_action,
        })

    results_df = pd.DataFrame(results)
    log.info("Tier 4: %d AI calls, %d failures", ai_calls, ai_failures)
    print(f"  Tier 4: {ai_calls} AI calls, {ai_failures} failures (gracefully degraded)")

    return results_df
