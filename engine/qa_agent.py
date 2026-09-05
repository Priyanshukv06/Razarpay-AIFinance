"""Conversational Q&A agent over reconciliation results.

Combines lexical (sparse) search for keyword/ID matching with semantic
understanding via the LLM for natural language questions.

Architecture:
  1. Lexical extraction: regex for order_ids, settlement_ids, payment_ids
  2. Keyword mapping: terms like "orphan", "refund" → category filters
  3. Data filtering: build relevant context from extracted matches
  4. LLM call: generate conversational answer using filtered context
  5. Fallback: if LLM fails, return data-driven response without AI

Usage:
    from engine.qa_agent import ReconciliationQA
    from engine.llm.router import RoutedLLMBackend

    qa = ReconciliationQA(results_df, settlements_df, orders_df)
    answer = qa.ask("Why wasn't ORD1079 matched?")
"""

import logging
import re

import pandas as pd

from engine.llm.qa_prompts import build_qa_prompt
from engine.llm.router import RoutedLLMBackend
from engine import cache as supabase_cache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexical search patterns
# ---------------------------------------------------------------------------

# Regex patterns for ID extraction from questions
ID_PATTERNS = {
    "order_id": re.compile(r"\b(ORD\d+)\b", re.IGNORECASE),
    "settlement_id": re.compile(r"\b(STL\d+)\b", re.IGNORECASE),
    "payment_id": re.compile(r"\b(pay_\w+)\b", re.IGNORECASE),
}

# Keyword → category/filter mapping for lexical matching
# Maps natural language terms to DataFrame filter values
KEYWORD_FILTERS = {
    # Category keywords
    "orphan": {"column": "category", "values": ["orphan_settlement", "orphan_order"]},
    "orphan settlement": {"column": "category", "values": ["orphan_settlement"]},
    "orphan order": {"column": "category", "values": ["orphan_order"]},
    "duplicate": {"column": "category", "values": ["duplicate", "duplicate_order"]},
    "refund": {"column": "category", "values": ["partial_refund"]},
    "partial refund": {"column": "category", "values": ["partial_refund"]},
    "split": {"column": "category", "values": ["split_settlement"]},
    "split settlement": {"column": "category", "values": ["split_settlement"]},
    "fee": {"column": "category", "values": ["fee_adjusted"]},
    "date drift": {"column": "category", "values": ["date_drift"]},
    "exact match": {"column": "category", "values": ["exact"]},
    # Status keywords
    "unresolved": {"column": "match_status", "values": ["unresolved"]},
    "matched": {"column": "match_status", "values": ["matched", "ai_resolved"]},
    "exception": {"column": "match_status", "values": ["unresolved"]},
    # Tier keywords
    "tier 1": {"column": "tier", "values": [1]},
    "tier 2": {"column": "tier", "values": [2]},
    "tier 3": {"column": "tier", "values": [3]},
    "tier 4": {"column": "tier", "values": [4]},
    "ai": {"column": "tier", "values": [3, 4]},
    "deterministic": {"column": "tier", "values": [1, 2]},
    # Payment method keywords
    "upi": {"column": "payment_method", "values": ["UPI"]},
    "card": {"column": "payment_method", "values": ["card"]},
    "netbanking": {"column": "payment_method", "values": ["netbanking"]},
    "wallet": {"column": "payment_method", "values": ["wallet"]},
    # Confidence keywords
    "low confidence": {"column": "confidence", "filter_type": "range", "min": 0.0, "max": 0.7},
    "high confidence": {"column": "confidence", "filter_type": "range", "min": 0.9, "max": 1.0},
    "uncertain": {"column": "confidence", "filter_type": "range", "min": 0.0, "max": 0.75},
}


class ReconciliationQA:
    """Q&A agent that answers questions about reconciliation results.

    Uses lexical search to find relevant records, then the LLM to
    generate a conversational response.
    """

    def __init__(self, results: pd.DataFrame,
                 settlements: pd.DataFrame,
                 orders: pd.DataFrame,
                 router: RoutedLLMBackend = None,
                 dataset_hash: str = None):
        """Initialize with data context.

        Args:
            results: Pipeline results DataFrame.
            settlements: Settlement report DataFrame.
            orders: Order ledger DataFrame.
            router: LLM router instance. If None, will be created lazily.
            dataset_hash: Optional hash for Supabase Q&A caching.
        """
        self.results = results
        self.settlements = settlements
        self.orders = orders
        self._router = router
        self._dataset_hash = dataset_hash
        self._summary_stats = self._compute_summary()

    @property
    def router(self) -> RoutedLLMBackend:
        """Lazy-initialize the LLM router on first Q&A call."""
        if self._router is None:
            self._router = RoutedLLMBackend()
        return self._router

    def _compute_summary(self) -> dict:
        """Pre-compute summary stats for prompt context."""
        matched = self.results[
            self.results["match_status"].isin(["matched", "ai_resolved"])
        ]
        matched_stl_ids = matched["settlement_id"].dropna()
        value_reconciled = self.settlements[
            self.settlements["settlement_id"].isin(matched_stl_ids)
        ]["amount_settled"].sum()
        total_value = self.settlements["amount_settled"].sum()

        # Category breakdown
        cat_counts = self.results["category"].value_counts()
        cat_lines = [f"  {cat}: {count}" for cat, count in cat_counts.items()]

        return {
            "total_settlements": len(self.settlements),
            "total_orders": len(self.orders),
            "total_results": len(self.results),
            "match_rate": len(matched) / len(self.settlements) * 100
                if len(self.settlements) > 0 else 0,
            "value_reconciled": value_reconciled,
            "value_reconciled_pct": (value_reconciled / total_value * 100)
                if total_value > 0 else 0,
            "tier12_count": len(self.results[self.results["tier"].isin([1, 2])]),
            "tier3_count": len(self.results[self.results["tier"] == 3]),
            "tier4_count": len(self.results[self.results["tier"] == 4]),
            "category_breakdown": "\n".join(cat_lines),
        }

    def _extract_ids(self, question: str) -> dict[str, list[str]]:
        """Extract order_ids, settlement_ids, payment_ids from question text.

        Returns dict of {id_type: [matched_values]}.
        """
        found = {}
        for id_type, pattern in ID_PATTERNS.items():
            matches = pattern.findall(question)
            if matches:
                # Normalize: order_ids to uppercase, payment_ids as-is
                if id_type in ("order_id", "settlement_id"):
                    matches = [m.upper() for m in matches]
                found[id_type] = matches
        return found

    def _extract_keywords(self, question: str) -> list[dict]:
        """Map keywords in the question to data filters.

        Returns list of filter dicts that matched.
        Checks longer phrases first to avoid partial matches.
        """
        question_lower = question.lower()
        matched_filters = []

        # Sort by length descending so "orphan settlement" matches before "orphan"
        sorted_keywords = sorted(KEYWORD_FILTERS.keys(), key=len, reverse=True)

        matched_terms = set()
        for keyword in sorted_keywords:
            if keyword in question_lower:
                # Skip if a longer keyword containing this one already matched
                already_covered = any(
                    keyword in longer and longer != keyword
                    for longer in matched_terms
                )
                if not already_covered:
                    matched_filters.append(KEYWORD_FILTERS[keyword])
                    matched_terms.add(keyword)

        return matched_filters

    def _filter_records(self, ids: dict, keyword_filters: list) -> pd.DataFrame:
        """Filter results based on extracted IDs and keywords.

        Returns a DataFrame of relevant records.
        """
        relevant = pd.DataFrame()

        # Filter by extracted IDs
        if "order_id" in ids:
            id_match = self.results[
                self.results["order_id"].isin(ids["order_id"])
            ]
            relevant = pd.concat([relevant, id_match])

        if "settlement_id" in ids:
            id_match = self.results[
                self.results["settlement_id"].isin(ids["settlement_id"])
            ]
            relevant = pd.concat([relevant, id_match])

        # Filter by keyword-matched categories/statuses
        for filt in keyword_filters:
            col = filt["column"]
            if col not in self.results.columns:
                # Payment method is in orders, not results -- skip
                continue
            if filt.get("filter_type") == "range":
                match = self.results[
                    (self.results[col] >= filt["min"]) &
                    (self.results[col] <= filt["max"])
                ]
            else:
                match = self.results[self.results[col].isin(filt["values"])]
            relevant = pd.concat([relevant, match])

        # Deduplicate
        if not relevant.empty:
            relevant = relevant.drop_duplicates()

        return relevant

    def _format_records(self, records: pd.DataFrame, max_records: int = 15) -> str:
        """Format records into a readable string for the LLM prompt."""
        if records.empty:
            return ""

        # Limit to prevent token overflow
        if len(records) > max_records:
            records = records.head(max_records)
            truncated = True
        else:
            truncated = False

        lines = []
        for _, row in records.iterrows():
            line_parts = []
            if pd.notna(row.get("settlement_id")):
                line_parts.append(f"STL: {row['settlement_id']}")
            if pd.notna(row.get("order_id")):
                line_parts.append(f"ORD: {row['order_id']}")
            line_parts.append(f"Tier: {row['tier']}")
            line_parts.append(f"Status: {row['match_status']}")
            line_parts.append(f"Category: {row['category']}")
            line_parts.append(f"Confidence: {row['confidence']:.2f}")
            if pd.notna(row.get("explanation")):
                line_parts.append(f"Explanation: {row['explanation']}")
            if pd.notna(row.get("suggested_action")):
                line_parts.append(f"Action: {row['suggested_action']}")
            lines.append("  " + " | ".join(line_parts))

        result = "\n".join(lines)
        if truncated:
            result += f"\n  ... ({len(records)} of {len(records)} shown, more available)"

        return result

    def _build_context(self, question: str) -> str:
        """Build relevant context for the LLM prompt.

        Combines lexical ID extraction, keyword filtering, and
        smart defaults for general questions.
        """
        ids = self._extract_ids(question)
        keyword_filters = self._extract_keywords(question)

        # Filter records based on what we found
        relevant = self._filter_records(ids, keyword_filters)

        if relevant.empty and not ids and not keyword_filters:
            # General question -- provide a sample of interesting records
            # (low confidence, unresolved, or AI-resolved)
            interesting = self.results[
                (self.results["confidence"] < 0.9) |
                (self.results["match_status"] == "unresolved")
            ].head(10)
            if interesting.empty:
                interesting = self.results.head(10)
            return self._format_records(interesting, max_records=10)

        return self._format_records(relevant)

    def _fallback_response(self, question: str, context: str) -> str:
        """Generate a data-driven response when LLM is unavailable."""
        ids = self._extract_ids(question)
        keyword_filters = self._extract_keywords(question)
        relevant = self._filter_records(ids, keyword_filters)

        response = "I couldn't connect to the AI service, but here's what I found in the data:\n\n"

        if not relevant.empty:
            response += f"Found {len(relevant)} matching records:\n"
            for _, row in relevant.head(5).iterrows():
                response += (
                    f"- {row.get('settlement_id', 'N/A')} / "
                    f"{row.get('order_id', 'N/A')}: "
                    f"{row['category']} (Tier {row['tier']}, "
                    f"confidence {row['confidence']:.2f})\n"
                )
            if len(relevant) > 5:
                response += f"  ... and {len(relevant) - 5} more records.\n"
        else:
            stats = self._summary_stats
            response += (
                f"Overall: {stats['total_results']} results, "
                f"{stats['match_rate']:.1f}% match rate, "
                f"INR {stats['value_reconciled']:,.2f} reconciled.\n"
                f"Tier 1+2: {stats['tier12_count']} | "
                f"Tier 3: {stats['tier3_count']} | "
                f"Tier 4: {stats['tier4_count']}\n"
            )

        return response

    def ask(self, question: str,
            conversation_history: list[dict] = None) -> str:
        """Process a question and return a conversational answer.

        Args:
            question: The user's question text.
            conversation_history: Previous Q&A turns for context.

        Returns:
            String answer (plain text, not JSON).
        """
        # Step 0: Check Supabase cache
        if self._dataset_hash:
            cached_answer = supabase_cache.get_cached_answer(
                self._dataset_hash, question
            )
            if cached_answer:
                log.info("Q&A cache hit for: %s...", question[:40])
                return cached_answer

        # Step 1: Build context via lexical extraction + filtering
        context = self._build_context(question)

        # Step 2: Build LLM prompt with context + conversation history
        messages = build_qa_prompt(
            question=question,
            summary_stats=self._summary_stats,
            relevant_records=context,
            conversation_history=conversation_history,
        )

        # Step 3: Route to LLM
        try:
            result = self.router.route(messages)
        except Exception as e:
            log.warning("Q&A LLM call failed: %s", e)
            result = None

        if result is not None and result.text:
            answer = result.text.strip()
            log.info("Q&A answered by %s (%s)", result.model.name, result.model.provider)
            # Cache the answer
            if self._dataset_hash:
                supabase_cache.store_answer(
                    self._dataset_hash, question, answer
                )
            return answer
        else:
            # Graceful degradation -- answer from data alone
            log.warning("Q&A: all providers failed, using data fallback")
            return self._fallback_response(question, context)
