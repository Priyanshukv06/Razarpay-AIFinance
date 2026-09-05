"""Q&A-specific prompts for the conversational reconciliation assistant.

The Q&A agent answers questions about reconciliation results using the
structured result set as context. No vector DB needed -- the dataset is
small enough to pass directly.

Prompt design:
  - System prompt establishes the assistant's role and constraints
  - Context includes relevant records (filtered via lexical search)
  - Conversation history (last N turns) for continuity
  - Response is plain text (conversational), NOT JSON
"""

SYSTEM_PROMPT = """You are a financial reconciliation assistant helping a finance analyst understand their reconciliation results. You have access to the full reconciliation output from an AI Finance Controller that matched a merchant's settlement records against their order ledger.

RULES:
- Answer based ONLY on the data provided in the context. Never fabricate records or numbers.
- Be concise and direct. Finance analysts want facts, not filler.
- When referring to specific records, always include the order_id or settlement_id.
- If the data doesn't contain enough information to answer, say so clearly.
- Use plain language. Avoid jargon unless the user used it first.
- When asked about amounts, include the currency (INR).
- If asked about a specific record that isn't in the context, say you don't have data for that specific ID."""


QA_PROMPT_TEMPLATE = """{system}

RECONCILIATION SUMMARY:
  Total settlements: {total_settlements}
  Total orders: {total_orders}
  Total results: {total_results}
  Match rate: {match_rate:.1f}%
  Value reconciled: INR {value_reconciled:,.2f} ({value_reconciled_pct:.1f}%)
  Deterministic matches (Tier 1+2): {tier12_count}
  AI-resolved matches (Tier 3): {tier3_count}
  Unresolved exceptions (Tier 4): {tier4_count}

CATEGORY BREAKDOWN:
{category_breakdown}

{relevant_records_section}

{conversation_history_section}

USER QUESTION: {question}

Answer the question based on the data above. Be concise and specific."""


def build_qa_prompt(question: str,
                    summary_stats: dict,
                    relevant_records: str,
                    conversation_history: list[dict] = None) -> list[dict]:
    """Build the messages list for a Q&A call.

    Args:
        question: The user's question text.
        summary_stats: Dict with total_settlements, total_orders, etc.
        relevant_records: Pre-formatted string of relevant records.
        conversation_history: List of {role, content} dicts for context.

    Returns:
        Messages list for the LLM router.
    """
    # Format conversation history
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        for turn in conversation_history[-10:]:  # Cap at last 10 turns
            role = "Analyst" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"  {role}: {turn['content']}")
        conv_section = "PREVIOUS CONVERSATION:\n" + "\n".join(history_lines)
    else:
        conv_section = ""

    # Format relevant records section
    if relevant_records:
        records_section = f"RELEVANT RECORDS:\n{relevant_records}"
    else:
        records_section = "No specific records matched the query."

    content = QA_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        total_settlements=summary_stats.get("total_settlements", 0),
        total_orders=summary_stats.get("total_orders", 0),
        total_results=summary_stats.get("total_results", 0),
        match_rate=summary_stats.get("match_rate", 0),
        value_reconciled=summary_stats.get("value_reconciled", 0),
        value_reconciled_pct=summary_stats.get("value_reconciled_pct", 0),
        tier12_count=summary_stats.get("tier12_count", 0),
        tier3_count=summary_stats.get("tier3_count", 0),
        tier4_count=summary_stats.get("tier4_count", 0),
        category_breakdown=summary_stats.get("category_breakdown", "N/A"),
        relevant_records_section=records_section,
        conversation_history_section=conv_section,
        question=question,
    )

    return [{"role": "user", "content": content}]
