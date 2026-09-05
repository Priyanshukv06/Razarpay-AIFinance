"""Reconciliation-specific LLM prompts with strict JSON schema enforcement.

Two prompt types:
  1. TIER3_CONFIRM: Settlement has a fuzzy candidate order -- confirm or reject
  2. TIER4_CATEGORIZE: No candidate found -- categorize the orphan/anomaly

Both enforce the SAME JSON output schema so the pipeline can parse uniformly
regardless of which provider answered.
"""

# The JSON schema is embedded in the prompt itself (not via tool-use) because
# not every free provider/model supports native tool-use reliably. This keeps
# the output contract identical regardless of which provider in the chain
# actually answers the call. (PRD Section 11.3)

SYSTEM_PROMPT = """You are a financial reconciliation analyst AI. Your job is to analyze payment settlement records and order ledger entries to determine if they match, and to categorize exceptions.

CRITICAL RULES:
- Return ONLY valid JSON. No markdown, no explanation text outside the JSON.
- Do not wrap the JSON in code blocks or backticks.
- Use exactly the field names specified.
- Be conservative with confidence scores -- only use high confidence when the evidence is strong.
- Never fabricate data or hallucinate order/settlement details not present in the input."""


TIER3_CONFIRM_TEMPLATE = """{system}

TASK: A settlement record has been paired with a candidate order record through fuzzy matching. Analyze whether this is a genuine match and explain why.

SETTLEMENT RECORD:
  settlement_id: {settlement_id}
  order_ref: {order_ref}
  amount_settled: {amount_settled}
  fee: {fee}
  tax: {tax}
  settlement_date: {settlement_date}
  status: {stl_status}

CANDIDATE ORDER RECORD:
  order_id: {order_id}
  customer: {customer}
  order_amount: {order_amount}
  order_date: {order_date}
  payment_method: {payment_method}
  status: {ord_status}

AMOUNT ANALYSIS:
  Amount difference: {amount_diff}
  Fee-adjusted expected: {fee_adjusted_expected}
  Fee-adjusted difference: {fee_adjusted_diff}

Return ONLY this JSON (no other text):
{{
  "match_confirmed": true or false,
  "category": one of "partial_refund" | "duplicate" | "split_settlement" | "fee_adjusted" | "date_drift" | "other",
  "confidence": float between 0.0 and 1.0,
  "explanation": "human-readable reason for the match/non-match decision",
  "suggested_action": "what a human analyst should check or do next"
}}"""


TIER4_CATEGORIZE_TEMPLATE = """{system}

TASK: This record could not be matched to any counterpart. Categorize it and suggest what a human should do.

{record_type} RECORD:
{record_details}

CONTEXT (summary of the full dataset):
  Total settlements: {total_settlements}
  Total orders: {total_orders}
  Already matched: {already_matched}
  This is one of {remaining_count} unmatched records.

Return ONLY this JSON (no other text):
{{
  "category": one of "orphan_settlement" | "orphan_order" | "duplicate" | "split_settlement" | "partial_refund" | "other",
  "confidence": float between 0.0 and 1.0,
  "explanation": "human-readable explanation of why this record has no match",
  "suggested_action": "specific next step for a human analyst"
}}"""


def build_tier3_prompt(settlement: dict, order: dict) -> list[dict]:
    """Build the messages list for a Tier 3 confirmation call."""
    amount_diff = round(abs(settlement["amount_settled"] - order["order_amount"]), 2)
    fee_adjusted = round(order["order_amount"] - settlement["fee"] - settlement["tax"], 2)
    fee_adjusted_diff = round(abs(settlement["amount_settled"] - fee_adjusted), 2)

    content = TIER3_CONFIRM_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        settlement_id=settlement["settlement_id"],
        order_ref=settlement["order_ref"],
        amount_settled=settlement["amount_settled"],
        fee=settlement["fee"],
        tax=settlement["tax"],
        settlement_date=settlement["settlement_date"],
        stl_status=settlement["status"],
        order_id=order["order_id"],
        customer=order["customer"],
        order_amount=order["order_amount"],
        order_date=order["order_date"],
        payment_method=order["payment_method"],
        ord_status=order["status"],
        amount_diff=amount_diff,
        fee_adjusted_expected=fee_adjusted,
        fee_adjusted_diff=fee_adjusted_diff,
    )
    return [{"role": "user", "content": content}]


def build_tier4_prompt(record: dict, record_type: str,
                       total_settlements: int, total_orders: int,
                       already_matched: int, remaining_count: int) -> list[dict]:
    """Build the messages list for a Tier 4 categorization call."""
    # Format the record details as key-value pairs
    details_lines = []
    for key, value in record.items():
        details_lines.append(f"  {key}: {value}")
    record_details = "\n".join(details_lines)

    content = TIER4_CATEGORIZE_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        record_type=record_type.upper(),
        record_details=record_details,
        total_settlements=total_settlements,
        total_orders=total_orders,
        already_matched=already_matched,
        remaining_count=remaining_count,
    )
    return [{"role": "user", "content": content}]
