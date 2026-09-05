"""Supabase caching layer for pipeline results and Q&A answers.

Caches pipeline results and Q&A responses by dataset hash (SHA256 of the
input CSV data). This avoids re-running the pipeline or making LLM calls
for repeated queries on the same data.

Cache behavior:
  - On dataset load: compute hash → check Supabase → if found, load cached
  - On pipeline run: store results + metadata in pipeline_cache
  - On Q&A question: check qa_cache → if found, return cached answer
  - On restart: delete all cached data for this dataset hash

All Supabase calls are wrapped in try/except so the app works without
caching if the connection fails (graceful degradation).
"""

import hashlib
import json
import logging
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase client (lazy singleton)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Lazy-initialize Supabase client."""
    global _client
    if _client is None:
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            log.warning("Supabase credentials not found in .env -- caching disabled")
            return None
        try:
            from supabase import create_client
            _client = create_client(url, key)
            log.info("Supabase client initialized: %s", url)
        except Exception as e:
            log.warning("Failed to initialize Supabase: %s", e)
            return None
    return _client


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------
def compute_dataset_hash(settlements_csv: str, orders_csv: str) -> str:
    """Compute SHA256 hash of settlement + order CSV content.

    Args:
        settlements_csv: CSV string of settlement data.
        orders_csv: CSV string of order data.

    Returns:
        Hex digest of the combined hash.
    """
    combined = settlements_csv + "|||" + orders_csv
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def compute_df_hash(settlements_df: pd.DataFrame, orders_df: pd.DataFrame) -> str:
    """Compute hash from DataFrames (convenience wrapper)."""
    return compute_dataset_hash(
        settlements_df.to_csv(index=False),
        orders_df.to_csv(index=False),
    )


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------
def get_cached_results(dataset_hash: str) -> dict | None:
    """Check if pipeline results exist in cache.

    Returns:
        Dict with 'results_json' and 'metadata_json' if found, None otherwise.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.table("pipeline_cache") \
            .select("results_json, metadata_json, created_at") \
            .eq("dataset_hash", dataset_hash) \
            .limit(1) \
            .execute()
        if response.data and len(response.data) > 0:
            log.info("Cache hit for dataset %s...", dataset_hash[:12])
            return response.data[0]
        return None
    except Exception as e:
        log.warning("Cache lookup failed: %s", e)
        return None


def store_results(dataset_hash: str, results_df: pd.DataFrame,
                  metadata: dict) -> bool:
    """Store pipeline results in Supabase cache.

    Args:
        dataset_hash: SHA256 of the input data.
        results_df: Pipeline results DataFrame.
        metadata: Dict with match_rate, tier_counts, timing, etc.

    Returns:
        True if stored successfully, False otherwise.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        results_json = json.loads(results_df.to_json(orient="records"))
        client.table("pipeline_cache").insert({
            "dataset_hash": dataset_hash,
            "results_json": results_json,
            "metadata_json": metadata,
        }).execute()
        log.info("Cached results for dataset %s...", dataset_hash[:12])
        return True
    except Exception as e:
        log.warning("Failed to cache results: %s", e)
        return False


def delete_cached_results(dataset_hash: str) -> bool:
    """Delete cached results for a dataset (used on pipeline restart).

    Also deletes associated Q&A cache entries.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("qa_cache").delete() \
            .eq("dataset_hash", dataset_hash).execute()
        client.table("pipeline_cache").delete() \
            .eq("dataset_hash", dataset_hash).execute()
        log.info("Deleted cache for dataset %s...", dataset_hash[:12])
        return True
    except Exception as e:
        log.warning("Failed to delete cache: %s", e)
        return False


# ---------------------------------------------------------------------------
# Q&A cache
# ---------------------------------------------------------------------------
def get_cached_answer(dataset_hash: str, question: str) -> str | None:
    """Check if a Q&A answer exists in cache for this dataset + question.

    Returns the cached answer string, or None if not found.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        # Normalize question for matching
        normalized = question.strip().lower()
        response = client.table("qa_cache") \
            .select("answer") \
            .eq("dataset_hash", dataset_hash) \
            .eq("question", normalized) \
            .limit(1) \
            .execute()
        if response.data and len(response.data) > 0:
            log.info("Q&A cache hit: %s...", question[:40])
            return response.data[0]["answer"]
        return None
    except Exception as e:
        log.warning("Q&A cache lookup failed: %s", e)
        return None


def store_answer(dataset_hash: str, question: str, answer: str) -> bool:
    """Store a Q&A answer in Supabase cache."""
    client = _get_client()
    if client is None:
        return False
    try:
        normalized = question.strip().lower()
        client.table("qa_cache").insert({
            "dataset_hash": dataset_hash,
            "question": normalized,
            "answer": answer,
        }).execute()
        log.info("Cached Q&A for: %s...", question[:40])
        return True
    except Exception as e:
        log.warning("Failed to cache Q&A: %s", e)
        return False
