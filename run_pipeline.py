"""Run the full pipeline with logging enabled and unbuffered output."""
import logging
import sys

# Force unbuffered stdout so progress shows in real-time
sys.stdout.reconfigure(line_buffering=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from engine.pipeline import load_data, run_pipeline, validate_against_answer_key, print_summary

# Load data
settlements, orders = load_data()
print(f"Loaded {len(settlements)} settlements, {len(orders)} orders")

# Run full pipeline (with AI)
results, remaining_stl, remaining_ord = run_pipeline(settlements, orders, use_ai=True)

# Validate against answer key
metrics = validate_against_answer_key(results, remaining_stl, remaining_ord)

# Print summary
print_summary(results, remaining_stl, remaining_ord, metrics)

# Save results to CSV for inspection
results.to_csv("data/pipeline_results.csv", index=False)
print(f"\nResults saved to data/pipeline_results.csv ({len(results)} rows)")
