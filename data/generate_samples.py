"""Generate 4 sample datasets for judges to test the pipeline.

Each sample has a different distribution and seed, producing varied
challenge levels. No answer key is included (judges shouldn't see
ground truth).

Usage:
    conda activate razorpay
    python data/generate_samples.py
"""

import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate, DISTRIBUTION

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# Sample configs: name, record_count, seed, distribution overrides
SAMPLE_CONFIGS = [
    {
        "name": "sample_small_easy",
        "records": 30,
        "seed": 101,
        "distribution": {
            "perfect_match": 0.55,
            "fee_adjusted": 0.15,
            "date_drift": 0.10,
            "partial_refund": 0.05,
            "duplicate_order": 0.05,
            "orphan_settlement": 0.04,
            "orphan_order": 0.03,
            "split_settlement": 0.03,
        },
        "description": "Small, mostly easy — 70% deterministic. Quick demo (~30s with AI).",
    },
    {
        "name": "sample_medium_balanced",
        "records": 60,
        "seed": 202,
        "distribution": {
            "perfect_match": 0.35,
            "fee_adjusted": 0.15,
            "date_drift": 0.10,
            "partial_refund": 0.10,
            "duplicate_order": 0.08,
            "orphan_settlement": 0.08,
            "orphan_order": 0.07,
            "split_settlement": 0.07,
        },
        "description": "Medium, balanced mix — ~40% needs AI. Good overall test.",
    },
    {
        "name": "sample_large_hard",
        "records": 100,
        "seed": 303,
        "distribution": {
            "perfect_match": 0.25,
            "fee_adjusted": 0.10,
            "date_drift": 0.10,
            "partial_refund": 0.10,
            "duplicate_order": 0.12,
            "orphan_settlement": 0.12,
            "orphan_order": 0.11,
            "split_settlement": 0.10,
        },
        "description": "Large, hard — 55% needs AI. Stress test for the pipeline.",
    },
    {
        "name": "sample_edge_cases",
        "records": 40,
        "seed": 404,
        "distribution": {
            "perfect_match": 0.10,
            "fee_adjusted": 0.08,
            "date_drift": 0.07,
            "partial_refund": 0.15,
            "duplicate_order": 0.15,
            "orphan_settlement": 0.15,
            "orphan_order": 0.15,
            "split_settlement": 0.15,
        },
        "description": "Edge-case heavy — 75% ambiguous. Most records need AI.",
    },
]


def generate_samples():
    """Generate all 4 sample datasets."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for config in SAMPLE_CONFIGS:
        name = config["name"]
        sample_dir = SAMPLES_DIR / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Generating: {name}")
        print(f"  Records: {config['records']}, Seed: {config['seed']}")
        print(f"  {config['description']}")
        print(f"{'=' * 60}")

        # Temporarily override the module-level DISTRIBUTION and SEED
        import data.generate_data as gen_mod
        original_dist = gen_mod.DISTRIBUTION.copy()
        original_seed = gen_mod.SEED

        gen_mod.DISTRIBUTION = config["distribution"]
        gen_mod.SEED = config["seed"]

        # Reset random + faker seeds
        random.seed(config["seed"])
        gen_mod.fake.seed_instance(config["seed"])

        try:
            generate(
                record_count=config["records"],
                output_dir=str(sample_dir),
            )
        finally:
            # Restore originals
            gen_mod.DISTRIBUTION = original_dist
            gen_mod.SEED = original_seed

        # Remove answer key (judges shouldn't see ground truth)
        answer_key = sample_dir / "answer_key.csv"
        if answer_key.exists():
            answer_key.unlink()
            print(f"  Removed answer_key.csv (judges don't see ground truth)")

        # Write sample README
        readme = sample_dir / "README.md"
        readme.write_text(
            f"# {name.replace('_', ' ').title()}\n\n"
            f"{config['description']}\n\n"
            f"- **Records**: {config['records']}\n"
            f"- **Seed**: {config['seed']}\n"
            f"- **Distribution**: {', '.join(f'{k}={v:.0%}' for k, v in config['distribution'].items())}\n\n"
            f"## Files\n"
            f"- `settlement_report.csv` — Settlement records from gateway\n"
            f"- `order_ledger.csv` — Merchant order records\n\n"
            f"## Usage\n"
            f"Upload both CSV files in the dashboard's **Upload & Reconcile** tab.\n",
            encoding="utf-8",
        )
        print(f"  Created README.md")

    print(f"\n{'=' * 60}")
    print(f"  All {len(SAMPLE_CONFIGS)} samples generated in: {SAMPLES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    generate_samples()
