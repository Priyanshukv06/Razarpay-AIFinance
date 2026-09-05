# AI Finance Controller

**Razorpay Buildathon 2026 — Track 04: AI Finance Controller**

A tiered reconciliation engine that matches a merchant's payment settlement records against their internal order ledger. It resolves as much as possible deterministically (fast, free, 100% reproducible), and uses AI — served through a **five-provider free-tier LLM chain** — only for the genuinely ambiguous remainder.

> **100% accuracy** on all 86 scorable records. **Zero false positives.** **Zero API cost.** Verified by automated test suite (9 suites, 27 checks, all passing).

![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Accuracy & Honesty](#accuracy--honesty)
- [What Broke & How We Fixed It](#what-broke--how-we-fixed-it)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Future Roadmap](#future-roadmap)

---

## Problem

Every merchant that processes payments through a gateway has **two records of the same money**:

1. **Settlement Report** (Source A) — what the gateway settled (net of fees, on its own timeline)
2. **Order Ledger** (Source B) — what the merchant's own system recorded (gross amounts, on order-creation time)

These records drift apart constantly for mundane, extremely common reasons:

| Mismatch Type | Example |
|---|---|
| Fee/tax deduction | ₹1,500 order → ₹1,462.30 settled (after fee + GST) |
| Date drift | Order on Sept 3, settlement on Sept 5 (T+2) |
| Partial refund | ₹2,000 order, ₹500 refunded, ₹1,500 settled |
| Duplicate order | Same order_id appears twice in the ledger |
| Orphan settlement | Settlement with no matching order in the ledger |
| Orphan order | Order with no corresponding settlement (pending/abandoned) |
| Split settlement | One order split across two settlement UTRs |
| Missing order | Settlement references an order_id that doesn't exist |

Today this reconciliation is done **manually in spreadsheets** by finance teams — slow, error-prone, and with no consistent audit trail.

---

## Solution

A **four-tier matching engine** that resolves records from most-certain to least-certain:

| Tier | Method | What It Does | Confidence |
|---|---|---|---|
| **Tier 1** | Exact Match | `order_ref == order_id` AND `amount_settled == order_amount` | 1.00 |
| **Tier 2** | Rule-Based | ID match + fee/tax-adjusted amount within tolerance | 0.98 |
| **Tier 3** | AI Fuzzy | Amount/date tolerance window → LLM confirms or rejects | 0.60–0.95 |
| **Tier 4** | AI Categorize | No candidate found → LLM categorizes the likely cause | 0.50–0.90 |

**Key design principle:** AI is used **only** where deterministic logic genuinely cannot decide. 65% of records are resolved with zero AI calls — pure, reproducible logic. The LLM layer only sees the genuinely ambiguous 35%.

---

## Key Results

| Metric | Value |
|---|---|
| Total records processed | 100 settlements + 99 orders |
| Deterministic resolution (Tier 1+2) | 65% of volume, 100% precision |
| AI-assisted matches (Tier 3) | 15 fuzzy matches confirmed |
| AI categorizations (Tier 4) | 39 records categorized |
| **Overall accuracy** | **86/86 scorable = 100.0%** |
| False positives | **0** across all tiers |
| Coverage | 86% of answer key addressed |
| Value reconciled | ₹5,75,147 of ₹6,95,259 (82.7%) |
| Mismatch types covered | All 8 types |
| Pipeline time (full) | ~190 seconds |
| Pipeline time (deterministic only) | ~0.08 seconds |
| API cost | **₹0** (all free-tier providers) |

---

## Architecture

```
Source A (Settlement Report)  ─┐
                                ├─→ Tier 1: Exact Match ─────────→ Matched (confidence 1.00)
Source B (Order Ledger)       ─┘         │
                                          ▼
                                Tier 2: Rule-Based Match ────────→ Matched (fee/date explained)
                                          │
                                          ▼
                                Tier 3: Fuzzy Candidate Gen ─────→ LLM confirms / rejects / scores
                                          │
                                          ▼
                                Tier 4: No Candidate Found ──────→ LLM categorizes / suggests action
                                          │
                                          ▼
                                Unified Result Set ──────────────→ Streamlit Dashboard
                                                                    ├─→ PDF / CSV Export
                                                                    ├─→ "Ask the Ledger" Q&A
                                                                    └─→ Upload & Reconcile
```

### Five-Provider LLM Failover Chain

All AI calls route through a **single OpenAI-compatible client** with automatic failover across five free-tier providers:

| Priority | Provider | Free Tier | Role |
|---|---|---|---|
| 1 | **Groq** | ~30 RPM, LPU hardware | Primary — fastest inference |
| 2 | **Google Gemini** | 500 req/day (Flash) | Fallback — best daily volume |
| 3 | **Mistral** | ~1B tokens/month | Fallback — EU-hosted diversity |
| 4 | **NVIDIA NIM** | 1,000 free credits | Reserve — credit-based |
| 5 | **OpenRouter** | 20 RPM, multiple free models | Last resort — meta-aggregator |

**10 models** across 5 providers. If one provider is rate-limited or down, the router automatically tries the next. Zero single point of failure.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Data generation | `faker` |
| Matching engine | `pandas`, `rapidfuzz` |
| AI layer | OpenAI-compatible client (`openai` SDK) → 5 free-tier providers |
| Dashboard | Streamlit |
| Charts | Plotly |
| PDF export | `fpdf2` |
| Caching | Supabase (optional) |

---

## Setup

### Prerequisites

- Python 3.13+ (recommended: [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- At least one API key from any of the five supported providers (all free, no credit card required)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ai-finance-controller.git
cd ai-finance-controller
```

### 2. Create and activate the environment

```bash
conda create -n razorpay python=3.13 -y
conda activate razorpay
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
# Windows
Copy-Item .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and paste your API keys. Each provider accepts comma-separated keys for quota spreading:

```env
# Get keys (all free, no card needed):
# Groq:       https://console.groq.com/keys
# Gemini:     https://aistudio.google.com/apikey
# OpenRouter:  https://openrouter.ai/keys
# Mistral:    https://console.mistral.ai/api-keys
# NVIDIA:     https://build.nvidia.com → "Get API Key"

GROQ_API_KEYS=gsk_your_key_1,gsk_your_key_2
GEMINI_API_KEYS=your_gemini_key
OPENROUTER_API_KEYS=sk-or-your_key
MISTRAL_API_KEYS=your_mistral_key
NVIDIA_API_KEYS=nvapi-your_key
```

**Optional** — For pipeline result caching via Supabase:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
```

> **Note:** The app works fully without Supabase — caching is disabled gracefully when these keys are absent.

### 5. Generate synthetic data (optional — already included)

The repository ships with pre-generated data (`data/settlement_report.csv`, `data/order_ledger.csv`, `data/answer_key.csv`). To regenerate:

```bash
python -m data.generate_data
```

This produces 100 settlement records + 99 order ledger records with all 8 mismatch types seeded at known ratios (seed=42 for reproducibility).

---

## Usage

### Launch the Dashboard

```bash
conda activate razorpay
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` with 5 interactive tabs.

### Run the Pipeline (CLI)

```bash
# Full pipeline (all 4 tiers, with AI)
python run_pipeline.py

# Full pipeline + answer key validation
python -m engine.pipeline --validate

# Deterministic only (Tier 1+2, no AI calls, ~0.08s)
python -m engine.pipeline --no-ai

# Deterministic + validation
python -m engine.pipeline --no-ai --validate
```

### Quick LLM Router Test

```bash
python test_router.py
```

Sends a single test call through the LLM chain to verify API keys and connectivity.

---

## Dashboard

The Streamlit dashboard has **5 tabs**:

### Tab 1 — Summary
- **KPI cards:** Total Settlements, Match Rate, Value Reconciled, Exceptions, False Positives
- **Tier waterfall chart:** visual breakdown of how records flow through the 4 tiers
- **Resolution donut:** matched vs. AI-resolved vs. unresolved
- **Value reconciliation bar:** ₹ matched vs. ₹ in exception

### Tab 2 — Exception Table
- Filterable, sortable table of all reconciliation results
- **Color-coded by confidence** (low-confidence AI calls visually stand out)
- **CSV download** + **PDF export** (professional audit-ready report)

### Tab 3 — Category Analysis
- Category distribution pie chart
- Tier-colored stacked bar chart
- Confidence histogram

### Tab 4 — Ask the Ledger (Q&A)
- Natural language questions about reconciliation results
- Lexical search: regex ID extraction (`ORD`/`STL`/`pay_` patterns) + keyword→category mapping
- Semantic search: LLM generates conversational answers from filtered context
- Conversation memory (last 10 turns) for follow-up questions
- Example question chips for quick exploration
- Graceful fallback: data-driven response if all LLM providers fail

### Tab 5 — Upload & Reconcile
- **4 pre-built sample datasets** of varying difficulty (30–100 records, 25–90% deterministic)
- **Custom CSV upload** with schema validation
- Step-by-step pipeline: Load → Tier 1+2 → Tier 3+4 → Results
- **Skip AI option:** run deterministic tiers only, download results without LLM calls
- **Supabase caching:** repeat runs on the same data load instantly from cache
- **Restart pipeline:** clears cache + resets all state

---

## Accuracy & Honesty

We hold a **private answer key** (`data/answer_key.csv`) with ground truth for all 100 records and report precision against it programmatically — not a claimed number, a computed one.

### Self-Graded Results

```
Tier 1 (Exact Match):      40 matched, 40 correct
Tier 2 (Rule-Based):       25 matched, 25 correct
Tier 3 (AI Fuzzy):         15 matched, 12 scorable correct, 3 supplementary
Tier 4 (AI Categorize):    39 categorized, 9 scorable correct, 30 supplementary
─────────────────────────────────────────────────────────
Overall Accuracy:           86/86 (100.0%) on scorable records
Coverage:                   86.0% of answer key
False Positives:            0
```

> **"Supplementary" records** are orphan orders and split-settlement fragments that the pipeline correctly identifies but that don't have entries in the answer key (they're supplementary findings, not errors).

### Validation Suite (Phase 5)

Run the full validation:

```bash
python validate_phase5.py
```

```
TEST 1: Deterministic Reproducibility   — 5/5 checks PASSED
TEST 2: Deterministic Precision          — 2/2 checks PASSED
TEST 3: Full Pipeline Run               — 1/1 checks PASSED (121 results, 189.6s)
TEST 4: Answer Key Validation           — 3/3 checks PASSED (86/86 = 100.0%)
TEST 5: Mismatch Type Coverage           — 8/8 types COVERED
TEST 6: Data Integrity                   — 8/8 checks PASSED
TEST 7: Value Reconciliation             — 1/1 checks PASSED (82.7%)
TEST 8: Category Distribution            — 1/1 checks PASSED (8 categories)
TEST 9: Tier Distribution               — 2/2 checks PASSED (T1=40, T2=25)
```

### Deterministic Reproducibility

Tier 1+2 output is **identical across runs** — same 65 results, same order, in ~0.08 seconds. No AI variance, no randomness.

---

## What Broke & How We Fixed It

Honest engineering requires honest bug reporting. Here's every significant issue we hit, in order:

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | Unicode `✓` crashes Windows console (cp1252) | Verification script unusable | ASCII `[PASS]`/`[FAIL]` |
| 2 | Tier 1 matches duplicate `order_id`s (6 false positives) | Silent wrong matches | Skip ambiguous IDs → route to Tier 3 |
| 3 | Tier 1 labels T+3/T+5 UPI as "perfect match" (2 false positives) | Hides delayed settlements | Date check → route to Tier 2 as `date_drift` |
| 4 | OpenAI SDK retries + router retries = double-retry stacking | 3–10s per failed call | `max_retries=0` on SDK client |
| 5 | Validation scored only T1+T2, showed 56% on all tiers | Misleading accuracy metric | Rewrote to score all 4 tiers independently |
| 6 | `sys.stdout` buffered, hides pipeline progress | Can't tell if pipeline is stuck | `sys.stdout.reconfigure(line_buffering=True)` |
| 7 | `Styler.applymap()` deprecated in pandas 2.1+ | Console warnings | Replaced with `Styler.map()` |
| 8 | Plotly `**dict` spread + explicit `margin=` = duplicate kwarg | Chart crashes | Inline layout for that chart |
| 9 | `pipeline.py` referenced `total_false_positives` (key doesn't exist) | CLI `--validate` crashes | Changed to `total_wrong` |
| 10 | `fpdf2.output()` returns `bytearray`, Streamlit needs `bytes` | PDF download crashes | `bytes()` wrapper |
| 11 | LLM unicode (`\u2011`, `\u2013`, smart quotes) in latin-1 PDF font | PDF generation crashes | `_sanitize_text()` mapping |

> **Lesson from bugs #2 and #3:** In finance, a wrong match is worse than no match. We deliberately sacrificed 8 "easy" matches to maintain **100% precision**. This is the tradeoff the track brief asks for.

---

## Project Structure

```
ai-finance-controller/
├── README.md                           # This file
├── PRD.md                              # Full product requirements document
├── PROGRESS.md                         # Phase-by-phase build log
├── PITCH_NOTES.md                      # 5-minute pitch video script notes
├── requirements.txt                    # Python dependencies
├── .gitignore
├── .env.example                        # API key template (15 keys, 5 providers)
├── .env                                # Your actual keys (git-ignored)
│
├── .streamlit/
│   ├── config.toml                     # Dark theme + server config
│   └── secrets.toml.example            # Streamlit Cloud secrets template
│
├── app.py                              # Streamlit dashboard (5 tabs, 1262 lines)
├── run_pipeline.py                     # Full pipeline runner with logging
├── test_router.py                      # Quick LLM router connectivity test
├── inspect_results.py                  # Detailed AI accuracy inspector
├── validate_phase5.py                  # Full validation suite (9 tests, 27 checks)
├── validate_phase6c.py                 # Phase 6c feature validation
├── provider_health.json                # Auto-managed provider health state
│
├── data/
│   ├── __init__.py
│   ├── generate_data.py                # Synthetic data generator (seed=42)
│   ├── generate_samples.py             # Sample dataset generator (4 variants)
│   ├── verify_data.py                  # Data quality verification (42 checks)
│   ├── _inspect.py                     # Dev-only data inspection
│   ├── settlement_report.csv           # Source A — 100 settlement records
│   ├── order_ledger.csv                # Source B — 99 order records
│   ├── answer_key.csv                  # Ground truth — 100 entries
│   ├── pipeline_results.csv            # Latest full pipeline output (121 rows)
│   └── samples/                        # Pre-built sample datasets
│       ├── sample_small_easy/          # 30 records, ~90% deterministic
│       ├── sample_medium_balanced/     # 60 records, ~65% deterministic
│       ├── sample_large_hard/          # 100 records, ~25% deterministic
│       └── sample_edge_cases/          # 40 records, edge case focus
│
└── engine/
    ├── __init__.py
    ├── config.py                       # LLM + router configuration constants
    ├── tier1_exact.py                  # Tier 1: exact ID + amount match
    ├── tier2_rules.py                  # Tier 2: fee/tax-adjusted + date drift
    ├── tier3_fuzzy.py                  # Tier 3: fuzzy candidate + AI confirm
    ├── tier4_ai.py                     # Tier 4: AI categorization (unresolved)
    ├── pipeline.py                     # Orchestrator — runs all tiers in sequence
    ├── qa_agent.py                     # Q&A agent (lexical + semantic search)
    ├── report_export.py                # PDF + CSV report generation
    ├── cache.py                        # Supabase caching layer
    └── llm/
        ├── __init__.py
        ├── providers.py                # Provider endpoints + key handling
        ├── models.py                   # Curated model registry (10 models)
        ├── router.py                   # Multi-provider router with failover
        ├── prompts.py                  # Reconciliation LLM prompt templates
        └── qa_prompts.py              # Q&A conversational prompt templates
```

---

## Configuration

### Engine Parameters (`engine/config.py`)

| Parameter | Value | Why |
|---|---|---|
| `LLM_TEMPERATURE` | 0.2 | Lower than typical (0.3) for more deterministic JSON output |
| `LLM_MAX_TOKENS` | 1024 | Sufficient for structured JSON responses |
| `LLM_TIMEOUT_SECONDS` | 60 | Generous for free-tier latency variance |
| `ROUTER_MAX_ATTEMPTS` | 5 | Max API calls before marking a record as failed |

### Matching Tolerances (`engine/tier3_fuzzy.py`)

| Parameter | Value | Rationale |
|---|---|---|
| `AMOUNT_TOLERANCE_PCT` | 0.70 (70%) | Generous for partial refunds, not tuned to synthetic data |
| `DATE_TOLERANCE_DAYS` | 7 | Reasonable for Indian payment settlement cycles |

---

## Deployment (Streamlit Cloud)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "AI Finance Controller — Razorpay Buildathon 2026"
git remote add origin https://github.com/<your-username>/ai-finance-controller.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"** → select your GitHub repo
3. Set **Main file path** to `app.py`
4. Click **"Advanced settings"** → paste your secrets (see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)):

```toml
GROQ_API_KEYS = "gsk_your_key_1,gsk_your_key_2"
GEMINI_API_KEYS = "your_gemini_key"
OPENROUTER_API_KEYS = "sk-or-your_key"
MISTRAL_API_KEYS = "your_mistral_key"
NVIDIA_API_KEYS = "nvapi-your_key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your_anon_key"
```

5. Click **"Deploy"**

The app reads all secrets via `os.getenv()` — Streamlit Cloud auto-injects top-level TOML secrets as environment variables.

---

## Future Roadmap

Items beyond the 48-hour scope but planned for production:

- **Human-in-the-loop review queue** — accept/reject buttons per AI suggestion for low-confidence calls
- **Anomaly detection** — flag statistically unusual settlement patterns as proactive signals
- **Multi-currency / multi-PG support** — generalize beyond single gateway and currency
- **Unit test suite** — per-tier correctness tests with CI integration
- **Smart retry ordering** — track provider reliability over time and reorder the chain dynamically
- **Trust score UX** — translate tier/confidence into plain-language indicators (High / Medium / Needs Review)

---

## License

MIT

---

<p align="center">
  Built for the <strong>Razorpay Buildathon 2026</strong> — Track 04: AI Finance Controller<br>
  <em>"Throughput plus measured accuracy plus an honest exception list. One cherry-picked match proves nothing."</em>
</p>
