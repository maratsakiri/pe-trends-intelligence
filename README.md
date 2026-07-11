# PE Trends Intelligence Pipeline

A signal-detection pipeline that identifies UK mid-market financial-services
companies undergoing private-equity ownership changes, using public data
sources — and surfaces them as a ranked, analyst-ready shortlist.

Developed as an MSc Business Analytics dissertation project (UCL) in partnership
with Palladium Digital.

---

## What it does

The pipeline detects mid-market PE activity that is **invisible in both free
public data and premium commercial deal databases** (Orbis M&A, PitchBook), yet
detectable from primary Companies House filings. It combines:

- **Detection** — Companies House Persons-with-Significant-Control (PSC) filings,
  filtered to financial-services SIC codes, flagging ownership changes since 2021.
- **Filtering** — intra-group exclusion, FTSE-350 mid-market scoping, and a
  non-financial-services contaminant filter.
- **Validation** — FCA register cross-check.
- **Sector context (NLP)** — news sentiment (Loughran-McDonald + FinBERT),
  topic modelling (LDA), and Google Trends search momentum.
- **Scoring** — a two-tier, non-blended score: per-deal detection confidence
  (Tier 1) reported alongside sector context (Tier 2).
- **Output** — a self-contained HTML dashboard for business-development triage.

> **Important:** the pipeline produces **triage leads for human review**, not
> confirmed deals. Detected ownership changes are validated against Companies
> House but are not recorded as deals in commercial databases (a central finding
> of the research). Treat all candidates as leads pending analyst verification.

---

## Requirements

- **Python 3.10+** (developed on 3.14)
- A free **Companies House API key** — register at
  https://developer.company-information.service.gov.uk/
- Optional (for the news / jobs / trends layers): **Guardian**, **Reed**, and
  **FRED** API keys. The core detection pipeline runs without these.

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/pe-trends-intelligence.git
cd pe-trends-intelligence

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file with your API keys (see below)
```

### Environment variables

Create a file named `.env` in the project root (it is git-ignored and will
never be committed):

```
COMPANIES_HOUSE_API_KEY=your_key_here
GUARDIAN_API_KEY=your_key_here
REED_API_KEY=your_key_here
FRED_API_KEY=your_key_here
FCA_API_KEY=your_key_here
FCA_API_EMAIL=your_email_here
```

Only `COMPANIES_HOUSE_API_KEY` is required for core detection.

---

## Running the pipeline

The pipeline is orchestrated through `orchestrator.py`, which runs stages
individually or end-to-end.

```bash
# See all options
python orchestrator.py --help

# Run the full pipeline end-to-end
python orchestrator.py --all

# Or run stages individually:
python orchestrator.py --companies-house   # detection
python orchestrator.py --filter            # mid-market FTSE filtering
python orchestrator.py --fca               # FCA validation
python orchestrator.py --triangulate       # news cross-check
python orchestrator.py --trends            # Google Trends momentum
```

Each stage writes its outputs to a local `data/` folder (created on first run;
git-ignored, so your outputs stay on your machine).

### NLP analysis (run after data collection)

```bash
python connectors/sentiment_lm.py     # Loughran-McDonald sentiment
python connectors/lda_topics.py       # LDA topic modelling
python connectors/lda_elbow.py        # topic-count selection (elbow + coherence)
python connectors/finbert_validate.py # FinBERT sentiment validation
python connectors/signal_scoring.py   # two-tier signal scoring
```

### Generating the dashboard

Once the pipeline has produced outputs in `data/`:

```bash
python generate_dashboard.py
```

This reads your local CSVs and writes `PE_Pipeline_Dashboard_LIVE.html` — a
single self-contained file you can open in any browser or share by email. No
server required.

---

## Repository contents

This repository contains **code and documentation only**. No data is included —
detection outputs contain company-specific results that each user generates
themselves by running the pipeline against the live APIs.

```
connectors/          # data-source connectors and analysis modules
orchestrator.py      # pipeline controller
generate_dashboard.py
_dashboard_template.html
requirements.txt
PROGRESS_LOG.md      # full methodology & findings log
README.md
.gitignore
```

Running the pipeline creates a local `data/` folder for your own outputs.

---

## Method & findings

See `PROGRESS_LOG.md` for the complete record of methodology decisions, results,
and limitations for every pipeline component.

---

## Notes & limitations

- Detection identifies **candidates**, not confirmed deals; PSC filings cannot
  distinguish a PE buyout from ordinary corporate ownership changes.
- NLP signals (sentiment, topics, trends) are **sector-level context**, not
  attributable to individual candidates.
- `pytrends` (Google Trends) is unofficial and rate-limited; runs may need
  retries.
- API keys and rate limits are the user's own responsibility.

---

## Licence

Academic project. Companies House data is Crown Copyright, licensed under the
Open Government Licence. Please respect the terms of service of all data
providers used.
