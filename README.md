# Berkshire GenAI Signal Engine

Extract structured financial signals from Berkshire Hathaway shareholder letters using generative AI, then compare those signals to stock behavior.

## What This Does

1. **Extracts Berkshire-specific signals** from shareholder letters:
   - Management confidence
   - Uncertainty level
   - Capital allocation posture (opportunistic vs defensive)
   - Cash deployment stance
   - Acquisition appetite
   - Market outlook
   - Insurance float emphasis

2. **Fetches BRK.B stock data** around letter release dates

3. **Analyzes correlations** between extracted signals and subsequent returns

4. **Visualizes results** in a Streamlit dashboard

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set up API key
cp .env.example .env
# Edit .env with your Anthropic API key

# Extract signals from letters
python scripts/extract_all.py

# Run the dashboard
streamlit run app.py
```

## Project Structure

```
├── data/
│   ├── letters/      # Raw letter text files (2020.txt, 2021.txt, etc.)
│   ├── signals/      # Extracted signals (JSON)
│   └── stock/        # Cached stock data
├── src/
│   ├── config.py     # Settings
│   ├── prompts.py    # LLM prompts for signal extraction
│   ├── extractor.py  # Signal extraction logic
│   ├── stock.py      # Stock data fetching
│   └── analyzer.py   # Signal vs stock analysis
├── app.py            # Streamlit dashboard
└── scripts/
    └── extract_all.py
```

## Adding Letters

1. Copy letter text to `data/letters/YYYY.txt` (e.g., `2023.txt`)
2. Run `python scripts/extract_all.py`
3. Refresh the dashboard

## Tech Stack

- **Claude API** — Structured signal extraction
- **Pydantic** — Output validation
- **yfinance** — Stock data
- **Streamlit** — Dashboard
- **Plotly** — Charts
