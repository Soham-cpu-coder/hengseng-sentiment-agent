# CrowdWisdom Trading Take-Home Submission

**Role:** HengSeng Sentiment Price Prediction Agent / AI Agents Developer Assessment  
**Recipient:** `gilad@crowdwisdomtrading.com`  

---

## 🔑 API Tokens Notice
Per the task instructions and security best practices, **API keys are strictly omitted from the project files and archive** (`.gitignore` enforces excluding `.env`). The **Apify API Token** and **Exa.com API Key** are provided separately in the email accompanying this submission.

---

## 🤖 Verified Apify Actors & Data Providers (Task 1 Audit)

Only actors confirmed accessible with public endpoints and tested schemas are marked **VERIFIED**:

| Source | Exchange | Apify Actor ID | Status | Real Input Schema / Verified Call | Genuine Social Source? |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Telegram** | HKEX & KRX | `tri_angle~telegram-scraper` | **VERIFIED** | `{"profiles": [...], "collectMessages": true, "scrapeLastNDays": 10, "maxMessagesPerChannel": 100}` | **Yes** (HKEX & KRX) |
| **LIHKG** | HKEX | `claude_code_reviewer~lihkg-scraper` | **VERIFIED** | `{"mode": "search", "bulkInput": "<ticker>", "catIds": ["5"], "maxItems": 50}` | **Yes** (HKEX) |
| **Naver Finance** | KRX | `hgservices~naver-stock-discussion-scraper` | **VERIFIED** | `{"stocks": ["<code6>"], "maxResults": 50, "dateFrom": "YYYY-MM-DD"}` | **Yes** (KRX) |
| **Futu / Moomoo** | HKEX | `apify~web-scraper` / `cheerio-scraper` | **VERIFIED** | Custom pageFunction extracting post cards and engagement | Fallback (HKEX satisfied by LIHKG + TG) |
| **AAStocks** | HKEX | `bovi~aastocks-scraper` | **VERIFIED** (News only) | `{"feed": "latest-news", "language": "tc", "maxResults": 50}` | **No** (News headlines only; dropped from social count) |
| **News Feed** | Both | **Exa.com** (`exa-py` SDK v1.8+) | **VERIFIED** | `exa.search_and_contents(query=f"{name} ({ticker}) stock financial news", num_results=5, start_published_date=..., text={"max_characters": 1000}, highlights=True)` | Financial Journalism |
| **Market Prices**| Both | `yfinance` + Yahoo v8 REST backup | **VERIFIED** | Daily OHLCV over 180 calendar days ($\ge 120$ trading days) | Local Currencies (HKD / KRW) |

### Minimum Social Sources Fulfillment
- **HKEX:** $\ge 2$ genuine social sources satisfied by **LIHKG** (`claude_code_reviewer~lihkg-scraper`) and **Telegram** (`tri_angle~telegram-scraper`).
- **KRX:** $\ge 2$ genuine social sources satisfied by **Naver Finance 종목토론실** (`hgservices~naver-stock-discussion-scraper`) and **Telegram** (`tri_angle~telegram-scraper`).

---

## 🚀 Step-by-Step Reproduction Instructions

### Step 1: Extract Archive and Enter Workspace
```bash
cd hengseng-sentiment-agent
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Credentials (`.env`)
Copy `.env.example` to `.env` and insert the tokens provided separately in the email:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
APIFY_API_KEY=<token_provided_in_email>
EXA_API_KEY=<token_provided_in_email>
```

### Step 4: Run Actor Smoke Tests (Task 2)
```bash
python scripts/smoke_test_actors.py
```
*Guarantees: Exits non-zero with exact actor error if token is missing or actor fails. Never writes placeholders or skips silently. Writes authentic raw JSON items to `docs/sample_raw_items/<source>.json`.*

### Step 5: Execute the Live Prediction Pipeline (Task 3)
```bash
# Execute for Hong Kong (HKEX):
python main.py --exchange HKEX --live

# Execute for South Korea (KRX):
python main.py --exchange KRX --live

# Execute for ALL 10 assets:
python main.py --exchange ALL --live
```

*(Note: To execute the calibrated offline demo mode without live API calls, omit `--live`: `python main.py --exchange ALL`)*

### Step 6: Inspect Results & Deliverables
- **Visual HTML Report:** Open `output/sentiment_report.html` in your web browser (fully self-contained, offline responsive SVG charts, client-side sortable tables):
  ```bash
  open output/sentiment_report.html
  ```
- **Terminal Summary:** High-contrast ASCII tables with Data Provenance Audit Block and out-of-sample walk-forward validation scorecard.
- **Machine-Readable JSON:** `output/pipeline_results.json` containing metadata, provenance, aggregate validation, per-ticker positions with reasons, baselines, and predicted-vs-realized tables.
- **Markdown Summary:** `output/report_summary.md`.

### Step 7: Run Automated Unit & Invariant Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📈 Assessment Deliverables & Technical Audit

1. **Task 1: Real Apify Actors Verified:** Every actor queried against Apify API `GET /v2/acts/<id>`. Non-functional or corporate actors replaced with genuine retail social discussion actors (`hgservices~naver-stock-discussion-scraper` for Naver 종목토론실, `tri_angle~telegram-scraper` for Telegram).
2. **Task 2: Honest Raw Samples:** Deleted all legacy hand-written mock samples. `smoke_test_actors.py` fails loudly on missing credentials or API errors without writing placeholder data.
3. **Task 3: Full End-to-End Pipeline & Real Market Data:** Real daily OHLCV prices ($\ge 120$ trading days) via Yahoo REST/yfinance (.HK in HKD, .KS in KRW). Real Exa news ingestion via `exa-py`. Provenance audit block saved to HTML, Markdown, and JSON. Low data warning badge (`⚠️ LOW DATA (<10 posts)`) for sparse assets.
4. **Task 4: Meaningful Position Sizing & S/R Logic:**
   - Targets skip levels closer than 1.0 ATR to entry.
   - Stop loss placed 0.5 to 1.0 ATR beyond nearest S/R level.
   - BUY invariant: $stop\_loss < entry < tp1 < tp2$ with $R/R \ge 1.5$.
   - SELL invariant: $tp2 < tp1 < entry < stop\_loss$ with $R/R \ge 1.5$.
   - Positions with $R/R < 1.5$ or low conviction are assigned **HOLD** with explicit reason shown (e.g. `R/R 1.09 < 1.5`).
5. **Task 5: Honest Out-of-Sample Validation & Baselines:**
   - Chronological walk-forward validation with zero lookahead bias (verified by automated causality tests).
   - Reports out-of-sample Hit Ratio, IC, RMSE, MAE, and Sharpe ratio.
   - Evaluated against two baselines: Zero-Return and Price-Only Ridge models, with an explicit verdict on whether sentiment improved performance.
   - Predicted-vs-Realized table for holdout bars with Error % and directional correctness.
6. **Task 6: Visual HTML Report:** Self-contained, zero external network dependencies, responsive inline SVG corridor and S/R charts, client-side sorting by mentions, sentiment, direction, and forecast.
