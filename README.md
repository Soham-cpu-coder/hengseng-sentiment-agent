# HengSeng & Multi-Exchange Sentiment Price Prediction Agent

An end-to-end Python pipeline and quantitative price-prediction agent developed for the **CrowdWisdom Trading** internship assessment.

The system ingests alternative social sentiment data across multiple verified platforms per exchange (LIHKG and Telegram for HKEX; Naver Finance 종목토론실 and Telegram for KRX), enriches it with semantic financial news intelligence via **Exa.com**, aligns feeds to UTC daily price bars over $\ge 120$ trading days, extracts structured sentiment and technical factors, and forecasts horizon $n+1$ price action using strict out-of-sample walk-forward validation compared against standard baselines.

---

## 📋 Features & Pipeline Architecture

- **Multi-Source Social Ingestion via Verified Apify Actors:**
  - Real, verified Apify actors with dedicated schemas (no reliance on generic scrapers with guessed CSS selectors).
  - Handles platform limitations cleanly: unexposed fields (e.g. LIHKG views, Naver followers) are explicitly set to `None` rather than fabricated with synthetic values.
  - Social items are isolated by market: Hong Kong assets only ingest HK-specific communities, and Korean assets only ingest KRX-specific channels.
  - At least 2 genuinely social platforms per exchange:
    - **HKEX:** LIHKG (`claude_code_reviewer~lihkg-scraper`) + Telegram (`tri_angle~telegram-scraper`).
    - **KRX:** Naver Finance Discussion (`hgservices~naver-stock-discussion-scraper`) + Telegram (`tri_angle~telegram-scraper`).
- **Financial News via Exa.com (`exa-py` SDK):**
  - Neural semantic search filtering financial news published within the last 10 days per ticker.
- **Real Market Daily OHLCV Price Feeds:**
  - Daily price bars fetched over at least 120 trading days (180 calendar days lookback) via `yfinance` with Yahoo Finance Chart v8 REST fallback.
  - Explicit missing sentiment handling: older bars outside the 10-day alternative data window use `NaN` sentiment values combined with a `has_social=0` indicator feature, preventing artificial data imputation.
  - Korean tickers formatted with `.KS` in KRW; Hong Kong tickers formatted with `.HK` in HKD.
- **Robust Position & Support/Resistance Logic:**
  - Classic floor pivots ($P, S_1, S_2, R_1, R_2$), Fibonacci retracements (0.382, 0.618), and swing extrema.
  - Stop loss: placed 0.5 to 1.0 ATR beyond the nearest support (long) or resistance (short).
  - Targets: skips levels closer than 1.0 ATR to entry, selecting the next two meaningful levels beyond 1.0 ATR.
  - Direction combines sentiment score, predicted $n+1$ return, and technical momentum with conviction scoring.
  - Strict ordering invariants enforced:
    - **BUY:** $\text{stop\_loss} < \text{entry} < \text{take\_profit\_1} < \text{take\_profit\_2}$ with $R/R \ge 1.5$.
    - **SELL:** $\text{take\_profit\_2} < \text{take\_profit\_1} < \text{entry} < \text{stop\_loss}$ with $R/R \ge 1.5$.
    - If Risk/Reward $< 1.5$ or conviction is low, the recommendation is strictly **HOLD** with the explicit reason shown (e.g. `R/R 1.09 < 1.5`).
- **Honest Walk-Forward Validation & Baselines:**
  - Strictly chronological walk-forward validation with zero lookahead bias.
  - Reports out-of-sample Hit Ratio, Information Coefficient (IC), RMSE, MAE, strategy return vs buy-and-hold, and strategy Sharpe ratio.
  - Evaluated against two baselines per ticker and in aggregate:
    1. **Predict-Zero-Return** (predicting 0% daily return).
    2. **Price-Only Model** (technical momentum without sentiment features).
  - Clear baseline comparison verdict stating whether sentiment features beat price-only out of sample.
  - Realized vs Predicted Close table holding out the last $K$ bars ($K = 5$ to $10$) with error % and directional hit.
- **Data Provenance & Clear Demarcation:**
  - Strict `--live` flag: halts immediately with an explicit error naming the missing source if credentials or network calls fail. Never falls back to mock data when live mode is requested.
  - In demo mode (without `--live`), every output displays a `SYNTHETIC DEMO DATA` banner and per-source `"data_mode"` metadata.

---

## 🤖 Verified Apify Actor Registry (Task 1 Audit)

| Source Platform | Exchange | Verified Apify Actor ID | Public? | Required Input Fields | Verified Input Parameters | Genuine Social Source? |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **Telegram** | HKEX & KRX | `tri_angle~telegram-scraper` | **Yes** | `['profiles']` | `{"profiles": [...], "collectMessages": true, "scrapeLastNDays": 10}` | **Yes** (Both) |
| **LIHKG** | HKEX | `claude_code_reviewer~lihkg-scraper` | **Yes** | `[]` | `{"mode": "search", "bulkInput": "<ticker>", "catIds": ["5"], "maxItems": 50}` | **Yes** (HKEX) |
| **Naver Finance** | KRX | `hgservices~naver-stock-discussion-scraper` | **Yes** | `['stocks']` | `{"stocks": ["<code6>"], "maxResults": 50, "dateFrom": "YYYY-MM-DD"}` | **Yes** (KRX) |
| **AAStocks** | HKEX | `bovi~aastocks-scraper` | **Yes** | `[]` | `{"feed": "latest-news", "language": "tc", "maxResults": 50}` | **No** (News headlines only; dropped from social quota) |
| **Futu / Moomoo** | HKEX | `apify~web-scraper` / `cheerio-scraper` | **Yes** | `['pageFunction']` | Custom Cheerio pageFunction extracting post cards | Fallback |

---

## 🗂️ Data Field Extraction & Platform Support (Task 1.5)

Platforms vary in the metrics they expose. Where a platform does not expose likes, reposts, views, or follower counts, the value is left as `None` rather than estimated or invented:

| Source Platform | Exchange | Primary Apify Actor / Endpoint | `timestamp_utc` | `text` | `likes` | `reposts` | `views` | `author_followers` | Notes / Platform Omissions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Telegram** | HKEX & KRX | `tri_angle~telegram-scraper` | Yes | Yes | Yes (reactions) | Yes (forwards) | Yes | None | Channel follower counts not exposed per message payload. |
| **LIHKG** | HKEX | `claude_code_reviewer~lihkg-scraper` | Yes | Yes | Yes (upvotes) | Yes (replies) | None | None | LIHKG does not expose thread view counters or author follower counts. |
| **Naver Finance** | KRX | `hgservices~naver-stock-discussion-scraper` | Yes | Yes | Yes (recommend) | None | Yes | None | Naver 종목토론실 exposes views and up/down recommendations; no user followers or reposts. |
| **Futu / Moomoo** | HKEX | `apify~web-scraper` (Custom PageFunction) | Yes | Yes | Yes | Yes | Yes | None | Brokerage community cards; author follower count omitted on card previews. |
| **AAStocks** | HKEX | `bovi~aastocks-scraper` | Yes | Yes | None | None | None | None | Financial news feed; exposes headlines and timestamps; no social engagement metrics. |
| **Exa News** | Both | `exa-py` Neural Search SDK | Yes | Yes (summary) | None | None | None | None | 10-day lookback financial journalism & corporate disclosures. |

---

## 🎯 Assets Monitored

- **HKEX (Hong Kong Exchange):**
  - `0700.HK` (Tencent Holdings Ltd.)
  - `9988.HK` (Alibaba Group Holding Ltd.)
  - `3690.HK` (Meituan)
  - `0005.HK` (HSBC Holdings plc)
  - `1211.HK` (BYD Company Limited)
- **KRX (Korea Exchange):**
  - `005930.KS` (Samsung Electronics Co., Ltd.)
  - `000660.KS` (SK Hynix Inc.)
  - `035420.KS` (NAVER Corporation)
  - `005380.KS` (Hyundai Motor Company)
  - `035720.KS` (Kakao Corp.)

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.9 macOS and Linux)
- pip

### 2. Install Dependencies
```bash
cd hengseng-sentiment-agent
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your active API keys:
```ini
APIFY_API_KEY=your_apify_api_token_here
EXA_API_KEY=your_exa_api_key_here
```
> **Security Note:** Real API keys are never stored in the repository. `.env` is ignored by `.gitignore`.

---

## 🚀 Execution Guide

### Option 1: Live Production Mode (`--live`)
Requires active `APIFY_API_KEY` and `EXA_API_KEY` in `.env`. If any source returns no real data or credentials are missing, the pipeline halts immediately with a clear error:

```bash
# Run live for HKEX assets
python main.py --exchange HKEX --live

# Run live for KRX assets
python main.py --exchange KRX --live

# Run live across all 10 assets
python main.py --exchange ALL --live
```

### Option 2: Synthetic Demo Mode (Offline Verification)
If no API keys are provided or `--live` is omitted, the pipeline runs using calibrated synthetic demo data with visible warnings and `"data_mode": "synthetic"`:

```bash
# Run demo for HKEX
python main.py --exchange HKEX

# Run demo across all 10 assets
python main.py --exchange ALL
```

### Option 3: Actor Smoke Test (Task 2)
```bash
python scripts/smoke_test_actors.py
```
This queries each verified actor, checks connectivity and input schemas, and downloads the first 3 authentic raw JSON items per source to `docs/sample_raw_items/`. Exits non-zero if credentials or actors fail.

### Option 4: Interactive Streamlit Dashboard
```bash
streamlit run dashboard.py
```

---

## 📖 How to Read the Visual Report

Open `output/sentiment_report.html` in any web browser. The report is fully self-contained and functions offline without external CDN dependencies.

1. **Header & Data Mode Banner:**
   - Displays whether the run was `LIVE PRODUCTION` or `SYNTHETIC DEMO DATA`.
2. **Data Provenance Audit Block:**
   - Shows the date range of price bars, total social posts and news articles ingested, and per-ticker breakdown pills (`<ticker>: <posts>p/<news>n`).
   - Sparse tickers with $< 10$ social posts display a yellow warning badge: `⚠️ LOW DATA (<10 posts)`.
3. **Sortable Exchange Tables (HKEX & KRX Tabs):**
   - Click column headers (**Mentions**, **Buzz Z**, **Sentiment**, **Forecast %**, **R/R**) to sort client-side ascending or descending.
   - Shows current price, aggregated sentiment score, sentiment direction, horizon $n+1$ forecast with 95% confidence bounds, position status, and risk/reward.
4. **Deep-Dive Ticker Cards:**
   - **Position Reason:** Explains why a trade was taken or held (e.g. `BUY 1:1.79 Bullish momentum & sentiment` or `HOLD 1:1.09 R/R 1.09 < 1.5`).
   - **Inline SVG Corridor Chart:** Shows historical closes, floor pivot, support $S_1$, resistance $R_1$, and horizon $n+1$ forecast corridor.
   - **Recent Exa News:** Top articles with clickable links and sentiment scores.
5. **Walk-Forward Validation & Baselines Tab:**
   - **Aggregate Scorecard:** Cross-asset total out-of-sample samples, mean hit ratio, mean Information Coefficient, mean RMSE, and mean Sharpe ratio.
   - **Baseline Verdict:** Explicitly declares whether the sentiment-augmented model beat the price-only model on directional accuracy and RMSE.
   - **Holdout Realized vs Predicted Table:** Displays the last $K$ bars with date, predicted close, realized close, error %, and directional match.

---

## 📊 Outputs & Deliverables

Every pipeline run generates:
1. `output/sentiment_report.html`: Self-contained visual HTML report.
2. `output/pipeline_results.json`: Full machine-readable payload with metadata, provenance, aggregate validation, per-ticker positions with reasons, baselines, and predicted-vs-realized tables.
3. `output/report_summary.md`: Markdown summary of provenance, performance, and positions.
4. Terminal Output: High-contrast ASCII tables with Data Provenance and out-of-sample walk-forward results.

---

## 🧪 Unit Testing

Run the automated test suite covering ordering invariants, live mode exception handling, strict zero-lookahead feature causality, and missing metric handling:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## ⚠️ Limitations & Quantitative Disclosures (Task 5)

1. **Asymmetric Alternative Data Lookback:** Social discussion and news feeds cover a 10-day lookback window due to platform constraints, whereas daily market OHLCV bars require $\ge 120$ trading days for technical indicator and model stability. Pre-sentiment bars are tagged with `has_social=0` and `NaN` sentiment rather than fabricated data.
2. **Retail Noise & Attention Cascades:** Forum discussions on LIHKG, Telegram, and Naver can be emotional, volatile, or influenced by retail attention cascades. Engagement weighting mitigates this but does not eliminate noise.
3. **Statistical Sample Size:** Daily bars over 120–180 trading days yield modest out-of-sample sample counts. Statistical significance is noisy, and no claim of long-term profitability is made unless substantiated by long-term live execution.
4. **Frictionless Model Assumptions:** Simulated returns do not deduct transaction costs, exchange stamp duties (e.g. HKEX 0.1% stamp duty), brokerage fees, or bid-ask execution slippage.
5. **Low Data Coverage:** Tickers with $< 10$ social posts in the 10-day window are explicitly flagged as "low data" rather than concealed.
