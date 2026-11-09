# CrowdWisdom Trading - Sentiment Price Prediction Summary

> **WARNING: SYNTHETIC DEMO DATA** — Simulated alternative data and price paths.

*Generated at: 2026-11-08 20:05:48 UTC*

## 1. Data Provenance Audit

- **Price Coverage:** 2026-10-29 to 2026-11-08 (~180 days) (synthetic)
- **Social Posts Ingested:** 918 posts (synthetic)
- **News Articles Ingested:** 18 articles (synthetic)
- **Overall Pipeline Mode:** `SYNTHETIC`

### Per-Ticker Provenance Breakdown

| Ticker | Exchange | Social Posts | News Articles | Low Data? |
|:---|:---|:---:|:---:|:---:|
| `0700.HK` | HKEX | 171 | 5 | No |
| `9988.HK` | HKEX | 188 | 4 | No |
| `3690.HK` | HKEX | 195 | 3 | No |
| `0005.HK` | HKEX | 182 | 3 | No |
| `1211.HK` | HKEX | 182 | 3 | No |

## 2. Cross-Asset Aggregate Out-of-Sample Performance

- **Total OOS Samples:** 560
- **Directional Hit Ratio:** **51.1%** (Price-Only Baseline: 51.1%)
- **Information Coefficient (IC):** **+0.037**
- **Model RMSE:** **0.0216** (Price-Only: 0.0220 | Zero-Return: 0.0193)
- **Strategy Sharpe Ratio:** **0.11**
- **Strategy vs Buy & Hold Return:** **+0.64%** vs **+7.49%**
- **Baseline Comparison Verdict:** Sentiment model reduced forecasting error over price-only baseline (RMSE 0.0216 vs 0.0220), with hit ratio of 51.1% vs 51.1%.

## 3. Executive Summary & Ticker Breakdown

```

=============================================================================================================================
 *** SYNTHETIC DEMO DATA *** (Simulation Mode — Pass --live to run against live APIs)
 CROWDWISDOM TRADING - HENG SENG & EXCHANGE SENTIMENT PRICE PREDICTION REPORT
 Generated: 2026-11-08 20:05:48 UTC | Data Mode: SYNTHETIC
=============================================================================================================================

>>> DATA PROVENANCE AUDIT BLOCK
-----------------------------------------------------------------------------------------------------------------------------
Coverage: 2026-10-29 to 2026-11-08 (~180 days) | Social: 918 posts (synthetic) | News: 18 articles (synthetic)
By Ticker: 0700.HK: 171 posts, 5 news | 9988.HK: 188 posts, 4 news | 3690.HK: 195 posts, 3 news | 0005.HK: 182 posts, 3 news | 1211.HK: 182 posts, 3 news

>>> EXCHANGE: HKEX (HONG KONG EXCHANGE)
-----------------------------------------------------------------------------------------------------------------------------
Ticker     Name                   Price        Mentions  Buzz Z   Sent     Dir       n+1 Pred      95% CI              Position   R/R    Reason
-----------------------------------------------------------------------------------------------------------------------------
3690.HK    Meituan                HKD 194.75   195       +0.69  +0.723 BULLISH   213.37 (+9.56%) [206.5, 220.2]      HOLD       1:1.09 R/R 1.09 < 1.5
9988.HK    Alibaba Group Holding  HKD 71.04    188       +1.92  +0.245 BULLISH   74.88 (+5.41%) [72.0, 77.8]        HOLD       1:1.44 R/R 1.44 < 1.5
0005.HK    HSBC Holdings plc      HKD 85.54    182       +1.39  +0.476 BULLISH   89.21 (+4.29%) [85.9, 92.5]        HOLD       1:1.28 R/R 1.28 < 1.5
1211.HK    BYD Company Limited    HKD 338.61   182       +1.39  +0.616 BULLISH   354.93 (+4.82%) [341.9, 368.0]      HOLD       1:1.28 R/R 1.28 < 1.5
0700.HK    Tencent Holdings Ltd.  HKD 370.38   171       +1.22  +0.473 BULLISH   387.03 (+4.49%) [373.2, 400.8]      BUY        1:1.79 Bullish momentum & sentiment (R/

=============================================================================================================================
 OUT-OF-SAMPLE WALK-FORWARD VALIDATION & BASELINE COMPARISON (ZERO LOOKAHEAD)
-----------------------------------------------------------------------------------------------------------------------------
Aggregate OOS: N=560 | Mean Hit=51.1% (vs Price-Only 51.1%) | Mean IC=+0.037 | Mean RMSE=0.0216 (vs Price-Only 0.0220) | Sharpe=0.11
Verdict: Sentiment model reduced forecasting error over price-only baseline (RMSE 0.0216 vs 0.0220), with hit ratio of 51.1% vs 51.1%.
-----------------------------------------------------------------------------------------------------------------------------
Ticker     OOS N   Hit %    IC       RMSE      Price-Only Hit%  Price-Only RMSE  Sent Better?   Sharpe  
-----------------------------------------------------------------------------------------------------------------------------
0700.HK    112     38.4%    -0.035   0.0209    38.4%            0.0213           YES (RMSE)     -2.17   
9988.HK    112     56.2%    -0.018   0.0217    56.2%            0.0218           YES (RMSE)     0.67    
3690.HK    112     51.8%    +0.051   0.0237    51.8%            0.0245           YES (RMSE)     0.47    
0005.HK    112     54.5%    +0.093   0.0209    54.5%            0.0212           YES (RMSE)     0.80    
1211.HK    112     54.5%    +0.093   0.0209    54.5%            0.0212           YES (RMSE)     0.80    

=============================================================================================================================
 RECENT HOLDOUT PREDICTION VS REALIZED CLOSE CHECK (Sample: 0700.HK)
-----------------------------------------------------------------------------------------------------------------------------
Bar Date     Pred Close    Real Close    Error %    Pred Return %    Real Return %    Sign Match
-----------------------------------------------------------------------------------------------------------------------------
2026-10-23   408.41        393.86             3.69%           -0.00%           -3.57% CORRECT   
2026-10-26   393.30        389.80             0.90%           -0.14%           -1.03% CORRECT   
2026-10-27   389.54        384.43             1.33%           -0.07%           -1.38% CORRECT   
2026-10-28   384.16        386.81             0.69%           -0.07%           +0.62% INCORRECT 
2026-10-29   386.46        373.05             3.60%           -0.09%           -3.56% CORRECT   
2026-10-30   371.79        355.12             4.69%           -0.34%           -4.81% CORRECT   
2026-11-02   343.49        358.67             4.23%           -3.27%           +1.00% INCORRECT 
2026-11-03   345.42        363.05             4.85%           -3.69%           +1.22% INCORRECT 
2026-11-04   353.37        368.64             4.14%           -2.67%           +1.54% INCORRECT 
2026-11-05   369.87        370.38             0.14%           +0.33%           +0.47% CORRECT   

=============================================================================================================================
 MODEL LIMITATIONS & RESEARCH DISCLOSURES:
 1. Short alternative data lookback: Social and news scrapers capture recent 10-day sentiment.
 2. Non-stationarity & noise: Retail sentiment is subject to sudden attention shifts and herd behavior.
 3. Out-of-sample sample size: Daily bars yield small sample sizes; statistical significance requires larger samples.
 4. Frictionless assumptions: Transaction costs, exchange fees, and execution slippage are not subtracted.
 5. Low data warning: Tickers marked with '*' have fewer than 10 social mentions in the lookback window.
=============================================================================================================================

```

## 4. Limitations & Research Disclosures

1. **Lookback Window:** Social and news scrapers capture recent 10-day alternative data.
2. **Microstructure & Noise:** Social media discussions exhibit sudden attention spikes and retail sentiment noise.
3. **Statistical Power:** Daily bars yield compact out-of-sample sample counts; caution should be used before claiming definitive long-term edge.
4. **Execution Frictions:** Commission, bid-ask spread, borrowing costs for shorting, and stamp duty are omitted.
5. **Low Data Coverage:** Tickers with `< 10` social posts in the 10-day window are flagged with low data warnings.
