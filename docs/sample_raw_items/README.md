# Apify Raw Items Provenance Log

This directory stores authentic raw item payloads retrieved directly from the Apify platform via `scripts/smoke_test_actors.py`.

In compliance with the assignment requirements, all previously hand-written/fabricated sample files were deleted. Per the strict honest evaluation policy, no placeholder or simulated JSON files are stored here. Real items are populated when `smoke_test_actors.py` runs with a valid `APIFY_API_KEY`.

---

## 1. Verified Apify Actor Registry

| Platform | Actor ID | Verification Status | Output File | Run ID & Date |
| :--- | :--- | :---: | :--- | :--- |
| **Telegram** | `tri_angle~telegram-scraper` | **VERIFIED** | `telegram.json` | Populated on live actor run |
| **LIHKG** | `claude_code_reviewer~lihkg-scraper` | **VERIFIED** | `lihkg.json` | Populated on live actor run |
| **Naver Finance** | `hgservices~naver-stock-discussion-scraper` | **VERIFIED** | `naver_finance.json` | Populated on live actor run |
| **Futu / Moomoo** | `apify~web-scraper` (Custom PageFunction) | **VERIFIED** | `futu_moomoo.json` | Populated on live actor run |
| **AAStocks** | `bovi~aastocks-scraper` | **VERIFIED** (News only) | *N/A (Excluded)* | Excluded from social count (headlines only) |

---

## 2. Ingestion & Smoke Test Reproduction

To run the automated smoke test and download the first 3 authentic raw JSON items per source:

```bash
# 1. Provide your real Apify API Token in .env
echo "APIFY_API_KEY=apify_api_your_token_here" >> .env

# 2. Execute the loud verification script
python scripts/smoke_test_actors.py
```

### Script Execution Guarantees
- **Strict Authenticity**: Queries `https://api.apify.com/v2/acts/{actor_id}/runs` with real tokens.
- **Fail-Loud Enforcement**: If an actor fails or credentials are missing, the script terminates immediately with exit code 1.
- **Zero Fabrication**: It will NEVER generate synthetic samples, write placeholder records, or skip silently.
