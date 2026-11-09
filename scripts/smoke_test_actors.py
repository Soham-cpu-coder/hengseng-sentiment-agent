"""
Smoke test script for verified Apify scrapers.
Runs each configured Apify actor with a small limit and persists the first 3 genuine
raw items to docs/sample_raw_items/<source>.json.
Fails loudly (exits non-zero) if API token is missing, actor fails, or returns no items.
Never writes placeholders or skips silently.
"""

from datetime import datetime, timezone
import json
import logging
import os
import sys

# Ensure repository root is in search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from scrapers.apify_client import ApifyHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmokeTestActors")

DOCS_SAMPLE_DIR = os.path.join(REPO_ROOT, "docs", "sample_raw_items")
os.makedirs(DOCS_SAMPLE_DIR, exist_ok=True)


def test_actor(source: str, actor_id: str, run_input: dict, apify: ApifyHandler) -> dict:
    logger.info(f"\n=======================================================")
    logger.info(f"Testing Source: '{source}' | Actor: '{actor_id}'")
    logger.info(f"Input: {json.dumps(run_input)}")
    logger.info(f"=======================================================")

    if not apify.is_configured:
        err_msg = (
            f"Apify API token (APIFY_API_KEY) is not configured in .env. "
            f"Cannot run live actor '{actor_id}' for '{source}'."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    try:
        items = apify.run_actor_and_fetch_items(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=90,
        )
        if not items:
            err_msg = f"Actor '{actor_id}' for '{source}' returned 0 items from dataset."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        sample = items[:3]
        sample_path = os.path.join(DOCS_SAMPLE_DIR, f"{source}.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(sample)} genuine raw items to {sample_path}")

        return {
            "source": source,
            "actor_id": actor_id,
            "items_count": len(items),
            "sample_path": sample_path,
            "status": "SUCCESS",
        }
    except Exception as e:
        logger.error(f"Execution failed for actor '{actor_id}' ({source}): {e}")
        raise


def main():
    apify = ApifyHandler(config.APIFY_API_KEY)
    logger.info(f"Apify Key Status: {'CONFIGURED' if apify.is_configured else 'MISSING / EMPTY'}")

    if not apify.is_configured:
        logger.error(
            "FATAL: APIFY_API_KEY is missing from environment and .env file. "
            "Smoke testing requires a valid Apify API token."
        )
        sys.exit(1)

    actors_to_test = [
        (
            "telegram",
            config.APIFY_ACTOR_TELEGRAM,
            {
                "profiles": ["hkstockgains"],
                "collectMessages": True,
                "scrapeLastNDays": 3,
                "maxMessagesPerChannel": 5,
            },
        ),
        (
            "lihkg",
            config.APIFY_ACTOR_LIHKG,
            {
                "mode": "search",
                "bulkInput": "700",
                "catIds": ["5"],
                "maxItems": 5,
            },
        ),
        (
            "naver_finance",
            config.APIFY_ACTOR_NAVER,
            {
                "stocks": ["005930"],
                "maxResults": 5,
                "dateFrom": "2026-11-01",
            },
        ),
        (
            "aastocks",
            config.APIFY_ACTOR_AASTOCKS,
            {
                "feed": "latest-news",
                "language": "tc",
                "maxResults": 5,
            },
        ),
    ]

    failed_actors = []
    run_records = []

    for source, actor_id, run_input in actors_to_test:
        try:
            rec = test_actor(source, actor_id, run_input, apify)
            run_records.append(rec)
        except Exception as e:
            failed_actors.append((source, actor_id, str(e)))

    # Update docs/sample_raw_items/README.md with real run details
    readme_path = os.path.join(DOCS_SAMPLE_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Verified Actor Smoke Test Samples\n\n")
        f.write(f"Test Run Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write("| Source | Actor ID | Items Captured | Status |\n")
        f.write("|:---|:---|:---:|:---:|\n")
        for r in run_records:
            f.write(f"| {r['source']} | `{r['actor_id']}` | {r['items_count']} | {r['status']} |\n")
        for src, act, err in failed_actors:
            f.write(f"| {src} | `{act}` | 0 | FAILED: {err} |\n")

    if failed_actors:
        logger.error(f"\nSmoke tests FAILED for {len(failed_actors)} actor(s):")
        for src, act, err in failed_actors:
            logger.error(f"  • {src} ({act}): {err}")
        sys.exit(1)

    logger.info("\nALL ACTOR SMOKE TESTS PASSED SUCCESSFULLY.")
    sys.exit(0)


if __name__ == "__main__":
    main()
