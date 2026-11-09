"""
Apify REST API client and wrapper.
Supports both the official apify-client package and a pure urllib fallback
for zero-dependency runtime resilience.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ApifyHandler:
    """Handles communication with Apify platform for web scraping actors."""

    BASE_URL = "https://api.apify.com/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ""
        self._has_client = False
        try:
            from apify_client import ApifyClient
            if self.api_key:
                self.client = ApifyClient(self.api_key)
                self._has_client = True
        except ImportError:
            self._has_client = False

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def run_actor_and_fetch_items(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
        timeout_secs: int = 120,
        memory_mbytes: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Execute an Apify Actor and retrieve the items stored in the resulting default dataset.
        """
        if not self.is_configured:
            logger.warning(f"Apify API key not provided. Cannot run actor {actor_id}.")
            return []

        # If apify_client is available, use official SDK
        if self._has_client:
            try:
                logger.info(f"Triggering Apify actor {actor_id} via official SDK...")
                run = self.client.actor(actor_id).call(
                    run_input=run_input,
                    timeout_secs=timeout_secs,
                    memory_mbytes=memory_mbytes,
                )
                if not run or "defaultDatasetId" not in run:
                    logger.error(f"Actor {actor_id} execution failed or returned no dataset.")
                    return []
                dataset_id = run["defaultDatasetId"]
                dataset = self.client.dataset(dataset_id)
                items = list(dataset.iterate_items())
                logger.info(f"Successfully fetched {len(items)} items from dataset {dataset_id}")
                return items
            except Exception as e:
                logger.error(f"Error running Apify actor {actor_id} via SDK: {e}")
                return []

        # Pure urllib REST API execution fallback
        try:
            logger.info(f"Triggering Apify actor {actor_id} via REST API...")
            encoded_actor = actor_id.replace("/", "~")
            run_url = f"{self.BASE_URL}/acts/{encoded_actor}/runs?token={self.api_key}&timeout={timeout_secs}&memory={memory_mbytes}"
            data_bytes = json.dumps(run_input).encode("utf-8")
            req = urllib.request.Request(
                run_url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            run_id = resp_data.get("data", {}).get("id")
            dataset_id = resp_data.get("data", {}).get("defaultDatasetId")
            if not dataset_id:
                logger.error("No dataset ID in Apify run response")
                return []

            # Poll for run completion if needed
            status = resp_data.get("data", {}).get("status", "READY")
            start_poll = time.time()
            while status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                if time.time() - start_poll > timeout_secs:
                    logger.warning(f"Actor run {run_id} timed out while polling.")
                    break
                time.sleep(3)
                check_url = f"{self.BASE_URL}/actor-runs/{run_id}?token={self.api_key}"
                with urllib.request.urlopen(check_url, timeout=15) as check_resp:
                    check_json = json.loads(check_resp.read().decode("utf-8"))
                    status = check_json.get("data", {}).get("status", "RUNNING")

            # Fetch dataset items
            items_url = f"{self.BASE_URL}/datasets/{dataset_id}/items?token={self.api_key}&format=json"
            with urllib.request.urlopen(items_url, timeout=30) as dataset_resp:
                items = json.loads(dataset_resp.read().decode("utf-8"))
                logger.info(f"Fetched {len(items)} items from Apify REST dataset {dataset_id}")
                return items
        except Exception as e:
            logger.error(f"Apify REST API error for actor {actor_id}: {e}")
            return []
