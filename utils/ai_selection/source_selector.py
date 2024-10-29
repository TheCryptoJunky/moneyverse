# Full file path: moneyverse/utils/ai_selection/source_selector.py

import time
import random
import logging
import requests
from collections import defaultdict
from centralized_logger import CentralizedLogger  # Leverage centralized logging
from utils.retry_decorator import retry  # Add retry logic for resiliency

# Initialize centralized logging
logger = CentralizedLogger()

class SourceSelector:
    """
    AI-driven SourceSelector class to optimize API source selection based on cost, reliability,
    and response times, with retry logic and real-time metrics tracking.
    """

    def __init__(self, api_configs):
        """
        Initialize with a list of API configurations.
        Each API config includes 'name', 'url', 'headers', 'cost_per_request', and 'rate_limit'.
        """
        self.api_configs = api_configs
        self.source_metrics = defaultdict(lambda: {
            "usage_count": 0,
            "fail_count": 0,
            "avg_response_time": 0,
            "total_cost": 0
        })
        self.response_time_window = 10  # Window for calculating average response time
        self.api_usage_limits = {api["name"]: api.get("rate_limit", 1000) for api in api_configs}
        self.last_usage_reset = time.time()

    @retry(retries=3, delay=2)  # Adding retry decorator for robustness
    def choose_best_source(self):
        """
        Chooses the best source based on a weighted score that considers cost, reliability,
        and efficiency. Sources nearing rate limits or with high costs are deprioritized.
        """
        scores = {}
        for api in self.api_configs:
            name = api["name"]
            metrics = self.source_metrics[name]
            usage_ratio = metrics["usage_count"] / self.api_usage_limits[name]
            fail_rate = metrics["fail_count"] / max(metrics["usage_count"], 1)
            avg_response_time = metrics["avg_response_time"]
            cost_per_use = metrics["total_cost"] / max(metrics["usage_count"], 1)

            # Heuristic scoring with weights
            score = (
                (1 - usage_ratio) * 0.4 +         # Less usage ratio is better
                (1 - fail_rate) * 0.3 +           # Lower failure rate is better
                (1 / (avg_response_time + 0.1)) * 0.2 +  # Lower response time is better
                (1 / (cost_per_use + 0.01)) * 0.1  # Lower cost per use is better
            )
            scores[name] = score

        best_source = max(scores, key=scores.get)
        logger.log("info", f"Selected best source: {best_source} with score: {scores[best_source]}")
        return next(api for api in self.api_configs if api["name"] == best_source)

    @retry(retries=3, delay=1)  # Ensures API call resiliency
    def call_api(self, api_config, params=None):
        """
        Calls the specified API and updates metrics based on success, response time, and cost.
        """
        try:
            start_time = time.time()
            response = requests.get(api_config["url"], headers=api_config.get("headers", {}), params=params)
            response.raise_for_status()
            elapsed_time = time.time() - start_time
            cost = api_config.get("cost_per_request", 0.01)  # Default cost per request
            self._update_metrics(api_config["name"], elapsed_time, cost, success=True)
            return response.json()
        except requests.RequestException as e:
            logger.log("error", f"API call to {api_config['name']} failed: {e}")
            self._update_metrics(api_config["name"], None, 0, success=False)
            return None

    def _update_metrics(self, api_name, response_time, cost, success):
        """
        Updates source metrics based on response time, cost, and success of the API call.
        """
        metrics = self.source_metrics[api_name]
        if success:
            metrics["usage_count"] += 1
            metrics["total_cost"] += cost
            if metrics["usage_count"] <= self.response_time_window:
                # Calculate cumulative average response time
                metrics["avg_response_time"] = (
                    (metrics["avg_response_time"] * (metrics["usage_count"] - 1) + response_time) / metrics["usage_count"]
                )
            else:
                # Rolling average for recent response times
                metrics["avg_response_time"] = (
                    (metrics["avg_response_time"] * (self.response_time_window - 1) + response_time) / self.response_time_window
                )
        else:
            metrics["fail_count"] += 1

        # Periodic reset to manage rate limits
        if time.time() - self.last_usage_reset > 86400:  # Reset every 24 hours
            self.reset_usage_counts()

    def reset_usage_counts(self):
        """
        Resets usage and fail counts for all sources daily to manage rate limits.
        """
        for api_name in self.source_metrics:
            self.source_metrics[api_name]["usage_count"] = 0
            self.source_metrics[api_name]["fail_count"] = 0
        self.last_usage_reset = time.time()
        logger.log("info", "Daily usage counts reset for all sources.")

    async def get_next_best_source(self, params=None):
        """
        Retrieves data from the best available source based on AI-driven selection.
        Attempts fallback sources if the primary one fails.
        """
        primary_source = self.choose_best_source()
        response = await self.call_api(primary_source, params=params)
        
        if response is None:  # Attempt to use fallback sources if primary fails
            alternative_sources = [api for api in self.api_configs if api["name"] != primary_source["name"]]
            for alt_source in alternative_sources:
                logger.log("info", f"Attempting alternative source: {alt_source['name']}")
                response = await self.call_api(alt_source, params=params)
                if response is not None:
                    break

        return response
