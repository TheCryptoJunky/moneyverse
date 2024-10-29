# Full file path: moneyverse/utils/api_integrations.py

import os
import requests
import asyncio
from aiocache import Cache
from centralized_logger import CentralizedLogger

# Initialize centralized logger
logger = CentralizedLogger()

class APIIntegrations:
    """
    Handles API integrations for gas estimation, asset swapping, and aggregator calls with error handling and caching.
    """

    def __init__(self):
        # Securely load API keys from environment variables
        self.api_keys = {
            "eth_gas_station": os.getenv("ETH_GAS_STATION_API_KEY"),
            "uniswap": os.getenv("UNISWAP_API_KEY"),
            "sushiswap": os.getenv("SUSHISWAP_API_KEY"),
            "1inch": os.getenv("1INCH_API_KEY"),
            "paraswap": os.getenv("PARASWAP_API_KEY")
        }
        # Setup cache for gas estimation responses
        self.cache = Cache(Cache.MEMORY)
    
    async def estimate_gas(self, tx_hash):
        """
        Estimates gas using a caching mechanism to reduce external API calls.
        
        Args:
            tx_hash (str): The transaction hash for gas estimation.

        Returns:
            float: Estimated gas price.
        """
        try:
            cached_gas = await self.cache.get(tx_hash)
            if cached_gas:
                return cached_gas

            # Example call to an external gas API
            url = f"https://ethgasstation.info/api/ethgasAPI.json?api-key={self.api_keys['eth_gas_station']}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            gas_price = data.get("average") / 10  # Example: converting Gwei to Ether
            await self.cache.set(tx_hash, gas_price, ttl=60)  # Cache for 1 minute
            logger.log("info", f"Estimated gas for {tx_hash}: {gas_price}")
            return gas_price

        except requests.RequestException as e:
            logger.log("error", f"Error estimating gas for {tx_hash}: {str(e)}")
            return None

    async def swap_assets(self, asset1, asset2, amount):
        """
        Swaps assets using a decentralized exchange API with error handling.
        
        Args:
            asset1 (str): Symbol or contract address of the asset to swap from.
            asset2 (str): Symbol or contract address of the asset to swap to.
            amount (float): Amount of asset1 to swap.

        Returns:
            dict: Swap transaction details.
        """
        try:
            url = "https://api.uniswap.org/v2/swap"  # Replace with actual Uniswap endpoint
            payload = {
                "fromToken": asset1,
                "toToken": asset2,
                "amount": amount,
                "api_key": self.api_keys["uniswap"]
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            swap_data = response.json()
            logger.log("info", f"Swapped {amount} {asset1} to {asset2}: {swap_data}")
            return swap_data

        except requests.RequestException as e:
            logger.log("error", f"Error swapping {asset1} to {asset2}: {str(e)}")
            return None

    async def aggregator_api_call(self, aggregator_url, payload):
        """
        Executes an API call to a liquidity aggregator with centralized error handling.
        
        Args:
            aggregator_url (str): URL of the aggregator API endpoint.
            payload (dict): Payload for the API request.

        Returns:
            dict: Aggregator API response data.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_keys['1inch']}"}
            response = requests.post(aggregator_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.log("info", f"Aggregator call successful: {data}")
            return data

        except requests.RequestException as e:
            logger.log("error", f"Error in aggregator API call to {aggregator_url}: {str(e)}")
            return None
