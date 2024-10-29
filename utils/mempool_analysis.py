import asyncio
from datetime import datetime
from web3 import Web3
import pandas as pd  # For data storage and export
from ai.rl_agent import MEVOpportunityAgent
from centralized_logger import CentralizedLogger
from sqlalchemy import create_engine
import os

# Initialize centralized logger and database engine
logger = CentralizedLogger()
engine = create_engine(os.getenv("DATABASE_URL"))  # Ensure this points to your database

class MempoolAnalysis:
    """
    Analyzes the mempool to identify MEV opportunities, store historical data in chunks,
    and support data export for offline training.
    """

    def __init__(self, web3_provider, chunk_size=100, export_path="exports/"):
        self.web3 = Web3(Web3.HTTPProvider(web3_provider))
        self.mev_agent = MEVOpportunityAgent()
        self.chunk_size = chunk_size
        self.collected_data = []
        self.is_collecting = False  # Toggle for data collection mode
        self.collection_interval = 60  # Default time interval in seconds
        self.export_path = export_path
        os.makedirs(export_path, exist_ok=True)  # Ensure export directory exists

    async def fetch_mempool_data(self):
        """Asynchronously fetch mempool data for real-time analysis."""
        try:
            await asyncio.sleep(0.1)  # Simulate async data fetch
            mempool_data = []  # Placeholder
            logger.log("info", "Fetched mempool data.")
            return mempool_data
        except Exception as e:
            logger.log("error", f"Failed to fetch mempool data: {str(e)}")
            return None

    async def analyze_mempool(self):
        """Analyzes the fetched mempool data."""
        mempool_data = await self.fetch_mempool_data()
        if self.is_collecting and mempool_data:
            await self.store_mempool_data(mempool_data)  # Store for offline training
        if mempool_data:
            await self.identify_mev_opportunities(mempool_data)

    async def identify_mev_opportunities(self, mempool_data):
        """Identifies MEV opportunities within the fetched data."""
        try:
            opportunities = [tx for tx in mempool_data if self.mev_agent.evaluate_opportunity(tx)]
            if opportunities:
                logger.log("info", f"Identified {len(opportunities)} MEV opportunities.")
        except Exception as e:
            logger.log("error", f"MEV opportunity identification error: {str(e)}")

    async def store_mempool_data(self, data_chunk):
        """Stores mempool data in the database in chunks, suitable for training."""
        self.collected_data.extend(data_chunk)
        if len(self.collected_data) >= self.chunk_size:
            df = pd.DataFrame(self.collected_data)
            timestamp = datetime.utcnow()
            try:
                df.to_sql("mempool_history", engine, if_exists="append", index=False)
                logger.log("info", f"Stored mempool chunk of size {len(df)} at {timestamp}")
                self.collected_data = []  # Reset chunk
            except Exception as e:
                logger.log("error", f"Failed to store data chunk: {str(e)}")

    def export_data(self, start_time=None, end_time=None):
        """
        Exports historical mempool data within a time range to CSV for offline training.
        
        Parameters:
            start_time (datetime): Starting time for data export.
            end_time (datetime): Ending time for data export.
        """
        query = "SELECT * FROM mempool_history"
        if start_time and end_time:
            query += f" WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'"
        
        try:
            df = pd.read_sql(query, engine)
            file_name = os.path.join(self.export_path, f"mempool_data_{start_time}_{end_time}.csv")
            df.to_csv(file_name, index=False)
            logger.log("info", f"Exported data to {file_name}")
        except Exception as e:
            logger.log("error", f"Failed to export data: {str(e)}")

    async def run_analysis_loop(self, collect_mode=False, interval_type="time", interval_value=60):
        """
        Continuously run the analysis loop, optionally collecting data for offline training.

        Parameters:
            collect_mode (bool): Enable or disable data collection mode.
            interval_type (str): 'time' for time-based collection or 'blocks' for block-based.
            interval_value (int): Interval for collection in seconds (time-based) or blocks.
        """
        self.is_collecting = collect_mode
        self.collection_interval = interval_value
        logger.log("info", f"Starting mempool analysis loop. Collect mode: {collect_mode}, interval: {interval_value} {interval_type}.")
        
        if interval_type == "time":
            while True:
                await self.analyze_mempool()
                await asyncio.sleep(self.collection_interval)
        elif interval_type == "blocks":
            block_count = 0
            while True:
                await self.analyze_mempool()
                block_count += 1
                if block_count >= self.collection_interval:
                    await asyncio.sleep(5)  # Adjust based on block time
                    block_count = 0

