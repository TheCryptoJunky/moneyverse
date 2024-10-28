# Full file path: /moneyverse/managers/transaction_manager.py

import asyncio
import mysql.connector
from centralized_logger import CentralizedLogger
from src.utils.error_handler import handle_errors
from src.database.database_manager import DatabaseManager

logger = CentralizedLogger()
db_manager = DatabaseManager()

class TransactionManager:
    """
    Manages the lifecycle of trading agents (bots) by initializing, tracking, and stopping them.
    Each agent operates based on a specified strategy and updates its state in the database.
    """

    def __init__(self):
        self.agents = []

    async def initialize_agents(self, strategies):
        """
        Asynchronously initialize multiple agents based on the provided strategies.
        Logs each initialization and stores lifecycle events in the database.
        """
        try:
            logger.log("info", "Initializing agents...")
            tasks = [self.start_agent(strategy) for strategy in strategies]
            await asyncio.gather(*tasks)
            logger.log("info", "All agents initialized successfully.")

        except Exception as e:
            logger.log("error", f"Error initializing agents: {str(e)}")
            handle_errors(e)

    async def start_agent(self, strategy):
        """
        Start an individual agent with the specified strategy.
        The agent's state is stored in the MySQL database for monitoring.
        
        Parameters:
            strategy (dict): Strategy details including the name and execution interval.
        """
        try:
            logger.log("info", f"Starting agent for strategy: {strategy['name']}")

            while True:
                await self.store_agent_state_in_db(strategy, "running")
                await asyncio.sleep(strategy["interval"])  # Defines the agent's operational cycle
                logger.log("info", f"Agent for {strategy['name']} completed a cycle.")

        except Exception as e:
            logger.log("error", f"Error in agent {strategy['name']}: {str(e)}")
            handle_errors(e)

    async def store_agent_state_in_db(self, strategy, status):
        """
        Store the current operational state of the agent in the database.
        This helps track whether agents are active, paused, or stopped.
        
        Parameters:
            strategy (dict): Strategy details.
            status (str): Current status of the agent ("running", "stopped", etc.).
        """
        try:
            connection = db_manager.get_connection()
            cursor = connection.cursor()

            # Insert the agent's state into the `agents` table
            query = """
                INSERT INTO agents (strategy_name, status, timestamp)
                VALUES (%s, %s, NOW())
            """
            data = (strategy["name"], status)
            cursor.execute(query, data)
            connection.commit()
            cursor.close()

            logger.log("info", f"Agent {strategy['name']} state stored in database: {status}")

        except mysql.connector.Error as err:
            logger.log("error", f"MySQL error during agent state storage: {err}")
            handle_errors(err)

    def stop_all_agents(self):
        """
        Gracefully stop all active agents and log their states in the database.
        """
        logger.log("info", "Stopping all agents.")
        for agent in self.agents:
            self.store_agent_state_in_db(agent, "stopped")
        logger.log("info", "All agents have been stopped and their states updated.")
