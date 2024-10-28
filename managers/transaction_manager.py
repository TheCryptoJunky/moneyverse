import asyncio
import mysql.connector
from centralized_logger import CentralizedLogger
from src.utils.error_handler import handle_errors
from src.database.database_manager import DatabaseManager

logger = CentralizedLogger()
db_manager = DatabaseManager()

class MultiAgentManager:
    def __init__(self):
        self.agents = []

    async def initialize_agents(self, strategies):
        """
        Asynchronously initialize multiple agents (bots) based on the given strategies.
        Stores the lifecycle events of the agents in the MySQL database.
        """
        try:
            logger.log("info", "Initializing agents...")
            tasks = []

            for strategy in strategies:
                tasks.append(self.start_agent(strategy))

            await asyncio.gather(*tasks)
            logger.log("info", "All agents initialized successfully.")

        except Exception as e:
            logger.log("error", f"Error initializing agents: {str(e)}")
            handle_errors(e)

    async def start_agent(self, strategy):
        """
        Start an individual agent (bot) with the given strategy.
        Store the agent's state in the MySQL database for tracking purposes.
        """
        try:
            logger.log("info", f"Starting agent for strategy: {strategy['name']}")

            while True:
                await self.store_agent_state_in_db(strategy, "running")
                await asyncio.sleep(strategy["interval"])
                logger.log("info", f"Agent for {strategy['name']} completed a cycle.")

        except Exception as e:
            logger.log("error", f"Error in agent {strategy['name']}: {str(e)}")
            handle_errors(e)

    async def store_agent_state_in_db(self, strategy, status):
        """
        Store the current state of the agent in the MySQL database.
        This helps keep track of active and paused agents.
        """
        try:
            connection = db_manager.get_connection()
            cursor = connection.cursor()

            # Insert agent state into the agents table
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
        Stop all active agents and log the state in the MySQL database.
        """
        logger.log("info", "Stopping all agents.")
        # Logic for safely stopping agents
        for agent in self.agents:
            self.store_agent_state_in_db(agent, "stopped")
