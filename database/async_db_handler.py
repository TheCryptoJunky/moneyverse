import asyncpg
from database.config import DB_CONFIG
from centralized_logger import CentralizedLogger

logger = CentralizedLogger()

class AsyncDBHandler:
    """
    Asynchronous Database Handler for managing all database operations.
    Provides centralized connection pooling, query execution, and transaction management.
    """

    def __init__(self):
        self.pool = None

    async def init_pool(self):
        """
        Initialize connection pool for async database operations.
        """
        try:
            self.pool = await asyncpg.create_pool(
                user=DB_CONFIG['USER'],
                password=DB_CONFIG['PASSWORD'],
                database=DB_CONFIG['DATABASE'],
                host=DB_CONFIG['HOST'],
                port=DB_CONFIG['PORT'],
                min_size=5,  # Minimum connections in the pool
                max_size=20  # Maximum connections in the pool
            )
            logger.log("info", "Database connection pool initialized.")
        except Exception as e:
            logger.log("error", f"Failed to initialize database pool: {e}")

    async def fetch(self, query, *args):
        """
        Execute a SELECT query asynchronously with error handling.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetch(query, *args)
                logger.log("info", f"Fetch query executed successfully: {query}")
                return result
        except Exception as e:
            logger.log("error", f"Error executing fetch query: {query} - {e}")
            return None

    async def execute(self, query, *args):
        """
        Execute an INSERT, UPDATE, or DELETE query asynchronously with error handling.
        """
        try:
            async with self.pool.acquire() as connection:
                result = await connection.execute(query, *args)
                logger.log("info", f"Execute query executed successfully: {query}")
                return result
        except Exception as e:
            logger.log("error", f"Error executing query: {query} - {e}")
            return None

    async def execute_transaction(self, queries):
        """
        Execute a series of queries within a single transaction asynchronously.
        
        Parameters:
            queries (list): A list of (query, args) tuples for execution.
        """
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                try:
                    for query, args in queries:
                        await connection.execute(query, *args)
                    logger.log("info", "Transaction executed successfully.")
                except Exception as e:
                    logger.log("error", f"Transaction failed: {e}")
                    raise

    async def close_pool(self):
        """
        Gracefully close the connection pool.
        """
        if self.pool:
            await self.pool.close()
            logger.log("info", "Database connection pool closed.")
        else:
            logger.log("warning", "Attempted to close an uninitialized pool.")
