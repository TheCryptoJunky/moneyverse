# Full file path: moneyverse/utils/memory.py

import asyncpg
import json
from datetime import datetime
from typing import Any, Dict, Tuple, List

class Memory:
    """
    Memory class for experience replay in reinforcement learning with database storage
    and Hierarchical Memory Networks (HMNs).
    """
    def __init__(self, db_config: Dict[str, Any], prioritized_replay: bool = False, episodic_replay: bool = False):
        """
        Initialize the Memory buffer with database connection and prioritized replay options.

        Args:
            db_config (Dict[str, Any]): Database connection configuration.
            prioritized_replay (bool): Enable prioritized experience replay.
            episodic_replay (bool): Enable episodic memory grouping.
        """
        self.db_config = db_config
        self.prioritized_replay = prioritized_replay
        self.episodic_replay = episodic_replay

    async def init_db(self):
        """
        Initializes the database connection pool.
        """
        self.pool = await asyncpg.create_pool(**self.db_config)

    async def add_experience(self, state: Any, action: Any, reward: float, next_state: Any, td_error: float, episode: int):
        """
        Stores an experience in the database.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (float): The reward received.
            next_state (Any): The next state.
            td_error (float): The TD error, used for prioritized replay.
            episode (int): Episode ID for episodic replay.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO experiences (state, action, reward, next_state, td_error, episode)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                json.dumps(state), json.dumps(action), reward, json.dumps(next_state), td_error, episode
            )

    async def sample_experiences(self, batch_size: int) -> List[Tuple]:
        """
        Sample experiences from the database based on prioritized or episodic replay.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            List[Tuple]: Sampled experiences.
        """
        async with self.pool.acquire() as conn:
            if self.prioritized_replay:
                rows = await conn.fetch(
                    """
                    SELECT * FROM experiences ORDER BY td_error DESC LIMIT $1
                    """, batch_size
                )
            elif self.episodic_replay:
                rows = await conn.fetch(
                    """
                    SELECT * FROM experiences ORDER BY episode DESC LIMIT $1
                    """, batch_size
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM experiences ORDER BY RANDOM() LIMIT $1
                    """, batch_size
                )
            return [(row["state"], row["action"], row["reward"], row["next_state"]) for row in rows]

    async def add_hierarchical_memory(self, high_level_memory: Dict[str, Any], low_level_memory: Dict[str, Any]):
        """
        Adds hierarchical memory to the database.

        Args:
            high_level_memory (Dict[str, Any]): High-level memory data.
            low_level_memory (Dict[str, Any]): Low-level memory data.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO hierarchical_memories (high_level_memory, low_level_memory)
                VALUES ($1, $2)
                """,
                json.dumps(high_level_memory), json.dumps(low_level_memory)
            )

    async def fetch_hierarchical_memory(self, num_records: int) -> List[Tuple]:
        """
        Retrieves hierarchical memories from the database for training multi-level decision-making.

        Args:
            num_records (int): Number of hierarchical memories to retrieve.

        Returns:
            List[Tuple]: Retrieved hierarchical memory data.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT high_level_memory, low_level_memory FROM hierarchical_memories ORDER BY timestamp DESC LIMIT $1
                """, num_records
            )
            return [(json.loads(row["high_level_memory"]), json.loads(row["low_level_memory"])) for row in rows]
