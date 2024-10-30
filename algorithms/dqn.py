import numpy as np
import logging
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from ..database.db_connection import DatabaseConnection
from .replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent that learns optimal actions through reinforcement learning.
    
    Attributes:
    - db (DatabaseConnection): Database for logging performance.
    - state_size (int): Size of the state space.
    - action_size (int): Size of the action space.
    - memory (ReplayBuffer): Experience replay buffer for training.
    - epsilon (float): Exploration-exploitation parameter.
    """

    def __init__(self, state_size, action_size, db: DatabaseConnection, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(max_size=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self._build_model()
        self.db = db
        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        """
        Builds the neural network model for the DQN.
        
        Returns:
        - keras.Sequential: Compiled neural network model.
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        self.logger.info("DQN model built and compiled.")
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        
        Args:
        - state: Current state.
        - action: Action taken.
        - reward: Reward received.
        - next_state: Next state after action.
        - done (bool): Whether the episode has ended.
        """
        self.memory.store((state, action, reward, next_state, done))
        self.logger.debug("Experience stored in replay buffer.")

    def act(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.
        
        Args:
        - state: Current state.
        
        Returns:
        - int: Action index.
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
            self.logger.debug("Exploration: Random action selected.")
        else:
            action_values = self.model.predict(state)
            action = np.argmax(action_values[0])
            self.logger.debug("Exploitation: Best action selected.")
        return action

    def replay(self):
        """
        Trains the DQN model using experiences from the replay buffer.
        """
        if len(self.memory) < self.batch_size:
            self.logger.warning("Insufficient experiences for replay.")
            return

        minibatch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.logger.info(f"Trained on replay batch; epsilon now {self.epsilon}")

    def save(self, name):
        """
        Saves the model to disk.
        
        Args:
        - name (str): File path to save the model.
        """
        self.model.save(name)
        self.logger.info(f"Model saved to {name}")

    def load(self, name):
        """
        Loads a model from disk.
        
        Args:
        - name (str): File path to load the model from.
        """
        self.model.load_weights(name)
        self.logger.info(f"Model loaded from {name}")
