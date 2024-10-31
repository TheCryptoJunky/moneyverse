import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from .replay_buffer import ReplayBuffer

class ReinforcementLearningAgent:
    """
    Reinforcement Learning Agent that learns optimal actions through Q-learning.
    
    Attributes:
    - state_size (int): Size of the input state space.
    - action_size (int): Number of possible actions.
    - gamma (float): Discount factor for future rewards.
    - epsilon (float): Exploration-exploitation parameter.
    - model (Sequential): Q-network for action-value estimation.
    - replay_buffer (ReplayBuffer): Experience replay buffer for training.
    """

    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.replay_buffer = ReplayBuffer(max_size=2000)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Reinforcement Learning Agent initialized.")

    def _build_model(self):
        """
        Builds and compiles the Q-network model.

        Returns:
        - Sequential: Compiled Q-network model.
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        self.logger.info("Q-network model built and compiled.")
        return model

    def act(self, state):
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
        - state (np.ndarray): Current state.

        Returns:
        - int: Selected action index.
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)  # Exploration
            self.logger.debug(f"Exploration: Random action selected {action}")
        else:
            action_values = self.model.predict(state)
            action = np.argmax(action_values[0])  # Exploitation
            self.logger.debug(f"Exploitation: Best action selected {action}")
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.

        Args:
        - state (np.ndarray): Current state.
        - action (int): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state.
        - done (bool): Whether the episode ended.
        """
        self.replay_buffer.store((state, action, reward, next_state, done))
        self.logger.info("Stored experience in replay buffer.")

    def replay(self, batch_size=32):
        """
        Trains the model using experiences from the replay buffer.

        Args:
        - batch_size (int): Number of experiences to sample for training.
        """
        if len(self.replay_buffer) < batch_size:
            self.logger.warning("Not enough experiences to train.")
            return

        minibatch, weights = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.logger.info(f"Trained on batch; epsilon decayed to {self.epsilon}")

    def save(self, path):
        """
        Saves the model to a file.

        Args:
        - path (str): File path to save the model.
        """
        self.model.save(path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path):
        """
        Loads the model from a file.

        Args:
        - path (str): File path to load the model.
        """
        self.model.load_weights(path)
        self.logger.info(f"Model loaded from {path}")
