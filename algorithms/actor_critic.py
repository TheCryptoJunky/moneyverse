import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from .replay_buffer import ReplayBuffer

class ActorCriticAgent:
    """
    Actor-Critic agent that learns optimal actions through policy and value updates.

    Attributes:
    - state_size (int): Size of the input state space.
    - action_size (int): Size of the output action space.
    - gamma (float): Discount factor for future rewards.
    - actor_model (Sequential): Neural network model for policy (actor).
    - critic_model (Sequential): Neural network model for value estimation (critic).
    - replay_buffer (ReplayBuffer): Shared experience buffer.
    - learning_rate (float): Learning rate for adaptive updates.
    """

    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.actor_model = self._build_actor()
        self.critic_model = self._build_critic()
        self.replay_buffer = ReplayBuffer(max_size=2000)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Actor-Critic agent initialized.")

    def _build_actor(self):
        """
        Builds and compiles the actor model.
        
        Returns:
        - Sequential: Compiled actor model.
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="softmax")
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="categorical_crossentropy")
        self.logger.info("Actor model built and compiled.")
        return model

    def _build_critic(self):
        """
        Builds and compiles the critic model.
        
        Returns:
        - Sequential: Compiled critic model.
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        self.logger.info("Critic model built and compiled.")
        return model

    def act(self, state):
        """
        Selects an action based on the current policy.

        Args:
        - state (np.ndarray): Current state.

        Returns:
        - int: Action index.
        """
        policy = self.actor_model.predict(state)[0]
        action = np.random.choice(self.action_size, p=policy)
        self.logger.debug(f"Selected action {action} with policy {policy}.")
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.

        Args:
        - state: Current state.
        - action: Action taken.
        - reward: Reward received.
        - next_state: Next state after action.
        - done (bool): Whether the episode has ended.
        """
        self.replay_buffer.store((state, action, reward, next_state, done))
        self.logger.debug("Stored experience in replay buffer.")

    def train(self, batch_size=32):
        """
        Trains the actor and critic networks using experiences from the replay buffer.

        Args:
        - batch_size (int): Number of experiences to sample for training.
        """
        if len(self.replay_buffer) < batch_size:
            self.logger.warning("Insufficient experiences for training.")
            return

        experiences, _ = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state, done in experiences:
            target = reward + (1 - done) * self.gamma * self.critic_model.predict(next_state)[0]
            td_error = target - self.critic_model.predict(state)[0]
            
            # Update critic
            self.critic_model.fit(state, target, verbose=0)

            # Update actor
            action_onehot = np.zeros(self.action_size)
            action_onehot[action] = 1
            self.actor_model.fit(state, action_onehot * td_error, verbose=0)
        
        # Adjust learning rate dynamically based on training progress
        self.learning_rate = max(0.0001, self.learning_rate * 0.995)
        self.logger.info(f"Trained on batch; learning rate adjusted to {self.learning_rate}.")

    def adaptive_learning_rate(self, initial_lr, decay_rate, min_lr):
        """
        Adjusts the learning rate adaptively.

        Args:
        - initial_lr (float): Starting learning rate.
        - decay_rate (float): Rate of learning rate decay.
        - min_lr (float): Minimum learning rate.
        """
        self.learning_rate = max(min_lr, initial_lr * decay_rate)
        self.logger.info(f"Learning rate updated to {self.learning_rate}.")
