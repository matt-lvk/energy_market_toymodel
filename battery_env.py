import numpy as np
import pandas as pd
from collections import defaultdict
import random
from dataclasses import dataclass, field


@dataclass
class BatteryEnv:
    price_series: np.ndarray
    max_capacity: int = 1
    max_rate: int = 1
    features: pd.DataFrame | None = None
    battery: float | None = None

    def __post_init__(self):
        self.T = len(self.price_series)
        self.reset()

    def reset(self):
        self.t = 0
        self.battery = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # Example state: (battery_level, hour)
        hour = self.t % 24
        state = (self.battery, hour)
        # Optionally add more features
        if self.features:
            feat = tuple(f[self.t] for f in self.features.values())
            state += feat
        return state

    def step(self, action):
        """
        action: 0=hold, 1=charge, 2=discharge
        """
        prev_battery = self.battery
        price = self.price_series[self.t]
        reward = 0

        if action == 1 and self.battery < self.max_capacity:
            # Charge
            self.battery = min(self.battery + self.max_rate, self.max_capacity)
            reward = -price  # Pay to charge
        elif action == 2 and self.battery > 0:
            # Discharge
            self.battery = max(self.battery - self.max_rate, 0)
            reward = price  # Earn from discharge
        # else: hold or invalid action (no penalty)

        self.t += 1
        self.done = (self.t >= self.T)
        return self._get_state(), reward, self.done, {}

    def action_space(self):
        return [0, 1, 2]  # hold, charge, discharge

    def state_space(self):
        # Discretize battery and hour, plus any features
        battery_states = np.arange(self.max_capacity + 1)
        hour_states = np.arange(24)
        # Optionally add more
        return battery_states, hour_states


@dataclass
class QLearningAgent:
    env: BatteryEnv
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    Q: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(3)))  # 3 actions: hold, charge, discharge

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.env.action_space())
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        a = action
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.Q[state][a]
        self.Q[state][a] += self.alpha * td_error

    def train(self, episodes=1000, verbose=True):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            if verbose and (ep + 1) % 100 == 0:
                print(f"Episode {ep + 1}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        return rewards

    def get_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy
