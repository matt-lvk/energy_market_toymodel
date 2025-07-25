import numpy as np
import pandas as pd
from collections import defaultdict
import random
from dataclasses import dataclass, field


@dataclass
class BatteryEnv:
    price_series: np.ndarray
    max_capacity: float = 1.0   # MWh
    max_rate: float = 1.0       # MWh per step
    features: pd.DataFrame | None = None
    battery: float = field(init=False)
    prev_battery: float = field(init=False)

    def __post_init__(self):
        self.T = len(self.price_series)
        self.reset()

    def reset(self):
        self.t = 0
        self.battery = 0.0
        self.prev_battery = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        hour = self.t % 24
        state = (round(self.battery, 3), hour)
        if self.features is not None:
            feat = tuple(self.features.iloc[self.t])
            state += feat
        return state

    def step(self, action):
        """
        action: 0=hold, 1=charge, 2=discharge
        """
        if self.done:
            raise Exception("Episode is done. Call reset().")

        self.prev_battery = self.battery
        price = self.price_series[self.t]
        reward = 0.0

        if action == 1:  # charge
            charge_amt = min(self.max_rate, self.max_capacity - self.battery)
            self.battery += charge_amt
            reward = -price * charge_amt
        elif action == 2:  # discharge
            discharge_amt = min(self.max_rate, self.battery)
            self.battery -= discharge_amt
            reward = price * discharge_amt
        # else: hold = 0  # hold is doing nothing, left it be

        # Check battery max capa
        self.battery = max(0.0, min(self.battery, self.max_capacity))

        self.t += 1
        self.done = self.t >= self.T
        return self._get_state(), reward, self.done, {}

    def action_space(self):
        return [0, 1, 2]  # hold, charge, discharge

    def state_space(self):
        battery_states = np.linspace(0, self.max_capacity, int(self.max_capacity * 10) + 1)
        hour_states = np.arange(24)
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
