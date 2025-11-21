# ai/rabbit_q_learning.py - Q-Learning implementation
"""
SurvAIval Q-Learning System
Simple reinforcement learning for agent decision making
"""

import numpy as np
import random
from typing import Tuple, Dict, Optional
import pickle
import os


class QLearningAgent:
    """Q-Learning agent for decision making"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.3, exploration_decay: float = 0.995):
        """Initialize Q-Learning agent

        Args:
            learning_rate: How much to update Q-values (alpha)
            discount_factor: Future reward importance (gamma)
            exploration_rate: Probability of random action (epsilon)
            exploration_decay: How fast exploration rate decreases
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.05

        # Q-Table: stores state-action values
        # Format: q_table[state] = {action: value}
        self.q_table: Dict[Tuple, Dict[str, float]] = {}

        # Statistics
        self.total_episodes = 0
        self.total_rewards = 0.0
        self.actions_taken = 0

        # Available actions for rabbits
        self.actions = [
            'flee_up',
            'flee_down',
            'flee_left',
            'flee_right',
            'seek_food',
            'wander',
            'stay'
        ]

    def get_state(self, world_state: dict) -> Tuple:
        """Convert world state to simplified discrete state

        Args:
            world_state: Dictionary with agent's perception

        Returns:
            Tuple representing discrete state
        """
        # Discretize continuous values into bins

        # 1. Energy level (low, medium, high)
        energy = world_state.get('energy', 50)
        if energy < 30:
            energy_state = 0  # Critical
        elif energy < 60:
            energy_state = 1  # Low
        else:
            energy_state = 2  # Good

        # 2. Nearest predator (very close, close, medium, far, none)
        predators = world_state.get('predators', [])
        if predators:
            nearest_pred = min(predators,
                               key=lambda p: np.linalg.norm(p.position - world_state['position']))
            distance = np.linalg.norm(nearest_pred.position - world_state['position'])

            if distance < 30:
                predator_state = 0  # Very close - DANGER!
            elif distance < 60:
                predator_state = 1  # Close - Caution
            elif distance < 100:
                predator_state = 2  # Medium distance
            else:
                predator_state = 3  # Far
        else:
            predator_state = 4  # No predators

        # 3. Food availability (none, far, close)
        food_sources = world_state.get('food_sources', [])
        if food_sources:
            # Food sources are FoodSource objects, not dicts
            nearest_food = food_sources[0]

            # Calculate distance to food
            if hasattr(nearest_food, 'position'):
                food_pos = nearest_food.position
                agent_pos = world_state['position']
                food_distance = np.linalg.norm(food_pos - agent_pos)
            else:
                food_distance = 999

            if food_distance < 30:
                food_state = 2  # Very close
            elif food_distance < 80:
                food_state = 1  # Medium distance
            else:
                food_state = 0  # Far
        else:
            food_state = 0  # No food visible

        # Return discrete state tuple
        return (energy_state, predator_state, food_state)

    def get_action(self, state: Tuple, exploring: bool = True) -> str:
        """Get action using epsilon-greedy policy

        Args:
            state: Current state tuple
            exploring: Whether to use exploration

        Returns:
            Action string
        """
        # Initialize state in Q-table if not seen before
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # Epsilon-greedy: explore or exploit
        if exploring and random.random() < self.exploration_rate:
            # Explore: random action
            action = random.choice(self.actions)
        else:
            # Exploit: best known action
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            # Get all actions with max Q-value (in case of ties)
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)

        self.actions_taken += 1
        return action

    def update_q_value(self, state: Tuple, action: str, reward: float,
                       next_state: Tuple) -> None:
        """Update Q-value using Q-learning formula

        Q(s,a) = Q(s,a) + α * [reward + γ * max(Q(s',a')) - Q(s,a)]
        """
        # Initialize states if not in table
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        # Current Q-value
        current_q = self.q_table[state][action]

        # Max Q-value for next state
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning update
        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q
        self.total_rewards += reward

    def decay_exploration(self) -> None:
        """Decay exploration rate over time"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

    def calculate_reward(self, agent, old_energy: float, action: str,
                         died: bool = False, ate_food: bool = False) -> float:
        """Calculate reward for the agent's action

        Args:
            agent: The agent object
            old_energy: Energy before action
            action: Action taken
            died: Whether agent died
            ate_food: Whether agent successfully ate

        Returns:
            Reward value
        """
        reward = 0.0

        # Major penalties/rewards
        if died:
            reward -= 100.0  # Very bad!
            return reward

        if ate_food:
            reward += 15.0  # Good!

        # Energy-based rewards
        energy_change = agent.energy - old_energy

        if agent.energy < 20:
            reward -= 5.0  # Penalty for very low energy
        elif agent.energy > 70:
            reward += 2.0  # Reward for good energy

        # Action-specific rewards
        if action == 'flee_up' or action == 'flee_down' or action == 'flee_left' or action == 'flee_right':
            # Check if there was actually a predator nearby
            # (This should be tracked by the agent)
            if hasattr(agent, 'panic_timer') and agent.panic_timer > 0:
                reward += 3.0  # Good decision to flee

        # Small survival reward each step
        reward += 0.1

        # Penalty for wasting energy
        if energy_change < -0.5:
            reward -= 0.5

        return reward

    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'exploration_rate': self.exploration_rate,
                    'total_episodes': self.total_episodes,
                    'total_rewards': self.total_rewards
                }, f)
            print(f"💾 Q-table saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save Q-table: {e}")

    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from file"""
        try:
            if not os.path.exists(filepath):
                print(f"⚠️ No saved Q-table found at {filepath}")
                return False

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
                self.total_episodes = data.get('total_episodes', 0)
                self.total_rewards = data.get('total_rewards', 0.0)

            print(f"📂 Q-table loaded from {filepath}")
            print(f"   States learned: {len(self.q_table)}")
            print(f"   Episodes: {self.total_episodes}")
            return True
        except Exception as e:
            print(f"❌ Failed to load Q-table: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get learning statistics"""
        return {
            'q_table_size': len(self.q_table),
            'exploration_rate': self.exploration_rate,
            'total_episodes': self.total_episodes,
            'total_rewards': self.total_rewards,
            'avg_reward': self.total_rewards / max(1, self.total_episodes),
            'actions_taken': self.actions_taken
        }

    def print_best_policies(self, top_n: int = 10) -> None:
        """Print best learned policies"""
        print("\n🎓 Best Learned Policies:")
        print("-" * 50)

        # Sort states by total Q-value
        state_values = [(state, sum(actions.values()))
                        for state, actions in self.q_table.items()]
        state_values.sort(key=lambda x: x[1], reverse=True)

        state_names = {
            0: "Critical Energy",
            1: "Low Energy",
            2: "Good Energy"
        }

        predator_names = {
            0: "Predator VERY CLOSE",
            1: "Predator Close",
            2: "Predator Medium",
            3: "Predator Far",
            4: "No Predator"
        }

        food_names = {
            0: "No Food",
            1: "Food Medium",
            2: "Food Close"
        }

        for i, (state, total_value) in enumerate(state_values[:top_n]):
            energy, predator, food = state
            best_action = max(self.q_table[state].items(), key=lambda x: x[1])

            print(f"\n{i + 1}. State: {state_names[energy]}, {predator_names[predator]}, {food_names[food]}")
            print(f"   Best Action: {best_action[0]} (Q={best_action[1]:.2f})")
            print(f"   Total Value: {total_value:.2f}")