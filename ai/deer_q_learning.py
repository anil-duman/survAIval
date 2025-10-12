# ai/deer_q_learning.py - Q-Learning for deer
"""
SurvAIval Deer Q-Learning System
Deer learn herd behavior and survival strategies
"""

import numpy as np
import random
from typing import Tuple, Dict, Optional
import pickle
import os


class DeerQLearningAgent:
    """Q-Learning agent specialized for deer herd behavior"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.3, exploration_decay: float = 0.9995):
        """Initialize Deer Q-Learning agent

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

        # Q-Table
        self.q_table: Dict[Tuple, Dict[str, float]] = {}

        # Statistics
        self.total_episodes = 0
        self.total_rewards = 0.0
        self.times_saved_by_herd = 0
        self.times_lost_herd = 0

        # Available actions for deer
        self.actions = [
            'flee_to_herd',
            'flee_alone',
            'stay_with_herd',
            'seek_food_near_herd',
            'seek_food_alone',
            'follow_leader',
            'wander_near_herd',
            'rest'
        ]

    def get_state(self, world_state: dict) -> Tuple:
        """Convert world state to discrete state for deer

        Args:
            world_state: Dictionary with deer's perception

        Returns:
            Tuple representing discrete state
        """
        # 1. Energy level
        energy = world_state.get('energy', 50)
        if energy < 25:
            energy_state = 0  # Critical
        elif energy < 50:
            energy_state = 1  # Low
        elif energy < 75:
            energy_state = 2  # Medium
        else:
            energy_state = 3  # High

        # 2. Predator threat level
        predators = world_state.get('predators', [])
        if predators:
            nearest_pred = min(predators,
                               key=lambda p: np.linalg.norm(p.position - world_state['position']))
            distance = np.linalg.norm(nearest_pred.position - world_state['position'])

            if distance < 40:
                predator_state = 0  # IMMEDIATE DANGER
            elif distance < 70:
                predator_state = 1  # Close danger
            elif distance < 110:
                predator_state = 2  # Medium distance
            else:
                predator_state = 3  # Far
        else:
            predator_state = 4  # No predators

        # 3. Herd status (critical for deer)
        herd_mates = world_state.get('herd_mates', [])
        herd_size = len(herd_mates)

        if herd_size == 0:
            herd_state = 0  # Alone - DANGEROUS!
        elif herd_size <= 1:
            herd_state = 1  # Small group
        elif herd_size <= 3:
            herd_state = 2  # Medium herd
        else:
            herd_state = 3  # Large herd - SAFE!

        # 4. Distance to herd center
        if herd_size > 0:
            herd_center = np.mean([mate.position for mate in herd_mates] + [world_state['position']], axis=0)
            distance_to_herd = np.linalg.norm(herd_center - world_state['position'])

            if distance_to_herd < 30:
                herd_distance_state = 2  # In herd
            elif distance_to_herd < 70:
                herd_distance_state = 1  # Near herd
            else:
                herd_distance_state = 0  # Far from herd
        else:
            herd_distance_state = 0

        return (energy_state, predator_state, herd_state, herd_distance_state)

    def get_action(self, state: Tuple, exploring: bool = True) -> str:
        """Get action using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        if exploring and random.random() < self.exploration_rate:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)

        return action

    def update_q_value(self, state: Tuple, action: str, reward: float,
                       next_state: Tuple) -> None:
        """Update Q-value using Q-learning formula"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q
        self.total_rewards += reward

    def calculate_reward(self, deer, old_energy: float, action: str,
                         herd_size: int, distance_to_herd: float,
                         died: bool = False, ate_food: bool = False,
                         was_in_danger: bool = False, escaped: bool = False) -> float:
        """Calculate reward for deer's action

        Args:
            deer: The deer agent
            old_energy: Energy before action
            action: Action taken
            herd_size: Number of herd mates
            distance_to_herd: Distance to herd center
            died: Whether deer died
            ate_food: Whether deer ate
            was_in_danger: Whether there was a predator nearby
            escaped: Whether deer escaped from danger

        Returns:
            Reward value
        """
        reward = 0.0

        # Major events
        if died:
            # Worse penalty if died alone
            if herd_size == 0:
                reward -= 150.0
                self.times_lost_herd += 1
            else:
                reward -= 100.0
            return reward

        if escaped and was_in_danger:
            # Big reward for escaping
            if herd_size > 0:
                reward += 40.0  # Extra bonus for escaping with herd
                self.times_saved_by_herd += 1
            else:
                reward += 25.0

        if ate_food:
            reward += 12.0

        # Herd behavior rewards (critical for deer)
        if herd_size > 0:
            reward += min(8.0, herd_size * 2.0)  # Reward for being in herd

            # Distance to herd matters
            if distance_to_herd < 30:
                reward += 5.0  # In the center of herd
            elif distance_to_herd < 70:
                reward += 2.0  # Near herd
            else:
                reward -= 3.0  # Too far from herd
        else:
            reward -= 8.0  # Big penalty for being alone

        # Action-specific rewards
        if action == 'flee_to_herd':
            if was_in_danger and herd_size > 0:
                reward += 10.0  # Excellent decision

        elif action == 'stay_with_herd':
            if herd_size > 1:
                reward += 3.0  # Good to stay with herd

        elif action == 'flee_alone':
            if herd_size > 0:
                reward -= 5.0  # Bad decision to abandon herd

        elif action == 'follow_leader':
            if herd_size > 1:
                reward += 4.0  # Good herd behavior

        # Energy management
        if deer.energy < 20:
            reward -= 5.0
        elif deer.energy > 75:
            reward += 2.0

        # Small survival reward
        reward += 0.15

        return reward

    def decay_exploration(self) -> None:
        """Decay exploration rate"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'exploration_rate': self.exploration_rate,
                    'total_episodes': self.total_episodes,
                    'total_rewards': self.total_rewards,
                    'times_saved_by_herd': self.times_saved_by_herd,
                    'times_lost_herd': self.times_lost_herd
                }, f)
            print(f"💾 Deer Q-table saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save deer Q-table: {e}")

    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from file"""
        try:
            if not os.path.exists(filepath):
                print(f"⚠️ No saved deer Q-table found at {filepath}")
                return False

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
                self.total_episodes = data.get('total_episodes', 0)
                self.total_rewards = data.get('total_rewards', 0.0)
                self.times_saved_by_herd = data.get('times_saved_by_herd', 0)
                self.times_lost_herd = data.get('times_lost_herd', 0)

            print(f"📂 Deer Q-table loaded from {filepath}")
            print(f"   States learned: {len(self.q_table)}")
            print(f"   Saved by herd: {self.times_saved_by_herd} times")
            return True
        except Exception as e:
            print(f"❌ Failed to load deer Q-table: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get learning statistics"""
        return {
            'q_table_size': len(self.q_table),
            'exploration_rate': self.exploration_rate,
            'total_episodes': self.total_episodes,
            'total_rewards': self.total_rewards,
            'avg_reward': self.total_rewards / max(1, self.total_episodes),
            'times_saved_by_herd': self.times_saved_by_herd,
            'times_lost_herd': self.times_lost_herd,
            'herd_survival_rate': (self.times_saved_by_herd / max(1,
                                                                  self.times_saved_by_herd + self.times_lost_herd)) * 100
        }

    def print_best_policies(self, top_n: int = 10) -> None:
        """Print best learned herd policies"""
        print("\n🦌 Best Learned Deer Policies:")
        print("-" * 60)

        state_values = [(state, sum(actions.values()))
                        for state, actions in self.q_table.items()]
        state_values.sort(key=lambda x: x[1], reverse=True)

        energy_names = {0: "Critical", 1: "Low", 2: "Medium", 3: "High"}
        predator_names = {0: "DANGER!", 1: "Close", 2: "Medium", 3: "Far", 4: "None"}
        herd_names = {0: "Alone", 1: "Small", 2: "Medium", 3: "Large"}
        distance_names = {0: "Far", 1: "Near", 2: "In Herd"}

        for i, (state, total_value) in enumerate(state_values[:top_n]):
            energy, predator, herd, distance = state
            best_action = max(self.q_table[state].items(), key=lambda x: x[1])

            print(f"\n{i + 1}. Energy: {energy_names[energy]}, Predator: {predator_names[predator]}")
            print(f"   Herd: {herd_names[herd]}, Position: {distance_names[distance]}")
            print(f"   Best Action: {best_action[0]} (Q={best_action[1]:.2f})")