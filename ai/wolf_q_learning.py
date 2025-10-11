# ai/wolf_q_learning.py - Q-Learning for wolves
"""
SurvAIval Wolf Q-Learning System
Wolves learn pack hunting strategies
"""

import numpy as np
import random
from typing import Tuple, Dict, Optional
import pickle
import os


class WolfQLearningAgent:
    """Q-Learning agent specialized for wolf pack hunting"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.25, exploration_decay: float = 0.9997):
        """Initialize Wolf Q-Learning agent

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

        # Q-Table for wolf decisions
        self.q_table: Dict[Tuple, Dict[str, float]] = {}

        # Statistics
        self.total_hunts = 0
        self.successful_hunts = 0
        self.total_rewards = 0.0
        self.pack_hunts = 0

        # Available actions for wolves
        self.actions = [
            'hunt_alone',
            'hunt_with_pack',
            'coordinate_left',
            'coordinate_right',
            'wait_for_pack',
            'chase_aggressively',
            'patrol',
            'rest'
        ]

    def get_state(self, world_state: dict) -> Tuple:
        """Convert world state to discrete state for wolves

        Args:
            world_state: Dictionary with wolf's perception

        Returns:
            Tuple representing discrete state
        """
        # 1. Energy level (critical, low, medium, high)
        energy = world_state.get('energy', 60)
        if energy < 30:
            energy_state = 0  # Critical - need food urgently
        elif energy < 50:
            energy_state = 1  # Low - should hunt soon
        elif energy < 80:
            energy_state = 2  # Medium - can hunt
        else:
            energy_state = 3  # High - can afford to rest

        # 2. Prey availability (none, far, medium, close, very_close)
        prey = world_state.get('prey', [])
        if prey:
            nearest_prey = min(prey, key=lambda p: np.linalg.norm(p.position - world_state['position']))
            distance = np.linalg.norm(nearest_prey.position - world_state['position'])

            # Also check prey energy (weaker prey = easier)
            prey_energy_ratio = nearest_prey.energy / nearest_prey.max_energy

            if distance < 25:
                prey_state = 4  # Very close - attack range
            elif distance < 50:
                prey_state = 3  # Close - chase
            elif distance < 100:
                prey_state = 2  # Medium - can pursue
            elif distance < 150:
                prey_state = 1  # Far - visible
            else:
                prey_state = 0  # Very far

            # Modify based on prey weakness
            if prey_energy_ratio < 0.3 and prey_state > 0:
                prey_state = min(4, prey_state + 1)  # Boost state for weak prey
        else:
            prey_state = 0  # No prey

        # 3. Pack status (alone, small_pack, medium_pack, large_pack)
        pack_mates = world_state.get('pack_mates', [])
        pack_size = len(pack_mates)

        if pack_size == 0:
            pack_state = 0  # Alone
        elif pack_size <= 1:
            pack_state = 1  # Small pack (2 wolves)
        elif pack_size <= 2:
            pack_state = 2  # Medium pack (3 wolves)
        else:
            pack_state = 3  # Large pack (4+ wolves)

        # 4. Pack coordination (are pack mates nearby and hunting?)
        if pack_size > 0:
            hunting_mates = sum(1 for mate in pack_mates
                                if hasattr(mate, 'current_state') and
                                'hunt' in mate.current_state.lower())

            if hunting_mates >= pack_size * 0.7:  # Most pack is hunting
                coordination_state = 2  # High coordination
            elif hunting_mates > 0:
                coordination_state = 1  # Some coordination
            else:
                coordination_state = 0  # No coordination
        else:
            coordination_state = 0

        return (energy_state, prey_state, pack_state, coordination_state)

    def get_action(self, state: Tuple, exploring: bool = True) -> str:
        """Get action using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        if exploring and random.random() < self.exploration_rate:
            # Explore
            action = random.choice(self.actions)
        else:
            # Exploit
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

    def calculate_reward(self, wolf, old_energy: float, action: str,
                         killed_prey: bool = False, pack_size: int = 0,
                         died: bool = False, pack_hunt: bool = False) -> float:
        """Calculate reward for wolf's action

        Args:
            wolf: The wolf agent
            old_energy: Energy before action
            action: Action taken
            killed_prey: Whether wolf successfully killed prey
            pack_size: Number of pack mates nearby
            died: Whether wolf died
            pack_hunt: Whether this was a coordinated pack hunt

        Returns:
            Reward value
        """
        reward = 0.0

        # Major events
        if died:
            reward -= 150.0  # Very bad
            return reward

        if killed_prey:
            base_reward = 80.0

            # Bonus for pack hunting (more efficient)
            if pack_hunt and pack_size > 0:
                pack_bonus = min(50.0, pack_size * 15.0)
                reward += base_reward + pack_bonus
                self.pack_hunts += 1
                print(f"🐺 Pack hunt bonus: +{pack_bonus:.0f}")
            else:
                reward += base_reward

            self.successful_hunts += 1
            self.total_hunts += 1

        # Energy management
        energy_change = wolf.energy - old_energy

        if wolf.energy < 20:
            reward -= 10.0  # Penalty for very low energy
        elif wolf.energy > 90:
            reward += 3.0  # Reward for good energy

        # Action-specific rewards
        if action == 'hunt_with_pack':
            if pack_size > 0:
                reward += 5.0  # Good decision to coordinate
            else:
                reward -= 3.0  # Bad decision, no pack

        elif action == 'wait_for_pack':
            if pack_size > 0:
                reward += 2.0  # Good patience

        elif action == 'hunt_alone':
            if pack_size > 1:
                reward -= 2.0  # Should have coordinated

        elif action == 'coordinate_left' or action == 'coordinate_right':
            if pack_size > 0:
                reward += 3.0  # Good flanking behavior

        elif action == 'rest':
            if wolf.energy > 80:
                reward += 1.0  # Good to rest when full
            elif wolf.energy < 40:
                reward -= 5.0  # Bad, need to hunt

        # Small survival reward
        reward += 0.15

        # Penalty for energy waste
        if energy_change < -1.0:
            reward -= 1.0

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
                    'total_hunts': self.total_hunts,
                    'successful_hunts': self.successful_hunts,
                    'pack_hunts': self.pack_hunts,
                    'total_rewards': self.total_rewards
                }, f)
            print(f"💾 Wolf Q-table saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save wolf Q-table: {e}")

    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from file"""
        try:
            if not os.path.exists(filepath):
                print(f"⚠️ No saved wolf Q-table found at {filepath}")
                return False

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
                self.total_hunts = data.get('total_hunts', 0)
                self.successful_hunts = data.get('successful_hunts', 0)
                self.pack_hunts = data.get('pack_hunts', 0)
                self.total_rewards = data.get('total_rewards', 0.0)

            success_rate = (self.successful_hunts / max(1, self.total_hunts)) * 100
            print(f"📂 Wolf Q-table loaded from {filepath}")
            print(f"   States learned: {len(self.q_table)}")
            print(f"   Hunts: {self.total_hunts} (Success: {success_rate:.1f}%)")
            print(f"   Pack hunts: {self.pack_hunts}")
            return True
        except Exception as e:
            print(f"❌ Failed to load wolf Q-table: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get learning statistics"""
        success_rate = (self.successful_hunts / max(1, self.total_hunts)) * 100
        pack_hunt_rate = (self.pack_hunts / max(1, self.successful_hunts)) * 100

        return {
            'q_table_size': len(self.q_table),
            'exploration_rate': self.exploration_rate,
            'total_hunts': self.total_hunts,
            'successful_hunts': self.successful_hunts,
            'success_rate': success_rate,
            'pack_hunts': self.pack_hunts,
            'pack_hunt_rate': pack_hunt_rate,
            'total_rewards': self.total_rewards,
            'avg_reward': self.total_rewards / max(1, self.total_hunts)
        }

    def print_best_policies(self, top_n: int = 10) -> None:
        """Print best learned hunting policies"""
        print("\n🐺 Best Learned Wolf Policies:")
        print("-" * 60)

        state_values = [(state, sum(actions.values()))
                        for state, actions in self.q_table.items()]
        state_values.sort(key=lambda x: x[1], reverse=True)

        energy_names = {0: "Critical", 1: "Low", 2: "Medium", 3: "High"}
        prey_names = {0: "No Prey", 1: "Far", 2: "Medium", 3: "Close", 4: "Very Close"}
        pack_names = {0: "Alone", 1: "Small Pack", 2: "Medium Pack", 3: "Large Pack"}
        coord_names = {0: "No Coord", 1: "Some Coord", 2: "High Coord"}

        for i, (state, total_value) in enumerate(state_values[:top_n]):
            energy, prey, pack, coord = state
            best_action = max(self.q_table[state].items(), key=lambda x: x[1])

            print(f"\n{i + 1}. Energy: {energy_names[energy]}, Prey: {prey_names[prey]}")
            print(f"   Pack: {pack_names[pack]}, Coordination: {coord_names[coord]}")
            print(f"   Best Action: {best_action[0]} (Q={best_action[1]:.2f})")
            print(f"   Total Value: {total_value:.2f}")