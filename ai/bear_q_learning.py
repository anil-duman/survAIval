# ai/bear_q_learning.py - Q-Learning for bears
"""
SurvAIval Bear Q-Learning System
Bears learn territorial defense and hunting strategies
"""

import numpy as np
import random
from typing import Tuple, Dict, Optional
import pickle
import os



class BearQLearningAgent:
    """Q-Learning agent specialized for bear territorial behavior"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.2, exploration_decay: float = 0.9998):
        """Initialize Bear Q-Learning agent"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.05

        # Q-Table
        self.q_table: Dict[Tuple, Dict[str, float]] = {}

        # Statistics
        self.total_hunts = 0
        self.successful_hunts = 0
        self.territory_defenses = 0
        self.total_rewards = 0.0

        # Available actions for bears
        self.actions = [
            'hunt_deer',
            'hunt_wolf',
            'hunt_rabbit',
            'defend_territory',
            'return_to_territory',
            'patrol_territory',
            'rest',
            'ambush'
        ]

    def get_state(self, world_state: dict) -> Tuple:
        """Convert world state to discrete state for bears"""
        # 1. Energy level
        energy = world_state.get('energy', 75)
        if energy < 40:
            energy_state = 0  # Critical - must hunt
        elif energy < 70:
            energy_state = 1  # Low - should hunt
        elif energy < 110:
            energy_state = 2  # Medium
        else:
            energy_state = 3  # High - can be selective

        # 2. Best prey available
        prey = world_state.get('prey', [])
        if prey:
            # Categorize prey by type
            has_deer = any(p.agent_type == 'deer' for p in prey)
            has_wolf = any(p.agent_type == 'wolf' for p in prey)
            has_rabbit = any(p.agent_type == 'rabbit' for p in prey)

            nearest_prey = min(prey, key=lambda p: np.linalg.norm(p.position - world_state['position']))
            distance = np.linalg.norm(nearest_prey.position - world_state['position'])

            if distance < 30:
                prey_state = 4  # Very close prey
            elif distance < 60:
                prey_state = 3  # Close prey
            elif distance < 100:
                if has_deer:
                    prey_state = 2  # Medium distance, large prey
                else:
                    prey_state = 1  # Medium distance, small prey
            else:
                prey_state = 0  # Far prey
        else:
            prey_state = 0  # No prey

        # 3. Territory status
        territory_established = world_state.get('territory_established', False)
        if territory_established:
            territory_center = world_state.get('territory_center', world_state['position'])
            distance_from_territory = np.linalg.norm(world_state['position'] - territory_center)

            if distance_from_territory < 50:
                territory_state = 2  # In territory center
            elif distance_from_territory < 150:
                territory_state = 1  # In territory
            else:
                territory_state = 0  # Outside territory
        else:
            territory_state = 0  # No territory

        # 4. Intruder status
        same_species = world_state.get('same_species', [])
        intruders = []
        if territory_established:
            territory_center = world_state.get('territory_center', world_state['position'])
            for bear in same_species:
                if bear.id != world_state.get('self_id', ''):
                    dist = np.linalg.norm(bear.position - territory_center)
                    if dist < 150:  # territorial_radius
                        intruders.append(bear)

        if intruders:
            intruder_state = 1  # Territory threatened
        else:
            intruder_state = 0  # No intruders

        return (energy_state, prey_state, territory_state, intruder_state)

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

    def calculate_reward(self, bear, old_energy: float, action: str,
                         killed_prey: bool = False, prey_type: str = None,
                         defended_territory: bool = False,
                         died: bool = False) -> float:
        """Calculate reward for bear's action"""
        reward = 0.0

        # Major events
        if died:
            reward -= 200.0  # Very bad for apex predator
            return reward

        if killed_prey:
            # Different rewards for different prey
            prey_rewards = {
                'deer': 100.0,  # Best prey
                'wolf': 80.0,  # Good prey
                'rabbit': 40.0  # Small prey
            }
            base_reward = prey_rewards.get(prey_type, 60.0)
            reward += base_reward

            self.successful_hunts += 1
            self.total_hunts += 1

        if defended_territory:
            reward += 30.0  # Good territorial behavior
            self.territory_defenses += 1

        # Energy management (critical for bears)
        if bear.energy < 30:
            reward -= 15.0  # Very low energy is dangerous
        elif bear.energy > 120:
            reward += 5.0  # Well-fed bear

        # Action-specific rewards
        if action == 'hunt_deer':
            if bear.energy < 80:
                reward += 5.0  # Good decision when hungry

        elif action == 'rest':
            if bear.energy > 110:
                reward += 3.0  # Good to rest when full
            elif bear.energy < 50:
                reward -= 10.0  # Bad decision, need food

        elif action == 'defend_territory':
            if defended_territory:
                reward += 15.0  # Successful defense

        elif action == 'patrol_territory':
            reward += 2.0  # Good territorial behavior

        elif action == 'ambush':
            if killed_prey:
                reward += 20.0  # Bonus for successful ambush

        # Small survival reward
        reward += 0.2

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
                    'territory_defenses': self.territory_defenses,
                    'total_rewards': self.total_rewards
                }, f)
            print(f"💾 Bear Q-table saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save bear Q-table: {e}")

    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from file"""
        try:
            if not os.path.exists(filepath):
                print(f"⚠️ No saved bear Q-table found at {filepath}")
                return False

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
                self.total_hunts = data.get('total_hunts', 0)
                self.successful_hunts = data.get('successful_hunts', 0)
                self.territory_defenses = data.get('territory_defenses', 0)
                self.total_rewards = data.get('total_rewards', 0.0)

            success_rate = (self.successful_hunts / max(1, self.total_hunts)) * 100
            print(f"📂 Bear Q-table loaded from {filepath}")
            print(f"   States learned: {len(self.q_table)}")
            print(f"   Hunt success: {success_rate:.1f}%")
            print(f"   Territory defenses: {self.territory_defenses}")
            return True
        except Exception as e:
            print(f"❌ Failed to load bear Q-table: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get learning statistics"""
        success_rate = (self.successful_hunts / max(1, self.total_hunts)) * 100

        return {
            'q_table_size': len(self.q_table),
            'exploration_rate': self.exploration_rate,
            'total_hunts': self.total_hunts,
            'successful_hunts': self.successful_hunts,
            'success_rate': success_rate,
            'territory_defenses': self.territory_defenses,
            'total_rewards': self.total_rewards,
            'avg_reward': self.total_rewards / max(1, self.total_hunts)
        }

    def print_best_policies(self, top_n: int = 10) -> None:
        """Print best learned bear policies"""
        print("\n🐻 Best Learned Bear Policies:")
        print("-" * 60)

        state_values = [(state, sum(actions.values()))
                        for state, actions in self.q_table.items()]
        state_values.sort(key=lambda x: x[1], reverse=True)

        energy_names = {0: "Critical", 1: "Low", 2: "Medium", 3: "High"}
        prey_names = {0: "No Prey", 1: "Small/Far", 2: "Large/Med", 3: "Close", 4: "Very Close"}
        territory_names = {0: "No Territory", 1: "In Territory", 2: "Territory Center"}
        intruder_names = {0: "No Intruders", 1: "Territory Threatened"}

        for i, (state, total_value) in enumerate(state_values[:top_n]):
            energy, prey, territory, intruder = state
            best_action = max(self.q_table[state].items(), key=lambda x: x[1])

            print(f"\n{i + 1}. Energy: {energy_names[energy]}, Prey: {prey_names[prey]}")
            print(f"   Territory: {territory_names[territory]}, Intruders: {intruder_names[intruder]}")
            print(f"   Best Action: {best_action[0]} (Q={best_action[1]:.2f})")