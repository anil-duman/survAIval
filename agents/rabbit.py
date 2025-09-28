# agents/rabbit.py - Rabbit agent implementation
"""
SurvAIval Rabbit Agent
Herbivore agent with basic survival AI
"""

import numpy as np
import random
from agents.base_agent import BaseAgent
import config


class Rabbit(BaseAgent):
    """Rabbit agent - Herbivore with survival instincts"""

    def __init__(self, position):
        """Initialize rabbit agent

        Args:
            position: [x, y] starting position
        """
        super().__init__(position, "rabbit", max_energy=80)

        # Rabbit-specific properties
        self.size = 12
        self.max_speed = 3.5
        self.max_force = 0.4
        self.vision_range = 100.0
        self.energy_drain_rate = 0.03

        # Behavioral parameters
        self.fear_distance = 60.0  # Distance to start fleeing from predators
        self.wander_strength = 0.2
        self.flee_strength = 1.0

        # Reproduction settings
        self.min_reproduction_energy = 60
        self.reproduction_cooldown = 0

        # AI state
        self.wander_angle = random.uniform(0, 2 * np.pi)
        self.panic_timer = 0

    def get_color(self):
        """Return rabbit display color"""
        if self.panic_timer > 0:
            return (255, 150, 150)  # Light red when panicking
        elif self.energy < 30:
            return (200, 150, 150)  # Pale when low energy
        else:
            return config.RABBIT_COLOR

    def decide_action(self, world_state):
        """AI decision making for rabbit behavior

        Args:
            world_state: Dictionary with world information

        Returns:
            Action dictionary
        """
        # Update panic timer
        if self.panic_timer > 0:
            self.panic_timer -= 1

        # Priority 1: FLEE from predators (highest priority)
        nearest_predator = self._find_nearest_predator(world_state['predators'])
        if nearest_predator:
            distance = np.linalg.norm(nearest_predator.position - self.position)
            if distance < self.fear_distance:
                self.panic_timer = 60  # Panic for 1 second
                self.current_state = "fleeing"
                return {
                    'type': 'flee',
                    'threat_position': nearest_predator.position,
                    'urgency': 1.0 - (distance / self.fear_distance)
                }

        # Priority 2: Seek food (when energy is low)
        if self.energy < 50:
            food_targets = self._find_food_sources(world_state)
            if food_targets:
                nearest_food = min(food_targets,
                                   key=lambda f: np.linalg.norm(f['position'] - self.position))
                self.current_state = "seeking_food"
                return {
                    'type': 'seek',
                    'target_position': nearest_food['position']
                }

        # Priority 3: Reproduction (when energy is high)
        if (self.energy > self.min_reproduction_energy and
                self.reproduction_cooldown == 0):
            potential_mate = self._find_mate(world_state['same_species'])
            if potential_mate:
                distance = np.linalg.norm(potential_mate.position - self.position)
                if distance < 30:  # Close enough to attempt reproduction
                    self.current_state = "reproducing"
                    return {
                        'type': 'reproduce',
                        'partner': potential_mate
                    }
                else:
                    # Move towards potential mate
                    self.current_state = "seeking_mate"
                    return {
                        'type': 'seek',
                        'target_position': potential_mate.position
                    }

        # Default: Wander around
        self.current_state = "wandering"
        return self._wander_behavior()

    def _find_nearest_predator(self, predators):
        """Find the closest predator"""
        if not predators:
            return None

        return min(predators, key=lambda p: np.linalg.norm(p.position - self.position))

    def _find_food_sources(self, world_state):
        """Find available food sources

        For now, rabbits can eat grass (we'll add grass later)
        Returns list of food source positions
        """
        # Temporary: Create some imaginary food sources
        # In the future, this will be actual grass/plants in the world
        food_sources = []

        # For now, create random food spots for testing
        if random.random() < 0.1:  # 10% chance to "find" food nearby
            food_pos = self.position + np.random.normal(0, 50, 2)
            # Keep within bounds
            food_pos[0] = max(0, min(config.SCREEN_WIDTH, food_pos[0]))
            food_pos[1] = max(0, min(config.SCREEN_HEIGHT, food_pos[1]))

            food_sources.append({
                'position': food_pos,
                'nutrition': 20
            })

        return food_sources

    def _find_mate(self, same_species):
        """Find a potential mate"""
        eligible_mates = []

        for rabbit in same_species:
            if (rabbit.id != self.id and
                    rabbit.energy > rabbit.min_reproduction_energy and
                    rabbit.reproduction_cooldown == 0):
                eligible_mates.append(rabbit)

        if eligible_mates:
            # Find closest eligible mate
            return min(eligible_mates,
                       key=lambda r: np.linalg.norm(r.position - self.position))

        return None

    def _wander_behavior(self):
        """Generate wandering movement"""
        # Random walk with some direction persistence
        self.wander_angle += random.uniform(-0.3, 0.3)

        # Calculate wander direction
        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        # Add some randomness
        random_force = np.random.normal(0, 0.1, 2)

        total_force = wander_force + random_force

        return {
            'type': 'move',
            'direction': total_force
        }

    def get_food_targets(self, nearby_entities):
        """Return entities this rabbit can eat (none for herbivores)"""
        # Rabbits are herbivores, they don't eat other animals
        return []

    def _create_offspring(self, position):
        """Create a new rabbit offspring"""
        # Add some genetic variation
        offspring_pos = position + np.random.normal(0, 10, 2)

        # Keep within world bounds
        offspring_pos[0] = max(20, min(config.SCREEN_WIDTH - 20, offspring_pos[0]))
        offspring_pos[1] = max(20, min(config.SCREEN_HEIGHT - 20, offspring_pos[1]))

        new_rabbit = Rabbit(offspring_pos)

        # Inherit some traits with slight variation
        new_rabbit.max_speed = self.max_speed + random.uniform(-0.2, 0.2)
        new_rabbit.vision_range = self.vision_range + random.uniform(-10, 10)
        new_rabbit.energy = 40  # Start with moderate energy

        print(f"🐰 New rabbit born! {new_rabbit.id}")
        return new_rabbit

    def _is_predator(self, other_agent):
        """Override to define what threatens rabbits"""
        return other_agent.agent_type in ['wolf', 'bear', 'fox']

    def _is_prey(self, other_agent):
        """Rabbits don't hunt other animals"""
        return False