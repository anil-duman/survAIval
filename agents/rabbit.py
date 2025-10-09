# agents/rabbit.py - Rabbit with integrated Q-Learning
"""
SurvAIval Rabbit Agent
Herbivore agent with optional Q-Learning capability
"""

import pygame
import numpy as np
import random
from agents.base_agent import BaseAgent
import config
from typing import Optional
from utils.animation import AnimatedSprite

# Import Q-Learning (optional dependency)
try:
    from ai.q_learning import QLearningAgent

    Q_LEARNING_AVAILABLE = True
except ImportError:
    Q_LEARNING_AVAILABLE = False
    print("⚠️ Q-Learning not available")


class Rabbit(BaseAgent):
    """Rabbit agent - Herbivore with survival instincts and optional learning"""

    # Shared Q-Learning agent for all rabbits (collective learning)
    shared_q_agent = None
    learning_enabled = False  # Global toggle for all rabbits

    @classmethod
    def enable_learning(cls, enable: bool = True):
        """Enable or disable Q-Learning for all rabbits"""
        if not Q_LEARNING_AVAILABLE and enable:
            print("❌ Cannot enable learning: Q-Learning module not found")
            return False

        cls.learning_enabled = enable

        if enable and cls.shared_q_agent is None:
            cls.shared_q_agent = QLearningAgent(
                learning_rate=0.1,
                discount_factor=0.95,
                exploration_rate=0.3,
                exploration_decay=0.9995
            )
            print("🧠 Q-Learning enabled for rabbits!")
        elif not enable:
            print("🔒 Q-Learning disabled for rabbits")

        return True

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
        self.fear_distance = 80.0
        self.food_search_radius = 150.0
        self.eating_distance = 15.0
        self.wander_strength = 0.2
        self.flee_strength = 1.0

        # Reproduction settings
        self.min_reproduction_energy = 60
        self.reproduction_cooldown = 0

        # AI state
        self.wander_angle = random.uniform(0, 2 * np.pi)
        self.panic_timer = 0
        self.target_food = None
        self.eating_timer = 0

        # Learning state (only used if learning enabled)
        self.q_state = None
        self.q_action = None
        self.last_energy = self.energy
        self.steps_alive = 0
        self.last_ate = False

        # Animation state
        self.animated_sprite: Optional[AnimatedSprite] = None
        self.facing_right = True
        self.last_velocity = np.array([1.0, 0.0])

    def set_animated_sprite(self, animated_sprite):
        """Set the animated sprite for this rabbit"""
        self.animated_sprite = animated_sprite

    def get_color(self):
        """Return rabbit display color"""
        # Different colors for learning vs normal
        if self.learning_enabled:
            if self.panic_timer > 0:
                return (255, 150, 255)  # Purple when fleeing (learned)
            elif self.eating_timer > 0:
                return (150, 255, 150)  # Bright green when eating
            elif self.steps_alive > 500:
                return (255, 200, 255)  # Light purple for experienced
            else:
                return (255, 180, 220)  # Pinkish purple (learning mode)
        else:
            # Normal colors
            if self.panic_timer > 0:
                return (255, 150, 150)  # Light red when panicking
            elif self.eating_timer > 0:
                return (255, 220, 180)  # Orange when eating
            elif self.energy < 30:
                return (200, 150, 150)  # Pale when low energy
            else:
                return config.RABBIT_COLOR

    def decide_action(self, world_state):
        """AI decision making - uses Q-Learning if enabled"""
        # Update timers
        if self.panic_timer > 0:
            self.panic_timer -= 1
        if self.eating_timer > 0:
            self.eating_timer -= 1

        # Use Q-Learning if enabled
        if self.learning_enabled and self.shared_q_agent:
            return self._decide_with_learning(world_state)
        else:
            return self._decide_normal(world_state)

    def _decide_with_learning(self, world_state):
        """Decision making using Q-Learning"""
        # Get Q-Learning state
        q_state = self.shared_q_agent.get_state(world_state)

        # Update Q-values from previous action
        if self.q_state is not None and self.q_action is not None:
            reward = self.shared_q_agent.calculate_reward(
                self, self.last_energy, self.q_action,
                died=not self.is_alive, ate_food=self.last_ate
            )
            self.shared_q_agent.update_q_value(
                self.q_state, self.q_action, reward, q_state
            )

        # Get new action from Q-Learning
        q_action = self.shared_q_agent.get_action(q_state, exploring=True)

        # Store for next update
        self.q_state = q_state
        self.q_action = q_action
        self.last_energy = self.energy
        self.last_ate = False
        self.steps_alive += 1

        # Convert Q-action to game action
        return self._convert_q_action(q_action, world_state)

    def _decide_normal(self, world_state):
        """Normal decision making (original behavior)"""
        # Priority 1: FLEE from predators
        nearest_predator = self._find_nearest_predator(world_state['predators'])
        if nearest_predator:
            distance = np.linalg.norm(nearest_predator.position - self.position)
            if distance < self.fear_distance:
                self.panic_timer = 60
                self.target_food = None
                self.current_state = "fleeing"

                if self.animated_sprite:
                    self.animated_sprite.play_animation('walk')

                return {
                    'type': 'flee',
                    'threat_position': nearest_predator.position,
                    'urgency': 1.0 - (distance / self.fear_distance)
                }

        # Priority 2: Seek and eat food
        if self.energy < 70:
            nearby_food = self._get_nearby_food(world_state)

            if nearby_food:
                closest_food = nearby_food[0]
                distance_to_food = closest_food['distance']

                if distance_to_food < self.eating_distance:
                    self.current_state = "eating"
                    self.eating_timer = 30

                    if self.animated_sprite:
                        self.animated_sprite.play_animation('idle')

                    return {
                        'type': 'eat',
                        'food_position': closest_food['position']
                    }
                else:
                    self.target_food = closest_food
                    self.current_state = "seeking_food"

                    if self.animated_sprite:
                        self.animated_sprite.play_animation('walk')

                    return {
                        'type': 'seek',
                        'target_position': closest_food['position']
                    }

        # Priority 3: Reproduction
        if (self.energy > self.min_reproduction_energy and
                self.reproduction_cooldown == 0):
            potential_mate = self._find_mate(world_state['same_species'])
            if potential_mate:
                distance = np.linalg.norm(potential_mate.position - self.position)
                if distance < 30:
                    self.current_state = "reproducing"

                    if self.animated_sprite:
                        self.animated_sprite.play_animation('idle')

                    return {
                        'type': 'reproduce',
                        'partner': potential_mate
                    }
                else:
                    self.current_state = "seeking_mate"

                    if self.animated_sprite:
                        self.animated_sprite.play_animation('walk')

                    return {
                        'type': 'seek',
                        'target_position': potential_mate.position
                    }

        # Default: Wander
        self.current_state = "wandering"

        if self.animated_sprite:
            speed = np.linalg.norm(self.velocity)
            if speed < 0.5:
                self.animated_sprite.play_animation('idle')
            else:
                self.animated_sprite.play_animation('walk')

        return self._wander_behavior()

    def _convert_q_action(self, q_action: str, world_state: dict):
        """Convert Q-Learning action to game action"""
        # Handle flee actions
        if q_action.startswith('flee_'):
            direction = q_action.split('_')[1]
            predators = world_state['predators']

            if predators:
                nearest_predator = min(predators,
                                       key=lambda p: np.linalg.norm(p.position - self.position))
                self.panic_timer = 30
                self.current_state = "fleeing_learned"

                if self.animated_sprite:
                    self.animated_sprite.play_animation('walk')

                flee_pos = nearest_predator.position.copy()

                if direction == 'up':
                    flee_pos[1] += 50
                elif direction == 'down':
                    flee_pos[1] -= 50
                elif direction == 'left':
                    flee_pos[0] += 50
                elif direction == 'right':
                    flee_pos[0] -= 50

                return {'type': 'flee', 'threat_position': flee_pos}

        # Seek food
        elif q_action == 'seek_food':
            if self.energy < 75:
                nearby_food = self._get_nearby_food(world_state)
                if nearby_food:
                    closest_food = nearby_food[0]

                    if closest_food['distance'] < self.eating_distance:
                        self.current_state = "eating_learned"
                        self.eating_timer = 30
                        if self.animated_sprite:
                            self.animated_sprite.play_animation('idle')
                        return {'type': 'eat', 'food_position': closest_food['position']}
                    else:
                        self.current_state = "seeking_food_learned"
                        if self.animated_sprite:
                            self.animated_sprite.play_animation('walk')
                        return {'type': 'seek', 'target_position': closest_food['position']}

        # Stay
        elif q_action == 'stay':
            self.current_state = "staying"
            if self.animated_sprite:
                self.animated_sprite.play_animation('idle')
            return {'type': 'move', 'direction': np.array([0.0, 0.0])}

        # Wander (default)
        self.current_state = "wandering_learned"
        if self.animated_sprite:
            speed = np.linalg.norm(self.velocity)
            if speed < 0.5:
                self.animated_sprite.play_animation('idle')
            else:
                self.animated_sprite.play_animation('walk')
        return self._wander_behavior()

    def update(self, world):
        """Update rabbit state and animation"""
        super().update(world)

        if np.linalg.norm(self.velocity) > 0.1:
            self.last_velocity = self.velocity.copy()

            if self.velocity[0] > 0.1:
                self.facing_right = True
            elif self.velocity[0] < -0.1:
                self.facing_right = False

        if self.animated_sprite:
            self.animated_sprite.set_flip(not self.facing_right)

    def _execute_action(self, action, world):
        """Override to add eating behavior"""
        if not action:
            return

        action_type = action.get('type', 'idle')

        if action_type == 'eat':
            self._attempt_eat_food(world)
        else:
            super()._execute_action(action, world)

    def _attempt_eat_food(self, world):
        """Attempt to eat food"""
        if hasattr(world, 'food_manager'):
            nutrition_gained = world.food_manager.consume_food_at(
                self.position,
                consumption_radius=self.eating_distance,
                amount=25
            )

            if nutrition_gained > 0:
                self.energy = min(self.max_energy, self.energy + nutrition_gained)
                self.eating_timer = 20
                self.energy -= 0.5

                # Track for learning
                if self.learning_enabled:
                    self.last_ate = True

    def _die(self, cause="natural"):
        """Override to give final learning update"""
        if self.learning_enabled and self.shared_q_agent:
            if self.q_state is not None and self.q_action is not None:
                final_reward = -100.0 if cause == "hunted" else -80.0
                terminal_state = (0, 0, 0)

                self.shared_q_agent.update_q_value(
                    self.q_state, self.q_action, final_reward, terminal_state
                )

        super()._die(cause)

    def _get_nearby_food(self, world_state):
        """Get nearby food sources"""
        food_list = []

        if 'food_sources' in world_state:
            for food in world_state['food_sources']:
                distance = np.linalg.norm(food.position - self.position)
                if distance <= self.food_search_radius and not food.is_depleted:
                    food_list.append({
                        'position': food.position,
                        'nutrition': food.nutrition_value,
                        'type': food.food_type,
                        'distance': distance,
                        'object': food
                    })

            food_list.sort(key=lambda f: f['distance'])

        return food_list

    def _perceive_world(self, world):
        """Override to include food sources"""
        world_state = super()._perceive_world(world)

        if hasattr(world, 'food_manager'):
            nearby_food = world.food_manager.get_food_in_range(
                self.position, self.food_search_radius
            )
            world_state['food_sources'] = nearby_food
        else:
            world_state['food_sources'] = []

        return world_state

    def _find_nearest_predator(self, predators):
        """Find closest predator"""
        if not predators:
            return None
        return min(predators, key=lambda p: np.linalg.norm(p.position - self.position))

    def _find_mate(self, same_species):
        """Find potential mate"""
        eligible_mates = []

        for rabbit in same_species:
            if (rabbit.id != self.id and
                    rabbit.energy > rabbit.min_reproduction_energy and
                    rabbit.reproduction_cooldown == 0):
                eligible_mates.append(rabbit)

        if eligible_mates:
            return min(eligible_mates,
                       key=lambda r: np.linalg.norm(r.position - self.position))
        return None

    def _wander_behavior(self):
        """Wander movement"""
        self.wander_angle += random.uniform(-0.3, 0.3)

        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        random_force = np.random.normal(0, 0.1, 2)

        return {
            'type': 'move',
            'direction': wander_force + random_force
        }

    def get_food_targets(self, nearby_entities):
        """Rabbits don't eat other animals"""
        return []

    def _create_offspring(self, position):
        """Create rabbit offspring"""
        offspring_pos = position + np.random.normal(0, 10, 2)

        offspring_pos[0] = max(20, min(config.SCREEN_WIDTH - 20, offspring_pos[0]))
        offspring_pos[1] = max(20, min(config.SCREEN_HEIGHT - 20, offspring_pos[1]))

        new_rabbit = Rabbit(offspring_pos)

        new_rabbit.max_speed = self.max_speed + random.uniform(-0.2, 0.2)
        new_rabbit.vision_range = self.vision_range + random.uniform(-10, 10)
        new_rabbit.fear_distance = self.fear_distance + random.uniform(-5, 5)
        new_rabbit.energy = 50

        mode = "LEARNING" if self.learning_enabled else "normal"
        print(f"🐰 New {mode} rabbit born! {new_rabbit.id}")
        return new_rabbit

    def _is_predator(self, other_agent):
        """Define what threatens rabbits"""
        return other_agent.agent_type in ['wolf', 'bear', 'fox']

    def _is_prey(self, other_agent):
        """Rabbits don't hunt"""
        return False

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the rabbit"""
        if not self.is_alive:
            return

        pos = (int(self.position[0] - camera_offset[0]),
               int(self.position[1] - camera_offset[1]))

        if self.animated_sprite:
            self.animated_sprite.draw(screen, pos)
        else:
            super().draw(screen, camera_offset)

        if self.energy < self.max_energy * 0.5:
            self._draw_energy_bar(screen, pos)

        # Debug
        if getattr(config, 'DEBUG_MODE', False):
            if self.target_food and self.current_state == "seeking_food":
                start_pos = pos
                end_pos = (int(self.target_food['position'][0]),
                           int(self.target_food['position'][1]))
                pygame.draw.line(screen, (255, 255, 0), start_pos, end_pos, 1)

    # Class methods for learning management
    @classmethod
    def save_learning(cls, filepath: str = "data/rabbit_q_table.pkl"):
        """Save Q-table"""
        if cls.shared_q_agent:
            cls.shared_q_agent.save_q_table(filepath)
            cls.shared_q_agent.print_best_policies(top_n=5)

    @classmethod
    def load_learning(cls, filepath: str = "data/rabbit_q_table.pkl"):
        """Load Q-table"""
        if cls.shared_q_agent:
            return cls.shared_q_agent.load_q_table(filepath)
        return False

    @classmethod
    def get_learning_stats(cls):
        """Get learning statistics"""
        if cls.shared_q_agent:
            return cls.shared_q_agent.get_statistics()
        return None

    @classmethod
    def decay_exploration(cls):
        """Decay exploration rate"""
        if cls.shared_q_agent:
            cls.shared_q_agent.decay_exploration()