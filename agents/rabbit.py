# agents/rabbit.py - Rabbit with animation support
"""
SurvAIval Rabbit Agent
Herbivore agent with animated sprites
"""

import pygame
import numpy as np
import random
from agents.base_agent import BaseAgent
import config


class Rabbit(BaseAgent):
    """Rabbit agent - Herbivore with survival instincts and animations"""

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

        # Animation state
        self.animated_sprite = None
        self.facing_right = True
        self.last_velocity = np.array([1.0, 0.0])

    def set_animated_sprite(self, animated_sprite):
        """Set the animated sprite for this rabbit

        Args:
            animated_sprite: AnimatedSprite instance from animation manager
        """
        self.animated_sprite = animated_sprite

    def get_color(self):
        """Return rabbit display color (fallback when no sprite)"""
        if self.panic_timer > 0:
            return (255, 150, 150)
        elif self.eating_timer > 0:
            return (255, 220, 180)
        elif self.energy < 30:
            return (200, 150, 150)
        else:
            return config.RABBIT_COLOR

    def decide_action(self, world_state):
        """AI decision making for rabbit behavior"""
        # Update timers
        if self.panic_timer > 0:
            self.panic_timer -= 1
        if self.eating_timer > 0:
            self.eating_timer -= 1

        # Priority 1: FLEE from predators
        nearest_predator = self._find_nearest_predator(world_state['predators'])
        if nearest_predator:
            distance = np.linalg.norm(nearest_predator.position - self.position)
            if distance < self.fear_distance:
                self.panic_timer = 60
                self.target_food = None
                self.current_state = "fleeing"

                # Update animation to walk (running animation)
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

                    # Update animation to idle when eating
                    if self.animated_sprite:
                        self.animated_sprite.play_animation('idle')

                    return {
                        'type': 'eat',
                        'food_position': closest_food['position']
                    }
                else:
                    self.target_food = closest_food
                    self.current_state = "seeking_food"

                    # Walking animation
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

        # Idle or walk animation based on movement
        if self.animated_sprite:
            speed = np.linalg.norm(self.velocity)
            if speed < 0.5:
                self.animated_sprite.play_animation('idle')
            else:
                self.animated_sprite.play_animation('walk')

        return self._wander_behavior()

    def update(self, world):
        """Update rabbit state and animation"""
        # Update base agent
        super().update(world)

        # Update facing direction based on velocity
        if np.linalg.norm(self.velocity) > 0.1:
            self.last_velocity = self.velocity.copy()

            # Determine facing direction
            if self.velocity[0] > 0.1:
                self.facing_right = True
            elif self.velocity[0] < -0.1:
                self.facing_right = False

        # Update animation flip
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
        """Attempt to eat food from the world's food system"""
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

    def _get_nearby_food(self, world_state):
        """Get nearby food sources from the world"""
        food_list = []

        if 'food_sources' in world_state:
            food_sources = world_state['food_sources']

            for food in food_sources:
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
        """Override to include food sources in perception"""
        world_state = super()._perceive_world(world)

        if hasattr(world, 'food_manager'):
            nearby_food = world.food_manager.get_food_in_range(
                self.position,
                self.food_search_radius
            )
            world_state['food_sources'] = nearby_food
        else:
            world_state['food_sources'] = []

        return world_state

    def _find_nearest_predator(self, predators):
        """Find the closest predator"""
        if not predators:
            return None

        return min(predators, key=lambda p: np.linalg.norm(p.position - self.position))

    def _find_mate(self, same_species):
        """Find a potential mate"""
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
        """Generate wandering movement"""
        self.wander_angle += random.uniform(-0.3, 0.3)

        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        random_force = np.random.normal(0, 0.1, 2)

        total_force = wander_force + random_force

        return {
            'type': 'move',
            'direction': total_force
        }

    def get_food_targets(self, nearby_entities):
        """Return entities this rabbit can eat (none for herbivores)"""
        return []

    def _create_offspring(self, position):
        """Create a new rabbit offspring"""
        offspring_pos = position + np.random.normal(0, 10, 2)

        offspring_pos[0] = max(20, min(config.SCREEN_WIDTH - 20, offspring_pos[0]))
        offspring_pos[1] = max(20, min(config.SCREEN_HEIGHT - 20, offspring_pos[1]))

        new_rabbit = Rabbit(offspring_pos)

        # Inherit traits with variation
        new_rabbit.max_speed = self.max_speed + random.uniform(-0.2, 0.2)
        new_rabbit.vision_range = self.vision_range + random.uniform(-10, 10)
        new_rabbit.fear_distance = self.fear_distance + random.uniform(-5, 5)
        new_rabbit.energy = 50

        print(f"🐰 New rabbit born! {new_rabbit.id}")
        return new_rabbit

    def _is_predator(self, other_agent):
        """Override to define what threatens rabbits"""
        return other_agent.agent_type in ['wolf', 'bear', 'fox']

    def _is_prey(self, other_agent):
        """Rabbits don't hunt other animals"""
        return False

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the rabbit with animation or fallback to circle"""
        if not self.is_alive:
            return

        pos = (int(self.position[0] - camera_offset[0]),
               int(self.position[1] - camera_offset[1]))

        # Try to use animated sprite
        if self.animated_sprite:
            self.animated_sprite.draw(screen, pos)
        else:
            # Fallback to parent class drawing (circles)
            super().draw(screen, camera_offset)

        # Draw energy bar if low
        if self.energy < self.max_energy * 0.5:
            self._draw_energy_bar(screen, pos)

        # Debug info
        if getattr(config, 'DEBUG_MODE', False):
            if self.target_food and self.current_state == "seeking_food":
                start_pos = pos
                end_pos = (int(self.target_food['position'][0]),
                           int(self.target_food['position'][1]))
                pygame.draw.line(screen, (255, 255, 0), start_pos, end_pos, 1)