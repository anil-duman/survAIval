# agents/deer.py - Deer herbivore agent
"""
SurvAIval Deer Agent
Large herbivore with herd behavior
"""

import pygame
import numpy as np
import random
from agents.base_agent import BaseAgent
import config
from typing import Optional
from utils.animation import AnimatedSprite


class Deer(BaseAgent):
    """Deer agent - Large herbivore with herd instincts"""

    def __init__(self, position):
        """Initialize deer agent

        Args:
            position: [x, y] starting position
        """
        super().__init__(position, "deer", max_energy=100)

        # Deer-specific properties
        self.size = 18
        self.max_speed = 2.8
        self.max_force = 0.35
        self.vision_range = 120.0
        self.energy_drain_rate = 0.04

        # Behavioral parameters
        self.fear_distance = 90.0
        self.food_search_radius = 140.0
        self.eating_distance = 18.0
        self.herd_range = 100.0
        self.wander_strength = 0.15

        # Reproduction settings
        self.min_reproduction_energy = 70
        self.reproduction_cooldown = 0

        # AI state
        self.wander_angle = random.uniform(0, 2 * np.pi)
        self.panic_timer = 0
        self.target_food = None
        self.eating_timer = 0
        self.herd_mates = []

        # Animation state
        self.animated_sprite: Optional[AnimatedSprite] = None
        self.facing_right = True
        self.last_velocity = np.array([1.0, 0.0])

    def set_animated_sprite(self, animated_sprite):
        """Set the animated sprite for this deer"""
        self.animated_sprite = animated_sprite

    def get_color(self):
        """Return deer display color"""
        if self.panic_timer > 0:
            return (200, 100, 80)
        elif self.eating_timer > 0:
            return (200, 150, 100)
        elif len(self.herd_mates) > 2:
            return (180, 120, 80)  # Slightly different when in herd
        else:
            return config.DEER_COLOR

    def decide_action(self, world_state):
        """AI decision making for deer behavior"""
        if self.panic_timer > 0:
            self.panic_timer -= 1
        if self.eating_timer > 0:
            self.eating_timer -= 1

        # Update herd awareness
        self._update_herd_mates(world_state['same_species'])

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

                # Flee towards herd center if possible
                if len(self.herd_mates) > 0:
                    herd_center = self._calculate_herd_center()
                    # Flee away from predator but towards herd
                    flee_direction = self.position - nearest_predator.position
                    herd_direction = herd_center - self.position
                    combined_direction = flee_direction * 0.7 + herd_direction * 0.3

                    return {
                        'type': 'flee',
                        'threat_position': self.position - combined_direction
                    }

                return {
                    'type': 'flee',
                    'threat_position': nearest_predator.position,
                    'urgency': 1.0 - (distance / self.fear_distance)
                }

        # Priority 2: Seek and eat food
        if self.energy < 75:
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

        # Priority 3: Stay with herd
        if len(self.herd_mates) > 0:
            herd_center = self._calculate_herd_center()
            distance_to_herd = np.linalg.norm(herd_center - self.position)

            # If too far from herd, move towards it
            if distance_to_herd > self.herd_range * 1.2:
                self.current_state = "regrouping"

                if self.animated_sprite:
                    self.animated_sprite.play_animation('walk')

                return {
                    'type': 'seek',
                    'target_position': herd_center
                }

        # Priority 4: Reproduction
        if (self.energy > self.min_reproduction_energy and
                self.reproduction_cooldown == 0):
            potential_mate = self._find_mate(world_state['same_species'])
            if potential_mate:
                distance = np.linalg.norm(potential_mate.position - self.position)
                if distance < 35:
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

        # Default: Wander with herd influence
        self.current_state = "wandering"

        if self.animated_sprite:
            speed = np.linalg.norm(self.velocity)
            if speed < 0.4:
                self.animated_sprite.play_animation('idle')
            else:
                self.animated_sprite.play_animation('walk')

        return self._wander_with_herd()

    def update(self, world):
        """Update deer state and animation"""
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
                amount=30
            )

            if nutrition_gained > 0:
                self.energy = min(self.max_energy, self.energy + nutrition_gained)
                self.eating_timer = 25
                self.energy -= 0.3

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
                self.position,
                self.food_search_radius
            )
            world_state['food_sources'] = nearby_food
        else:
            world_state['food_sources'] = []

        return world_state

    def _update_herd_mates(self, same_species):
        """Update list of nearby herd members"""
        self.herd_mates = []

        for deer in same_species:
            if deer.id != self.id:
                distance = np.linalg.norm(deer.position - self.position)
                if distance < self.herd_range:
                    self.herd_mates.append(deer)

    def _calculate_herd_center(self):
        """Calculate center of the herd"""
        if not self.herd_mates:
            return self.position

        positions = [deer.position for deer in self.herd_mates]
        positions.append(self.position)

        return np.mean(positions, axis=0)

    def _find_nearest_predator(self, predators):
        """Find closest predator"""
        if not predators:
            return None

        return min(predators, key=lambda p: np.linalg.norm(p.position - self.position))

    def _find_mate(self, same_species):
        """Find potential mate"""
        eligible_mates = []

        for deer in same_species:
            if (deer.id != self.id and
                    deer.energy > deer.min_reproduction_energy and
                    deer.reproduction_cooldown == 0):
                eligible_mates.append(deer)

        if eligible_mates:
            return min(eligible_mates,
                       key=lambda d: np.linalg.norm(d.position - self.position))

        return None

    def _wander_with_herd(self):
        """Wander but influenced by herd"""
        self.wander_angle += random.uniform(-0.25, 0.25)

        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        # Add herd influence
        if len(self.herd_mates) > 0:
            herd_center = self._calculate_herd_center()
            herd_direction = herd_center - self.position
            herd_distance = np.linalg.norm(herd_direction)

            if herd_distance > 0:
                herd_force = (herd_direction / herd_distance) * 0.05
                wander_force += herd_force

        return {
            'type': 'move',
            'direction': wander_force
        }

    def get_food_targets(self, nearby_entities):
        """Deer don't eat other animals"""
        return []

    def _create_offspring(self, position):
        """Create new deer offspring"""
        offspring_pos = position + np.random.normal(0, 12, 2)

        offspring_pos[0] = max(20, min(config.SCREEN_WIDTH - 20, offspring_pos[0]))
        offspring_pos[1] = max(20, min(config.SCREEN_HEIGHT - 20, offspring_pos[1]))

        new_deer = Deer(offspring_pos)

        new_deer.max_speed = self.max_speed + random.uniform(-0.15, 0.15)
        new_deer.vision_range = self.vision_range + random.uniform(-10, 10)
        new_deer.fear_distance = self.fear_distance + random.uniform(-8, 8)
        new_deer.energy = 55

        print(f"🦌 New deer born! {new_deer.id}")
        return new_deer

    def _is_predator(self, other_agent):
        """Define what threatens deer"""
        return other_agent.agent_type in ['wolf', 'bear']

    def _is_prey(self, other_agent):
        """Deer don't hunt"""
        return False

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the deer"""
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

        # Debug: show herd connections
        if getattr(config, 'DEBUG_MODE', False) and self.herd_mates:
            for herd_mate in self.herd_mates[:3]:  # Only show 3 nearest
                start_pos = pos
                end_pos = (int(herd_mate.position[0]), int(herd_mate.position[1]))
                pygame.draw.line(screen, (100, 180, 100), start_pos, end_pos, 1)