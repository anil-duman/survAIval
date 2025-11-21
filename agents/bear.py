# agents/bear.py - Bear apex predator agent
"""
SurvAIval Bear Agent
Apex predator - hunts both herbivores and wolves
"""

import pygame
import numpy as np
import random
from agents.base_agent import BaseAgent
import config
from typing import Optional
from utils.animation import AnimatedSprite

# Import Q-Learning
try:
    from ai.bear_q_learning import BearQLearningAgent

    BEAR_Q_LEARNING_AVAILABLE = True
except ImportError:
    BEAR_Q_LEARNING_AVAILABLE = False


class Bear(BaseAgent):
    """Bear agent - Apex predator with territorial behavior"""
    # Shared Q-Learning
    shared_q_agent = None
    learning_enabled = False

    @classmethod
    def enable_learning(cls, enable: bool = True):
        if not BEAR_Q_LEARNING_AVAILABLE and enable:
            print("❌ Cannot enable bear learning")
            return False

        cls.learning_enabled = enable

        if enable and cls.shared_q_agent is None:
            cls.shared_q_agent = BearQLearningAgent()
            print("🐻🧠 Q-Learning enabled for bears!")
        elif not enable:
            print("🔒 Bear Q-Learning disabled")

        return True

    @classmethod
    def save_learning(cls, filepath="data/bear_q_table.pkl"):
        if cls.shared_q_agent:
            cls.shared_q_agent.save_q_table(filepath)
            cls.shared_q_agent.print_best_policies(5)

    @classmethod
    def load_learning(cls, filepath="data/bear_q_table.pkl"):
        if cls.shared_q_agent:
            return cls.shared_q_agent.load_q_table(filepath)
        return False

    @classmethod
    def get_learning_stats(cls):
        if cls.shared_q_agent:
            return cls.shared_q_agent.get_statistics()
        return None

    @classmethod
    def decay_exploration(cls):
        if cls.shared_q_agent:
            cls.shared_q_agent.decay_exploration()

    def __init__(self, position):
        """Initialize bear agent

        Args:
            position: [x, y] starting position
        """
        super().__init__(position, "bear", max_energy=150)

        # Bear-specific properties
        self.size = 24
        self.max_speed = 2.5
        self.max_force = 0.6
        self.vision_range = 180.0
        self.energy_drain_rate = 0.1  # Burns lots of energy

        # Hunting parameters
        self.hunt_range = 140.0
        self.attack_distance = 25.0
        self.hunt_energy_threshold = 80  # Only hunt when hungry
        self.territorial_radius = 150.0

        # Behavioral parameters
        self.aggression = 0.95
        self.territorial_aggression = 0.7
        self.wander_strength = 0.12

        # Reproduction settings
        self.min_reproduction_energy = 110
        self.reproduction_cooldown = 0

        # AI state
        self.wander_angle = random.uniform(0, 2 * np.pi)
        self.target_prey = None
        self.hunting_timer = 0
        self.last_kill_timer = 0
        self.territory_center = np.array(position, dtype=float)
        self.territory_established = False

        # Learning state
        self.q_state = None
        self.q_action = None
        self.last_energy = self.energy
        self.last_killed_prey = False
        self.prey_type_killed = None

        # Animation state
        self.animated_sprite: Optional[AnimatedSprite] = None
        self.facing_right = True
        self.last_velocity = np.array([1.0, 0.0])

    def set_animated_sprite(self, animated_sprite):
        """Set the animated sprite for this bear"""
        self.animated_sprite = animated_sprite

    def get_color(self):
        """Return bear display color"""
        if self.hunting_timer > 0:
            return (180, 50, 30)  # Red when hunting
        elif self.energy < 60:
            return (80, 50, 30)  # Dark when low energy
        elif self.last_kill_timer > 0:
            return (120, 80, 50)  # Satisfied after kill
        else:
            return config.BEAR_COLOR

    def decide_action(self, world_state):
        """AI decision making for bear behavior"""
        if self.hunting_timer > 0:
            self.hunting_timer -= 1
        if self.last_kill_timer > 0:
            self.last_kill_timer -= 1

        # Establish territory if not done yet
        if not self.territory_established and self.age > 100:
            self.territory_center = self.position.copy()
            self.territory_established = True
            print(f"🐻 {self.id} established territory at {self.territory_center}")

        # Priority 1: HUNT when hungry
        if self.energy < self.hunt_energy_threshold:
            prey_list = self._find_prey(world_state['prey'])

            if prey_list:
                best_prey = self._select_best_prey(prey_list)

                if best_prey:
                    distance = best_prey['distance']

                    if distance < self.attack_distance:
                        self.current_state = "attacking"
                        self.hunting_timer = 40

                        if self.animated_sprite:
                            self.animated_sprite.play_animation('attack', reset=True)

                        return {
                            'type': 'hunt',
                            'target': best_prey['object']
                        }
                    else:
                        self.target_prey = best_prey
                        self.current_state = "hunting"
                        self.hunting_timer = 80

                        if self.animated_sprite:
                            self.animated_sprite.play_animation('walk')

                        return {
                            'type': 'chase',
                            'target_position': best_prey['position']
                        }

        # Priority 2: Defend territory from other bears
        if self.territory_established:
            nearby_bears = [a for a in world_state['same_species']
                            if a.id != self.id]

            for bear in nearby_bears:
                distance_to_territory = np.linalg.norm(bear.position - self.territory_center)

                if distance_to_territory < self.territorial_radius:
                    self.current_state = "defending_territory"

                    if self.animated_sprite:
                        self.animated_sprite.play_animation('walk')

                    return {
                        'type': 'chase',
                        'target_position': bear.position
                    }

        # Priority 3: Return to territory if too far
        if self.territory_established:
            distance_from_territory = np.linalg.norm(self.position - self.territory_center)

            if distance_from_territory > self.territorial_radius * 1.5:
                self.current_state = "returning_to_territory"

                if self.animated_sprite:
                    self.animated_sprite.play_animation('walk')

                return {
                    'type': 'seek',
                    'target_position': self.territory_center
                }

        # Priority 4: Reproduction (when well-fed and in territory)
        if (self.energy > self.min_reproduction_energy and
                self.reproduction_cooldown == 0 and
                self.last_kill_timer == 0):

            potential_mate = self._find_mate(world_state['same_species'])
            if potential_mate:
                distance = np.linalg.norm(potential_mate.position - self.position)
                if distance < 40:
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

        # Default: Patrol territory or wander
        self.current_state = "patrolling"

        if self.animated_sprite:
            speed = np.linalg.norm(self.velocity)
            if speed < 0.3:
                self.animated_sprite.play_animation('idle')
            else:
                self.animated_sprite.play_animation('walk')

        return self._patrol_behavior()

    def update(self, world):
        """Update bear state and animation"""
        super().update(world)

        if np.linalg.norm(self.velocity) > 0.1:
            self.last_velocity = self.velocity.copy()

            if self.velocity[0] > 0.1:
                self.facing_right = True
            elif self.velocity[0] < -0.1:
                self.facing_right = False

        if self.animated_sprite:
            self.animated_sprite.set_flip(not self.facing_right)

            # Animation management
            if not self.is_alive:
                if self.animated_sprite.current_animation != 'death':
                    self.animated_sprite.play_animation('death', reset=True)
            elif self.hunting_timer > 0 and self.current_state == "attacking":
                if self.animated_sprite.current_animation != 'attack':
                    self.animated_sprite.play_animation('attack', reset=True)
            else:
                speed = np.linalg.norm(self.velocity)
                if speed < 0.3:
                    self.animated_sprite.play_animation('idle')
                else:
                    self.animated_sprite.play_animation('walk')

    def _execute_action(self, action, world):
        """Override to add bear-specific actions"""
        if not action:
            return

        action_type = action.get('type', 'idle')

        if action_type == 'chase':
            target_pos = action.get('target_position')
            if target_pos is not None:
                self._seek_target(target_pos)
        else:
            super()._execute_action(action, world)

    def _hunt_target(self, target, world):
        """Override hunting behavior for bears with attack animation"""
        distance = np.linalg.norm(self.position - target.position)

        self._seek_target(target.position)

        if distance < self.attack_distance:
            # Trigger attack animation
            if self.animated_sprite:
                self.animated_sprite.play_animation('attack', reset=True)

            # Bears are very successful hunters
            base_chance = 0.6
            energy_bonus = (self.energy / self.max_energy) * 0.2

            success_chance = min(0.95, base_chance + energy_bonus)

            if random.random() < success_chance:
                # Massive energy gain from kill
                energy_gain = min(target.energy * 0.9, self.max_energy - self.energy)
                self.energy += energy_gain

                # Trigger death animation on target
                if hasattr(target, 'animated_sprite') and target.animated_sprite:
                    target.animated_sprite.play_animation('death', reset=True)

                target._die("hunted")

                self.last_kill_timer = 240
                print(f"🐻 {self.id} killed {target.agent_type} {target.id}! Energy: {self.energy:.1f}")
            else:
                self.energy -= 8

    def _die(self, cause="natural"):
        """Handle death with animation"""
        if self.animated_sprite:
            self.animated_sprite.play_animation('death', reset=True)

        super()._die(cause)

    def _find_prey(self, prey_list):
        """Find available prey (everything is prey for bears)"""
        available_prey = []

        for prey in prey_list:
            distance = np.linalg.norm(prey.position - self.position)

            if distance <= self.hunt_range:
                # Prefer larger prey (more energy)
                size_preference = {
                    'deer': 1.2,
                    'wolf': 1.0,
                    'rabbit': 0.7
                }.get(prey.agent_type, 0.8)

                desirability = size_preference * (1.0 - (prey.energy / prey.max_energy) * 0.3)

                available_prey.append({
                    'object': prey,
                    'position': prey.position,
                    'distance': distance,
                    'energy': prey.energy,
                    'desirability': desirability
                })

        return available_prey

    def _select_best_prey(self, prey_list):
        """Select best prey target"""
        if not prey_list:
            return None

        for prey in prey_list:
            distance_score = 1.0 - (prey['distance'] / self.hunt_range)
            prey['score'] = distance_score * 0.5 + prey['desirability'] * 0.5

        return max(prey_list, key=lambda p: p['score'])

    def _find_mate(self, same_species):
        """Find potential mate"""
        eligible_mates = []

        for bear in same_species:
            if (bear.id != self.id and
                    bear.energy > bear.min_reproduction_energy and
                    bear.reproduction_cooldown == 0):
                eligible_mates.append(bear)

        if eligible_mates:
            return min(eligible_mates,
                       key=lambda b: np.linalg.norm(b.position - self.position))

        return None

    def _patrol_behavior(self):
        """Patrol behavior - wander within territory"""
        if self.territory_established:
            # Wander but stay near territory
            to_center = self.territory_center - self.position
            distance_from_center = np.linalg.norm(to_center)

            if distance_from_center > self.territorial_radius:
                # Pull towards territory
                pull_force = (to_center / distance_from_center) * 0.2
                return {
                    'type': 'move',
                    'direction': pull_force
                }

        # Normal wandering
        self.wander_angle += random.uniform(-0.15, 0.15)

        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        if random.random() < 0.015:
            self.wander_angle += random.uniform(-0.8, 0.8)

        return {
            'type': 'move',
            'direction': wander_force
        }

    def get_food_targets(self, nearby_entities):
        """Bears eat everything"""
        return [e for e in nearby_entities if e.agent_type in ['rabbit', 'deer', 'wolf']]

    def _create_offspring(self, position):
        """Create new bear offspring"""
        offspring_pos = position + np.random.normal(0, 20, 2)

        offspring_pos[0] = max(25, min(config.WORLD_WIDTH - 25, offspring_pos[0]))
        offspring_pos[1] = max(25, min(config.WORLD_HEIGHT - 25, offspring_pos[1]))

        new_bear = Bear(offspring_pos)

        new_bear.max_speed = self.max_speed + random.uniform(-0.1, 0.1)
        new_bear.vision_range = self.vision_range + random.uniform(-15, 15)
        new_bear.aggression = max(0.7, min(1.0, self.aggression + random.uniform(-0.05, 0.05)))
        new_bear.energy = 90

        print(f"🐻 New bear born! {new_bear.id}")
        return new_bear

    def _is_predator(self, other_agent):
        """Bears have no natural predators"""
        return False

    def _is_prey(self, other_agent):
        """Bears hunt everything"""
        return other_agent.agent_type in ['rabbit', 'deer', 'wolf']

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the bear"""
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

        # Debug: show territory
        if getattr(config, 'DEBUG_MODE', False) and self.territory_established:
            territory_pos = (int(self.territory_center[0] - camera_offset[0]),
                             int(self.territory_center[1] - camera_offset[1]))
            pygame.draw.circle(screen, (255, 100, 0), territory_pos, int(self.territorial_radius), 2)
            pygame.draw.line(screen, (255, 150, 0), pos, territory_pos, 1)

# FILE ENDS HERE - NO MORE CODE AFTER THIS!