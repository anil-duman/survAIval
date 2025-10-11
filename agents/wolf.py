# agents/wolf.py - Wolf with integrated Q-Learning
"""
SurvAIval Wolf Agent
Carnivore predator with pack hunting behavior and optional Q-Learning
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
    from ai.wolf_q_learning import WolfQLearningAgent

    WOLF_Q_LEARNING_AVAILABLE = True
except ImportError:
    WOLF_Q_LEARNING_AVAILABLE = False
    print("⚠️ Wolf Q-Learning not available")


class Wolf(BaseAgent):
    """Wolf agent - Pack hunter with optional learning"""

    # Shared Q-Learning agent for all wolves (collective pack learning)
    shared_q_agent = None
    learning_enabled = False  # Global toggle

    @classmethod
    def enable_learning(cls, enable: bool = True):
        """Enable or disable Q-Learning for all wolves"""
        if not WOLF_Q_LEARNING_AVAILABLE and enable:
            print("❌ Cannot enable wolf learning: Q-Learning module not found")
            return False

        cls.learning_enabled = enable

        if enable and cls.shared_q_agent is None:
            cls.shared_q_agent = WolfQLearningAgent(
                learning_rate=0.1,
                discount_factor=0.95,
                exploration_rate=0.25,
                exploration_decay=0.9997
            )
            print("🐺🧠 Q-Learning enabled for wolves!")
        elif not enable:
            print("🔒 Wolf Q-Learning disabled")

        return True

    def __init__(self, position):
        """Initialize wolf agent"""
        super().__init__(position, "wolf", max_energy=120)

        # Wolf-specific properties
        self.size = 16
        self.max_speed = 3.0
        self.max_force = 0.5
        self.vision_range = 150.0
        self.energy_drain_rate = 0.08

        # Hunting parameters
        self.hunt_range = 120.0
        self.attack_distance = 20.0
        self.pack_coordination_range = 100.0
        self.hunt_energy_threshold = 60

        # Behavioral parameters
        self.aggression = 0.8
        self.pack_bonus = 0.3
        self.wander_strength = 0.15

        # Reproduction settings
        self.min_reproduction_energy = 90
        self.reproduction_cooldown = 0

        # AI state
        self.wander_angle = random.uniform(0, 2 * np.pi)
        self.target_prey = None
        self.hunting_timer = 0
        self.pack_mates = []
        self.last_kill_timer = 0

        # Learning state (only used if learning enabled)
        self.q_state = None
        self.q_action = None
        self.last_energy = self.energy
        self.last_killed_prey = False
        self.last_pack_hunt = False

        # Animation state
        self.animated_sprite: Optional[AnimatedSprite] = None
        self.facing_right = True
        self.last_velocity = np.array([1.0, 0.0])

    def set_animated_sprite(self, sprite: AnimatedSprite) -> None:
        """Assigns the AnimatedSprite instance to this agent"""
        self.animated_sprite = sprite

    def _update_animation_state(self) -> None:
        """Determines the correct animation and flip state"""
        if self.animated_sprite is None or not self.is_alive:
            return

        speed = np.linalg.norm(self.velocity)

        if speed > 0.1:
            if self.velocity[0] > 0.1:
                self.animated_sprite.set_flip(False)
            elif self.velocity[0] < -0.1:
                self.animated_sprite.set_flip(True)

        if self.current_state == "attacking":
            self.animated_sprite.play_animation('attack', reset=True)
        elif self.current_state == "chasing" or speed > 0.5:
            self.animated_sprite.play_animation('walk')
        elif speed > 0.05:
            self.animated_sprite.play_animation('walk')
        else:
            self.animated_sprite.play_animation('idle')

    def get_color(self):
        """Return wolf display color"""
        # Different colors for learning vs normal
        if self.learning_enabled:
            if self.hunting_timer > 0:
                return (200, 50, 200)  # Purple when hunting (learned)
            elif len(self.pack_mates) > 1:
                return (150, 100, 200)  # Purple-ish when in pack
            else:
                return (180, 120, 180)  # Light purple (learning mode)
        else:
            if self.hunting_timer > 0:
                return (150, 30, 30)
            elif self.energy < 40:
                return (80, 80, 80)
            elif len(self.pack_mates) > 0:
                return (120, 120, 140)
            else:
                return config.WOLF_COLOR

    def decide_action(self, world_state):
        """AI decision making - uses Q-Learning if enabled"""
        self._update_animation_state()

        if self.hunting_timer > 0:
            self.hunting_timer -= 1
        if self.last_kill_timer > 0:
            self.last_kill_timer -= 1

        self._update_pack_mates(world_state['same_species'])

        # Use Q-Learning if enabled
        if self.learning_enabled and self.shared_q_agent:
            return self._decide_with_learning(world_state)
        else:
            return self._decide_normal(world_state)

    def _decide_with_learning(self, world_state):
        """Decision making using Q-Learning"""
        # Get Q-Learning state
        world_state['pack_mates'] = self.pack_mates
        q_state = self.shared_q_agent.get_state(world_state)

        # Update Q-values from previous action
        if self.q_state is not None and self.q_action is not None:
            reward = self.shared_q_agent.calculate_reward(
                self, self.last_energy, self.q_action,
                killed_prey=self.last_killed_prey,
                pack_size=len(self.pack_mates),
                died=not self.is_alive,
                pack_hunt=self.last_pack_hunt
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
        self.last_killed_prey = False
        self.last_pack_hunt = False

        # Convert Q-action to game action
        return self._convert_q_action(q_action, world_state)

    def _decide_normal(self, world_state):
        """Normal decision making (original behavior)"""
        # Priority 1: HUNT when hungry
        if self.energy < self.hunt_energy_threshold:
            prey_list = self._find_prey(world_state['prey'])

            if prey_list:
                best_prey = self._select_best_prey(prey_list)

                if best_prey:
                    distance = best_prey['distance']

                    if distance < self.attack_distance:
                        self.current_state = "attacking"
                        self.hunting_timer = 30
                        return {
                            'type': 'hunt',
                            'target': best_prey['object']
                        }
                    else:
                        self.target_prey = best_prey
                        self.current_state = "chasing"
                        self.hunting_timer = 60

                        speed_multiplier = 1.0
                        if len(self.pack_mates) > 0:
                            speed_multiplier += self.pack_bonus

                        return {
                            'type': 'chase',
                            'target_position': best_prey['position'],
                            'speed_multiplier': speed_multiplier
                        }

        # Priority 2: Avoid bears
        threats = [p for p in world_state['predators'] if p.agent_type == 'bear']
        if threats:
            nearest_threat = min(threats, key=lambda t: np.linalg.norm(t.position - self.position))
            distance = np.linalg.norm(nearest_threat.position - self.position)

            if distance < 60:
                self.current_state = "avoiding"
                return {
                    'type': 'flee',
                    'threat_position': nearest_threat.position
                }

        # Priority 3: Reproduction
        if (self.energy > self.min_reproduction_energy and
                self.reproduction_cooldown == 0 and
                self.last_kill_timer == 0):

            potential_mate = self._find_mate(world_state['same_species'])
            if potential_mate:
                distance = np.linalg.norm(potential_mate.position - self.position)
                if distance < 35:
                    self.current_state = "reproducing"
                    return {
                        'type': 'reproduce',
                        'partner': potential_mate
                    }
                else:
                    self.current_state = "seeking_mate"
                    return {
                        'type': 'seek',
                        'target_position': potential_mate.position
                    }

        # Priority 4: Pack behavior
        if len(self.pack_mates) > 0 and random.random() < 0.3:
            pack_center = self._calculate_pack_center()
            distance_to_pack = np.linalg.norm(pack_center - self.position)

            if distance_to_pack > self.pack_coordination_range * 1.5:
                self.current_state = "regrouping"
                return {
                    'type': 'seek',
                    'target_position': pack_center
                }

        # Default: Patrol
        self.current_state = "patrolling"
        return self._patrol_behavior()

    def _convert_q_action(self, q_action: str, world_state: dict):
        """Convert Q-Learning action to game action"""
        prey_list = self._find_prey(world_state['prey'])

        if q_action == 'hunt_with_pack':
            if prey_list and len(self.pack_mates) > 0:
                best_prey = self._select_best_prey(prey_list)
                self.current_state = "hunting_with_pack_learned"
                self.last_pack_hunt = True
                return {
                    'type': 'chase',
                    'target_position': best_prey['position'],
                    'speed_multiplier': 1.0 + self.pack_bonus
                }

        elif q_action == 'hunt_alone':
            if prey_list:
                best_prey = self._select_best_prey(prey_list)
                self.current_state = "hunting_alone_learned"
                return {
                    'type': 'chase',
                    'target_position': best_prey['position']
                }

        elif q_action == 'wait_for_pack':
            self.current_state = "waiting_for_pack_learned"
            # Slow down and wait
            return {
                'type': 'move',
                'direction': np.array([0.0, 0.0])
            }

        elif q_action == 'chase_aggressively':
            if prey_list:
                best_prey = self._select_best_prey(prey_list)
                self.current_state = "chasing_aggressively_learned"
                return {
                    'type': 'chase',
                    'target_position': best_prey['position'],
                    'speed_multiplier': 1.3
                }

        elif q_action in ['coordinate_left', 'coordinate_right']:
            if prey_list and len(self.pack_mates) > 0:
                best_prey = self._select_best_prey(prey_list)
                # Flank the prey
                offset = np.array([50, 0]) if q_action == 'coordinate_left' else np.array([-50, 0])
                flank_pos = best_prey['position'] + offset
                self.current_state = "flanking_learned"
                self.last_pack_hunt = True
                return {
                    'type': 'seek',
                    'target_position': flank_pos
                }

        elif q_action == 'rest':
            self.current_state = "resting_learned"
            return {
                'type': 'move',
                'direction': np.array([0.0, 0.0])
            }

        # Default: patrol
        self.current_state = "patrolling_learned"
        return self._patrol_behavior()

    def _execute_action(self, action, world):
        """Override to add hunting-specific actions"""
        if not action:
            return

        action_type = action.get('type', 'idle')

        if action_type == 'chase':
            target_pos = action.get('target_position')
            speed_mult = action.get('speed_multiplier', 1.0)

            if target_pos is not None:
                old_max_speed = self.max_speed
                self.max_speed *= speed_mult
                self._seek_target(target_pos)
                self.max_speed = old_max_speed
        else:
            super()._execute_action(action, world)

    def _hunt_target(self, target, world):
        """Override hunting behavior for wolves"""
        distance = np.linalg.norm(self.position - target.position)
        self._seek_target(target.position)

        if distance < self.attack_distance:
            base_chance = 0.3
            pack_bonus = len(self.pack_mates) * 0.15
            energy_bonus = (self.energy / self.max_energy) * 0.2

            success_chance = min(0.9, base_chance + pack_bonus + energy_bonus)

            if random.random() < success_chance:
                energy_gain = min(target.energy * 0.8, self.max_energy - self.energy)
                self.energy += energy_gain
                target._die("hunted")

                self.last_kill_timer = 180

                # Track for learning
                if self.learning_enabled:
                    self.last_killed_prey = True
                    if len(self.pack_mates) > 0:
                        self.last_pack_hunt = True

                mode = "PACK" if len(self.pack_mates) > 0 else "solo"
                print(f"🐺 {self.id} killed {target.id} ({mode})! Energy: {self.energy:.1f}")

                self._share_kill_with_pack(energy_gain * 0.3)
            else:
                self.energy -= 5

    def _die(self, cause="natural"):
        """Override to give final learning update"""
        if self.learning_enabled and self.shared_q_agent:
            if self.q_state is not None and self.q_action is not None:
                final_reward = -150.0
                terminal_state = (0, 0, 0, 0)

                self.shared_q_agent.update_q_value(
                    self.q_state, self.q_action, final_reward, terminal_state
                )

        super()._die(cause)

    def _find_prey(self, prey_list):
        """Find available prey within hunting range"""
        available_prey = []

        for prey in prey_list:
            distance = np.linalg.norm(prey.position - self.position)

            if distance <= self.hunt_range:
                desirability = 1.0 - (prey.energy / prey.max_energy) * 0.5

                available_prey.append({
                    'object': prey,
                    'position': prey.position,
                    'distance': distance,
                    'energy': prey.energy,
                    'desirability': desirability
                })

        return available_prey

    def _select_best_prey(self, prey_list):
        """Select the best prey target"""
        if not prey_list:
            return None

        for prey in prey_list:
            distance_score = 1.0 - (prey['distance'] / self.hunt_range)
            prey['score'] = distance_score * 0.6 + prey['desirability'] * 0.4

        return max(prey_list, key=lambda p: p['score'])

    def _update_pack_mates(self, same_species):
        """Update list of nearby pack members"""
        self.pack_mates = []

        for wolf in same_species:
            if wolf.id != self.id:
                distance = np.linalg.norm(wolf.position - self.position)
                if distance < self.pack_coordination_range:
                    self.pack_mates.append(wolf)

    def _calculate_pack_center(self):
        """Calculate the center position of the pack"""
        if not self.pack_mates:
            return self.position

        positions = [wolf.position for wolf in self.pack_mates]
        positions.append(self.position)

        return np.mean(positions, axis=0)

    def _share_kill_with_pack(self, energy_amount):
        """Share energy from kill with nearby pack members"""
        if not self.pack_mates:
            return

        close_pack = [w for w in self.pack_mates
                      if np.linalg.norm(w.position - self.position) < 40]

        if close_pack:
            energy_per_wolf = energy_amount / len(close_pack)
            for wolf in close_pack:
                wolf.energy = min(wolf.max_energy, wolf.energy + energy_per_wolf)

    def _find_mate(self, same_species):
        """Find a potential mate"""
        eligible_mates = []

        for wolf in same_species:
            if (wolf.id != self.id and
                    wolf.energy > wolf.min_reproduction_energy and
                    wolf.reproduction_cooldown == 0):
                eligible_mates.append(wolf)

        if eligible_mates:
            return min(eligible_mates,
                       key=lambda w: np.linalg.norm(w.position - self.position))

        return None

    def _patrol_behavior(self):
        """Patrol behavior"""
        self.wander_angle += random.uniform(-0.2, 0.2)

        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        if random.random() < 0.02:
            self.wander_angle += random.uniform(-1.0, 1.0)

        return {
            'type': 'move',
            'direction': wander_force
        }

    def get_food_targets(self, nearby_entities):
        """Return entities this wolf can eat"""
        return [e for e in nearby_entities if e.agent_type in ['rabbit', 'deer']]

    def _create_offspring(self, position):
        """Create a new wolf offspring"""
        offspring_pos = position + np.random.normal(0, 15, 2)

        offspring_pos[0] = max(20, min(config.SCREEN_WIDTH - 20, offspring_pos[0]))
        offspring_pos[1] = max(20, min(config.SCREEN_HEIGHT - 20, offspring_pos[1]))

        new_wolf = Wolf(offspring_pos)

        new_wolf.max_speed = self.max_speed + random.uniform(-0.15, 0.15)
        new_wolf.vision_range = self.vision_range + random.uniform(-15, 15)
        new_wolf.aggression = max(0.3, min(1.0, self.aggression + random.uniform(-0.1, 0.1)))
        new_wolf.energy = 70

        mode = "LEARNING" if self.learning_enabled else "normal"
        print(f"🐺 New {mode} wolf born! {new_wolf.id}")
        return new_wolf

    def _is_predator(self, other_agent):
        """Wolves only fear bears"""
        return other_agent.agent_type == 'bear'

    def _is_prey(self, other_agent):
        """What wolves hunt"""
        return other_agent.agent_type in ['rabbit', 'deer']

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the wolf"""
        if not self.is_alive:
            return

        if self.animated_sprite:
            display_pos = (
                int(self.position[0] - camera_offset[0]),
                int(self.position[1] - camera_offset[1])
            )
            self.animated_sprite.draw(screen, display_pos)
        else:
            super().draw(screen, camera_offset)

        if getattr(config, 'DEBUG_MODE', False) and self.target_prey and self.current_state in ["chasing", "attacking"]:
            start_pos = (int(self.position[0] - camera_offset[0]), int(self.position[1] - camera_offset[1]))
            end_pos = (int(self.target_prey['position'][0] - camera_offset[0]),
                       int(self.target_prey['position'][1] - camera_offset[1]))
            pygame.draw.line(screen, (255, 0, 0), start_pos, end_pos, 2)

        if getattr(config, 'DEBUG_MODE', False) and self.pack_mates:
            for pack_mate in self.pack_mates:
                start_pos = (int(self.position[0] - camera_offset[0]), int(self.position[1] - camera_offset[1]))
                end_pos = (int(pack_mate.position[0] - camera_offset[0]), int(pack_mate.position[1] - camera_offset[1]))
                pygame.draw.line(screen, (100, 100, 255), start_pos, end_pos, 1)

    # Class methods for learning management
    @classmethod
    def save_learning(cls, filepath: str = "data/wolf_q_table.pkl"):
        """Save Q-table"""
        if cls.shared_q_agent:
            cls.shared_q_agent.save_q_table(filepath)
            cls.shared_q_agent.print_best_policies(top_n=5)

    @classmethod
    def load_learning(cls, filepath: str = "data/wolf_q_table.pkl"):
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