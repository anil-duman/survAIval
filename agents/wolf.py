# agents/wolf.py - Wolf predator agent
"""
SurvAIval Wolf Agent
Carnivore predator with pack hunting behavior
"""
import pygame
import numpy as np
import random
from agents.base_agent import BaseAgent
import config


class Wolf(BaseAgent):
    """Wolf agent - Pack hunter that preys on herbivores"""

    def __init__(self, position):
        """Initialize wolf agent

        Args:
            position: [x, y] starting position
        """
        super().__init__(position, "wolf", max_energy=120)

        # Wolf-specific properties
        self.size = 16
        self.max_speed = 3.0
        self.max_force = 0.5
        self.vision_range = 150.0
        self.energy_drain_rate = 0.08  # Predators burn more energy

        # Hunting parameters
        self.hunt_range = 120.0
        self.attack_distance = 20.0
        self.pack_coordination_range = 100.0
        self.hunt_energy_threshold = 60  # Only hunt if energy below this

        # Behavioral parameters
        self.aggression = 0.8
        self.pack_bonus = 0.3  # Extra speed when hunting in pack
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

    def get_color(self):
        """Return wolf display color"""
        if self.hunting_timer > 0:
            return (150, 30, 30)  # Dark red when hunting
        elif self.energy < 40:
            return (80, 80, 80)  # Dark gray when low energy
        elif len(self.pack_mates) > 0:
            return (120, 120, 140)  # Slightly blue when in pack
        else:
            return config.WOLF_COLOR

    def decide_action(self, world_state):
        """AI decision making for wolf behavior

        Args:
            world_state: Dictionary with world information

        Returns:
            Action dictionary
        """
        # Update timers
        if self.hunting_timer > 0:
            self.hunting_timer -= 1
        if self.last_kill_timer > 0:
            self.last_kill_timer -= 1

        # Update pack awareness
        self._update_pack_mates(world_state['same_species'])

        # Priority 1: HUNT when hungry
        if self.energy < self.hunt_energy_threshold:
            prey_list = self._find_prey(world_state['prey'])

            if prey_list:
                # Choose best prey target
                best_prey = self._select_best_prey(prey_list)

                if best_prey:
                    distance = best_prey['distance']

                    # If close enough to attack
                    if distance < self.attack_distance:
                        self.current_state = "attacking"
                        self.hunting_timer = 30
                        return {
                            'type': 'hunt',
                            'target': best_prey['object']
                        }

                    # Otherwise, chase
                    else:
                        self.target_prey = best_prey
                        self.current_state = "chasing"
                        self.hunting_timer = 60

                        # Pack coordination - adjust speed
                        speed_multiplier = 1.0
                        if len(self.pack_mates) > 0:
                            speed_multiplier += self.pack_bonus

                        return {
                            'type': 'chase',
                            'target_position': best_prey['position'],
                            'speed_multiplier': speed_multiplier
                        }

        # Priority 2: Avoid stronger predators (bears)
        threats = [p for p in world_state['predators'] if p.agent_type == 'bear']
        if threats:
            nearest_threat = min(threats, key=lambda t: np.linalg.norm(t.position - self.position))
            distance = np.linalg.norm(nearest_threat.position - self.position)

            if distance < 60:  # Too close to a bear
                self.current_state = "avoiding"
                return {
                    'type': 'flee',
                    'threat_position': nearest_threat.position
                }

        # Priority 3: Reproduction (when well-fed)
        if (self.energy > self.min_reproduction_energy and
                self.reproduction_cooldown == 0 and
                self.last_kill_timer == 0):  # Not distracted by recent kill

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

        # Priority 4: Pack behavior - stay somewhat close to pack
        if len(self.pack_mates) > 0 and random.random() < 0.3:
            # Occasionally move towards pack center
            pack_center = self._calculate_pack_center()
            distance_to_pack = np.linalg.norm(pack_center - self.position)

            if distance_to_pack > self.pack_coordination_range * 1.5:
                self.current_state = "regrouping"
                return {
                    'type': 'seek',
                    'target_position': pack_center
                }

        # Default: Patrol/wander (looking for prey)
        self.current_state = "patrolling"
        return self._patrol_behavior()

    def _execute_action(self, action, world):
        """Override to add hunting-specific actions"""
        if not action:
            return

        action_type = action.get('type', 'idle')

        if action_type == 'chase':
            # Chase with potential speed boost
            target_pos = action.get('target_position')
            speed_mult = action.get('speed_multiplier', 1.0)

            if target_pos is not None:
                # Temporarily boost speed
                old_max_speed = self.max_speed
                self.max_speed *= speed_mult
                self._seek_target(target_pos)
                self.max_speed = old_max_speed

        else:
            # Use parent class implementation
            super()._execute_action(action, world)

    def _hunt_target(self, target, world):
        """Override hunting behavior for wolves"""
        distance = np.linalg.norm(self.position - target.position)

        # Chase the target
        self._seek_target(target.position)

        # If close enough, attempt to kill
        if distance < self.attack_distance:
            # Attack success chance based on energy and pack size
            base_chance = 0.3
            pack_bonus = len(self.pack_mates) * 0.15
            energy_bonus = (self.energy / self.max_energy) * 0.2

            success_chance = min(0.9, base_chance + pack_bonus + energy_bonus)

            if random.random() < success_chance:
                # Successful kill!
                energy_gain = min(target.energy * 0.8, self.max_energy - self.energy)
                self.energy += energy_gain
                target._die("hunted")

                self.last_kill_timer = 180  # 3 seconds of satisfaction
                print(f"🐺 {self.id} killed {target.id}! Energy: {self.energy:.1f}")

                # Share kill with nearby pack mates
                self._share_kill_with_pack(energy_gain * 0.3)
            else:
                # Failed attack - prey escapes, lose some energy
                self.energy -= 5

    def _find_prey(self, prey_list):
        """Find available prey within hunting range"""
        available_prey = []

        for prey in prey_list:
            distance = np.linalg.norm(prey.position - self.position)

            if distance <= self.hunt_range:
                # Prefer low-energy prey (easier to catch)
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
        """Select the best prey target based on distance and desirability"""
        if not prey_list:
            return None

        # Score each prey: closer and weaker = better
        for prey in prey_list:
            distance_score = 1.0 - (prey['distance'] / self.hunt_range)
            prey['score'] = distance_score * 0.6 + prey['desirability'] * 0.4

        # Return highest scored prey
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

        # Only share with very close pack members
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
        """Patrol behavior - purposeful wandering while hunting"""
        # More directed wandering than rabbits
        self.wander_angle += random.uniform(-0.2, 0.2)

        wander_force = np.array([
            np.cos(self.wander_angle),
            np.sin(self.wander_angle)
        ]) * self.wander_strength

        # Occasionally change direction more dramatically
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

        # Keep within bounds
        offspring_pos[0] = max(20, min(config.SCREEN_WIDTH - 20, offspring_pos[0]))
        offspring_pos[1] = max(20, min(config.SCREEN_HEIGHT - 20, offspring_pos[1]))

        new_wolf = Wolf(offspring_pos)

        # Inherit traits with variation
        new_wolf.max_speed = self.max_speed + random.uniform(-0.15, 0.15)
        new_wolf.vision_range = self.vision_range + random.uniform(-15, 15)
        new_wolf.aggression = max(0.3, min(1.0, self.aggression + random.uniform(-0.1, 0.1)))
        new_wolf.energy = 70

        print(f"🐺 New wolf born! {new_wolf.id}")
        return new_wolf

    def _is_predator(self, other_agent):
        """Wolves only fear bears"""
        return other_agent.agent_type == 'bear'

    def _is_prey(self, other_agent):
        """What wolves hunt"""
        return other_agent.agent_type in ['rabbit', 'deer']

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the wolf with visual feedback"""
        if not self.is_alive:
            return

        # Draw using parent method
        super().draw(screen, camera_offset)

        # Draw hunting target line in debug mode
        if (getattr(config, 'DEBUG_MODE', False) and
                self.target_prey and
                self.current_state in ["chasing", "attacking"]):
            start_pos = (int(self.position[0]), int(self.position[1]))
            end_pos = (int(self.target_prey['position'][0]),
                       int(self.target_prey['position'][1]))

            pygame.draw.line(screen, (255, 0, 0), start_pos, end_pos, 2)

        # Draw pack connections in debug mode
        if getattr(config, 'DEBUG_MODE', False) and self.pack_mates:
            for pack_mate in self.pack_mates:
                start_pos = (int(self.position[0]), int(self.position[1]))
                end_pos = (int(pack_mate.position[0]), int(pack_mate.position[1]))
                pygame.draw.line(screen, (100, 100, 255), start_pos, end_pos, 1)