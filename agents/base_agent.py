# agents/base_agent.py - Base class for all AI agents
"""
SurvAIval Base Agent Class
Abstract base class for all animal agents in the ecosystem
"""

import pygame
import numpy as np
import random
from abc import ABC, abstractmethod
import config


class BaseAgent(ABC):
    """Abstract base class for all AI agents in the simulation"""

    def __init__(self, position, agent_type, max_energy=100):
        """Initialize base agent properties

        Args:
            position: [x, y] starting position
            agent_type: string identifier (rabbit, wolf, etc.)
            max_energy: maximum energy capacity
        """
        # Position and movement
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])

        # Agent identity
        self.agent_type = agent_type
        self.id = self._generate_id()

        # Life properties
        self.energy = max_energy
        self.max_energy = max_energy
        self.age = 0
        self.is_alive = True

        # Physical properties (can be overridden by subclasses)
        self.size = 16
        self.max_speed = 2.0
        self.max_force = 0.3
        self.vision_range = 80.0
        self.energy_drain_rate = 0.05

        # AI state
        self.current_state = "wandering"
        self.target = None
        self.memory = {}

        # Reproduction
        self.reproduction_cooldown = 0
        self.min_reproduction_energy = 80

        # Debug
        self.debug_info = {}

    def _generate_id(self):
        """Generate unique ID for this agent"""
        return f"{self.agent_type}_{random.randint(1000, 9999)}"

    # Abstract methods (must be implemented by subclasses)
    @abstractmethod
    def get_color(self):
        """Return the display color for this agent type"""
        pass

    @abstractmethod
    def decide_action(self, world_state):
        """AI decision making - returns action to take

        Args:
            world_state: Dictionary containing world information

        Returns:
            Dictionary with action type and parameters
        """
        pass

    @abstractmethod
    def get_food_targets(self, nearby_entities):
        """Return list of entities this agent can eat

        Args:
            nearby_entities: List of nearby agents/resources

        Returns:
            List of valid food targets
        """
        pass

    # Core update methods
    def update(self, world):
        """Main update loop for the agent"""
        if not self.is_alive:
            return

        # Age and natural energy drain
        self.age += 1
        self.energy -= self.energy_drain_rate

        # Death conditions
        if self.energy <= 0:
            self._die("starvation")
            return

        # Get world state for AI decision making
        world_state = self._perceive_world(world)

        # AI decides what to do
        action = self.decide_action(world_state)

        # Execute the action
        self._execute_action(action, world)

        # Apply physics
        self._update_physics()

        # Update cooldowns
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

    def _perceive_world(self, world):
        """Gather information about the world around this agent"""
        # Get nearby entities
        nearby_entities = world.get_entities_in_range(
            self.position, self.vision_range
        )

        # Remove self from the list
        nearby_entities = [e for e in nearby_entities if e.id != self.id]

        # Categorize entities
        predators = []
        prey = []
        same_species = []

        for entity in nearby_entities:
            if self._is_predator(entity):
                predators.append(entity)
            elif self._is_prey(entity):
                prey.append(entity)
            elif entity.agent_type == self.agent_type:
                same_species.append(entity)

        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'energy': self.energy,
            'age': self.age,
            'nearby_entities': nearby_entities,
            'predators': predators,
            'prey': prey,
            'same_species': same_species,
            'world_bounds': [config.SCREEN_WIDTH, config.SCREEN_HEIGHT]
        }

    def _execute_action(self, action, world):
        """Execute the action decided by AI"""
        if not action:
            return

        action_type = action.get('type', 'idle')

        if action_type == 'move':
            direction = action.get('direction', [0, 0])
            self._apply_force(direction)

        elif action_type == 'flee':
            threat_pos = action.get('threat_position')
            if threat_pos is not None:
                self._flee_from(threat_pos)

        elif action_type == 'seek':
            target_pos = action.get('target_position')
            if target_pos is not None:
                self._seek_target(target_pos)

        elif action_type == 'hunt':
            target = action.get('target')
            if target:
                self._hunt_target(target, world)

        elif action_type == 'reproduce':
            partner = action.get('partner')
            if partner:
                self._attempt_reproduction(partner, world)

    # Movement and physics methods
    def _apply_force(self, force):
        """Apply force to agent's acceleration"""
        force = np.array(force)
        # Limit force magnitude
        force_mag = np.linalg.norm(force)
        if force_mag > self.max_force:
            force = force / force_mag * self.max_force

        self.acceleration += force

    def _seek_target(self, target_pos):
        """Move towards a target position"""
        direction = np.array(target_pos) - self.position
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize and scale by desired speed
            desired_velocity = (direction / distance) * self.max_speed
            # Calculate steering force
            steering_force = desired_velocity - self.velocity
            self._apply_force(steering_force)

    def _flee_from(self, threat_pos):
        """Move away from a threat position"""
        direction = self.position - np.array(threat_pos)
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize and scale by desired speed
            desired_velocity = (direction / distance) * self.max_speed
            # Calculate steering force
            steering_force = desired_velocity - self.velocity
            self._apply_force(steering_force)

    def _update_physics(self):
        """Update position based on velocity and acceleration"""
        # Update velocity
        self.velocity += self.acceleration

        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # Update position
        self.position += self.velocity

        # Handle screen boundaries (bounce)
        if self.position[0] < 0 or self.position[0] > config.SCREEN_WIDTH:
            self.velocity[0] *= -0.8  # Bounce with some energy loss
            self.position[0] = max(0, min(config.SCREEN_WIDTH, self.position[0]))

        if self.position[1] < 0 or self.position[1] > config.SCREEN_HEIGHT:
            self.velocity[1] *= -0.8
            self.position[1] = max(0, min(config.SCREEN_HEIGHT, self.position[1]))

        # Reset acceleration
        self.acceleration = np.array([0.0, 0.0])

        # Add small amount of friction
        self.velocity *= 0.95

    # Interaction methods
    def _hunt_target(self, target, world):
        """Attempt to hunt/eat a target"""
        distance = np.linalg.norm(self.position - target.position)

        # Move towards target
        self._seek_target(target.position)

        # If close enough, attempt to eat
        if distance < (self.size + target.size):
            success = self._attempt_eat(target)
            if success:
                # Gain energy from eating
                energy_gain = min(target.energy * 0.7, self.max_energy - self.energy)
                self.energy += energy_gain
                target._die("eaten")

    def _attempt_eat(self, target):
        """Attempt to eat another agent"""
        # Simple success based on energy difference
        success_chance = self.energy / (self.energy + target.energy)
        return random.random() < success_chance

    def _attempt_reproduction(self, partner, world):
        """Attempt to reproduce with another agent"""
        if (self.reproduction_cooldown == 0 and
                partner.reproduction_cooldown == 0 and
                self.energy >= self.min_reproduction_energy and
                partner.energy >= partner.min_reproduction_energy):

            # Create offspring
            offspring_pos = (self.position + partner.position) / 2
            offspring_pos += np.random.normal(0, 20, 2)  # Small random offset

            # Create new agent (this would be implemented by subclasses)
            offspring = self._create_offspring(offspring_pos)
            if offspring:
                world.add_entity(offspring)

                # Pay energy cost
                energy_cost = 30
                self.energy -= energy_cost
                partner.energy -= energy_cost

                # Set reproduction cooldown
                self.reproduction_cooldown = 300  # 5 seconds at 60 FPS
                partner.reproduction_cooldown = 300

    @abstractmethod
    def _create_offspring(self, position):
        """Create offspring of the same type (implemented by subclasses)"""
        pass

    # Helper methods
    def _is_predator(self, other_agent):
        """Check if another agent is a predator to this one"""
        # This will be defined by each species
        predator_relationships = {
            'rabbit': ['wolf', 'bear'],
            'deer': ['wolf', 'bear'],
            'wolf': ['bear'],  # Bears can threaten wolves
            'bear': []  # Bears have no natural predators
        }

        return other_agent.agent_type in predator_relationships.get(self.agent_type, [])

    def _is_prey(self, other_agent):
        """Check if another agent is prey to this one"""
        prey_relationships = {
            'wolf': ['rabbit', 'deer'],
            'bear': ['rabbit', 'deer', 'wolf'],
            'rabbit': [],  # Rabbits don't hunt
            'deer': []  # Deer don't hunt
        }

        return other_agent.agent_type in prey_relationships.get(self.agent_type, [])

    def _die(self, cause="natural"):
        """Handle agent death"""
        self.is_alive = False
        self.current_state = "dead"
        print(f"🪦 {self.id} died from {cause} at age {self.age}")

    def get_distance_to(self, other_agent):
        """Get distance to another agent"""
        return np.linalg.norm(self.position - other_agent.position)

    # Rendering method
    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the agent on screen"""
        if not self.is_alive:
            return

        # Calculate screen position
        screen_pos = (
            int(self.position[0] - camera_offset[0]),
            int(self.position[1] - camera_offset[1])
        )

        # Draw agent body
        color = self.get_color()
        pygame.draw.circle(screen, color, screen_pos, self.size)
        pygame.draw.circle(screen, config.BLACK, screen_pos, self.size, 2)

        # Draw energy bar if energy is low
        if self.energy < self.max_energy * 0.5:
            self._draw_energy_bar(screen, screen_pos)

        # Draw state indicator (for debugging)
        if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
            self._draw_debug_info(screen, screen_pos)

    def _draw_energy_bar(self, screen, pos):
        """Draw energy bar above agent"""
        bar_width = self.size * 2
        bar_height = 4
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] - self.size - 10

        # Background
        pygame.draw.rect(screen, config.BLACK, (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2))

        # Energy level
        energy_ratio = self.energy / self.max_energy
        energy_width = int(bar_width * energy_ratio)

        # Color based on energy level
        if energy_ratio > 0.6:
            energy_color = config.GREEN
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)  # Yellow
        else:
            energy_color = (255, 0, 0)  # Red

        pygame.draw.rect(screen, energy_color, (bar_x, bar_y, energy_width, bar_height))

    def _draw_debug_info(self, screen, pos):
        """Draw debug information"""
        font = pygame.font.Font(None, 16)
        debug_text = f"{self.current_state}"
        text_surface = font.render(debug_text, True, config.WHITE)
        screen.blit(text_surface, (pos[0] - 20, pos[1] + self.size + 5))