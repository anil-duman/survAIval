# engine/simulation.py - Simulation with animations fixed
"""
SurvAIval Simulation Controller
Complete version with animation system properly integrated
"""

import pygame
import random
import sys
import os
import config
from agents.rabbit import Rabbit
from agents.wolf import Wolf
from environment.food_system import FoodManager
from utils.animation import init_animations, get_animation_manager


class Simulation:
    """Main simulation controller class"""

    def __init__(self, screen):
        """Initialize the simulation

        Args:
            screen: Pygame display surface
        """
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

        # Agent management
        self.agents = []
        self.dead_agents = []

        # Food system
        self.food_manager = FoodManager()
        self.food_manager.initialize_food_sources(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        # Animation system - CRITICAL: Load before creating agents
        print("🎬 Initializing animation system...")
        self.animation_manager = init_animations()

        # Statistics
        self.stats = {
            'total_born': 0,
            'total_died': 0,
            'generation': 1,
            'rabbits_eaten': 0,
            'wolves_starved': 0
        }

        # Initialize agents AFTER animation system
        self._create_initial_agents()

        print("✅ Simulation initialized with animated predator-prey ecosystem!")

    def _create_initial_agents(self):
        """Create initial population of agents with animations"""
        print("🌱 Creating initial ecosystem...")

        # Get animation manager
        anim_mgr = get_animation_manager()

        # Create rabbits with animations
        for i in range(config.INITIAL_RABBITS):
            pos = [
                random.randint(50, config.SCREEN_WIDTH - 50),
                random.randint(50, config.SCREEN_HEIGHT - 50)
            ]
            rabbit = Rabbit(pos)

            # Assign animated sprite
            rabbit_anim = anim_mgr.get_sprite('rabbit')
            if rabbit_anim:
                rabbit.set_animated_sprite(rabbit_anim)

            self.agents.append(rabbit)
            self.stats['total_born'] += 1

        # Create wolves with animations
        for i in range(config.INITIAL_WOLVES):
            pos = [
                random.randint(100, config.SCREEN_WIDTH - 100),
                random.randint(100, config.SCREEN_HEIGHT - 100)
            ]
            wolf = Wolf(pos)

            # Assign animated sprite
            wolf_anim = anim_mgr.get_sprite('wolf')
            if wolf_anim:
                wolf.set_animated_sprite(wolf_anim)

            self.agents.append(wolf)
            self.stats['total_born'] += 1

        rabbits = len([a for a in self.agents if a.agent_type == 'rabbit'])
        wolves = len([a for a in self.agents if a.agent_type == 'wolf'])
        print(f"🐰 Created {rabbits} animated rabbits")
        print(f"🐺 Created {wolves} animated wolves")

    def handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self._add_random_rabbit()

                elif event.key == pygame.K_w:
                    self._add_random_wolf()

                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    status = "paused" if self.paused else "resumed"
                    print(f"⏸️ Simulation {status}")

                elif event.key == pygame.K_r:
                    print("🔄 Resetting simulation...")
                    self._reset_simulation()

                elif event.key == pygame.K_d:
                    config.DEBUG_MODE = not getattr(config, 'DEBUG_MODE', False)
                    status = "enabled" if config.DEBUG_MODE else "disabled"
                    print(f"🐛 Debug mode {status}")

                elif event.key == pygame.K_f:
                    mouse_pos = pygame.mouse.get_pos()
                    from environment.food_system import FoodSource
                    food = FoodSource(mouse_pos, "grass")
                    self.food_manager.food_sources.append(food)
                    print(f"🌱 Spawned grass at mouse position")

    def _add_random_rabbit(self):
        """Add a new rabbit with animation"""
        pos = [
            random.randint(50, config.SCREEN_WIDTH - 50),
            random.randint(50, config.SCREEN_HEIGHT - 50)
        ]
        rabbit = Rabbit(pos)

        # Assign animated sprite
        anim_mgr = get_animation_manager()
        rabbit_anim = anim_mgr.get_sprite('rabbit')
        if rabbit_anim:
            rabbit.set_animated_sprite(rabbit_anim)

        self.agents.append(rabbit)
        self.stats['total_born'] += 1
        print(f"🐰 Added new rabbit! Total: {self.get_agent_count_by_type('rabbit')}")

    def _add_random_wolf(self):
        """Add a new wolf with animation"""
        pos = [
            random.randint(100, config.SCREEN_WIDTH - 100),
            random.randint(100, config.SCREEN_HEIGHT - 100)
        ]
        wolf = Wolf(pos)

        # Assign animated sprite
        anim_mgr = get_animation_manager()
        wolf_anim = anim_mgr.get_sprite('wolf')
        if wolf_anim:
            wolf.set_animated_sprite(wolf_anim)

        self.agents.append(wolf)
        self.stats['total_born'] += 1
        print(f"🐺 Added new wolf! Total: {self.get_agent_count_by_type('wolf')}")

    def _reset_simulation(self):
        """Reset the simulation to initial state"""
        self.agents.clear()
        self.dead_agents.clear()
        self.food_manager = FoodManager()
        self.food_manager.initialize_food_sources(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.stats = {
            'total_born': 0,
            'total_died': 0,
            'generation': 1,
            'rabbits_eaten': 0,
            'wolves_starved': 0
        }
        self._create_initial_agents()

    def update(self):
        """Update simulation state"""
        if self.paused:
            return

        # Update animation system - IMPORTANT
        self.animation_manager.update_all()

        # Update food system
        self.food_manager.update(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        # Update all agents
        for agent in self.agents[:]:
            agent.update(self)

            if not agent.is_alive:
                self.agents.remove(agent)
                self.dead_agents.append(agent)
                self.stats['total_died'] += 1

                if agent.agent_type == 'rabbit':
                    self.stats['rabbits_eaten'] += 1
                elif agent.agent_type == 'wolf':
                    self.stats['wolves_starved'] += 1

        # Population management
        rabbit_count = self.get_agent_count_by_type('rabbit')
        wolf_count = self.get_agent_count_by_type('wolf')

        if rabbit_count == 0:
            print("💀 Rabbits extinct! Adding new population...")
            for _ in range(5):
                self._add_random_rabbit()

        if wolf_count == 0 and rabbit_count > 15:
            print("🐺 Reintroducing wolves...")
            self._add_random_wolf()

    def render(self):
        """Render the simulation"""
        self.screen.fill(config.GREEN)

        # Render food sources
        self.food_manager.draw_all(self.screen)

        # Render agents (herbivores first, then predators)
        herbivores = [a for a in self.agents if a.agent_type in ['rabbit', 'deer']]
        predators = [a for a in self.agents if a.agent_type in ['wolf', 'bear']]

        for agent in herbivores + predators:
            agent.draw(self.screen)

        # Render UI
        self._render_ui()

        pygame.display.flip()

    def _render_ui(self):
        """Render user interface"""
        font_large = pygame.font.Font(None, 36)
        font_medium = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 20)

        title = font_large.render("SurvAIval - AI Ecosystem", True, config.WHITE)
        self.screen.blit(title, (10, 10))

        controls = [
            "SPACE: Add rabbit",
            "W: Add wolf",
            "F: Spawn food",
            "P: Pause/Resume",
            "D: Toggle debug",
            "R: Reset",
            "ESC: Exit"
        ]

        y_offset = 50
        for control in controls:
            text = font_small.render(control, True, config.WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 18

        # Statistics
        rabbit_count = self.get_agent_count_by_type('rabbit')
        wolf_count = self.get_agent_count_by_type('wolf')
        food_stats = self.food_manager.get_statistics()

        if wolf_count > 0:
            ratio = rabbit_count / wolf_count
        else:
            ratio = rabbit_count

        stats = [
            f"🐰 Rabbits: {rabbit_count}",
            f"🐺 Wolves: {wolf_count}",
            f"Ratio: {ratio:.1f}:1",
            "",
            f"🌱 Food: {food_stats['available_food']}/{food_stats['total_food_sources']}",
            f"Grass: {food_stats.get('grass_count', 0)}",
            f"Berries: {food_stats.get('berry_count', 0)}",
            "",
            f"Total Born: {self.stats['total_born']}",
            f"Total Died: {self.stats['total_died']}",
            f"Rabbits Eaten: {self.stats['rabbits_eaten']}",
            "",
            f"FPS: {int(self.clock.get_fps())}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]

        x_pos = config.SCREEN_WIDTH - 220
        y_offset = 50

        stats_bg = pygame.Surface((210, len(stats) * 20 + 10))
        stats_bg.set_alpha(128)
        stats_bg.fill(config.BLACK)
        self.screen.blit(stats_bg, (x_pos - 10, y_offset - 5))

        for stat in stats:
            if stat:
                text = font_medium.render(stat, True, config.WHITE)
                self.screen.blit(text, (x_pos, y_offset))
            y_offset += 20

        if getattr(config, 'DEBUG_MODE', False):
            self._render_debug_info()

    def _render_debug_info(self):
        """Render debug information"""
        font_small = pygame.font.Font(None, 16)

        state_counts = {}
        for agent in self.agents:
            state = getattr(agent, 'current_state', 'unknown')
            agent_type = agent.agent_type
            key = f"{agent_type}: {state}"
            state_counts[key] = state_counts.get(key, 0) + 1

        y_offset = config.SCREEN_HEIGHT - (len(state_counts) * 18 + 30)

        debug_bg = pygame.Surface((200, len(state_counts) * 18 + 25))
        debug_bg.set_alpha(128)
        debug_bg.fill(config.BLACK)
        self.screen.blit(debug_bg, (config.SCREEN_WIDTH - 210, y_offset - 10))

        title = font_small.render("Agent States:", True, config.WHITE)
        self.screen.blit(title, (config.SCREEN_WIDTH - 200, y_offset))
        y_offset += 20

        for state, count in sorted(state_counts.items()):
            text = font_small.render(f"{state}: {count}", True, config.WHITE)
            self.screen.blit(text, (config.SCREEN_WIDTH - 190, y_offset))
            y_offset += 18

    def run(self):
        """Main simulation loop"""
        print("🚀 Starting animated predator-prey simulation...")
        print("📝 Controls: SPACE=Rabbit, W=Wolf, F=Food, P=Pause, D=Debug, R=Reset")

        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(config.FPS)

        print("✅ Simulation ended")

    # Helper methods
    def get_entities_in_range(self, position, range_distance):
        """Get all entities within range"""
        nearby = []

        for agent in self.agents:
            if not agent.is_alive:
                continue

            distance = ((position[0] - agent.position[0]) ** 2 +
                        (position[1] - agent.position[1]) ** 2) ** 0.5

            if distance <= range_distance:
                nearby.append(agent)

        return nearby

    def add_entity(self, agent):
        """Add a new agent to the simulation"""
        # Assign animation if available
        anim_mgr = get_animation_manager()
        sprite = anim_mgr.get_sprite(agent.agent_type)
        if sprite:
            agent.set_animated_sprite(sprite)

        self.agents.append(agent)
        self.stats['total_born'] += 1
        print(f"🌱 New {agent.agent_type} added!")

    def get_agent_count_by_type(self, agent_type):
        """Get count of agents by type"""
        return len([a for a in self.agents if a.agent_type == agent_type and a.is_alive])