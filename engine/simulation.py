# engine/simulation.py - Simulation with food system integration
"""
SurvAIval Simulation Controller
Now includes food system for herbivore feeding
"""

import pygame
import random
import sys
import os
import config
from agents.rabbit import Rabbit
from environment.food_system import FoodManager


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

        # Statistics
        self.stats = {
            'total_born': 0,
            'total_died': 0,
            'generation': 1
        }

        # Load assets
        self._load_assets()

        # Initialize agents
        self._create_initial_agents()

        print("✅ Simulation initialized with AI agents and food system!")

    def _load_assets(self):
        """Load game assets (sprites, sounds, etc.)"""
        self.assets = {}

        # Try to load animal sprites
        asset_paths = {
            'rabbit': 'assets/animals/rabbit.png',
            'deer': 'assets/animals/deer.png',
            'wolf': 'assets/animals/wolf.png',
            'bear': 'assets/animals/bear.png'
        }

        for animal, path in asset_paths.items():
            try:
                if os.path.exists(path):
                    image = pygame.image.load(path)
                    size = 24 if animal == 'rabbit' else 32
                    self.assets[animal] = pygame.transform.scale(image, (size, size))
                    print(f"📁 Loaded {animal} sprite")
                else:
                    self.assets[animal] = None
                    print(f"⚠️ Asset not found: {path}")
            except Exception as e:
                print(f"❌ Failed to load {animal}: {e}")
                self.assets[animal] = None

    def _create_initial_agents(self):
        """Create initial population of agents"""
        print("🌱 Creating initial population...")

        # Create rabbits
        for i in range(config.INITIAL_RABBITS):
            position = [
                random.randint(50, config.SCREEN_WIDTH - 50),
                random.randint(50, config.SCREEN_HEIGHT - 50)
            ]
            rabbit = Rabbit(position)
            self.agents.append(rabbit)
            self.stats['total_born'] += 1

        print(f"🐰 Created {len(self.agents)} rabbits")

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
                    # Manually spawn food
                    mouse_pos = pygame.mouse.get_pos()
                    from environment.food_system import FoodSource
                    food = FoodSource(mouse_pos, "grass")
                    self.food_manager.food_sources.append(food)
                    print(f"🌱 Spawned grass at mouse position")

    def _add_random_rabbit(self):
        """Add a new rabbit at random position"""
        position = [
            random.randint(50, config.SCREEN_WIDTH - 50),
            random.randint(50, config.SCREEN_HEIGHT - 50)
        ]
        rabbit = Rabbit(position)
        self.agents.append(rabbit)
        self.stats['total_born'] += 1
        print(f"🐰 Added new rabbit! Total: {len(self.agents)}")

    def _reset_simulation(self):
        """Reset the simulation to initial state"""
        self.agents.clear()
        self.dead_agents.clear()
        self.food_manager = FoodManager()
        self.food_manager.initialize_food_sources(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.stats = {
            'total_born': 0,
            'total_died': 0,
            'generation': 1
        }
        self._create_initial_agents()

    def update(self):
        """Update simulation state"""
        if self.paused:
            return

        # Update food system
        self.food_manager.update(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        # Update all agents
        for agent in self.agents[:]:
            agent.update(self)

            # Remove dead agents
            if not agent.is_alive:
                self.agents.remove(agent)
                self.dead_agents.append(agent)
                self.stats['total_died'] += 1

        # Handle population extinction
        if len(self.agents) == 0:
            print("💀 Population extinct! Restarting...")
            self._create_initial_agents()

    def render(self):
        """Render the simulation"""
        # Clear screen with grass color
        self.screen.fill(config.GREEN)

        # Render food sources (below agents)
        self.food_manager.draw_all(self.screen)

        # Render agents
        self._render_agents()

        # Render UI
        self._render_ui()

        # Update display
        pygame.display.flip()

    def _render_agents(self):
        """Render all agents in the simulation"""
        for agent in self.agents:
            if self.assets.get(agent.agent_type):
                sprite = self.assets[agent.agent_type]
                sprite_rect = sprite.get_rect(center=(int(agent.position[0]), int(agent.position[1])))
                self.screen.blit(sprite, sprite_rect)
            else:
                agent.draw(self.screen)

    def _render_ui(self):
        """Render user interface elements"""
        font_large = pygame.font.Font(None, 36)
        font_medium = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 20)

        # Title
        title = font_large.render("SurvAIval - AI Ecosystem", True, config.WHITE)
        self.screen.blit(title, (10, 10))

        # Controls
        controls = [
            "SPACE: Add rabbit",
            "F: Spawn food (at mouse)",
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

        # Get statistics
        alive_rabbits = len([a for a in self.agents if a.agent_type == 'rabbit'])
        food_stats = self.food_manager.get_statistics()

        stats = [
            f"Population: {len(self.agents)}",
            f"Rabbits: {alive_rabbits}",
            f"",
            f"Food Sources: {food_stats['available_food']}/{food_stats['total_food_sources']}",
            f"Grass: {food_stats.get('grass_count', 0)}",
            f"Berries: {food_stats.get('berry_count', 0)}",
            f"",
            f"Total Born: {self.stats['total_born']}",
            f"Total Died: {self.stats['total_died']}",
            f"",
            f"FPS: {int(self.clock.get_fps())}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]

        # Display on right side
        x_pos = config.SCREEN_WIDTH - 220
        y_offset = 50

        # Background for stats
        stats_bg = pygame.Surface((210, len(stats) * 20 + 10))
        stats_bg.set_alpha(128)
        stats_bg.fill(config.BLACK)
        self.screen.blit(stats_bg, (x_pos - 10, y_offset - 5))

        for stat in stats:
            if stat:  # Skip empty lines in rendering
                text = font_medium.render(stat, True, config.WHITE)
                self.screen.blit(text, (x_pos, y_offset))
            y_offset += 20

        # Agent state information (if debug mode)
        if getattr(config, 'DEBUG_MODE', False):
            self._render_debug_info()

    def _render_debug_info(self):
        """Render debug information about agents"""
        font_small = pygame.font.Font(None, 16)

        # Count agents by state
        state_counts = {}
        for agent in self.agents:
            state = getattr(agent, 'current_state', 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1

        # Display state counts
        y_offset = config.SCREEN_HEIGHT - 150

        # Background
        debug_bg = pygame.Surface((180, len(state_counts) * 18 + 20))
        debug_bg.set_alpha(128)
        debug_bg.fill(config.BLACK)
        self.screen.blit(debug_bg, (config.SCREEN_WIDTH - 190, y_offset - 10))

        # Title
        title = font_small.render("Agent States:", True, config.WHITE)
        self.screen.blit(title, (config.SCREEN_WIDTH - 180, y_offset))
        y_offset += 20

        for state, count in state_counts.items():
            text = font_small.render(f"{state}: {count}", True, config.WHITE)
            self.screen.blit(text, (config.SCREEN_WIDTH - 170, y_offset))
            y_offset += 18

    def run(self):
        """Main simulation loop"""
        print("🚀 Starting simulation with food system...")
        print("📝 Controls: SPACE=Rabbit, F=Food, P=Pause, D=Debug, R=Reset, ESC=Exit")

        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(config.FPS)

        print("✅ Simulation loop ended")

    # Helper methods for agents
    def get_entities_in_range(self, position, range_distance):
        """Get all entities within range of a position"""
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
        self.agents.append(agent)
        self.stats['total_born'] += 1
        print(f"🌱 New {agent.agent_type} added to simulation!")

    def get_agent_count_by_type(self, agent_type):
        """Get count of agents by type"""
        return len([a for a in self.agents if a.agent_type == agent_type and a.is_alive])