# engine/simulation.py - Simulation with wolves
"""
SurvAIval Simulation Controller
Now includes predator-prey dynamics with wolves
"""

import pygame
import random
import sys
import os
import config
from agents.rabbit import Rabbit
from agents.wolf import Wolf
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
            'generation': 1,
            'rabbits_eaten': 0,
            'wolves_starved': 0
        }

        # Load assets
        self._load_assets()

        # Initialize agents
        self._create_initial_agents()

        print("✅ Simulation initialized with predator-prey ecosystem!")

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
                    size = 24 if animal == 'rabbit' else 32 if animal == 'wolf' else 36
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
        print("🌱 Creating initial ecosystem...")

        # Create rabbits
        for i in range(config.INITIAL_RABBITS):
            position = [
                random.randint(50, config.SCREEN_WIDTH - 50),
                random.randint(50, config.SCREEN_HEIGHT - 50)
            ]
            rabbit = Rabbit(position)
            self.agents.append(rabbit)
            self.stats['total_born'] += 1

        # Create wolves
        for i in range(config.INITIAL_WOLVES):
            # Spawn wolves away from rabbit clusters
            position = [
                random.randint(100, config.SCREEN_WIDTH - 100),
                random.randint(100, config.SCREEN_HEIGHT - 100)
            ]
            wolf = Wolf(position)
            self.agents.append(wolf)
            self.stats['total_born'] += 1

        rabbits = len([a for a in self.agents if a.agent_type == 'rabbit'])
        wolves = len([a for a in self.agents if a.agent_type == 'wolf'])
        print(f"🐰 Created {rabbits} rabbits")
        print(f"🐺 Created {wolves} wolves")

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
                    # Add wolf with W key
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
                    # Spawn food at mouse position
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
        print(f"🐰 Added new rabbit! Total rabbits: {self.get_agent_count_by_type('rabbit')}")

    def _add_random_wolf(self):
        """Add a new wolf at random position"""
        position = [
            random.randint(100, config.SCREEN_WIDTH - 100),
            random.randint(100, config.SCREEN_HEIGHT - 100)
        ]
        wolf = Wolf(position)
        self.agents.append(wolf)
        self.stats['total_born'] += 1
        print(f"🐺 Added new wolf! Total wolves: {self.get_agent_count_by_type('wolf')}")

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

        # Update food system
        self.food_manager.update(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        # Update all agents
        for agent in self.agents[:]:
            agent.update(self)

            # Track death causes for statistics
            if not agent.is_alive:
                self.agents.remove(agent)
                self.dead_agents.append(agent)
                self.stats['total_died'] += 1

                # Track specific death causes
                if agent.agent_type == 'rabbit' and hasattr(agent, 'current_state'):
                    if agent.current_state == "dead":
                        self.stats['rabbits_eaten'] += 1
                elif agent.agent_type == 'wolf':
                    self.stats['wolves_starved'] += 1

        # Population management
        rabbit_count = self.get_agent_count_by_type('rabbit')
        wolf_count = self.get_agent_count_by_type('wolf')

        # If rabbits extinct, respawn some
        if rabbit_count == 0:
            print("💀 Rabbits extinct! Adding new population...")
            for _ in range(5):
                self._add_random_rabbit()

        # If wolves extinct and rabbits overpopulated, add wolves
        if wolf_count == 0 and rabbit_count > 15:
            print("🐺 Reintroducing wolves to control rabbit population...")
            self._add_random_wolf()

    def render(self):
        """Render the simulation"""
        # Clear screen with grass color
        self.screen.fill(config.GREEN)

        # Render food sources (below agents)
        self.food_manager.draw_all(self.screen)

        # Render agents (herbivores first, then predators on top)
        herbivores = [a for a in self.agents if a.agent_type in ['rabbit', 'deer']]
        predators = [a for a in self.agents if a.agent_type in ['wolf', 'bear']]

        for agent in herbivores + predators:
            if self.assets.get(agent.agent_type):
                sprite = self.assets[agent.agent_type]
                sprite_rect = sprite.get_rect(center=(int(agent.position[0]), int(agent.position[1])))
                self.screen.blit(sprite, sprite_rect)
            else:
                agent.draw(self.screen)

        # Render UI
        self._render_ui()

        # Update display
        pygame.display.flip()

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

        # Get statistics
        rabbit_count = self.get_agent_count_by_type('rabbit')
        wolf_count = self.get_agent_count_by_type('wolf')
        food_stats = self.food_manager.get_statistics()

        # Calculate ratio
        if wolf_count > 0:
            prey_predator_ratio = rabbit_count / wolf_count
        else:
            prey_predator_ratio = rabbit_count

        stats = [
            f"🐰 Rabbits: {rabbit_count}",
            f"🐺 Wolves: {wolf_count}",
            f"Ratio: {prey_predator_ratio:.1f}:1",
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

        # Display on right side
        x_pos = config.SCREEN_WIDTH - 220
        y_offset = 50

        # Background for stats
        stats_bg = pygame.Surface((210, len(stats) * 20 + 10))
        stats_bg.set_alpha(128)
        stats_bg.fill(config.BLACK)
        self.screen.blit(stats_bg, (x_pos - 10, y_offset - 5))

        for stat in stats:
            if stat:
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
            agent_type = agent.agent_type
            key = f"{agent_type}: {state}"
            state_counts[key] = state_counts.get(key, 0) + 1

        # Display state counts
        y_offset = config.SCREEN_HEIGHT - (len(state_counts) * 18 + 30)

        # Background
        debug_bg = pygame.Surface((200, len(state_counts) * 18 + 25))
        debug_bg.set_alpha(128)
        debug_bg.fill(config.BLACK)
        self.screen.blit(debug_bg, (config.SCREEN_WIDTH - 210, y_offset - 10))

        # Title
        title = font_small.render("Agent States:", True, config.WHITE)
        self.screen.blit(title, (config.SCREEN_WIDTH - 200, y_offset))
        y_offset += 20

        for state, count in sorted(state_counts.items()):
            text = font_small.render(f"{state}: {count}", True, config.WHITE)
            self.screen.blit(text, (config.SCREEN_WIDTH - 190, y_offset))
            y_offset += 18

    def run(self):
        """Main simulation loop"""
        print("🚀 Starting predator-prey simulation...")
        print("📝 Controls: SPACE=Rabbit, W=Wolf, F=Food, P=Pause, D=Debug, R=Reset")

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
        print(f"🌱 New {agent.agent_type} added!")

    def get_agent_count_by_type(self, agent_type):
        """Get count of agents by type"""
        return len([a for a in self.agents if a.agent_type == agent_type and a.is_alive])A