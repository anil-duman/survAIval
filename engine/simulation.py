# engine/simulation.py - Updated simulation class with real agents
"""
SurvAIval Simulation Controller
Manages the main simulation loop, events, and rendering with real AI agents
"""

import pygame
import random
import sys
import os
import config
from agents.rabbit import Rabbit


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

        print("✅ Simulation initialized with real AI agents!")

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
                    # Scale to appropriate size
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
                    # Add new rabbit
                    self._add_random_rabbit()

                elif event.key == pygame.K_p:
                    # Toggle pause
                    self.paused = not self.paused
                    status = "paused" if self.paused else "resumed"
                    print(f"⏸️ Simulation {status}")

                elif event.key == pygame.K_r:
                    # Reset simulation
                    print("🔄 Resetting simulation...")
                    self._reset_simulation()

                elif event.key == pygame.K_d:
                    # Toggle debug mode
                    config.DEBUG_MODE = not getattr(config, 'DEBUG_MODE', False)
                    status = "enabled" if config.DEBUG_MODE else "disabled"
                    print(f"🐛 Debug mode {status}")

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

        # Update all agents
        for agent in self.agents[:]:  # Copy list to avoid modification issues
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

        # Render agents
        self._render_agents()

        # Render UI
        self._render_ui()

        # Update display
        pygame.display.flip()

    def _render_agents(self):
        """Render all agents in the simulation"""
        for agent in self.agents:
            # Try to use sprite asset first
            if self.assets.get(agent.agent_type):
                sprite = self.assets[agent.agent_type]
                sprite_rect = sprite.get_rect(center=(int(agent.position[0]), int(agent.position[1])))
                self.screen.blit(sprite, sprite_rect)
            else:
                # Use agent's own draw method (fallback to circles)
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
        alive_rabbits = len([a for a in self.agents if a.agent_type == 'rabbit'])

        stats = [
            f"Population: {len(self.agents)}",
            f"Rabbits: {alive_rabbits}",
            f"Total Born: {self.stats['total_born']}",
            f"Total Died: {self.stats['total_died']}",
            f"FPS: {int(self.clock.get_fps())}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]

        # Display on right side of screen
        x_pos = config.SCREEN_WIDTH - 200
        y_offset = 50

        # Background for stats
        stats_bg = pygame.Surface((180, len(stats) * 22 + 10))
        stats_bg.set_alpha(128)
        stats_bg.fill(config.BLACK)
        self.screen.blit(stats_bg, (x_pos - 10, y_offset - 5))

        for stat in stats:
            text = font_medium.render(stat, True, config.WHITE)
            self.screen.blit(text, (x_pos, y_offset))
            y_offset += 22

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
        y_offset = 200
        for state, count in state_counts.items():
            text = font_small.render(f"{state}: {count}", True, config.WHITE)
            self.screen.blit(text, (config.SCREEN_WIDTH - 150, y_offset))
            y_offset += 18

    def run(self):
        """Main simulation loop"""
        print("🚀 Starting simulation loop with AI agents...")
        print("📝 Controls: SPACE=Add rabbit, P=Pause, D=Debug, R=Reset, ESC=Exit")

        while self.running:
            # Handle events
            self.handle_events()

            # Update simulation
            self.update()

            # Render
            self.render()

            # Control frame rate
            self.clock.tick(config.FPS)

        print("✅ Simulation loop ended")

    # Helper methods for agents
    def get_entities_in_range(self, position, range_distance):
        """Get all entities within range of a position

        Args:
            position: Center position [x, y]
            range_distance: Search radius

        Returns:
            List of agents within range
        """
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
        """Add a new agent to the simulation (used for reproduction)"""
        self.agents.append(agent)
        self.stats['total_born'] += 1
        print(f"🌱 New {agent.agent_type} added to simulation!")

    def get_agent_count_by_type(self, agent_type):
        """Get count of agents by type"""
        return len([a for a in self.agents if a.agent_type == agent_type and a.is_alive])