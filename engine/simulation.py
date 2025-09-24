# engine/simulation.py - Main simulation class
"""
SurvAIval Simulation Controller
Manages the main simulation loop, events, and rendering
"""

import pygame
import random
import sys
import os
import config


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

        # Load assets
        self._load_assets()

        # Initialize test data (temporary)
        self._init_test_data()

        print("✅ Simulation initialized successfully!")

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
                    size = 32 if animal == 'rabbit' else 40
                    self.assets[animal] = pygame.transform.scale(image, (size, size))
                    print(f"📁 Loaded {animal} sprite")
                else:
                    self.assets[animal] = None
                    print(f"⚠️ Asset not found: {path}")
            except Exception as e:
                print(f"❌ Failed to load {animal}: {e}")
                self.assets[animal] = None

    def _init_test_data(self):
        """Initialize temporary test data"""
        # Test rabbits for demonstration
        self.test_entities = []

        for i in range(5):
            x = random.randint(50, config.SCREEN_WIDTH - 50)
            y = random.randint(50, config.SCREEN_HEIGHT - 50)
            velocity_x = random.uniform(-1, 1)
            velocity_y = random.uniform(-1, 1)

            entity = {
                'type': 'rabbit',
                'position': [x, y],
                'velocity': [velocity_x, velocity_y],
                'energy': 100,
                'age': 0
            }
            self.test_entities.append(entity)

    def handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    # Add new test rabbit
                    x = random.randint(50, config.SCREEN_WIDTH - 50)
                    y = random.randint(50, config.SCREEN_HEIGHT - 50)
                    entity = {
                        'type': 'rabbit',
                        'position': [x, y],
                        'velocity': [random.uniform(-1, 1), random.uniform(-1, 1)],
                        'energy': 100,
                        'age': 0
                    }
                    self.test_entities.append(entity)
                    print(f"🐰 Added new rabbit! Total: {len(self.test_entities)}")

                elif event.key == pygame.K_p:
                    # Toggle pause
                    self.paused = not self.paused
                    status = "paused" if self.paused else "resumed"
                    print(f"⏸️ Simulation {status}")

                elif event.key == pygame.K_r:
                    # Reset simulation
                    print("🔄 Resetting simulation...")
                    self._init_test_data()

    def update(self):
        """Update simulation state"""
        if self.paused:
            return

        # Update test entities
        for entity in self.test_entities[:]:  # Copy list to avoid modification issues
            # Simple movement
            entity['position'][0] += entity['velocity'][0]
            entity['position'][1] += entity['velocity'][1]

            # Bounce off screen edges
            if entity['position'][0] <= 0 or entity['position'][0] >= config.SCREEN_WIDTH:
                entity['velocity'][0] *= -1
            if entity['position'][1] <= 0 or entity['position'][1] >= config.SCREEN_HEIGHT:
                entity['velocity'][1] *= -1

            # Keep within bounds
            entity['position'][0] = max(16, min(config.SCREEN_WIDTH - 16, entity['position'][0]))
            entity['position'][1] = max(16, min(config.SCREEN_HEIGHT - 16, entity['position'][1]))

            # Add small random movement
            entity['velocity'][0] += random.uniform(-0.1, 0.1)
            entity['velocity'][1] += random.uniform(-0.1, 0.1)

            # Limit speed
            max_speed = 2.0
            speed = (entity['velocity'][0] ** 2 + entity['velocity'][1] ** 2) ** 0.5
            if speed > max_speed:
                entity['velocity'][0] = entity['velocity'][0] / speed * max_speed
                entity['velocity'][1] = entity['velocity'][1] / speed * max_speed

            # Age and energy
            entity['age'] += 1
            entity['energy'] -= 0.01

    def render(self):
        """Render the simulation"""
        # Clear screen with grass color
        self.screen.fill(config.GREEN)

        # Render UI
        self._render_ui()

        # Render entities
        self._render_entities()

        # Update display
        pygame.display.flip()

    def _render_ui(self):
        """Render user interface elements"""
        font_large = pygame.font.Font(None, 36)
        font_medium = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 20)

        # Title
        title = font_large.render("SurvAIval - AI Ecosystem", True, config.WHITE)
        title_rect = title.get_rect()
        title_rect.topleft = (10, 10)
        self.screen.blit(title, title_rect)

        # Controls
        controls = [
            "SPACE: Add rabbit",
            "P: Pause/Resume",
            "R: Reset",
            "ESC: Exit"
        ]

        y_offset = 50
        for control in controls:
            text = font_small.render(control, True, config.WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

        # Statistics
        stats = [
            f"Entities: {len(self.test_entities)}",
            f"FPS: {int(self.clock.get_fps())}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]

        y_offset = 150
        for stat in stats:
            text = font_medium.render(stat, True, config.WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25

    def _render_entities(self):
        """Render all entities in the simulation"""
        for entity in self.test_entities:
            pos = (int(entity['position'][0]), int(entity['position'][1]))
            entity_type = entity['type']

            # Try to use sprite asset first
            if self.assets.get(entity_type):
                sprite = self.assets[entity_type]
                sprite_rect = sprite.get_rect(center=pos)
                self.screen.blit(sprite, sprite_rect)
            else:
                # Fallback to colored circles
                color = getattr(config, f"{entity_type.upper()}_COLOR", config.WHITE)
                pygame.draw.circle(self.screen, color, pos, 16)
                pygame.draw.circle(self.screen, config.BLACK, pos, 16, 2)

            # Energy bar (optional)
            if entity['energy'] < 50:
                bar_width = 30
                bar_height = 4
                bar_x = pos[0] - bar_width // 2
                bar_y = pos[1] - 25

                # Background
                pygame.draw.rect(self.screen, config.BLACK, (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2))
                # Energy level
                energy_width = int(bar_width * (entity['energy'] / 100))
                energy_color = config.GREEN if entity['energy'] > 25 else (255, 0, 0)  # Red when low
                pygame.draw.rect(self.screen, energy_color, (bar_x, bar_y, energy_width, bar_height))

    def run(self):
        """Main simulation loop"""
        print("🚀 Starting simulation loop...")
        print("📝 Controls: SPACE=Add rabbit, P=Pause, R=Reset, ESC=Exit")

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