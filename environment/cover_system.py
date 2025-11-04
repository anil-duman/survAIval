# environment/cover_system.py - Trees and bushes for hiding
"""
SurvAIval Cover System
Manages trees, bushes, and hiding mechanics
"""

import pygame
import numpy as np
import random
import config


class CoverObject:
    """A tree, bush, or rock that provides cover"""

    def __init__(self, position, cover_type="bush", sprite=None):
        """Initialize cover object

        Args:
            position: [x, y] position
            cover_type: "bush", "tree", "rock"
            sprite: pygame.Surface sprite image (optional)
        """
        self.position = np.array(position, dtype=float)
        self.cover_type = cover_type
        self.size = self._get_size()
        self.hide_radius = self._get_hide_radius()
        self.vision_block = self._get_vision_block()
        self.id = f"{cover_type}_{random.randint(1000, 9999)}"

        # 🎨 Sprite support
        self.sprite = sprite
        self.sprite_scale = self._get_sprite_scale()

    def _get_size(self):
        """Get display size based on type"""
        sizes = {
            'bush': 25,
            'tree': 40,
            'rock': 30,
            'tall_grass': 20
        }
        return sizes.get(self.cover_type, 25)

    def _get_sprite_scale(self):
        """Get sprite scale multiplier"""
        scales = {
            'bush': 2.0,
            'tree': 2.5,
            'rock': 2.0,
            'tall_grass': 1.5
        }
        return scales.get(self.cover_type, 2.0)

    def _get_hide_radius(self):
        """Get radius where animals can hide"""
        radii = {
            'bush': 35,
            'tree': 50,
            'rock': 40,
            'tall_grass': 30
        }
        return radii.get(self.cover_type, 35)

    def _get_vision_block(self):
        """Does this object block line of sight?"""
        blockers = {
            'bush': True,
            'tree': True,
            'rock': True,
            'tall_grass': False  # Can see through grass
        }
        return blockers.get(self.cover_type, True)

    def is_position_hidden(self, pos):
        """Check if a position is hidden by this cover

        Args:
            pos: [x, y] position to check

        Returns:
            bool: True if position is hidden
        """
        distance = np.linalg.norm(np.array(pos) - self.position)
        return distance <= self.hide_radius

    def blocks_line_of_sight(self, pos1, pos2):
        """Check if this object blocks line of sight between two positions

        Args:
            pos1, pos2: [x, y] positions

        Returns:
            bool: True if line of sight is blocked
        """
        if not self.vision_block:
            return False

        # Check if line segment intersects with cover circle
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        center = self.position

        # Vector from p1 to p2
        d = p2 - p1
        # Vector from p1 to center
        f = p1 - center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - (self.size ** 2)

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return False

        # Ray intersects sphere, check if within segment
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if intersection is within line segment [0, 1]
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def get_color(self):
        """Get display color"""
        colors = {
            'bush': (34, 139, 34),  # Forest green
            'tree': (46, 125, 50),  # Darker green
            'rock': (120, 120, 120),  # Gray
            'tall_grass': (107, 142, 35)  # Yellow-green
        }
        return colors.get(self.cover_type, (34, 139, 34))

    def draw(self, screen, camera_offset=(0, 0)):
        """Draw the cover object"""
        pos = (
            int(self.position[0] - camera_offset[0]),
            int(self.position[1] - camera_offset[1])
        )

        # 🎨 Draw sprite if available
        if self.sprite:
            # Scale sprite
            scaled_width = int(self.sprite.get_width() * self.sprite_scale)
            scaled_height = int(self.sprite.get_height() * self.sprite_scale)
            scaled_sprite = pygame.transform.scale(self.sprite, (scaled_width, scaled_height))

            # Draw centered on position
            sprite_rect = scaled_sprite.get_rect(center=pos)
            screen.blit(scaled_sprite, sprite_rect)

            # Debug: show hide radius
            if getattr(config, 'DEBUG_MODE', False):
                pygame.draw.circle(screen, (255, 255, 0), pos, int(self.hide_radius), 1)

            return

        # 🎨 Fallback: draw simple shapes if no sprite
        color = self.get_color()

        if self.cover_type == 'tree':
            # Draw tree (trunk + foliage)
            trunk_color = (101, 67, 33)  # Brown
            trunk_width = 8
            trunk_height = 25

            # Trunk
            trunk_rect = pygame.Rect(
                pos[0] - trunk_width // 2,
                pos[1] - trunk_height // 2,
                trunk_width,
                trunk_height
            )
            pygame.draw.rect(screen, trunk_color, trunk_rect)

            # Foliage (3 circles for bushy look)
            pygame.draw.circle(screen, color, (pos[0] - 10, pos[1] - 20), 18)
            pygame.draw.circle(screen, color, (pos[0] + 10, pos[1] - 20), 18)
            pygame.draw.circle(screen, color, pos, 22)

        elif self.cover_type == 'bush':
            # Draw bush (cluster of circles)
            pygame.draw.circle(screen, color, pos, self.size)
            pygame.draw.circle(screen, color, (pos[0] - 12, pos[1] - 8), 15)
            pygame.draw.circle(screen, color, (pos[0] + 12, pos[1] - 8), 15)

            # Outline
            pygame.draw.circle(screen, (25, 100, 25), pos, self.size, 2)

        elif self.cover_type == 'rock':
            # Draw rock (irregular polygon)
            rock_points = [
                (pos[0], pos[1] - self.size),
                (pos[0] + self.size - 5, pos[1] - 10),
                (pos[0] + self.size, pos[1] + 10),
                (pos[0] + 5, pos[1] + self.size),
                (pos[0] - self.size + 5, pos[1] + 10),
                (pos[0] - self.size, pos[1] - 5),
            ]
            pygame.draw.polygon(screen, color, rock_points)
            pygame.draw.polygon(screen, (80, 80, 80), rock_points, 2)

        elif self.cover_type == 'tall_grass':
            # Draw tall grass (vertical lines)
            for i in range(6):
                offset_x = random.randint(-8, 8)
                offset_y = random.randint(-5, 5)
                start = (pos[0] + offset_x, pos[1] + offset_y)
                end = (pos[0] + offset_x, pos[1] + offset_y - 15)
                pygame.draw.line(screen, color, start, end, 2)

        # Debug: show hide radius
        if getattr(config, 'DEBUG_MODE', False):
            pygame.draw.circle(screen, (255, 255, 0), pos, int(self.hide_radius), 1)


class CoverManager:
    """Manages all cover objects in the world"""

    def __init__(self):
        """Initialize cover manager"""
        self.cover_objects = []
        self.max_cover = 60

        # 🎨 Load sprites
        self.sprites = {}
        self._load_cover_sprites()

    def _load_cover_sprites(self):
        """Load tree and bush sprites"""
        sprite_paths = {
            'tree': 'assets/environment/tree.png',
            'bush': 'assets/environment/bush.png',
            'rock': 'assets/environment/rock.png',
            'tall_grass': 'assets/environment/grass.png'
        }

        for cover_type, path in sprite_paths.items():
            try:
                sprite = pygame.image.load(path).convert_alpha()
                self.sprites[cover_type] = sprite
                print(f"✅ Loaded {cover_type} sprite: {path}")
            except Exception as e:
                print(f"⚠️ Could not load {cover_type} sprite: {e}")
                self.sprites[cover_type] = None

    def initialize_cover(self, world_width, world_height):
        """Generate initial cover distribution"""
        print("🌳 Generating trees and bushes...")

        # Create trees (sparse)
        for _ in range(12):
            pos = [
                random.randint(100, world_width - 100),
                random.randint(100, world_height - 100)
            ]
            tree = CoverObject(pos, "tree", sprite=self.sprites.get('tree'))
            self.cover_objects.append(tree)

        # Create bushes (common)
        for _ in range(30):
            pos = [
                random.randint(50, world_width - 50),
                random.randint(50, world_height - 50)
            ]
            bush = CoverObject(pos, "bush", sprite=self.sprites.get('bush'))
            self.cover_objects.append(bush)

        # Create rocks (occasional)
        for _ in range(10):
            pos = [
                random.randint(80, world_width - 80),
                random.randint(80, world_height - 80)
            ]
            rock = CoverObject(pos, "rock", sprite=self.sprites.get('rock'))
            self.cover_objects.append(rock)

        # Create tall grass patches (many)
        for _ in range(20):
            pos = [
                random.randint(60, world_width - 60),
                random.randint(60, world_height - 60)
            ]
            grass = CoverObject(pos, "tall_grass", sprite=self.sprites.get('tall_grass'))
            self.cover_objects.append(grass)

        print(f"🌿 Created {len(self.cover_objects)} cover objects")

    def get_nearest_cover(self, position, max_distance=150):
        """Find nearest cover object

        Args:
            position: [x, y] position to search from
            max_distance: Maximum search radius

        Returns:
            CoverObject or None
        """
        nearest = None
        min_dist = max_distance

        for cover in self.cover_objects:
            dist = np.linalg.norm(np.array(position) - cover.position)
            if dist < min_dist:
                min_dist = dist
                nearest = cover

        return nearest

    def is_position_hidden(self, position):
        """Check if a position is hidden by any cover

        Args:
            position: [x, y] position

        Returns:
            tuple: (is_hidden: bool, cover_object: CoverObject or None)
        """
        for cover in self.cover_objects:
            if cover.is_position_hidden(position):
                return True, cover

        return False, None

    def check_line_of_sight(self, pos1, pos2):
        """Check if line of sight is clear between two positions

        Args:
            pos1, pos2: [x, y] positions

        Returns:
            bool: True if line of sight is clear
        """
        for cover in self.cover_objects:
            if cover.blocks_line_of_sight(pos1, pos2):
                return False

        return True

    def draw_all(self, screen, camera_offset=(0, 0)):
        """Draw all cover objects"""
        # Sort by y-position for proper layering
        sorted_cover = sorted(self.cover_objects, key=lambda c: c.position[1])

        for cover in sorted_cover:
            cover.draw(screen, camera_offset)

    def get_statistics(self):
        """Get cover statistics"""
        stats = {
            'total_cover': len(self.cover_objects),
            'trees': len([c for c in self.cover_objects if c.cover_type == 'tree']),
            'bushes': len([c for c in self.cover_objects if c.cover_type == 'bush']),
            'rocks': len([c for c in self.cover_objects if c.cover_type == 'rock']),
            'tall_grass': len([c for c in self.cover_objects if c.cover_type == 'tall_grass'])
        }
        return stats