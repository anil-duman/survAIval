# environment/food_system.py - Food and grass management
"""
SurvAIval Food System
Manages grass, plants and other food sources in the ecosystem
"""

import pygame
import numpy as np
import random
import config


class FoodSource:
    """Individual food source (grass patch, plant, etc.)"""

    def __init__(self, position, food_type="grass"):
        """Initialize food source

        Args:
            position: [x, y] position
            food_type: Type of food (grass, berry, etc.)
        """
        self.position = np.array(position, dtype=float)
        self.food_type = food_type
        self.nutrition_value = self._get_nutrition_value()
        self.max_nutrition = self.nutrition_value
        self.size = self._get_size()
        self.growth_rate = self._get_growth_rate()
        self.is_depleted = False
        self.respawn_timer = 0
        self.id = f"{food_type}_{random.randint(1000, 9999)}"

    def _get_nutrition_value(self):
        """Get nutrition value based on food type"""
        nutrition_values = {
            'grass': 15,
            'berry': 25,
            'mushroom': 10,
            'leaves': 20
        }
        return nutrition_values.get(self.food_type, 15)

    def _get_size(self):
        """Get display size based on food type"""
        sizes = {
            'grass': 8,
            'berry': 6,
            'mushroom': 10,
            'leaves': 12
        }
        return sizes.get(self.food_type, 8)

    def _get_growth_rate(self):
        """Get regrowth rate (frames to respawn)"""
        growth_rates = {
            'grass': 300,  # 5 seconds at 60 FPS
            'berry': 600,  # 10 seconds
            'mushroom': 450,  # 7.5 seconds
            'leaves': 360  # 6 seconds
        }
        return growth_rates.get(self.food_type, 300)

    def consume(self, amount):
        """Consume some nutrition from this food source

        Args:
            amount: Amount of nutrition to consume

        Returns:
            Actual amount consumed
        """
        if self.is_depleted:
            return 0

        consumed = min(amount, self.nutrition_value)
        self.nutrition_value -= consumed

        if self.nutrition_value <= 0:
            self.is_depleted = True
            self.respawn_timer = self.growth_rate
            print(f"🌱 {self.food_type} depleted at {self.position}")

        return consumed

    def update(self):
        """Update food source (regrowth, etc.)"""
        if self.is_depleted and self.respawn_timer > 0:
            self.respawn_timer -= 1

            if self.respawn_timer <= 0:
                # Respawn
                self.nutrition_value = self.max_nutrition
                self.is_depleted = False
                print(f"🌿 {self.food_type} regrew at {self.position}")

    def get_color(self):
        """Get display color based on type and state"""
        base_colors = {
            'grass': (50, 200, 50),
            'berry': (200, 50, 200),
            'mushroom': (150, 100, 50),
            'leaves': (100, 180, 100)
        }

        base_color = base_colors.get(self.food_type, (50, 200, 50))

        if self.is_depleted:
            # Faded color when depleted
            return tuple(c // 3 for c in base_color)
        else:
            # Color intensity based on nutrition level
            intensity = self.nutrition_value / self.max_nutrition
            return tuple(int(c * (0.3 + 0.7 * intensity)) for c in base_color)

    def draw(self, screen, camera_offset=(0, 0)):  # ← Parametre ekle
        """Draw the food source"""
        if self.is_depleted and self.respawn_timer > self.growth_rate * 0.7:
            return

        # Camera offset
        pos = (int(self.position[0] - camera_offset[0]),
               int(self.position[1] - camera_offset[1]))

        color = self.get_color()

        if self.food_type == 'grass':
            for i in range(3):
                offset_x = random.randint(-3, 3)
                offset_y = random.randint(-2, 2)
                grass_pos = (pos[0] + offset_x, pos[1] + offset_y)
                pygame.draw.rect(screen, color, (grass_pos[0] - 1, grass_pos[1] - 3, 2, 6))

        elif self.food_type == 'berry':
            pygame.draw.circle(screen, color, pos, self.size)
            pygame.draw.circle(screen, (0, 0, 0), pos, self.size, 1)

        else:
            pygame.draw.circle(screen, color, pos, self.size)
            pygame.draw.circle(screen, (0, 0, 0), pos, self.size, 1)

class FoodManager:
    """Manages all food sources in the ecosystem"""

    def __init__(self):
        """Initialize food manager"""
        self.food_sources = []
        self.spawn_timer = 0
        self.max_food_sources = 150
        self.spawn_interval = 120  # Spawn new food every 2 seconds

    def initialize_food_sources(self, world_width, world_height):
        """Create initial food distribution"""
        print("🌱 Generating initial food sources...")

        # Create grass patches
        for _ in range(80):
            pos = [
                random.randint(20, world_width - 20),
                random.randint(20, world_height - 20)
            ]
            grass = FoodSource(pos, "grass")
            self.food_sources.append(grass)

        # Create berry bushes (less common)
        for _ in range(20):
            pos = [
                random.randint(30, world_width - 30),
                random.randint(30, world_height - 30)
            ]
            berry = FoodSource(pos, "berry")
            self.food_sources.append(berry)

        # Create mushrooms (rare)
        for _ in range(10):
            pos = [
                random.randint(40, world_width - 40),
                random.randint(40, world_height - 40)
            ]
            mushroom = FoodSource(pos, "mushroom")
            self.food_sources.append(mushroom)

        print(f"🍃 Created {len(self.food_sources)} food sources")

    def update(self, world_width, world_height):
        """Update all food sources"""
        # Update existing food sources
        for food in self.food_sources:
            food.update()

        # Spawn new food sources periodically
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and len(self.food_sources) < self.max_food_sources:
            self._spawn_random_food(world_width, world_height)
            self.spawn_timer = self.spawn_interval

    def _spawn_random_food(self, world_width, world_height):
        """Spawn a random food source"""
        # 70% grass, 20% berries, 10% mushrooms
        rand = random.random()
        if rand < 0.7:
            food_type = "grass"
        elif rand < 0.9:
            food_type = "berry"
        else:
            food_type = "mushroom"

        # Find empty spot
        attempts = 0
        while attempts < 10:
            pos = [
                random.randint(20, world_width - 20),
                random.randint(20, world_height - 20)
            ]

            # Check if too close to existing food
            too_close = False
            min_distance = 25 if food_type == "grass" else 40

            for existing_food in self.food_sources:
                distance = np.linalg.norm(np.array(pos) - existing_food.position)
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                new_food = FoodSource(pos, food_type)
                self.food_sources.append(new_food)
                break

            attempts += 1

    def get_food_in_range(self, position, range_distance):
        """Get food sources within range of position

        Args:
            position: Center position [x, y]
            range_distance: Search radius

        Returns:
            List of food sources within range
        """
        nearby_food = []

        for food in self.food_sources:
            if food.is_depleted:
                continue

            distance = np.linalg.norm(np.array(position) - food.position)
            if distance <= range_distance:
                nearby_food.append(food)

        return nearby_food

    def consume_food_at(self, position, consumption_radius=15, amount=20):
        """Consume food at a given position

        Args:
            position: Position to consume food at
            consumption_radius: Radius to search for food
            amount: Amount of nutrition to consume

        Returns:
            Actual nutrition gained
        """
        total_nutrition = 0

        for food in self.food_sources:
            if food.is_depleted:
                continue

            distance = np.linalg.norm(np.array(position) - food.position)
            if distance <= consumption_radius:
                nutrition = food.consume(amount)
                total_nutrition += nutrition

                if nutrition > 0:
                    print(f"🍃 Agent consumed {nutrition} from {food.food_type}")
                    break  # Only eat from one source per update

        return total_nutrition

    def draw_all(self, screen, camera_offset=(0, 0)):
        """Draw all food sources"""
        for food in self.food_sources:
            food.draw(screen, camera_offset)  # ← camera_offset

    def get_statistics(self):
        """Get food system statistics"""
        stats = {
            'total_food_sources': len(self.food_sources),
            'available_food': len([f for f in self.food_sources if not f.is_depleted]),
            'depleted_food': len([f for f in self.food_sources if f.is_depleted])
        }

        # Count by type
        for food_type in ['grass', 'berry', 'mushroom', 'leaves']:
            stats[f'{food_type}_count'] = len([f for f in self.food_sources if f.food_type == food_type])

        return stats


# ============================================

# agents/rabbit.py - Updated rabbit to use food system
"""
Updated Rabbit class to interact with food system
Add this method to the existing Rabbit class:
"""


def _find_food_sources_updated(self, world_state):
    """Updated method to find real food sources using food manager"""
    # Get food manager from world
    if hasattr(world_state, 'food_manager'):
        food_manager = world_state['food_manager']
        nearby_food = food_manager.get_food_in_range(self.position, self.vision_range)

        # Convert food sources to the format expected by AI
        food_list = []
        for food in nearby_food:
            if not food.is_depleted:
                food_list.append({
                    'position': food.position,
                    'nutrition': food.nutrition_value,
                    'type': food.food_type,
                    'distance': np.linalg.norm(food.position - self.position)
                })

        # Sort by distance (closest first)
        food_list.sort(key=lambda f: f['distance'])
        return food_list

    return []


def _attempt_eat_food_updated(self, world):
    """Updated method to eat real food sources"""
    if hasattr(world, 'food_manager'):
        nutrition_gained = world.food_manager.consume_food_at(
            self.position,
            consumption_radius=self.size + 5,
            amount=25
        )

        if nutrition_gained > 0:
            self.energy = min(self.max_energy, self.energy + nutrition_gained)
            return True

    return False