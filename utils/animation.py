# utils/animation.py - Animation system for sprites
"""
SurvAIval Animation System
Handles sprite animation, frame management, and state transitions
"""

import pygame
from typing import Dict, List, Tuple, Optional


class Animation:
    """Single animation (e.g., 'idle', 'walk')"""

    def __init__(self, name: str, frames: List[pygame.Surface], frame_duration: int = 10):
        """Initialize animation

        Args:
            name: Animation name (idle, walk, hit, death)
            frames: List of pygame surfaces for each frame
            frame_duration: How many game ticks per frame (60 FPS = 10 ticks = 6 FPS animation)
        """
        self.name = name
        self.frames = frames
        self.frame_duration = frame_duration
        self.frame_count = len(frames)

    def get_frame(self, frame_index: int) -> pygame.Surface:
        """Get a specific frame"""
        return self.frames[frame_index % self.frame_count]


class AnimatedSprite:
    """Manages animations for a single entity"""

    def __init__(self, spritesheet_path: str, frame_width: int = 32, frame_height: int = 32, scale: float = 1.5):
        """Initialize animated sprite

        Args:
            spritesheet_path: Path to spritesheet PNG
            frame_width: Width of each frame
            frame_height: Height of each frame
            scale: Scale multiplier for display (1.5 = 150% size)
        """
        self.spritesheet_path = spritesheet_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = scale

        # Animation storage
        self.animations: Dict[str, Animation] = {}

        # Current state
        self.current_animation: Optional[str] = None
        self.current_frame: int = 0
        self.frame_timer: int = 0
        self.is_flipped: bool = False
        self.is_playing: bool = True

        # Load spritesheet
        try:
            self.spritesheet = pygame.image.load(spritesheet_path).convert_alpha()
            print(f"✅ Loaded spritesheet: {spritesheet_path}")
        except Exception as e:
            print(f"❌ Failed to load spritesheet {spritesheet_path}: {e}")
            self.spritesheet = None

    def add_animation(self, name: str, start_frame: int, frame_count: int,
                      frame_duration: int = 10) -> None:
        """Add an animation from the spritesheet

        Args:
            name: Animation name (idle, walk, hit, death)
            start_frame: Starting frame index in spritesheet
            frame_count: Number of frames in this animation
            frame_duration: Ticks per frame
        """
        if self.spritesheet is None:
            return

        frames = []

        for i in range(frame_count):
            frame_index = start_frame + i
            x = frame_index * self.frame_width
            y = 0  # Assuming horizontal strip

            # Extract frame from spritesheet
            frame_rect = pygame.Rect(x, y, self.frame_width, self.frame_height)
            frame_surface = pygame.Surface((self.frame_width, self.frame_height), pygame.SRCALPHA)
            frame_surface.fill((0, 0, 0, 0))  # Clear with transparency
            frame_surface.blit(self.spritesheet, (0, 0), frame_rect)

            frames.append(frame_surface)

        animation = Animation(name, frames, frame_duration)
        self.animations[name] = animation

        # Set first animation as default
        if self.current_animation is None:
            self.current_animation = name

        print(f"  ➕ Added animation: {name} ({frame_count} frames, {self.frame_width}x{self.frame_height})")

    def play_animation(self, animation_name: str, reset: bool = False) -> None:
        """Switch to a different animation

        Args:
            animation_name: Name of animation to play
            reset: Whether to restart animation from frame 0
        """
        if animation_name not in self.animations:
            return

        # Only reset if switching to different animation or forced
        if self.current_animation != animation_name or reset:
            self.current_animation = animation_name
            if reset:
                self.current_frame = 0
                self.frame_timer = 0

    def update(self) -> None:
        """Update animation state (call every frame)"""
        if not self.is_playing or self.current_animation is None:
            return

        animation = self.animations.get(self.current_animation)
        if animation is None:
            return

        # Update frame timer
        self.frame_timer += 1

        if self.frame_timer >= animation.frame_duration:
            self.frame_timer = 0
            self.current_frame = (self.current_frame + 1) % animation.frame_count

    def get_current_frame(self) -> Optional[pygame.Surface]:
        """Get the current frame to render"""
        if self.current_animation is None:
            return None

        animation = self.animations.get(self.current_animation)
        if animation is None:
            return None

        frame = animation.get_frame(self.current_frame)

        # Flip if needed
        if self.is_flipped:
            frame = pygame.transform.flip(frame, True, False)

        return frame

    def set_flip(self, flipped: bool) -> None:
        """Set horizontal flip state"""
        self.is_flipped = flipped

    def draw(self, screen: pygame.Surface, position: Tuple[int, int]) -> None:
        """Draw current frame at position

        Args:
            screen: Surface to draw on
            position: (x, y) center position
        """
        frame = self.get_current_frame()
        if frame:
            # Scale the frame if needed
            if self.scale != 1.0:
                new_width = int(self.frame_width * self.scale)
                new_height = int(self.frame_height * self.scale)
                frame = pygame.transform.scale(frame, (new_width, new_height))

            rect = frame.get_rect(center=position)
            screen.blit(frame, rect)


class SpriteAnimationManager:
    """Manages all animated sprites in the game"""

    def __init__(self):
        """Initialize animation manager"""
        self.animated_sprites: Dict[str, AnimatedSprite] = {}
        print("🎬 Animation manager initialized")

    def load_animal_sprite(self, animal_name: str, spritesheet_path: str,
                           animation_config: Optional[Dict] = None, scale: float = 1.5) -> bool:
        """Load an animal's spritesheet with custom or standard animations

        Args:
            animal_name: Name of animal (rabbit, wolf, etc.)
            spritesheet_path: Path to spritesheet PNG
            animation_config: Optional dict with custom animation definitions
            scale: Display scale multiplier (1.5 = 150% size)

        Returns:
            True if loaded successfully
        """
        try:
            # Create animated sprite with scale
            sprite = AnimatedSprite(spritesheet_path, frame_width=32, frame_height=32, scale=scale)

            if animation_config:
                # Use custom animation configuration
                for anim_name, anim_data in animation_config.items():
                    sprite.add_animation(
                        anim_name,
                        start_frame=anim_data['start'],
                        frame_count=anim_data['count'],
                        frame_duration=anim_data.get('duration', 10)
                    )
            else:
                # Default: 4 frames each for simple animals (rabbit, deer)
                sprite.add_animation('idle', start_frame=0, frame_count=4, frame_duration=12)
                sprite.add_animation('walk', start_frame=4, frame_count=4, frame_duration=8)
                sprite.add_animation('hit', start_frame=8, frame_count=4, frame_duration=6)
                sprite.add_animation('death', start_frame=12, frame_count=4, frame_duration=10)

            self.animated_sprites[animal_name] = sprite
            print(f"🎨 Loaded animated sprite: {animal_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to load {animal_name}: {e}")
            return False

    def get_sprite(self, animal_name: str) -> Optional[AnimatedSprite]:
        """Get an animated sprite by name"""
        return self.animated_sprites.get(animal_name)

    def update_all(self) -> None:
        """Update all animations (call once per frame)"""
        for sprite in self.animated_sprites.values():
            sprite.update()


# Global animation manager
_animation_manager: Optional[SpriteAnimationManager] = None


def get_animation_manager() -> SpriteAnimationManager:
    """Get the global animation manager instance"""
    global _animation_manager
    if _animation_manager is None:
        _animation_manager = SpriteAnimationManager()
    return _animation_manager


def init_animations() -> SpriteAnimationManager:
    """Initialize and load all animations"""
    manager = get_animation_manager()

    # Rabbit config (simple - 16 frames) - SLOWER ANIMATION
    rabbit_config = {
        'idle': {'start': 0, 'count': 4, 'duration': 20},
        'walk': {'start': 4, 'count': 4, 'duration': 12},
        'hit': {'start': 8, 'count': 4, 'duration': 10},
        'death': {'start': 12, 'count': 4, 'duration': 15}
    }

    # Wolf config (detailed - 45 frames) - SLOWER ANIMATION

    wolf_config = {

        'idle': {'start': 0, 'count': 4, 'duration': 18},
        'walk': {'start': 4, 'count': 8, 'duration': 10},
        'jump': {'start': 12, 'count': 3, 'duration': 12},
        'attack': {'start': 15, 'count': 7, 'duration': 7},
        'attack2': {'start': 22, 'count': 5, 'duration': 7},
        'howl': {'start': 27, 'count': 9, 'duration': 12},
        'hit': {'start': 36, 'count': 3, 'duration': 8},
        'death': {'start': 39, 'count': 7, 'duration': 15}
    }

    # Bear config - Apex Predator
    bear_config = {
        'idle': {'start': 0, 'count': 4, 'duration': 18},
        'walk': {'start': 4, 'count': 8, 'duration': 10},
        'jump': {'start': 12, 'count': 3, 'duration': 12},
        'attack': {'start': 15, 'count': 7, 'duration': 7},
        'attack2': {'start': 22, 'count': 5, 'duration': 7},
        'howl': {'start': 27, 'count': 9, 'duration': 12},
        'hit': {'start': 36, 'count': 3, 'duration': 8},
        'death': {'start': 39, 'count': 7, 'duration': 15}
    }

    # Deer config - Herbivore
    deer_config = {

        'idle': {'start': 0, 'count': 5, 'duration': 18},
        'walk': {'start': 5, 'count': 3, 'duration': 10},
        'jump': {'start': 8, 'count': 3, 'duration': 12},
        'attack': {'start': 11, 'count': 7, 'duration': 7},
        'hit': {'start': 18, 'count': 3, 'duration': 8},
        'death': {'start': 21, 'count': 5, 'duration': 15}
    }


    # Load animations with BIGGER scale
    manager.load_animal_sprite('rabbit', 'assets/animals/rabbit_spritesheet.png', rabbit_config, scale=2.5)
    manager.load_animal_sprite('wolf', 'assets/animals/wolf_spritesheet.png', wolf_config, scale=2.5)
    manager.load_animal_sprite('bear', 'assets/animals/bear_spritesheet.png', bear_config, scale=2.5)
    manager.load_animal_sprite('deer', 'assets/animals/deer_spritesheet.png', deer_config, scale=2.5)

    return manager