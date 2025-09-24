# main.py - Main entry point
"""
SurvAIval - Main application entry point
Run this file to start the simulation
"""

import pygame
import sys
from engine.simulation import Simulation
import config


def main():
    """Main function - Initialize and start the simulation"""
    print("🎮 Starting SurvAIval...")

    # Initialize Pygame
    pygame.init()

    # Create display
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("SurvAIval - AI Ecosystem Simulation")

    # Create and run simulation
    try:
        simulation = Simulation(screen)
        simulation.run()
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return 1

    print("👋 SurvAIval shutting down...")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())