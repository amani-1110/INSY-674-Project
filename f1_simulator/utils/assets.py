"""Shared loading of bundled car images."""
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def load_car_images():
    """Load car sprites after a pygame display has been initialized."""
    import pygame

    return [
        pygame.image.load(str(ASSETS_DIR / f"car_{index}.png")).convert_alpha()
        for index in range(1, 5)
    ]
