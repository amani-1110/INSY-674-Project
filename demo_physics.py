"""Compatibility launcher for the detailed-physics plotting demo."""

from runpy import run_module


if __name__ == "__main__":
    run_module("examples.physics_demo", run_name="__main__")
