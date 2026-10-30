"""Headless checks of camera transforms and cached circuit geometry."""
import numpy as np


def test_follow_and_overview_reuse_spline_geometry(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from f1_simulator.ui import renderer
    from f1_simulator.core.track import Track
    from f1_simulator.core.simulator import reset_simulation
    track = Track()
    state, _ = reset_simulation(track)
    try:
        renderer.render(track, state, controller_name="imitation")
        context = renderer._renderer
        def unexpected_geometry_query(*args):
            raise AssertionError("Static geometry must not be recomputed per frame")
        monkeypatch.setattr(track, "boundaries", unexpected_geometry_query)
        monkeypatch.setattr(track, "position", unexpected_geometry_query)
        import pygame
        draw_polygon = pygame.draw.polygon
        def draw_visible_quad(surface, color, points, *args, **kwargs):
            assert len(points) <= 4, "Follow camera must cull and draw local road quads"
            return draw_polygon(surface, color, points, *args, **kwargs)
        monkeypatch.setattr(pygame.draw, "polygon", draw_visible_quad)
        renderer.render(track, state, camera_mode="follow", zoom=7)
        renderer.render(track, state, camera_mode="overview")
        assert renderer._renderer is context
        assert context["minimap"].get_size() == (264, 180)
    finally:
        renderer.close_renderer()
    assert renderer._renderer is None


def test_follow_camera_preserves_world_scale():
    from f1_simulator.ui.renderer import _pixels
    center = np.array([100., 200.])
    assert np.array_equal(_pixels(center, center, 5), [550, 415])
    assert np.array_equal(_pixels(center+[12., 0.], center, 5), [610, 415])
    assert np.array_equal(_pixels(center+[0., 12.], center, 5), [550, 355])
