"""Integration checks for saved policies after moving code and bundled inputs."""
import hashlib
import json

import joblib
import numpy as np
import pytest

from f1_simulator.core.full_policy import load_learned_driver
from f1_simulator.core.imitation import load_model, predict_action
from f1_simulator.core.simulator import get_observation, reset_simulation
from f1_simulator.core.track import Track
from f1_simulator.controllers.speed_planner import make_hybrid_controller
from f1_simulator.paths import MODEL_DIR


@pytest.mark.parametrize('filename', [
    'original_pedals.joblib', 'full_control.joblib', 'tuned_random_forest.joblib',
    'tuned_svr.joblib', 'tuned_neural_network.joblib', 'rf_svr_blend.joblib',
])
def test_bundled_policy_loads_and_controls_simulator(filename):
    """Bundled models must survive package moves and use the live observation API."""
    path = MODEL_DIR / filename
    manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['models'][filename]['sha256']
    track = Track('monza')
    state, _ = reset_simulation(track)
    if filename == 'full_control.joblib':
        controller = load_learned_driver(path, track)
    else:
        model = load_model(path) if filename == "original_pedals.joblib" else joblib.load(path)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        controller = make_hybrid_controller(model, predict_action, track=track)
    observer = getattr(controller, 'observe', None)
    observation = observer(state, track) if observer else get_observation(
        state, track, lookahead_offsets=controller.lookahead_offsets)
    action = controller(observation)
    assert set(action) == {'steering', 'throttle', 'brake'}
    assert all(np.isfinite(v) for v in action.values())
    assert 0 <= action['throttle'] <= 1
    assert 0 <= action['brake'] <= 1
