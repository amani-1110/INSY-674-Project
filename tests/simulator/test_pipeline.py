import pytest

from scripts.run_pipeline import run_pipeline, prediction_metrics
from f1_simulator.controllers.ai_controller import make_ai_controller
from f1_simulator.controllers.baseline import baseline_action
from f1_simulator.controllers.evaluation import evaluate_controller
from f1_simulator.core.imitation import prepare_training_data, train_model, predict_action, save_model, load_model
from f1_simulator.core.track import Track


@pytest.mark.parametrize("kwargs", [dict(dt=0), dict(max_time=0), dict(held_out_lap=17)])
def test_reject_invalid_pipeline_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_pipeline(tmp_path / "unused", **kwargs)
    assert not (tmp_path / "unused").exists()


def test_saved_imitation_model_drives_actual_simulator(tmp_path):
    track = Track()
    records = [dict(s=s, speed=20., lateral_offset=0., heading_error=0.,
                    throttle=.4, brake=0., lap_id="training")
               for s in [0., 30., 60., 90.]]
    X, y = prepare_training_data(records, track)
    model = train_model(X, y, n_estimators=5, min_samples_leaf=1)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    loaded.n_jobs = 1
    controller = make_ai_controller(loaded, predict_action)
    assert controller(X[0])["steering"] == baseline_action(X[0])["steering"]
    assert controller(X[0])["throttle"] == pytest.approx(.4)
    result = evaluate_controller(controller, track, dt=.1, max_time=.25)
    assert result["elapsed_time"] == pytest.approx(.25)
    assert result["termination"] == "timeout"
    assert not result["valid_lap"]
    assert result["distance_travelled"] > 0
    metrics = prediction_metrics(loaded, records, track, records)
    assert metrics["throttle_mae"] == pytest.approx(0)
    assert metrics["brake_mae"] == 0


def test_long_horizon_matches_training_and_runtime_and_keeps_steering(tmp_path):
    from f1_simulator.core.imitation import LONG_LOOKAHEAD_OFFSETS
    from f1_simulator.core.simulator import reset_simulation, get_observation
    track = Track()
    state, _ = reset_simulation(track)
    offsets = LONG_LOOKAHEAD_OFFSETS
    records = [dict(s=0., speed=0., lateral_offset=0., heading_error=0., throttle=.4, brake=0.)]
    X, y = prepare_training_data(records, track, offsets)
    observation = get_observation(state, track, lookahead_offsets=offsets)
    assert observation["curvature_ahead"] == pytest.approx(X[0]["curvature_ahead"])
    assert len(observation["curvature_ahead"]) == 10
    assert observation["curvature_ahead"][-1] == pytest.approx(track.curvature(250))
    model = train_model(X, y, n_estimators=3, min_samples_leaf=1, lookahead_offsets=offsets)
    path = tmp_path/"long.joblib"
    save_model(model, path)
    loaded = load_model(path)
    loaded.n_jobs = 1
    assert loaded.lookahead_offsets_ == offsets
    controller = make_ai_controller(loaded, predict_action)
    assert controller.lookahead_offsets == offsets
    assert controller(observation)["steering"] == baseline_action(get_observation(state, track))["steering"]
    result = evaluate_controller(controller, track, dt=.1, max_time=.2)
    assert result["distance_travelled"] > 0
    with pytest.raises(ValueError, match="10 values"):
        predict_action(loaded, get_observation(state, track))


def test_old_model_metadata_remains_compatible(tmp_path):
    import joblib
    from f1_simulator.core.imitation import FEATURE_NAMES, LOOKAHEAD_OFFSETS, TARGET_NAMES
    track = Track()
    X, y = prepare_training_data([dict(s=0., speed=0., lateral_offset=0.,
                  heading_error=0., throttle=.5, brake=0.)], track)
    model = train_model(X, y, n_estimators=2)
    del model.lookahead_offsets_
    path = tmp_path/"legacy.joblib"
    joblib.dump(dict(model=model, features=FEATURE_NAMES, targets=TARGET_NAMES, lookahead=LOOKAHEAD_OFFSETS), path)
    loaded = load_model(path)
    assert loaded.lookahead_offsets_ == LOOKAHEAD_OFFSETS
    assert predict_action(loaded, X[0])["throttle"] == pytest.approx(.5)


@pytest.mark.parametrize("horizon,count", [(500, 15), (1000, 20)])
def test_extended_horizon_round_trip_and_lap_wrapping(tmp_path, horizon, count):
    from f1_simulator.core.imitation import offsets_for_horizon
    from f1_simulator.core.simulator import get_observation
    track = Track()
    s = track.length - 5
    x, y = track.position(s)
    state = dict(x=x, y=y, heading=track.heading(s), speed=30.)
    offsets = offsets_for_horizon(horizon)
    records = [dict(s=s, speed=30., lateral_offset=0., heading_error=0., throttle=.3, brake=0.)]
    X, y = prepare_training_data(records, track, offsets)
    obs = get_observation(state, track, lookahead_offsets=offsets)
    assert len(obs["curvature_ahead"]) == count
    assert offsets[-1] == horizon
    assert obs["curvature_ahead"] == pytest.approx(X[0]["curvature_ahead"], abs=1e-7)
    model = train_model(X, y, n_estimators=2, lookahead_offsets=offsets)
    path = tmp_path/f"model_{horizon}.joblib"
    save_model(model, path)
    restored = load_model(path)
    restored.n_jobs = 1
    assert restored.lookahead_offsets_ == offsets
    assert predict_action(restored, obs)["throttle"] == pytest.approx(.3)
