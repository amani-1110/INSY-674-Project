import math

import pytest

from f1_simulator.controllers.ai import make_ai_controller
from f1_simulator.controllers.baseline import baseline_action
from f1_simulator.controllers.imitation import predict
from f1_simulator.controllers.telemetry_model import train_telemetry_model
from f1_simulator.core.simulator import RaceSession
from f1_simulator.core.telemetry_track import TelemetryTrack
from f1_simulator.data.telemetry_replay import TelemetryReplay


def test_bundled_telemetry_replay_closes_near_the_derived_track_start():
    track = TelemetryTrack.from_csv()
    replay = TelemetryReplay.from_csv()
    first = replay.state_at(0.0)
    last = replay.state_at(replay.duration)

    assert math.hypot(first["x"] - last["x"], first["y"] - last["y"]) < 1.0
    assert first["speed"] > 0
    assert last["finished"] is True
    assert track.project(first["x"], first["y"])[1] == pytest.approx(0.0, abs=1.0)


def test_player_and_ai_race_sessions_keep_independent_state():
    track = TelemetryTrack.from_csv()
    player = RaceSession(track, lateral_offset=-3.0)
    ai = RaceSession(track, lateral_offset=3.0)

    player.step({"steering": 0.0, "throttle": 1.0, "brake": 0.0}, 0.1)

    assert player.time == 0.1
    assert ai.time == 0.0
    assert player.state["speed"] > 0
    assert ai.state["speed"] == 0


def test_baseline_controller_runs_in_an_independent_race_session():
    track = TelemetryTrack.from_csv()
    baseline = RaceSession(track, start_s=track.length - 20.0)

    state, _ = baseline.step(baseline_action(baseline.observation()), 0.1)

    assert baseline.time == 0.1
    assert state["speed"] > 0
    assert baseline.progress >= 0


def test_telemetry_trained_ai_launches_and_advances_on_monza():
    track = TelemetryTrack.from_csv()
    model = train_telemetry_model(track)
    controller = make_ai_controller(model, predict)
    ai = RaceSession(track, lateral_offset=3.0)

    for _ in range(120):
        ai.step(controller(ai.observation()), 1 / 60)

    assert ai.state["speed"] > 5.0
    assert ai.progress > 5.0
