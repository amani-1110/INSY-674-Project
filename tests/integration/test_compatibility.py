import importlib.util
from pathlib import Path

import baseline as legacy_baseline
from f1_simulator.controllers.baseline import baseline_action
from f1_simulator.core.physics import update_car as prototype_update_car
from f1_simulator.evaluation import evaluate_controller
from f1_simulator.physics import update_car as detailed_update_car


def test_root_physics_import_remains_the_detailed_model():
    module_path = Path(__file__).resolve().parents[2] / "physics.py"
    spec = importlib.util.spec_from_file_location("legacy_physics", module_path)
    legacy_physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_physics)

    assert legacy_physics.update_car is detailed_update_car


def test_root_baseline_import_remains_the_packaged_controller():
    assert legacy_baseline.baseline_action is baseline_action


def test_core_physics_path_remains_the_prototype_model():
    assert prototype_update_car is not detailed_update_car


def test_evaluation_import_is_available_from_package():
    assert callable(evaluate_controller)
