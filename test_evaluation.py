from evaluation import evaluate_controller
from baseline import baseline_action

class MockTrack:
    pass

track = MockTrack()

try:
    results = evaluate_controller(
        baseline_action,
        track,
        dt=0.1
    )
    print("Evaluation results:", results)

except NotImplementedError:
    print("Evaluation framework is ready for simulator integration.")

    from evaluation import calculate_metrics

results = calculate_metrics(
    elapsed_time=95.0,
    distance_travelled=5200.0,
    off_track_events=2,
    completed_lap=True
)

print("Lap evaluation results:", results)

failed_results = calculate_metrics(
    elapsed_time=120.0,
    distance_travelled=3000.0,
    off_track_events=4,
    completed_lap=False
)

assert failed_results["lap_time"] is None
assert failed_results["completed_lap"] is False
assert failed_results["off_track_events"] == 4

print("Incomplete lap test passed!")

from evaluation import compare_controllers

baseline_results = {
    "lap_time": 100.0,
    "distance_travelled": 5200.0,
    "off_track_events": 3,
    "completed_lap": True
}

ai_results = {
    "lap_time": 90.0,
    "distance_travelled": 5200.0,
    "off_track_events": 1,
    "completed_lap": True
}

comparison = compare_controllers(baseline_results, ai_results)

assert comparison["time_improvement"] == 10.0
assert comparison["ai_off_track"] < comparison["baseline_off_track"]
assert comparison["ai_completed"] is True

print("Controller comparison test passed!")
print("Comparison results:", comparison)