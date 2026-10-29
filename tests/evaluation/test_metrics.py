from f1_simulator.evaluation import calculate_metrics, compare_controllers


def test_completed_lap_metrics_include_lap_time():
    result = calculate_metrics(
        elapsed_time=95.0,
        distance_travelled=5200.0,
        off_track_events=2,
        completed_lap=True,
    )

    assert result["lap_time"] == 95.0
    assert result["completed_lap"] is True
    assert result["off_track_events"] == 2


def test_incomplete_lap_does_not_report_a_lap_time():
    result = calculate_metrics(
        elapsed_time=120.0,
        distance_travelled=3000.0,
        off_track_events=4,
        completed_lap=False,
    )

    assert result["lap_time"] is None
    assert result["completed_lap"] is False


def test_controller_comparison_reports_improvement_and_off_track_counts():
    baseline = {
        "lap_time": 100.0,
        "distance_travelled": 5200.0,
        "off_track_events": 3,
        "completed_lap": True,
    }
    ai = {
        "lap_time": 90.0,
        "distance_travelled": 5200.0,
        "off_track_events": 1,
        "completed_lap": True,
    }

    comparison = compare_controllers(baseline, ai)

    assert comparison["time_improvement"] == 10.0
    assert comparison["ai_off_track"] < comparison["baseline_off_track"]
    assert comparison["ai_completed"] is True