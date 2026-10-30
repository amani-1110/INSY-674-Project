from data_eval.evaluation import calculate_metrics, compare_controllers


def test_finished_and_failed_metrics():
    finished = calculate_metrics(95.0, 5200.0, 2, True)
    failed = calculate_metrics(120.0, 3000.0, 4, False)
    assert finished['lap_time'] == 95.0
    assert failed['lap_time'] is None
    assert failed['completed_lap'] is False
    assert failed['off_track_events'] == 4


def test_comparison():
    baseline = calculate_metrics(100.0, 5200.0, 3, True)
    ai = calculate_metrics(90.0, 5200.0, 1, True)
    result = compare_controllers(baseline, ai)
    assert result['time_improvement'] == 10.0
    assert result['ai_off_track'] < result['baseline_off_track']
