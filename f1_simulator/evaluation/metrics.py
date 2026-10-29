"""Performance metrics and controller comparison helpers."""


def calculate_metrics(
    elapsed_time,
    distance_travelled,
    off_track_events,
    completed_lap,
    controller_name=None,
):
    """Return the standard performance result for one controller run."""
    return {
        "lap_time": elapsed_time if completed_lap else None,
        "distance_travelled": distance_travelled,
        "off_track_events": off_track_events,
        "completed_lap": completed_lap,
        "controller_name": controller_name,
    }


def compare_controllers(baseline_results, ai_results):
    """Compare baseline and AI controller results."""
    comparison = {
        "baseline_completed": baseline_results["completed_lap"],
        "ai_completed": ai_results["completed_lap"],
        "baseline_lap_time": baseline_results["lap_time"],
        "ai_lap_time": ai_results["lap_time"],
        "baseline_off_track": baseline_results["off_track_events"],
        "ai_off_track": ai_results["off_track_events"],
        "baseline_distance": baseline_results["distance_travelled"],
        "ai_distance": ai_results["distance_travelled"],
    }

    if baseline_results["lap_time"] is not None and ai_results["lap_time"] is not None:
        comparison["time_improvement"] = (
            baseline_results["lap_time"] - ai_results["lap_time"]
        )
    else:
        comparison["time_improvement"] = None

    return comparison
