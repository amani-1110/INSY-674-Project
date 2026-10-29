from f1_simulator.core import simulator

from .metrics import calculate_metrics


def evaluate_controller(controller, track, dt=0.1, controller_name=None):
    """
    Evaluate a driving controller on a racetrack.

    Parameters:
        controller: Function that makes driving decisions.
        track: The racetrack provided by your teammate.
        dt: Time between simulation steps (seconds).

    Returns:
        Dictionary containing the driver's performance.
    """

    # These will be updated during the simulation.
    lap_time = 0.0
    distance_travelled = 0.0
    off_track_events = 0
    completed_lap = False
    was_off_track = False

    # Start a new simulation on the selected track
    state, race_info = simulator.reset_simulation(track)

    # Set a maximum evaluation time of 5 minutes
    max_time = 300.0

    while lap_time + dt / 2 < max_time and not completed_lap:
        # Get information about the car and track
        observation = simulator.get_observation(state, track)

        # Ask the controller to make a driving decision
        action = controller(observation)

        # Move the car forward one simulation step
        new_state, race_info = simulator.simulate_step(state, action, dt, track)

        if race_info.get("lap_complete", False):
            completed_lap = True

        if "off_track" in race_info:
            is_off_track = race_info["off_track"]
            if is_off_track and not was_off_track:
                off_track_events += 1
            was_off_track = is_off_track
    
       # Calculate distance moved during this step
        dx = new_state["x"] - state["x"]
        dy = new_state["y"] - state["y"]

        step_distance = (dx ** 2 + dy ** 2) ** 0.5
        distance_travelled += step_distance

        # Update elapsed time
        lap_time += dt

        # Save the updated state
        state = new_state

    # Return the driver's performance results
    return calculate_metrics(
        elapsed_time=lap_time,
        distance_travelled=distance_travelled,
        off_track_events=off_track_events,
        completed_lap=completed_lap,
        controller_name=controller_name,
    )