
def baseline_action(observation: dict) -> dict:
    """
    Rule-based AI driver.

    Input: Information about the car and track.
    Output: Steering, throttle and brake decisions.
    """

    # Get information about the car and track
    speed = observation["speed"]
    lateral_offset = observation["lateral_offset"]
    heading_error = observation["heading_error"]
    curvature_ahead = observation["curvature_ahead"]

    # Correct the car's position and heading
    position_correction = -0.04 * lateral_offset
    heading_correction = -0.8 * heading_error

    # Anticipate upcoming corners
    corner_steering = 2.0 * curvature_ahead[0] if curvature_ahead else 0.0

    #Combine steering decisions
    steering = position_correction + heading_correction + corner_steering

    # Keep steering within allowed limits
    steering = max(-0.3, min(0.3, steering))


    # 2. SPEED: Slow down before sharp corners
    max_curvature = max(abs(c) for c in curvature_ahead)

    if max_curvature > 0.025:
        target_speed = 10.0
    elif max_curvature > 0.01:
        target_speed = 18.0
    else:
        target_speed = 30.0

    # 3. THROTTLE AND BRAKE
    if speed < target_speed - 1.0:
        throttle = 0.5
        brake = 0.0
    elif speed > target_speed + 1.0:
        throttle = 0.0
        brake = 0.5
    else:
        throttle = 0.0
        brake = 0.0

    return {
        "steering": float(steering),
        "throttle": float(throttle),
        "brake": float(brake)
    }


# Test the function using a sample observation
if __name__ == "__main__":
    observation = {
        "speed": 25.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.05, 0.04, 0.03]
    }

    action = baseline_action(observation)
    print("Driving AI action:", action)