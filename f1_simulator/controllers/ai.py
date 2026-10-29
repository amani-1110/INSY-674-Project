from .baseline import baseline_action


def make_ai_controller(model, predict_action, launch_speed=10.0):
    """
    Create a controller using baseline steering and learned longitudinal controls.

    The telemetry does not include stationary launch examples, so the AI uses
    a short, deterministic launch phase before relying on model predictions.
    """

    def ai_controller(observation):
        # Use baseline steering for both drivers
        baseline = baseline_action(observation)

        # Get throttle and brake predictions from the AI model
        prediction = predict_action(model, observation)
        if observation["speed"] < launch_speed:
            prediction["throttle"] = max(prediction["throttle"], 0.8)
            prediction["brake"] = 0.0

        # Combine steering with AI predictions
        return {
            "steering": baseline["steering"],
            "throttle": prediction["throttle"],
            "brake": prediction["brake"],
        }

    return ai_controller