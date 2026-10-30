from baseline import baseline_action


def make_ai_controller(model, predict_action):
    """
    Create a driving controller using AI throttle and brake
    predictions with the baseline steering controller.
    """

    def ai_controller(observation):
        # Use baseline steering for both drivers
        baseline = baseline_action(observation)

        # Get throttle and brake predictions from the AI model
        prediction = predict_action(model, observation)

        # Combine steering with AI predictions
        return {
            "steering": baseline["steering"],
            "throttle": prediction["throttle"],
            "brake": prediction["brake"],
        }

    return ai_controller