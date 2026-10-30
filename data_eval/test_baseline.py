from data_eval.baseline import baseline_action


def test_straight():
    observation = {
        "speed": 10.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.0, 0.0, 0.0]
    }

    action = baseline_action(observation)

    assert action["steering"] == 0.0
    assert action["throttle"] > 0
    assert action["brake"] == 0.0

    print("Straight driving test passed!")


def test_corner():
    observation = {
        "speed": 25.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.03, 0.04, 0.05]
    }

    action = baseline_action(observation)

    assert action["throttle"] == 0.0
    assert action["brake"] > 0

    print("Corner braking test passed!")


def test_steering():
    observation = {
        "speed": 15.0,
        "lateral_offset": -2.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.0, 0.0, 0.0]
    }

    
    
    action = baseline_action(observation)

    assert action["steering"] > 0

    print("Steering correction test passed!")
def test_corner_anticipation():
    observation = {
        "speed": 25.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.05, 0.04, 0.03]
    }

    action = baseline_action(observation)

    assert action["steering"] > 0
    assert action["brake"] > 0
    assert action["throttle"] == 0.0

    print("Corner anticipation test passed!")

if __name__ == "__main__":
    test_straight()
    test_corner()
    test_steering()
    test_corner_anticipation()

    print("All baseline tests passed!")
