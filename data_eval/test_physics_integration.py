from baseline import baseline_action
from physics import update_car

# 1. Set the car's starting position and speed
state = {
    "x": 0.0,
    "y": 0.0,
    "heading": 0.0,
    "speed": 10.0
}

# 2. Give the baseline driver information about the track
observation = {
    "speed": state["speed"],
    "lateral_offset": 0.0,
    "heading_error": 0.0,
    "curvature_ahead": [0.0, 0.0, 0.0]
}

# 3. Ask the baseline driver what to do
action = baseline_action(observation)

print("Baseline action:", action)

# 4. Apply that action to the car physics
new_state = update_car(state, action, dt=0.1)

print("Original car state:", state)
print("Updated car state:", new_state)

# 5. Check that the physics model returned a valid car state
assert isinstance(new_state, dict)
assert all(key in new_state for key in ["x", "y", "heading", "speed"])

print("Physics integration test passed!")

# 6. Simulate 5 seconds of straight-line driving
state = {
    "x": 0.0,
    "y": 0.0,
    "heading": 0.0,
    "speed": 0.0
}

dt = 0.1

for step in range(50):
    observation = {
        "speed": state["speed"],
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.0, 0.0, 0.0]
    }

    action = baseline_action(observation)
    state = update_car(state, action, dt)

print("Car state after 5 seconds:", state)

assert state["x"] > 0
assert state["speed"] > 0

print("Multi-step driving test passed!")

# 7. Test turning using the actual car physics
state = {
    "x": 0.0,
    "y": 0.0,
    "heading": 0.0,
    "speed": 10.0
}

observation = {
    "speed": state["speed"],
    "lateral_offset": 0.0,
    "heading_error": 0.0,
    "curvature_ahead": [0.05, 0.04, 0.03]
}

action = baseline_action(observation)
new_state = update_car(state, action, dt=0.1)

print("Corner steering action:", action)
print("Car state after turning:", new_state)

assert action["steering"] > 0
assert new_state["heading"] > 0

print("Turning integration test passed!")