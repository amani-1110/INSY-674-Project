import math

MAX_SPEED_KMH = 372.6
MAX_SPEED_MPS = MAX_SPEED_KMH / 3.6


def update_car(state, action, dt):
    speed = max(0, min(MAX_SPEED_MPS, state['speed'] + (8 * max(0, min(1, action['throttle'])) - 15 * max(0, min(1, action['brake'])) - .003 * state['speed']**2) * dt))
    steering = max(-.55, min(.55, action['steering']))
    heading = state['heading'] + speed / 3.6 * math.tan(steering) * dt
    return dict(x=state['x'] + speed * math.cos(heading) * dt, y=state['y'] + speed * math.sin(heading) * dt, heading=(heading + math.pi) % (2 * math.pi) - math.pi, speed=speed)
