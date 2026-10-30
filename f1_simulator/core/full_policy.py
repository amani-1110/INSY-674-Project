"""Learned steering and signed-pedal policy: no runtime driving-rule fallback."""
import math
import joblib
import numpy as np

OFFSETS=tuple(float(v) for v in range(0,501,5))


def model_features(observation):
    curvature=np.asarray(observation["curvature_ahead"],dtype=float)
    if curvature.shape != (len(OFFSETS),):
        raise ValueError("Full policy needs its 101 configured curvature samples")
    speed=float(observation["speed"])
    offset=float(observation["lateral_offset"])
    heading=float(observation["heading_error"])
    steering=np.asarray([offset,heading,curvature[0]],dtype=float)
    pedals=np.r_[speed/45.,abs(offset)/5.,abs(heading)/.5,np.abs(curvature)*30.]
    if not np.isfinite(steering).all() or not np.isfinite(pedals).all():
        raise ValueError("Policy observations must be finite")
    return steering,pedals


class LearnedDriver:
    """Models choose all driving actions; only actuator-range clipping remains.

    Positive signed pedal means throttle, negative means brake. This action-space
    encoding prevents overlap without a launch rule or a brake threshold.
    Track geometry is observed, but no target-speed/planning calculation is used.
    """
    lookahead_offsets=OFFSETS

    def __init__(self,bundle,track):
        if tuple(bundle["offsets"]) != OFFSETS or bundle["schema"] != "full_action_v1":
            raise ValueError("Unsupported full-policy model schema")
        self.steering_model=bundle["steering_model"]
        self.pedal_model=bundle["pedal_model"]
        self.pedal_model.n_jobs=1
        if self.steering_model.n_features_in_ != 3 or self.pedal_model.n_features_in_ != 104:
            raise ValueError("Full-policy feature schema does not match its models")
        self.metadata=bundle.get("metadata",{})
        self.track=track
        self.sample_s=np.linspace(0,track.length,math.ceil(track.length/5.)+1)
        self.sample_k=np.asarray([track.curvature(s) for s in self.sample_s])

    def observe(self,state,track):
        if track is not self.track:
            raise ValueError("Policy observer belongs to another track")
        s,offset=track.project(state["x"],state["y"])
        heading=(state["heading"]-track.heading(s)+math.pi)%(2*math.pi)-math.pi
        curvature=np.interp(np.mod(s+np.asarray(OFFSETS),track.length),self.sample_s,self.sample_k)
        return dict(speed=state["speed"],lateral_offset=offset,heading_error=heading,
                    curvature_ahead=curvature.tolist())

    def __call__(self,observation):
        steer_features,pedal_features=model_features(observation)
        steering=float(self.steering_model.predict(steer_features[None,:])[0])*.4
        signed=float(self.pedal_model.predict(pedal_features[None,:])[0])
        if not math.isfinite(steering) or not math.isfinite(signed):
            raise ValueError("Learned actions must be finite")
        steering=float(np.clip(steering,-.4,.4))
        signed=float(np.clip(signed,-1.,1.))
        return dict(steering=steering,throttle=max(0.,signed),brake=max(0.,-signed))


def load_learned_driver(path,track):
    return LearnedDriver(joblib.load(path),track)
