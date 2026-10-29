"""Demo plots for physics.py -> physics_demo.png"""
import math
import matplotlib.pyplot as plt
from physics import update_car

DT = 0.01
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# Left: full throttle 0 -> top speed, then full brake
s, t, ts, vs, phase = {"x": 0, "y": 0, "heading": 0, "speed": 0}, 0.0, [], [], "go"
while True:
    act = {"throttle": 1.0} if t < 25 else {"brake": 1.0}
    s = update_car(s, act, DT); t += DT
    ts.append(t); vs.append(s["speed"] * 3.6)
    if t > 25 and s["speed"] == 0: break
ax = axs[0]
ax.plot(ts, vs, color="#1f77b4", lw=2)
ax.axvspan(25, ts[-1], color="#d62728", alpha=.12, label="Full brake")
ax.axvspan(0, 25, color="#2ca02c", alpha=.08, label="Full throttle")
for k in (100, 200, 300):
    i = next(i for i, v in enumerate(vs) if v >= k)
    ax.annotate(f"{k} km/h at {ts[i]:.1f}s", (ts[i], k), xytext=(ts[i]+2, k-25),
                arrowprops=dict(arrowstyle="->", lw=.8), fontsize=9)
ax.set(xlabel="Time (s)", ylabel="Speed (km/h)",
       title=f"Straight line: top speed {max(vs):.0f} km/h, stops in {ts[-1]-25:.1f}s")
ax.legend(loc="center right"); ax.grid(alpha=.25)

# Right: same steering (0.1 rad) at different speeds -> grip limit widens circle
ax = axs[1]
for v, c in ((10, "#2ca02c"), (30, "#ff7f0e"), (60, "#d62728")):
    s = {"x": 0, "y": 0, "heading": 0, "speed": v}; xs, ys = [0], [0]
    for _ in range(int(12 / DT)):
        s = update_car(s, {"steering": 0.1}, DT); s["speed"] = v
        xs.append(s["x"]); ys.append(s["y"])
    ax.plot(xs, ys, color=c, lw=2, label=f"{v*3.6:.0f} km/h")
ax.plot(0, 0, "ko"); ax.annotate("start (heading east)", (0, 0), xytext=(-60, 8), fontsize=9)
ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(title="Constant speed")
ax.set(xlabel="x (m)", ylabel="y (m)",
       title="Same steering (0.1 rad), 12 s: faster = wider turn (grip limit)")
plt.tight_layout(); plt.savefig("physics_demo.png", dpi=150)
print("saved physics_demo.png")
