"""Validate and plot a bundled circuit or a custom track CSV."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from track import Track


def plot_track(track, output):
    ss = np.linspace(0, track.length, 2500)
    center = np.array([track.position(s) for s in ss])
    boundaries = np.array([track.boundaries(s) for s in ss])
    curvature = np.array([track.curvature(s) for s in ss])
    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    ax.plot(*boundaries[:, 0].T, color="#207c62", lw=0.8, label="Left boundary")
    ax.plot(*boundaries[:, 1].T, color="#b35c37", lw=0.8, label="Right boundary")
    segments = np.stack((center[:-1], center[1:]), axis=1)
    collection = LineCollection(segments, cmap="coolwarm", linewidth=1.6)
    collection.set_array(curvature[:-1])
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax, shrink=0.6, label="Signed curvature (1/m); + left")
    arrows_s = np.linspace(0, track.length, 16, endpoint=False)
    origins = np.array([track.position(s) for s in arrows_s])
    heading = np.array([track.heading(s) for s in arrows_s])
    ax.quiver(*origins.T, np.cos(heading), np.sin(heading), angles="xy",
              scale_units="xy", scale=1/65, width=0.003, color="#293d54")
    start = track.position(0)
    ax.scatter(*start, marker="*", s=110, color="#111111", zorder=6,
               label="s=0 (dataset start / simulation finish)")
    # Keep projections close to a well-separated centerline location.
    s0 = 0.17 * track.length
    c = np.array(track.position(s0))
    h = track.heading(s0)
    normal = np.array([-np.sin(h), np.cos(h)])
    for offset, color in [(3., "#116149"), (-3., "#b04b24")]:
        point = c + offset * normal
        s, recovered = track.project(*point)
        foot = np.array(track.position(s))
        ax.plot(*np.vstack((point, foot)).T, color=color, lw=2)
        ax.scatter(*point, s=25, color=color, zorder=7,
                   label=f"Projection: offset {recovered:+.2f} m")
    ax.set(title=f"{track.name.title()} | spline length {track.length:,.2f} m",
           xlabel="Local x (m)", ylabel="Local y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", nargs="?", default="monza")
    parser.add_argument("--csv", help="Custom TUM or x,y,left_width,right_width CSV")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", default="output/monza_track.png")
    args = parser.parse_args()
    track = (Track.from_csv(args.csv, reverse=args.reverse) if args.csv else
             Track(args.circuit, reverse=args.reverse))
    print(f"{track.name}: {track.length:.6f} m")
    print(plot_track(track, args.output))


if __name__ == "__main__":
    main()
