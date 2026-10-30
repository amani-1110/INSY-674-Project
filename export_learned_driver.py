"""Capture and verify the learned full-action benchmark replays."""
import json
import numpy as np

from track import Track
from f1_simulator.ui import renderer
from export_analysis_pack import (ROOT,OUT,MEDIA,MANIFEST,replay,native_screenshot,
                                 annotated_frame,save_gif,closest_index)


def main():
    original=json.loads((OUT/"manifest.json").read_text())
    for circuit,station in [("monza",934.8377658721791),("silverstone",1045.)]:
        track=Track(circuit)
        folder=ROOT/f"output/learned_driver/{circuit}"
        benchmark=json.loads((folder/"benchmark.json").read_text())
        samples=replay(track,folder/"controls.csv",benchmark["result"],circuit+"_full_action")
        apex=closest_index(samples,station,track.length)
        native_screenshot(track,samples[apex],circuit+"_full_action_screenshot.png","learned")
        context=renderer._renderer
        center=np.asarray(track.position(station))
        frames=[annotated_frame(track,context,samples[i],f"{circuit.title()} | learned full-action | no runtime driving helpers",
                  view="fixed",center=center,zoom=4. if circuit=="monza" else 5.)
                for i in range(max(0,apex-30),min(len(samples),apex+31))]
        save_gif(frames,circuit+"_full_action_turn_1x.gif",100,dict(circuit=circuit,source="verified full-action learned replay",
                   playback_rate="1x",centerline_station_m=station,model_sha256=benchmark["model_sha256"]))
        times=np.array([s["time"] for s in samples])
        indices=np.unique(np.r_[np.searchsorted(times,np.arange(0,times[-1],2.)),len(samples)-1])
        frames=[annotated_frame(track,context,samples[i],f"{circuit.title()} | learned full-action | no runtime driving helpers",playback="10x")
                for i in indices]
        save_gif(frames,circuit+"_full_action_lap_10x.gif",200,dict(circuit=circuit,source="verified full-action learned replay",
                   playback_rate="10x nominal",model_sha256=benchmark["model_sha256"]))
        renderer.close_renderer()
    original["runs"].update(MANIFEST["runs"])
    original["media"]=[r for r in original["media"] if "full_action" not in r["file"]]+MANIFEST["media"]
    original["full_action_training"]=json.loads((ROOT/"output/learned_driver/training.json").read_text())
    (OUT/"manifest.json").write_text(json.dumps(original,indent=2)+"\n")


if __name__ == "__main__":
    main()
