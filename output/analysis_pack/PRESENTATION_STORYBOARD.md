# Ten-minute presentation storyboard

Use one claim per slide and one short GIF at a time. Do not play entire laps
while explaining a dense table. Keep the numerical results visible after a clip.

| Time | Slide / spoken point | Visual |
|---|---|---|
| 0:00-0:45 | Research question: can learned pedals improve valid driving and transfer? | Two track maps, Figure 01 |
| 0:45-1:45 | Three laps are not three independent sessions; 73 km/h minimum versus rest start | EDA Figure 04 |
| 1:45-2:30 | Telemetry -> alignment -> features -> pedals -> steering/planner -> simulator | One process diagram; existing alignment overlay as backup |
| 2:30-3:30 | Compare model families with whole laps held out | CV Figure 08; state two folds and mean +/- SD |
| 3:30-4:30 | Accurate pedal predictions do not guarantee a clean lap | Figure 07 then the first-departure GIF |
| 4:30-5:30 | Longer lookahead alone failed; mutually conflicting pedals exposed a control mismatch | Horizon Figure 03; distinguish raw and corrected policies |
| 5:30-6:30 | A rule-based speed envelope makes the hybrid complete clean turns | Same-location planned GIF; speed/target Figure 09 |
| 6:30-7:30 | Does AI contribute beyond the planner? A matched ablation | Monza half of Figure 02 |
| 7:30-8:30 | Transfer to Silverstone without retraining or planner tuning | Silverstone turn GIF and the other half of Figure 02 |
| 8:30-9:30 | What is hand-written, what is learned, and what remains untested | Four concise limitations from ANALYSIS.md |
| 9:30-10:00 | Takeaway: planner enables clean laps; learned pedals add a small measured benefit | Valid-lap results table; no new claims |

Backup: full-lap 10x GIFs, feature importance/correlations, per-fold CSVs,
source hashes and timestep sensitivity results. Hyperparameter tuning and an
explicit ensemble comparison remain work to complete, not results to invent.
