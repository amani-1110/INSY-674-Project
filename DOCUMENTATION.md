# F1 Racing Simulator: the project story

[GitHub repository](https://github.com/amani-1110/INSY-674-Project) · [Visual document](output/analysis_pack/PROJECT_STORY.html) · [Presentation plan](output/analysis_pack/PRESENTATION_STORYBOARD.md) · [Detailed evidence](PROJECT_EVIDENCE.md)

## 01 · Can a learned driver finish a clean lap?

**Rubric:** Frame the Problem and Look at the Big Picture

- We built a 2D racing simulator for students and researchers to compare driving approaches.
- Success means finishing with zero road departures. Compare speed only after a lap is valid.

**Takeaway:** Clean driving comes first; speed comes second.

## 02 · One table: three stages and supporting comparisons

**Rubric:** Process and insights

- First, learned pedals without a speed planner left the road.
- Adding a planner produced clean laps. A separately trained driver then learned both steering and pedals.

**Takeaway:** The stages show how the system developed—not a comparison changing only assistance.

| Driving configuration | Steering | Pedals | Speed planner | Monza | Silverstone | Spa |
| --- | --- | --- | --- | --- | --- | --- |
| Rules only | Rule | Rule | Yes | 208.7 s; clean lap | 258.3 s; clean lap | 287.4 s; clean lap |
| Original RF, no speed planner | Rule | Original RF | No | 91.8 s; 13 departures; invalid | Timeout (600 s); 5 departures; invalid | Timeout (600 s); 9 departures; invalid |
| Original RF + speed planner | Rule | Original RF | Yes | 205.4 s; clean lap | 256.5 s; clean lap | 284.6 s; clean lap |
| Learned steering + pedals | Neural network | ExtraTrees | No | 209.1 s; clean lap | 266.3 s; clean lap | 294.3 s; clean lap |
| Tuned RF + planner | Rule | Tuned RF | Yes | 205.5 s; clean lap | 253.7 s; clean lap | 282.0 s; clean lap |
| Tuned SVR + planner | Rule | Tuned SVR | Yes | 204.1 s; clean lap | 253.6 s; clean lap | 281.9 s; clean lap |
| Tuned neural pedals + planner | Rule | Tuned neural net | Yes | 199.1 s; clean lap | 248.1 s; clean lap | 276.8 s; clean lap |
| 50/50 RF + SVR + planner | Rule | RF + SVR blend | Yes | 205.1 s; clean lap | 254.7 s; clean lap | 281.2 s; clean lap |

All 24 attempts use the same start, physics, 0.1-second timestep and 600-second limit. Clean means completed with zero departures. This compares complete driving configurations, not isolated algorithms. The original and tuned RF fits differ; full-control imitation also changes steering and training data. These are expanded development tests on already inspected circuits. See output/all_configurations/REPORT.md for the protocol and all outcomes.

## 03 · The examples did not cover every driving situation

**Rubric:** Explore the Data

- 1,774 samples from three Verstappen qualifying laps at Monza; one driver and one session.
- The slowest recorded speed was 73 km/h, but the simulator starts at rest. Only about 13% of samples contain braking.
- Saved cleaned tables have no missing values; original channels were interpolated. Real measured steering labels are unavailable.

**Takeaway:** Good qualifying examples do not automatically teach starts or recovery.

## 04 · Keep whole laps separate

**Rubric:** Prepare the Data

- Match driver positions to the circuit, standardize units and describe the road ahead.
- Use laps 5 and 11 for training. Train on one whole lap and check on the other, then swap.
- Keep scaling inside each training split. Lap 14 was inspected during development, so its scores are exploratory.

**Takeaway:** Nearby moments from the same lap should not be mixed between training and checking.

## 05 · Test settings—and test a combined model

**Rubric:** Measure and Compare Their Performance / Fine-Tune the System

- Screen five prediction approaches, then test 18 settings across Random Forest, support-vector and neural-network models.
- Compare the tuned models with a fixed 50/50 Random Forest + support-vector combination.
- Random Forest has the lowest selection error. The combination does not improve that error. Selection errors and fold variability are in the supporting evidence.

**Takeaway:** Choose using training laps, before looking at the fresh circuit.

## 06 · Check whether the learned pedals actually help

**Rubric:** Measure and Compare Their Performance

- Keep steering, planner, physics and starting conditions the same; remove only the learned pedal proposals.
- Monza: planner-only 208.7 s → assisted learned pedals 205.4 s.
- Silverstone: planner-only 258.3 s → assisted learned pedals 256.5 s. All four laps are valid.

**Takeaway:** The original learned pedals add a small measured benefit beyond the shared planner.

## 07 · Freeze the models, then test a new circuit

**Rubric:** Final evaluation

- Save the models and test protocol before retrieving Spa geometry or running the test.
- The preselected Random Forest hybrid finishes in 282.0 s; planner-only takes 287.4 s. Both have zero departures.
- The separate stage-3 learned driver finishes in 294.3 s, also with zero departures. All six predefined drivers complete cleanly.

**Takeaway:** The selected hybrid is 5.4 seconds faster here; the learned full-action driver also transfers cleanly.

## 08 · What we learned—and where it stops

**Rubric:** Assumptions and Limitations

- Matching recorded pedals and driving well are different tasks. Starts, recovery and the complete controller matter.
- Stage 3 learns a simulated teacher’s behaviour; it does not independently discover a racing strategy.
- Approximate physics, estimated track widths and reference-point road limits differ from real F1. The final test covers one deterministic starting condition.

**Takeaway:** We demonstrated clean simulated driving—not real-car performance or broad reliability.

## Evidence and reproduction

[Detailed methods, attributes, fold variability and limitations](PROJECT_EVIDENCE.md) · [Completed study](output/completed_study/REPORT.md) · [Saved protocol](output/completed_study/final_test_protocol.json)

Run `python build_concise_documentation.py` to rebuild the concise document, presentation outline and Overleaf ZIP from saved results. This does not retrain or rerun the final test. The experiment implementation is `complete_model_study.py`; preserve the original final-test outputs.
