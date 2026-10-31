# Presentation plan · 9 minutes 30 seconds

Use the same eight sections as the visual document. One takeaway per slide; leave 30 seconds as a buffer. Use one master table throughout: highlight the relevant rows or circuit column instead of introducing more tables.

| Time | Slide headline | Visual | Speaker cue |
|---|---|---|---|
| 0:00–0:45 | Can a learned driver finish a clean lap? | Track map: figures/01_track_maps.png | Open with the question and explain what counts as success. |
| 0:45–2:30 | One table: three stages and supporting comparisons | Master comparison table: highlight the three stage rows | Explain each card. Stage 1 still has steering and pedal rules; stage 3 uses different examples and models. |
| 2:30–3:30 | The examples did not cover every driving situation | figures/04_data_distributions.png | Explain the data gap before introducing any model terminology. |
| 3:30–4:30 | Keep whole laps separate | Telemetry → alignment → road features → learned controls → simulator | Introduce grouped cross-validation only after explaining the whole-lap split. |
| 4:30–5:45 | Test settings—and test a combined model | Master table: highlight the four tuned candidates | MAE means average prediction error. These two-fold selection scores are not final-test accuracy. |
| 5:45–6:45 | Check whether the learned pedals actually help | Two paired results; figures/02_valid_lap_ablation.png | These are historical original-model comparisons, separate from the later tuned-model Spa test. |
| 6:45–8:15 | Freeze the models, then test a new circuit | Master table: highlight the Spa column | All results are reported. The neural hybrid is faster despite worse prediction errors; do not reselect the model from final-test results. |
| 8:15–9:30 | What we learned—and where it stops | One closing takeaway with three short limitations | Close with the result. Keep the remaining 30 seconds as a timing buffer. |

## Backup slides only

- Full attribute profile and labelled data distributions; correlations and feature importance stay in the evidence appendix. No heatmap is needed in the main presentation.
- Model settings, all 18 trials, mean ± fold standard deviation, convergence checks.
- Original failed preview-distance experiments and pedal-conflict diagnosis.
- All six Spa results, frozen hashes and protocol; timestep sensitivity.

Use the visual document’s “Build slide” controls for speaker cues. Keep code paths and settings in backup material, not the main spoken story.

Technical definitions: supervised learning = learning from labelled examples; grouped cross-validation = checking on whole laps excluded from fitting; MAE = average size of a prediction error; expert imitation = copying a teacher’s decisions.
