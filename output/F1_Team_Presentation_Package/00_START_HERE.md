# Start here

Open **01_Presentation/F1_Team_Presentation.pdf**. This is the latest 13-slide version with full-page visuals, no commentary strips and the three-system teaser.

## Folders

1. **01_Presentation** — the current PDF.
2. **02_GIFs** — six replay clips. Names start with the slide(s), stage, circuit and playback speed.
3. **03_Screenshots** — original-resolution stills used in the PDF.
4. **04_Track_Maps** — Monza, Silverstone and Spa-Francorchamps.
5. **05_Charts_and_Diagrams** — physics demonstration, actual recorded-speed chart, alignment plot and one optional historical benchmark graphic.
6. **06_Slide_Previews** — every slide as a named PNG, for quick reference.
7. **07_Benchmark_Data** — master comparison CSV and measured tuning/final-test data.

8. **08_Complete_Driving_Benchmark** — all 24 attempts, protocol, results and per-step traces.
9. **09_Canva_Videos** — six MP4 replays successfully accepted as Canva video assets; use these for Canva playback.

## Using the media

- The PDF shows still images. Insert the matching GIF into PowerPoint or Google Slides to show motion.
- Open **MEDIA_PREVIEW.html** for an animated gallery; keep it beside the folders after extracting the ZIP.
- **1x** means real-time replay; **10x** means nominal accelerated playback. Clips are independent recorded-control simulator replays, not synchronized races or real onboard video.
- Spa is Spa-Francorchamps in Belgium. We have a geometry map and benchmark data here, not Spa replay footage. Do not label the Silverstone clip as Spa.
- The physics chart is a controlled code demonstration, not a racing benchmark.
- The three development configurations are original RF without speed planning, original RF with speed planning, and learned full control. Tuned pedal predictors are supporting experiments. All eight complete driving configurations are now tested on all three circuits.

## Slide-to-file index

| Asset | Where to use it |
|---|---|
| [02_GIFs/Slide_01_Monza_LearnedDriver_FullLap_10x.gif](02_GIFs/Slide_01_Monza_LearnedDriver_FullLap_10x.gif) | Slide 1: optional opening demo. Nominal 10x playback, not real time. |
| [02_GIFs/Slides_04_07_Stage1_Monza_NoSpeedPlanner_1x.gif](02_GIFs/Slides_04_07_Stage1_Monza_NoSpeedPlanner_1x.gif) | Slides 4 and 7: naive learned pedals, first road departure. Rule steering and pedal cleanup remain. |
| [02_GIFs/Slides_04_08_Stage2_Monza_WithSpeedPlanner_1x.gif](02_GIFs/Slides_04_08_Stage2_Monza_WithSpeedPlanner_1x.gif) | Slides 4 and 8: assisted learned pedals near the same turn. Planner can override pedals. |
| [02_GIFs/Slides_04_09_Stage3_Monza_LearnedSteeringAndPedals_1x.gif](02_GIFs/Slides_04_09_Stage3_Monza_LearnedSteeringAndPedals_1x.gif) | Slides 4 and 9: learned full control, no runtime driving-rule assistance; actuator limits remain. |
| [02_GIFs/Slides_11_13_Stage3_Silverstone_TransferTurn_1x.gif](02_GIFs/Slides_11_13_Stage3_Silverstone_TransferTurn_1x.gif) | Slides 11 and 13: actual Silverstone learned-driver replay. Not Spa footage. |
| [02_GIFs/Optional_Stage3_Silverstone_FullLap_10x.gif](02_GIFs/Optional_Stage3_Silverstone_FullLap_10x.gif) | Optional full-lap transfer demo; nominal 10x playback. |
| [03_Screenshots/Slides_01_04_09_Stage3_Monza_LearnedFullControl.png](03_Screenshots/Slides_01_04_09_Stage3_Monza_LearnedFullControl.png) | Slides 1, 4, 9: full-control driver at Monza. |
| [03_Screenshots/Slide_03_Monza_Simulator_FollowCamera.png](03_Screenshots/Slide_03_Monza_Simulator_FollowCamera.png) | Slide 3: simulator with timing, progress and road-status display. |
| [03_Screenshots/Slides_04_07_Stage1_Monza_FirstDeparture.png](03_Screenshots/Slides_04_07_Stage1_Monza_FirstDeparture.png) | Slides 4 and 7: original pedal driver departure. |
| [03_Screenshots/Slides_04_08_Stage2_Monza_AssistedTurn.png](03_Screenshots/Slides_04_08_Stage2_Monza_AssistedTurn.png) | Slides 4 and 8: hybrid assisted turn. |
| [03_Screenshots/Slides_11_13_Stage3_Silverstone_LearnedFullControl.png](03_Screenshots/Slides_11_13_Stage3_Silverstone_LearnedFullControl.png) | Slides 11 and 13: actual Silverstone replay. |
| [04_Track_Maps/Slides_02_11_Monza_TrainingCircuit.png](04_Track_Maps/Slides_02_11_Monza_TrainingCircuit.png) | Slides 2 and 11: training circuit map, Italy. |
| [04_Track_Maps/Slide_11_Silverstone_TransferCircuit.png](04_Track_Maps/Slide_11_Silverstone_TransferCircuit.png) | Slide 11: first transfer circuit, United Kingdom. |
| [04_Track_Maps/Slide_11_SpaFrancorchamps_FinalTestCircuit_Belgium.png](04_Track_Maps/Slide_11_SpaFrancorchamps_FinalTestCircuit_Belgium.png) | Slide 11: Spa-Francorchamps, Belgium; fresh frozen-model test. |
| [05_Charts_and_Diagrams/Slide_02_Physics_AccelerationAndBraking_Demonstration.png](05_Charts_and_Diagrams/Slide_02_Physics_AccelerationAndBraking_Demonstration.png) | Slide 2: controlled run of shared physics, not a racing benchmark. |
| [05_Charts_and_Diagrams/Slide_05_Monza_Lap5_RecordedSpeed_590Moments.png](05_Charts_and_Diagrams/Slide_05_Monza_Lap5_RecordedSpeed_590Moments.png) | Slide 5: speed over one saved lap; explains what recorded moments represent. |
| [05_Charts_and_Diagrams/Slide_06_Monza_Lap5_TelemetryToTrack_Alignment.png](05_Charts_and_Diagrams/Slide_06_Monza_Lap5_TelemetryToTrack_Alignment.png) | Slide 6: actual training-lap alignment output. |
| [05_Charts_and_Diagrams/Optional_OriginalModel_PlannerOnly_vs_LearnedPedals.png](05_Charts_and_Diagrams/Optional_OriginalModel_PlannerOnly_vs_LearnedPedals.png) | Optional historical planner-only versus original hybrid comparison; not tuned-model Spa results. |
| [07_Benchmark_Data/Slide_10_TunedModels_SelectionErrors.csv](07_Benchmark_Data/Slide_10_TunedModels_SelectionErrors.csv) | Tuned candidate selection means; two training-lap folds. |
| [07_Benchmark_Data/Slide_10_Tuning_All18Trials_AndFoldErrors.json](07_Benchmark_Data/Slide_10_Tuning_All18Trials_AndFoldErrors.json) | All settings, per-fold errors and selected candidate. |
| [07_Benchmark_Data/Slides_11_12_Spa_FinalBenchmarks_AllSixDrivers.json](07_Benchmark_Data/Slides_11_12_Spa_FinalBenchmarks_AllSixDrivers.json) | All six final Spa results, frozen model hashes and protocol. |
