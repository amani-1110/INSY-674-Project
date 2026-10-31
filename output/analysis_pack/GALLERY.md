# Visual evidence gallery

[View the three-stage comparison](DRIVING_STAGES.md)

![Three driving stages](figures/12_driving_stages.png)

Ready-to-use captions accompany actual simulator screenshots, verified replay GIFs and measurement plots.
See [analysis and remaining work](ANALYSIS.md) and the [10-minute storyboard](PRESENTATION_STORYBOARD.md).

## Start here: failure versus planned driving

### First model-only departure, 1x

![First model-only departure, 1x](media/monza_first_departure_1x.gif)

First departure from the corrected 500 m model run without a speed planner. The full run had 13 departures; the clip is not the whole lap.

### Same location with planning, 1x

![Same location with planning, 1x](media/monza_same_turn_with_planner_1x.gif)

The saved hybrid passes this location more slowly and stays inside. It is a different lap/time window with the same playback rate.

### A fast lap can be invalid

![A fast lap can be invalid](figures/10_first_departure_comparison.png)

Actual reconstructed trajectories near the first departure: corrected learned pedals alone run outside the boundaries; the hybrid remains inside.

## Two circuits and full-lap motion

### Two distinct geometries

![Two distinct geometries](figures/01_track_maps.png)

Monza trains the pedal model; Silverstone is excluded from training. The planner can query both maps.

### Monza full lap, 10x

![Monza full lap, 10x](media/monza_full_lap_10x.gif)

Verified replay of the 205.4 s valid hybrid lap. The model trained on Monza; the planner supplies the speed limits.

### Silverstone full lap, 10x

![Silverstone full lap, 10x](media/silverstone_full_lap_10x.gif)

Verified replay of the 256.5 s valid hybrid lap on an unseen-to-model circuit. Geometry remains available to the planner.

## Close-up turns at real-time playback

### Monza geometry-selected turn 1, 1x

![Monza geometry-selected turn 1, 1x](media/monza_turn_1_1x.gif)

Verified hybrid replay. Turns are selected by separated curvature peaks and labelled by dataset station, not official corner numbering.

### Monza geometry-selected turn 2, 1x

![Monza geometry-selected turn 2, 1x](media/monza_turn_2_1x.gif)

Verified hybrid replay. Turns are selected by separated curvature peaks and labelled by dataset station, not official corner numbering.

### Silverstone geometry-selected turn 1, 1x

![Silverstone geometry-selected turn 1, 1x](media/silverstone_turn_1_1x.gif)

Verified hybrid replay. Turns are selected by separated curvature peaks and labelled by dataset station, not official corner numbering.

### Silverstone geometry-selected turn 2, 1x

![Silverstone geometry-selected turn 2, 1x](media/silverstone_turn_2_1x.gif)

Verified hybrid replay. Turns are selected by separated curvature peaks and labelled by dataset station, not official corner numbering.

## Report figures

### Measured AI contribution

![Measured AI contribution](figures/02_valid_lap_ablation.png)

Matched ablations give a 3.3 s gain on Monza and 1.8 s on Silverstone; both drivers complete clean laps. Single-condition simulator results.

### Farther is not automatically better

![Farther is not automatically better](figures/03_horizon_failures.png)

Historical raw-pedal models never produced a valid lap. The 1,000 m zero-departure run timed out, rather than completing safely.

### Qualifying-to-simulator distribution shift

![Qualifying-to-simulator distribution shift](figures/04_data_distributions.png)

The saved data begins at 73 km/h, whereas simulation starts at rest. Brake labels are imbalanced; accuracy requires a baseline.

### Feature relationships

![Feature relationships](figures/05_training_correlations.png)

Pearson correlations use training laps 5/11 only. Correlation is descriptive and is not evidence of causal driving rules.

### What the Random Forest uses

![What the Random Forest uses](figures/06_feature_importance.png)

Curvature 40 m ahead has the largest impurity importance in the saved 500 m model; correlated features can share importance.

### Offline error versus closed-loop behavior

![Offline error versus closed-loop behavior](figures/07_offline_prediction_errors.png)

Predictions on repeatedly inspected lap 14 illustrate errors. These are exploratory validation plots, not a pristine final-test estimate.

### Several model families

![Several model families](figures/08_model_family_cv.png)

Whole-lap two-fold CV with scaling inside folds. Bars show mean MAE and sample SD; RF and SVR are close with limited data.

### Where corner speed comes from

![Where corner speed comes from](figures/09_monza_speed_and_pedals.png)

The red target is calculated by the hand-written planner. Actual speed follows that envelope; this is not independently learned cornering.

### Same planner on a new circuit

![Same planner on a new circuit](figures/09_silverstone_speed_and_pedals.png)

Silverstone speed targets come from its geometry with unchanged parameters. The Monza-trained model contributes only pedal proposals.

## Native simulator screenshots

### Monza native simulator screenshot

![Monza native simulator screenshot](media/monza_follow_screenshot.png)

Native pygame renderer applied to an exact saved-run state; HUD time, speed and departure count correspond to that replay moment.

### Monza native simulator screenshot

![Monza native simulator screenshot](media/monza_turn_1_screenshot.png)

Native pygame renderer applied to an exact saved-run state; HUD time, speed and departure count correspond to that replay moment.

### Monza native simulator screenshot

![Monza native simulator screenshot](media/monza_turn_2_screenshot.png)

Native pygame renderer applied to an exact saved-run state; HUD time, speed and departure count correspond to that replay moment.

### Silverstone native simulator screenshot

![Silverstone native simulator screenshot](media/silverstone_follow_screenshot.png)

Native pygame renderer applied to an exact saved-run state; HUD time, speed and departure count correspond to that replay moment.

### Silverstone native simulator screenshot

![Silverstone native simulator screenshot](media/silverstone_turn_1_screenshot.png)

Native pygame renderer applied to an exact saved-run state; HUD time, speed and departure count correspond to that replay moment.

### Silverstone native simulator screenshot

![Silverstone native simulator screenshot](media/silverstone_turn_2_screenshot.png)

Native pygame renderer applied to an exact saved-run state; HUD time, speed and departure count correspond to that replay moment.

### First-departure screenshot

![First-departure screenshot](media/monza_first_departure_screenshot.png)

Native simulator view of the first recorded departure, with off-track status visible.

## Provenance

manifest.json records source control-trace hashes, replay agreement, clip selection and playback rates.
The source benchmarks and model-family scores are exploratory measurements, not proof of real-world F1 performance.
GIF contact sheets are visual QA aids. Full-lap clips use nominal 10x playback; corner/failure clips use 1x.

## Full-action learned-policy follow-up

The third stage now works without runtime steering/planning rules. It is expert imitation, not reinforcement learning.

### Monza learned policy

![Learned full-action turn](media/monza_full_action_turn_1x.gif)

[Full lap at 10x](media/monza_full_action_lap_10x.gif)

### Silverstone learned policy

![Learned full-action turn](media/silverstone_full_action_turn_1x.gif)

[Full lap at 10x](media/silverstone_full_action_lap_10x.gif)

