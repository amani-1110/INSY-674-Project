# Submission readiness — INSY 674

Audit against the provided final-project rubric, pages 4–5. This maps evidence to requirements; it does not predict a grade.

## Required hand-in

- **GitHub repository:** https://github.com/amani-1110/INSY-674-Project
- **Documentation:** [plain-language narrative](DOCUMENTATION.md) and [detailed evidence](PROJECT_EVIDENCE.md). The rubric allows README/documentation; it does not require a separate formal report.
- **Slides:** [main 13-slide PDF](output/pdf/presentation/F1_Project_Presentation.pdf), [editable Canva](https://www.canva.com/d/PxP3hiYAmJOFCja), and [team media package](output/F1_Team_Presentation_Package.zip). PDF pages are static; use the Canva video elements or packaged MP4/GIF files for motion.

## Source-code rubric

| Requirement | Evidence | Assessment |
|---|---|---|
| Frame the problem: purpose, intended users, learning task and use | DOCUMENTATION.md; PROJECT_EVIDENCE.md; README.md | Covered in documentation. Make purpose and intended audience explicit in the spoken opening. |
| Explore attributes, missingness, distributions, correlations and useful features | output/analysis_pack/ANALYSIS.md; attribute_profile.csv; labeled figures | Covered. Saved exports have no missing fields, but interpolation does not create independent measurements. |
| Prepare data and explain cleaning, feature engineering and scaling | PROJECT_EVIDENCE.md; screen_model_families.py; telemetry/; alignment/ | Implemented, with the calibration caveat below. |
| Compare several model families using N-fold CV, mean and standard deviation; errors, variables, shortlist | output/analysis_pack/model_family_cv_summary.csv; ANALYSIS.md; screen_model_families.py | Five initial families including the mean baseline; three candidates shortlisted: Random Forest, SVR and neural network. Only two independent training laps/folds. |
| Tune candidates with CV; try an ensemble | complete_model_study.py; output/completed_study/tuning.json; comparison.csv | Six settings per candidate, 18 trials; fixed RF/SVR blend tested. Blend did not improve selection error. |
| Evaluate the final system, compare baseline and explain limitations | output/completed_study/final_test_protocol.json; final_test.json; output/all_configurations/REPORT.md | Original frozen Spa evaluation is recorded. All eight configurations now tested on all three tracks. Later expanded tests are development evaluation, not a new untouched test set. |

## What the comparison measures

**Three driving systems** describe the story: learned pedals without speed assistance; learned pedals with assistance; learned steering and pedals. These are not three equivalent algorithms trained on identical labels.

**Three tuned candidate families** are Random Forest, SVR and a neural network for telemetry pedal prediction. The full-control policy is a separate simulator-trained imitation model. The benchmark includes eight configurations to expose the reference, assistance choices, original versus tuned models and the ensemble.

Offline prediction error measures how closely pedal predictions match recordings. Driving tests measure lap completion, departures and time. A low prediction error does not guarantee a fast or clean lap; comparisons must state which controls and helpers each system uses.

All 24 driving attempts are recorded: 21 clean completions, one invalid completion, two timeouts. One attempt per configuration/track is not a reliability estimate.

## Evaluation caveats to disclose

- **Limited independent data:** 1,774 saved moments come from three laps of one driver/session; training uses two laps. Similar moments within a lap are correlated. Collect more drivers, sessions and tracks before claiming broad generalization.
- **Calibration before CV:** alignment uses all of lap 5 before the two whole-lap folds. When lap 5 is held out, calibration has seen its positions. Scalers are fitted inside each fold and pedal targets are not used for alignment, but the full preparation process is not entirely isolated from validation data. Refit calibration using each training fold, or use independent calibration data, for a stricter estimate.
- **Repeated selection:** tuning reuses the two selection folds; lap 14 was inspected repeatedly. Do not call lap 14 an untouched final test or treat selection scores as unbiased final accuracy. A new independent telemetry session is still needed for a post-tuning prediction test.
- **Simulator limits:** simplified physics, estimated track widths and point-based departures differ from real racing. Validate physics and track boundaries before making real-car claims.
- **Different training sources and assistance:** the full-control model imitates a rule teacher on simulated states; pedal candidates use real recordings. Geometry remains visible at runtime. These are system comparisons, not an isolated ranking of learning algorithms.

## Presentation: final team actions

The material covers the end-to-end process and results. Before submission:

1. State the motivation and audience on the opening slide: **“We built a repeatable simulator so our team can compare driving approaches on the same roads and physics, then test whether learned controls can complete unfamiliar tracks.”** Explain why the recorded laps fit learning pedal behavior and why simulation supplies repeatable driving tests.
2. Put limitations visibly on the closing slide, with consequences and next steps: **one session → uncertain generalization → collect new sessions; simplified physics → limited realism → calibrate; one run per setup → uncertain robustness → vary starts and conditions.** Include the calibration/test caveats when discussing accuracy.
3. Rehearse the 13 main slides within **10 minutes**. Keep the five optional visual slides as an appendix. Timing has not been verified by a live rehearsal.
4. Check video playback in Canva presentation mode and submit the repository link plus slides. Saved video elements are present, but full presentation-mode playback still needs the team’s final check.

Do not describe the fastest assisted candidate as the best standalone AI. Lead with clean completion, then time, and explain the assistance shown in the master table.

## Repository verification

Track CSV files use LF line endings so their original source-provenance hashes survive Windows checkouts. Restoring these line endings changes no numerical geometry. Historical benchmark manifests retain the hashes of the exact bytes used for those runs, which may differ solely because of prior CRLF checkout conversion.

Full local suite after installing the declared FastF1/Pygame dependencies and restoring track line endings: `python -m pytest -q` — **112 passed**, with dependency deprecation warnings (2026-10-31).
