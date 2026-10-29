# Three additional Monza qualifying laps

Real Max Verstappen telemetry from 2025 Monza qualifying. This folder contains exactly three distinct laps: 5, 11 and 14. Laps 8 and 17 are excluded.

Each lap has a six-column CSV, metadata JSON, speed/control plot, and XY trajectory plot. Time is lap-relative seconds; X/Y are metres in the telemetry source frame; speed is m/s; throttle is 0-1 and brake is binary 0/1.

| Lap | Lap time seconds | Rows | Largest native car gap seconds | Largest native position gap seconds |
| --- | ---: | ---: | ---: | ---: |
| 14 | 78.923 | 593 | 1.119 | 1.260 |
| 11 | 79.422 | 591 | 1.039 | 0.899 |
| 5 | 79.455 | 590 | 1.080 | 1.240 |

Quality policy: these exports use a 1.3-second maximum native gap, compared with the original starter policy of 1.0 second. No native channel values are missing. FastF1 merges and interpolates the car and position streams, including lap boundaries. These records are exploratory and need the sampling-gap information retained during alignment and training; they are not precise racing-line ground truth.

Validation: all three saved CSVs were reopened and checked for the exact six fields, finite numbers, increasing timestamps, nonnegative speed, throttle within 0-1, binary brake, distinct lap identities and full lap time coverage. Existing source code and exports were left unchanged.

Copy this entire folder into your project data folder if desired. Keep one CSV per lap and retain its matching metadata and plots.
