# All configurations on all circuits

This table measures whole driving systems: lap completion, road-departure episodes and elapsed time. It is not an isolated ranking of machine-learning algorithms. Every configuration uses the same circuit geometry, physics, aligned starting heading, standing start, 0.1-second timestep and 600-second limit. No model was retrained.

Original RF and tuned RF are different saved fits. Full-control imitation uses simulated-teacher training and learned steering; its results measure the complete policy, not the isolated effect of replacing a pedal algorithm.

Comparisons among planner-equipped pedal candidates hold rule steering and speed assistance fixed. Comparing the same original RF with/without the speed planner isolates the added planner. Rules-only versus planner-equipped RF isolates the marginal contribution of learned pedal proposals.

These are expanded development/reproduction tests on already inspected circuits. The earlier frozen Spa protocol is preserved; this run is not claimed as another untouched final test.

| Driving configuration | Steering | Pedals | Speed planner | Monza | Silverstone | Spa |
|---|---|---|---|---|---|---|
| Rules only | Rule | Rule | Yes | 208.7 s; Clean lap | 258.3 s; Clean lap | 287.4 s; Clean lap |
| Original RF, no speed planner | Rule | Original RF | No | 91.8 s; 13 departures; invalid | Timeout (600 s); 5 departures; invalid | Timeout (600 s); 9 departures; invalid |
| Original RF + speed planner | Rule | Original RF | Yes | 205.4 s; Clean lap | 256.5 s; Clean lap | 284.6 s; Clean lap |
| Learned steering + pedals | Neural network | ExtraTrees | No | 209.1 s; Clean lap | 266.3 s; Clean lap | 294.3 s; Clean lap |
| Tuned RF + speed planner | Rule | Tuned RF | Yes | 205.5 s; Clean lap | 253.7 s; Clean lap | 282.0 s; Clean lap |
| Tuned SVR + speed planner | Rule | Tuned SVR | Yes | 204.1 s; Clean lap | 253.6 s; Clean lap | 281.9 s; Clean lap |
| Tuned neural pedals + planner | Rule | Tuned neural net | Yes | 199.1 s; Clean lap | 248.1 s; Clean lap | 276.8 s; Clean lap |
| 50/50 RF + SVR + planner | Rule | RF + SVR blend | Yes | 205.1 s; Clean lap | 254.7 s; Clean lap | 281.2 s; Clean lap |

Clean lap means completed with zero departures. An invalid completion is not a time improvement. Each cell is one predefined deterministic attempt, not a statistical reliability estimate. Per-step control traces and all model/geometry/code hashes are preserved.
