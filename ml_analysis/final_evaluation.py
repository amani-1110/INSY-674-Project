"""Use the preserved simulator test record; do not relabel telemetry CV as a test."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml_analysis.preprocessing import ROOT

OUTPUT = ROOT / "output" / "rubric_analysis"
SIMULATOR_RECORD = ROOT / "output" / "completed_study" / "final_test.json"


def run_final_evaluation(output_dir: Path | None = None) -> dict:
    out = Path(output_dir or OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads(SIMULATOR_RECORD.read_text(encoding="utf-8"))
    systems = record["results"]
    table = pd.DataFrame(
        [
            {
                "system": name,
                "lap_time_s": metrics.get("lap_time"),
                "completed_lap": metrics.get("completed_lap"),
                "valid_lap": metrics.get("valid_lap"),
                "off_track_events": metrics.get("off_track_events"),
            }
            for name, metrics in systems.items()
        ]
    )
    table.to_csv(out / "simulator_frozen_evaluation.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#25845a" if valid else "#bb433f" for valid in table["valid_lap"]]
    ax.bar(table["system"], table["lap_time_s"], color=colors)
    ax.set(ylabel="Elapsed simulator time (s)", title="Frozen Spa closed-loop simulator evaluation")
    ax.tick_params(axis="x", rotation=35)
    for index, row in table.iterrows():
        ax.text(index, row["lap_time_s"], f"{row['off_track_events']} departures", ha="center", va="bottom", fontsize=8)
    fig.savefig(figure_dir / "frozen_simulator_results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    offline_plots = []
    throttle_path = out / "throttle_oof_predictions.csv"
    if throttle_path.exists():
        predictions = pd.read_csv(throttle_path)
        predictions = predictions[predictions["model"] == "Random Forest"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].scatter(predictions["throttle_true"], predictions["throttle_predicted"], s=8, alpha=0.4)
        axes[0].plot([0, 1], [0, 1], "--", color="black")
        axes[0].set(xlabel="Recorded throttle (0-1)", ylabel="Predicted throttle (0-1)",
                    title="Group-CV out-of-fold (exploratory)")
        residual = predictions["throttle_predicted"] - predictions["throttle_true"]
        axes[1].scatter(predictions["throttle_true"], residual, s=8, alpha=0.4)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set(xlabel="Recorded throttle (0-1)", ylabel="Prediction residual (0-1)",
                    title="Residuals; not a final test")
        fig.savefig(figure_dir / "throttle_oof_residuals.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        offline_plots.append("Group-CV out-of-fold throttle predictions/residuals; exploratory, not held-out final evaluation.")

    brake_path = out / "brake_oof_predictions.csv"
    if brake_path.exists():
        predictions = pd.read_csv(brake_path)
        predictions = predictions[predictions["model"] == "Random Forest"]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        errors = predictions[predictions["brake_true"] != predictions["brake_predicted"]]
        ax.scatter(predictions["lap_id"], predictions["brake_true"], s=8, alpha=0.2, label="All OOF samples")
        ax.scatter(errors["lap_id"], errors["brake_true"], s=14, color="#bd3e3a", label="Misclassified")
        ax.set(xlabel="Held-out lap ID", ylabel="Brake label (0/1)", title="Brake classification errors (group CV)")
        ax.legend()
        fig.savefig(figure_dir / "brake_oof_errors.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        offline_plots.append("Group-CV brake misclassification locations; exploratory, not held-out final evaluation.")

    segment_path = out / "error_analysis_by_segment.csv"
    if segment_path.exists():
        segments = pd.read_csv(segment_path)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for segment, frame in segments.groupby("segment"):
            ax.plot(frame["fold"], frame["throttle_mae"], marker="o", label=segment)
        ax.set(xlabel="CV fold", ylabel="Throttle MAE (0-1)", title="Error by track segment (group CV)")
        ax.legend()
        fig.savefig(figure_dir / "segment_error_analysis.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    result = {
        "telemetry_prediction_test": None,
        "telemetry_test_status": "Unavailable: laps 5 and 11 were used for selection; lap 14 has been repeatedly inspected and is not untouched.",
        "valid_independent_telemetry_test_available": False,
        "preserved_simulator_evaluation": {
            "source": "output/completed_study/final_test.json",
            "circuit": "Spa",
            "evaluation_kind": "frozen closed-loop driving evaluation, not real-telemetry prediction",
            "selection": record["protocol"]["selection"],
            "model_hashes_unchanged": record.get("models_unchanged", False),
            "results": systems,
        },
        "offline_figures": offline_plots,
        "generalization_limit": "The existing Spa test evaluates simulator driving, not prediction on an independent real telemetry session. It does not repair the lack of an untouched real-telemetry test group.",
    }
    (out / "final_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "final_evaluation_summary.md").write_text(
        "# Final evaluation status\n\n"
        "**No independent real-telemetry final test is available.** The two training laps (5 and 11) "
        "were used for model selection, and lap 14 has been repeatedly inspected. It is not reported "
        "as an untouched test. The supplied frozen Spa result is a closed-loop simulator evaluation, "
        "not real-telemetry prediction accuracy. The procedure can be rerun on a future untouched "
        "session by extending the source-data manifest and freezing the selected artifacts before evaluation.\n\n"
        "Group-CV predictions and residual plots are explicitly labeled exploratory. See "
        "`simulator_frozen_evaluation.csv` and `figures/frozen_simulator_results.png` for the historical "
        "simulator record; the original benchmark JSON and model hashes are preserved.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run_final_evaluation(), indent=2))
