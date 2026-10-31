"""Run additive rubric analyses in order without changing historical outputs."""

from ml_analysis.eda import run_eda
from ml_analysis.ensembles import run_ensembles
from ml_analysis.final_evaluation import run_final_evaluation
from ml_analysis.hyperparameter_tuning import run_tuning
from ml_analysis.model_comparison import run_model_comparison


def main():
    print("EDA")
    run_eda()
    print("Fold-aware model comparison")
    run_model_comparison()
    print("Hyperparameter tuning")
    run_tuning()
    print("Ensemble comparison")
    run_ensembles()
    print("Final evaluation status and plots")
    run_final_evaluation()


if __name__ == "__main__":
    main()
