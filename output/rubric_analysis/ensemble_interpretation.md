# Ensemble interpretation

The equal-weight RF/SVR regressor and soft-voting RF/logistic brake classifier are compared against their component estimators using the same two whole-lap folds. Stacking was deliberately excluded: two independent laps cannot support a reliable out-of-fold meta-training layer. Read the measured mean and fold SD from `ensemble_cv_summary.csv`; an ensemble that does not beat its constituent models is reported as non-improving rather than selected by assumption.
