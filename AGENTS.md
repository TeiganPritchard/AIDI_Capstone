# Running injury-risk experiments

Run the commands below from `Project/`. The project uses Python 3.13 or later and
`uv`. Use `uv run python` so the code uses the project dependencies.

## Choose the experiment workflow

| Task | Command |
| --- | --- |
| Cross-validate one registered model on the prepared data | `uv run python experiments.py --only xgboost --notes "Describe what changed"` |
| Cross-validate all registered models without a matching saved CV run | `uv run python experiments.py` |
| Repeat the registered CV runs | `uv run python experiments.py --rerun` |
| Score a selected registered model on the prepared holdout | `uv run python experiments.py --final xgboost` |
| Run the raw-data feature and split study for three classes | `uv run python feature_study.py` |
| Run the raw-data feature and split study for two classes | `uv run python feature_study.py --binary` |
| Run or resume 50 three-class XGBoost CV configurations | `uv run python feature_study.py --sweep` |
| Run or resume 50 binary XGBoost CV configurations | `uv run python feature_study.py --sweep --binary` |

The registered model names include `random_forest`, `catboost`, and `xgboost`.
Add a uniquely named entry to `EXPERIMENTS` in `Project/experiments.py` for a
hyperparameter variant that will use the same prepared features. For feature
experiments, use `Project/feature_study.py`, where `FEATURES` and `SWEEP_FEATURES`
define raw input columns and `SETTINGS` and `SWEEP_SETTINGS` define parameter
variants. Give changed configurations distinct names so the resume logic can
distinguish them from saved runs.

## Keep comparisons meaningful

- The three-class target is Healthy, Low Risk, and Injured. The binary target
  merges Healthy and Low Risk into Low Risk and maps Injured to High Risk.
- The prepared-data runner reads `X_train.csv`, `y_train.csv`, `X_test.csv`,
  and `y_test.csv` under `Project/src/Dataset/`. The current `DataPrep.ipynb`
  computes imputation medians before splitting. Prefer the raw-data study for
  feature comparisons because it fits the imputer inside each model fit.
- The raw-data study uses the existing stratified 70/30 session split with
  seed 42. The full study screens with three-fold cross-validation, compares
  finalists with five-fold cross-validation, and scores the selected model on
  the 30% holdout. A sweep uses three-fold cross-validation and leaves the
  holdout out of the evaluation.
- Compare runs on the same target, evaluation stage, and fold design. Use the
  combined score to reproduce the current leaderboard and report macro F1
  alongside it. The combined score equally weights macro ROC AUC and the
  normalized calibrated Brier score. Higher combined scores and lower Brier
  losses are better.
- The session split can place the same athlete in training and evaluation.
  For an athlete-disjoint check of a raw-study five-fold finalist, run
  `uv run python feature_study.py --athlete-check RUN_ID`. Add `--binary` for
  a binary finalist.

## Verify and report the results

The tracker appends results to `Project/experiments/results.sqlite3`. Each
successful run prints an ID. Use these commands to verify that the expected runs
were saved and inspect the exact settings, features when recorded, and metrics:

```powershell
uv run python experiments.py --history
uv run python experiments.py --best
uv run python experiments.py --run RUN_ID
```

When asked to execute a batch, wait for it to finish and check how many new run
IDs were saved. Report the target, evaluation stage, new run count, leading run
IDs and scores, and what changed. To show the results in the local dashboard,
run `uv run python experiment_view.py` and use **Refresh runs** after the batch.
