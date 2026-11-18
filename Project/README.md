Open terminal
cd ./Project
uv run jupyter lab

## Experiment tracking

From `Project`, run:

```powershell
uv run python experiment_view.py                      # Open http://127.0.0.1:8765
uv run python experiments.py                          # Run unrecorded five-fold CV experiments
uv run python experiments.py --only xgboost --notes "Baseline"
uv run python experiments.py --final catboost         # Evaluate one model on the test split
uv run python experiments.py --show                   # Latest run per experiment and split
uv run python experiments.py --history                # All saved runs
uv run python experiments.py --best                   # Best historical run per split by combined score
uv run python experiments.py --run 1                  # Full metrics and settings for run 1
```

The local dashboard reads the SQLite tracker directly. Search or sort the run history,
open a run by ID, and choose a baseline to compare recorded feature and settings changes.
Use **Refresh runs** after finishing another experiment.
The **Feature analysis** section recalculates the feature-set leaderboard and
selects an importance ranking from the saved runs for each target.

Open **Predict** in the dashboard to fit a local model from a saved study run.
Choose a binary or three-class run, load it, and enter one session's feature values.
The page shows calibrated probabilities and the class selected by the fitted
decision thresholds. Blank fields use training medians. The predictor uses the
recorded feature list and settings with the current raw data and keeps fitted
models in memory while the server runs.
New runs include each model's built-in feature importance, averaged across folds for CV.
To add importance for a saved study test configuration, use
`uv run python feature_study.py --retest RUN_ID` (add `--binary` for binary runs).
This records a new run.

Each completed model notebook evaluation also saves a test run. Runs are stored locally in
`experiments/results.sqlite3`, with the model settings, calibration and thresholds, per-class
metrics, confusion matrix, and Git commit. Repeated runs get new IDs. Use `--rerun` after changing
the prepared data or evaluation code. To add a named model variant, add an entry to
`EXPERIMENTS` in `experiments.py`.

The CV runner reads the prepared training CSVs. The current data preparation computes
imputation medians before splitting the raw data, so those medians include test rows.

### Multiclass feature and split study

Run `uv run python feature_study.py` to compare seven feature sets across all three
models. The study uses raw data and fits its median imputer inside each model fit.
It screens candidates with three-fold CV, compares finalists and calibration sizes
(10%, 20%, 30%, and 40% of the development split) with five-fold CV, and evaluates
the selected candidate on the existing 30% holdout.
`experiments.py --best` shows the leading run for each study stage, and
`experiments.py --run ID` shows the exact feature names and settings.
Pass `--binary` to study two classes. The binary target merges Healthy and Low Risk
into Low Risk and maps Injured to High Risk. Binary runs have separate leaderboard rows.
Pass `--athlete-check RUN_ID` to re-evaluate a five-fold finalist on validation folds
that hold out entire athletes. Add `--binary` when the run ID is from a binary study.

Run `uv run python feature_study.py --sweep` for 50 three-class XGBoost variations or
add `--binary` for 50 binary variations. Each sweep compares 10 feature sets and
five settings on the same three CV folds within its target. It resumes by skipping
saved configurations and leaves the holdout out of the sweep.
