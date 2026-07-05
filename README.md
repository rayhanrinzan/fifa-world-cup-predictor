# FIFA World Cup 2026 Knockout Match Prediction Project

This project is a beginner-friendly sports analytics and machine learning pipeline for predicting international football match outcomes, with the long-term goal of predicting and simulating the knockout stages of the 2026 FIFA World Cup.

The project uses historical international football match data to train models that predict match outcomes from Team A's perspective:

* Team A win
* Draw
* Team A loss

The longer-term goal is to use these match-level probabilities to simulate knockout-stage advancement and eventually run full tournament simulations.

## Project Goal

The main goal is to build a machine learning system that predicts international football match outcomes using historical results, recent team form, and team-strength features such as Elo ratings.

The long-term goal is to adapt the model for the 2026 FIFA World Cup by incorporating group-stage performance before predicting knockout-stage matches.

## Current Project Direction

Originally, this project was planned as a full World Cup prediction platform. The current focus has been narrowed to a more realistic machine learning workflow:

> Predict 2026 World Cup knockout-stage matches using historical international match data, recent team form, and Elo-based team strength features, eventually updated with 2026 group-stage performance.

This approach makes sense because football team strength changes over time, and group-stage results provide fresh information before the knockout rounds.

## Current Pipeline

```text
data/raw/results.csv
        ↓
data/processed/matches_with_results.csv
        ↓
data/processed/team_matches.csv
        ↓
data/processed/team_matches_with_form.csv
        ↓
data/processed/matchup_training_data.csv
        ↓
data/processed/matchup_training_data_with_elo.csv
        ↓
data/processed/matchup_training_data_with_clean_elo.csv
        ↓
models/current_best_model_with_elo.pkl
models/current_best_model_with_elo_metadata.json
```

In plain English:

```text
Raw historical match data
→ Add match result labels
→ Convert matches into team-perspective rows
→ Add recent-form features
→ Convert into matchup-level training data
→ Test logistic regression baselines
→ Test tree-based models
→ Add Elo rating features
→ Clean the Elo-enhanced dataset
→ Save the best current model
→ Build a reusable prediction function
```

## Project Structure

```text
fifa-world-cup-predictor/
│
├── data/
│   ├── raw/
│   │   └── results.csv
│   │
│   └── processed/
│       ├── matches_with_results.csv
│       ├── team_matches.csv
│       ├── team_matches_with_form.csv
│       ├── matchup_training_data.csv
│       ├── matchup_training_data_with_elo.csv
│       ├── matchup_training_data_with_clean_elo.csv
│       ├── final_elo_ratings.csv
│       └── baseline_predictions_with_draw_logic.csv   # experimental, not final
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_team_perspective_data.ipynb
│   ├── 03_recent_form_features.ipynb
│   ├── 04_matchup_training_data.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_tree_based_models.ipynb
│   ├── 07_add_elo_features.ipynb
│   └── 08_finalize_model_with_elo.ipynb
│
├── models/
│   ├── baseline_logistic_regression.pkl
│   ├── baseline_logistic_regression_balanced.pkl
│   ├── current_best_model_with_elo.pkl
│   └── current_best_model_with_elo_metadata.json
│
├── src/
│
├── README.md
└── requirements.txt
```

## Completed Work

### Notebook 01: Data Exploration

Loaded historical international match data and created a basic match result label.

Created:

```text
data/processed/matches_with_results.csv
```

Added the target column:

```text
home_win
draw
away_win
```

Main tasks completed:

* Loaded raw match data
* Inspected columns and data types
* Checked missing values
* Converted the date column
* Created match result labels
* Saved processed match-level dataset

### Notebook 02: Team-Perspective Dataset

Converted the original match-level data into team-perspective rows.

Original format:

```text
home_team | away_team | home_score | away_score
```

New format:

```text
team | opponent | goals_for | goals_against | result | points | goal_difference
```

Created:

```text
data/processed/team_matches.csv
```

Added columns:

```text
result
points
goal_difference
is_home
```

Each match now appears twice: once from each team's perspective.

Example:

```text
Brazil vs Argentina
Argentina vs Brazil
```

This structure makes it easier to calculate recent form and team-specific features.

### Notebook 03: Recent Form Features

Created rolling recent-form features for each team using only matches that happened before the current match.

Created:

```text
data/processed/team_matches_with_form.csv
```

Added features:

```text
win
draw
loss
last_5_points
last_5_goals_for
last_5_goals_against
last_5_goal_difference
last_5_win
last_5_draw
last_5_loss
last_5_points_per_match
last_5_goals_for_per_match
last_5_goals_against_per_match
last_5_goal_difference_per_match
previous_matches
form_matches_used
```

Important note:

The rolling features use `shift(1)` so the model only sees matches that happened before the current match. This prevents data leakage.

### Notebook 04: Matchup-Level Training Dataset

Converted team-perspective data into one row per match.

Created:

```text
data/processed/matchup_training_data.csv
```

Each row represents a matchup:

```text
Team A vs Team B
```

with Team A and Team B's recent form compared directly.

Example columns:

```text
date
team_a
team_b
team_a_goals
team_b_goals
tournament
neutral
target
```

Difference features:

```text
last_5_points_per_match_diff
last_5_goals_for_per_match_diff
last_5_goals_against_per_match_diff
last_5_goal_difference_per_match_diff
last_5_win_diff
last_5_draw_diff
last_5_loss_diff
```

Target column:

```text
target = win / draw / loss
```

The target is from Team A's perspective.

### Notebook 05: Baseline Logistic Regression Model

Trained the first baseline machine learning models.

Created:

```text
models/baseline_logistic_regression.pkl
models/baseline_logistic_regression_balanced.pkl
```

The baseline model used only recent-form difference features to predict:

```text
win / draw / loss
```

Features used:

```text
last_5_points_per_match_diff
last_5_goals_for_per_match_diff
last_5_goals_against_per_match_diff
last_5_goal_difference_per_match_diff
last_5_win_diff
last_5_draw_diff
last_5_loss_diff
```

The data was split chronologically:

```text
Training data: matches before 2018
Testing data: matches from 2018 onward
```

This is more realistic than a random split because sports models should train on past matches and predict future matches.

Evaluation included:

```text
accuracy
precision
recall
F1-score
macro F1
weighted F1
classification report
confusion matrix
prediction counts
actual vs predicted class distribution
```

## Logistic Regression Experiment Results

Several logistic-regression variations were tested:

```text
normal
balanced
mild_draw_boost
medium_draw_boost
strong_draw_boost
very_strong_draw_boost
```

| Model                  | Accuracy | Macro F1 | Win F1 | Loss F1 | Draw F1 | Predicted Wins | Predicted Losses | Predicted Draws |
| ---------------------- | -------: | -------: | -----: | ------: | ------: | -------------: | ---------------: | --------------: |
| normal                 |    0.513 |     0.36 |   0.65 |    0.43 |    0.00 |           1081 |              280 |               0 |
| balanced               |    0.484 |     0.44 |   0.60 |    0.49 |    0.22 |            650 |              475 |             236 |
| mild_draw_boost        |    0.514 |     0.37 |   0.65 |    0.45 |    0.00 |           1034 |              327 |               0 |
| medium_draw_boost      |    0.517 |     0.37 |   0.66 |    0.46 |    0.00 |            992 |              368 |               1 |
| strong_draw_boost      |    0.447 |     0.44 |   0.54 |    0.38 |    0.39 |            461 |              185 |             715 |
| very_strong_draw_boost |    0.354 |     0.31 |   0.30 |    0.20 |    0.43 |            161 |               70 |            1130 |

## Key Findings From Logistic Regression

The main finding was that class weighting changes the prediction distribution, but it does not fully solve the model problem.

### Normal Logistic Regression

The normal model had strong raw accuracy for a baseline, but it completely ignored draws.

```text
Accuracy: 0.513
Predicted draws: 0
Draw F1: 0.00
```

This model is not a good final match-outcome model because it is draw-blind.

### Balanced Logistic Regression

The balanced model had lower accuracy, but it was healthier as a three-class model.

```text
Accuracy: 0.484
Predicted draws: 236
Draw F1: 0.22
Macro F1: 0.44
```

This model treated wins, losses, and draws more seriously, but still was not strong enough to be considered final.

### Draw-Boosted Logistic Regression

Mild and medium draw boosting improved or matched raw accuracy, but still predicted almost no draws.

```text
mild_draw_boost predicted draws: 0
medium_draw_boost predicted draws: 1
```

Strong and very strong draw boosting predicted many more draws, but they overcorrected.

```text
strong_draw_boost predicted draws: 715
very_strong_draw_boost predicted draws: 1130
actual draws: 360
```

The project moved away from manual draw logic and focused on improving the feature set and model family instead.

## Notebook 06: Tree-Based Models

After logistic regression, tree-based models were tested to see whether stronger model families could better capture non-linear patterns.

Models tested:

```text
RandomForestClassifier
GradientBoostingClassifier
HistGradientBoostingClassifier
GradientBoostingClassifier with balanced sample weights
HistGradientBoostingClassifier with balanced sample weights
```

Evaluation metrics:

```text
accuracy
macro F1
weighted F1
win F1
draw F1
loss F1
prediction counts
classification report
confusion matrix
```

## Tree-Based Model Results

| Model                           | Accuracy | Macro F1 | Weighted F1 | Win F1 | Draw F1 | Loss F1 | Predicted Wins | Predicted Draws | Predicted Losses |
| ------------------------------- | -------: | -------: | ----------: | -----: | ------: | ------: | -------------: | --------------: | ---------------: |
| gradient_boosting_balanced      |    0.458 |    0.445 |       0.472 |  0.553 |   0.314 |   0.468 |           2917 |            2892 |             2346 |
| hist_gradient_boosting_balanced |    0.458 |    0.444 |       0.472 |  0.553 |   0.308 |   0.472 |           2963 |            2824 |             2368 |
| logistic_balanced               |    0.490 |    0.444 |       0.485 |  0.606 |   0.233 |   0.493 |           3717 |            1524 |             2914 |
| random_forest                   |    0.448 |    0.426 |       0.457 |  0.556 |   0.284 |   0.439 |           3254 |            2491 |             2410 |
| logistic_medium_draw_boost      |    0.468 |    0.396 |       0.443 |  0.635 |   0.343 |   0.209 |           4378 |            3330 |              447 |
| gradient_boosting               |    0.529 |    0.365 |       0.439 |  0.666 |   0.008 |   0.421 |           6604 |              12 |             1539 |
| logistic_normal                 |    0.526 |    0.361 |       0.436 |  0.663 |   0.000 |   0.420 |           6569 |               0 |             1586 |
| hist_gradient_boosting          |    0.524 |    0.360 |       0.434 |  0.663 |   0.007 |   0.409 |           6625 |              13 |             1517 |

## Key Findings From Tree-Based Models

Tree-based models confirmed the same tradeoff seen in logistic regression:

```text
The highest-accuracy models still tended to ignore draws.
Balanced models had lower raw accuracy but much healthier class balance.
```

The best tree-based model before Elo was:

```text
gradient_boosting_balanced
```

It had the strongest macro F1 among the non-Elo models:

```text
Accuracy: 0.458
Macro F1: 0.445
Draw F1: 0.314
```

However, the overall performance was still limited because the model only used recent-form difference features.

## Notebook 07: Elo Rating Features

The next improvement was adding Elo ratings as a team-strength feature.

Created:

```text
data/processed/matchup_training_data_with_elo.csv
```

Added features:

```text
team_a_elo_before
team_b_elo_before
elo_diff
```

The main Elo feature used for modeling was:

```text
elo_diff = team_a_elo_before - team_b_elo_before
```

This feature gives the model a stronger estimate of overall team quality, instead of relying only on recent form.

## Features With Elo

The Elo-enhanced models use:

```text
last_5_points_per_match_diff
last_5_goals_for_per_match_diff
last_5_goals_against_per_match_diff
last_5_goal_difference_per_match_diff
last_5_win_diff
last_5_draw_diff
last_5_loss_diff
elo_diff
```

## Elo Model Results

| Model                                    | Accuracy | Macro F1 | Weighted F1 | Win F1 | Draw F1 | Loss F1 | Predicted Wins | Predicted Draws | Predicted Losses |
| ---------------------------------------- | -------: | -------: | ----------: | -----: | ------: | ------: | -------------: | --------------: | ---------------: |
| gradient_boosting_balanced_with_elo      |    0.566 |    0.526 |       0.567 |  0.687 |   0.310 |   0.582 |           3760 |            1949 |             2450 |
| hist_gradient_boosting_balanced_with_elo |    0.569 |    0.525 |       0.567 |  0.690 |   0.300 |   0.586 |           3803 |            1812 |             2544 |
| logistic_balanced_with_elo               |    0.576 |    0.525 |       0.569 |  0.696 |   0.282 |   0.598 |           3972 |            1616 |             2571 |

## Key Findings From Elo

Adding Elo significantly improved the models.

Before Elo, the best macro F1 was about:

```text
0.445
```

After Elo, the best macro F1 was about:

```text
0.526
```

This was one of the most important improvements in the project so far.

The strongest confirmed model from Notebook 07 was:

```text
gradient_boosting_balanced_with_elo
```

Reason:

```text
It had the best macro F1 and strongest draw F1 among the Elo models.
```

Although `logistic_balanced_with_elo` had the highest raw accuracy, `gradient_boosting_balanced_with_elo` was preferred because the project needs balanced class performance and useful probabilities for simulation, not just accuracy.

## Notebook 08: Finalize Model With Clean Elo

Notebook 08 was created to clean up the Elo-enhanced workflow before moving on to prediction and simulation.

Main goals:

```text
Remove unplayed matches with missing scores
Rebuild Elo using only played matches
Prevent duplicate rows during Elo merging
Handle missing values cleanly
Retrain final candidate models
Select the best current model
Save the model and metadata
Create a reusable predict_match() function
```

Important cleanup decisions:

```text
Rows with missing match scores are removed before training.
Rows with missing Elo values are dropped.
Missing recent-form difference values are filled with 0.
```

The reasoning is:

```text
Missing scores should not be treated as draws.
Missing Elo usually means the Elo merge failed for that row.
Missing recent-form differences can reasonably be treated as neutral form difference.
```

Notebook 08 creates or updates:

```text
data/processed/matchup_training_data_with_clean_elo.csv
data/processed/final_elo_ratings.csv
models/current_best_model_with_elo.pkl
models/current_best_model_with_elo_metadata.json
```

It also introduces a first reusable prediction helper:

```python
predict_match("France", "Brazil")
```

The function returns:

```text
Team A win probability
Draw probability
Team B win probability
Simple Team A advancement probability
Simple Team B advancement probability
```

For knockout matches, the beginner advancement formula is:

```text
Team A advances = Team A win probability + 0.5 * draw probability
Team B advances = Team B win probability + 0.5 * draw probability
```

This can later be improved with extra-time, penalty, or team-strength assumptions.

## Current Best Model Status

The current best confirmed model before the final clean rerun is:

```text
gradient_boosting_balanced_with_elo
```

Why this model is preferred:

```text
It has the highest confirmed macro F1.
It has the strongest confirmed draw F1 among the Elo models.
It predicts a reasonable number of wins, draws, and losses.
It supports probability outputs through predict_proba().
```

The final saved model from Notebook 08 should be selected using:

```text
macro F1
draw F1
accuracy
prediction distribution
```

A model should not be chosen based only on raw accuracy because a model can achieve decent accuracy while ignoring draws.

## Current Status

```text
Completed:
- Data exploration
- Team-perspective dataset
- Recent-form feature engineering
- Matchup-level training dataset
- Baseline logistic regression model
- Balanced logistic regression model
- Custom class-weight experiments
- Draw logic experiment
- Model comparison visualizations
- Tree-based model testing
- Balanced tree-based model testing
- Elo rating feature engineering
- Elo-enhanced model comparison
- Clean Elo finalization notebook
- Missing-value cleanup strategy
- First reusable predict_match() helper

Current stage:
- Finalizing the best Elo-enhanced model
- Saving the model and metadata
- Preparing for knockout-stage simulation logic
```

## Current Limitations

The current model is much stronger than the original baseline, but it is still not a final high-confidence football predictor.

Current limitations include:

```text
No FIFA ranking features yet
No opponent-adjusted form yet
No last-10-match form features yet
No tournament importance weighting yet
No advanced home/neutral-site adjustment yet
No head-to-head history yet
No squad/player availability data
No injury data
No xG or advanced match statistics
No betting market comparison
No probability calibration yet
No full knockout bracket simulator yet
```

The model should currently be treated as a strong project milestone, not a production-quality predictor.

## Next Steps

### Immediate Next Steps

1. Finish running `08_finalize_model_with_elo.ipynb`
2. Confirm the final clean model results from `final_results_df`
3. Save the best model and metadata
4. Test `predict_match()` on several matchups
5. Start knockout-stage prediction logic

### Short-Term Modeling Improvements

1. Add probability calibration
2. Compare calibrated vs uncalibrated probabilities
3. Add last-10-match form features
4. Add home/neutral-site advantage
5. Add long-term team strength features
6. Add opponent-adjusted form

### Simulation Steps

1. Build a function that predicts a knockout match
2. Convert draw probability into advancement probability
3. Build a single-match advancement simulator
4. Build a bracket simulator
5. Run Monte Carlo simulations
6. Estimate advancement and championship probabilities

### Long-Term Goals

1. Add 2026 World Cup group-stage data
2. Predict knockout-stage matches
3. Simulate the full knockout bracket
4. Create a FastAPI backend
5. Create a React frontend
6. Deploy the project
7. Publish a polished GitHub repository

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repo-url>
cd fifa-world-cup-predictor
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the notebooks in order

```text
01_data_exploration.ipynb
02_team_perspective_data.ipynb
03_recent_form_features.ipynb
04_matchup_training_data.ipynb
05_baseline_model.ipynb
06_tree_based_models.ipynb
07_add_elo_features.ipynb
08_finalize_model_with_elo.ipynb
```

## Main Files

### Raw Data

```text
data/raw/results.csv
```

Historical international football match results.

### Processed Data

```text
data/processed/matches_with_results.csv
```

Match-level data with result labels.

```text
data/processed/team_matches.csv
```

Team-perspective version of the match data.

```text
data/processed/team_matches_with_form.csv
```

Team-perspective data with recent-form features.

```text
data/processed/matchup_training_data.csv
```

Matchup-level dataset with recent-form difference features.

```text
data/processed/matchup_training_data_with_elo.csv
```

Matchup-level dataset with Elo features added.

```text
data/processed/matchup_training_data_with_clean_elo.csv
```

Cleaned Elo-enhanced dataset used for final model selection.

```text
data/processed/final_elo_ratings.csv
```

Latest Elo rating for each team after processing historical matches.

### Models

```text
models/baseline_logistic_regression.pkl
```

Initial logistic regression model.

```text
models/baseline_logistic_regression_balanced.pkl
```

Balanced logistic regression model.

```text
models/current_best_model_with_elo.pkl
```

Current best saved model using recent-form and Elo features.

```text
models/current_best_model_with_elo_metadata.json
```

Metadata for the current best model, including features, target column, split date, and metrics.

## Notes

This project is still in progress. The current model is intended to build a complete and understandable sports analytics workflow rather than produce guaranteed predictions.

The most important accomplishment so far is that the project now has a full machine learning pipeline:

```text
data collection
→ data cleaning
→ team-perspective transformation
→ recent-form feature engineering
→ matchup-level dataset creation
→ baseline modeling
→ model refinement
→ tree-based model testing
→ Elo feature engineering
→ model finalization
→ prediction helper function
```

The biggest modeling lesson so far is that raw accuracy alone is not enough. Some models achieve decent accuracy by mostly ignoring draws, which is not acceptable for a full football match-outcome predictor. Macro F1, draw F1, and prediction distribution are important because the final tournament simulator needs meaningful probabilities for all three outcomes.

The next major milestone is to move from single-match prediction to knockout-stage advancement simulation.
