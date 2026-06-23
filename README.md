# FIFA World Cup 2026 Knockout Match Prediction Project

This project is a beginner-friendly sports analytics and machine learning pipeline for predicting international football match outcomes, with the long-term goal of predicting and simulating the knockout stages of the 2026 FIFA World Cup.

The project uses historical international football match data to train baseline models. Eventually, the project will incorporate 2026 World Cup group-stage performance to make updated knockout-stage predictions.

## Project Goal

The main goal is to build a machine learning system that can predict the outcome of a football match from Team A's perspective:

- Team A win
- Draw
- Team A loss

The longer-term goal is to use these match-level predictions to simulate the 2026 World Cup knockout bracket.

## Current Project Direction

Originally, the project was planned as a full World Cup prediction platform. The current focus has been narrowed to a more realistic version:

> Predict 2026 World Cup knockout-stage matches using historical international match data and recent team form, eventually updated with group-stage performance.

This approach makes sense because football team form can change quickly, and group-stage results provide fresh information before the knockout rounds.

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
notebooks/05_baseline_model.ipynb
        ↓
models/baseline_logistic_regression.pkl
models/baseline_logistic_regression_balanced.pkl
```

In plain English:

```text
Raw historical match data
→ Add match result labels
→ Convert matches into team-perspective rows
→ Add recent-form features
→ Convert into matchup-level training data
→ Train baseline logistic regression models
→ Evaluate accuracy, class-level performance, and prediction bias
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
│       └── baseline_predictions_with_draw_logic.csv   # experimental, not final
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_team_perspective_data.ipynb
│   ├── 03_recent_form_features.ipynb
│   ├── 04_matchup_training_data.ipynb
│   └── 05_baseline_model.ipynb
│
├── models/
│   ├── baseline_logistic_regression.pkl
│   └── baseline_logistic_regression_balanced.pkl
│
├── src/
│
├── README.md
└── requirements.txt
```

## Completed Work

### Day 2: Data Exploration

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

- Loaded raw match data
- Inspected columns and data types
- Checked missing values
- Converted date column
- Created match result labels
- Saved processed match-level dataset

### Day 3: Team-Perspective Dataset

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

### Day 4: Recent Form Features

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

### Day 5: Matchup-Level Training Dataset

Converted team-perspective data into one row per match.

Created:

```text
data/processed/matchup_training_data.csv
```

Each row now represents a matchup:

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

Team A form features:

```text
team_a_last_5_points_per_match
team_a_last_5_goals_for_per_match
team_a_last_5_goals_against_per_match
team_a_last_5_goal_difference_per_match
team_a_last_5_win
team_a_last_5_draw
team_a_last_5_loss
```

Team B form features:

```text
team_b_last_5_points_per_match
team_b_last_5_goals_for_per_match
team_b_last_5_goals_against_per_match
team_b_last_5_goal_difference_per_match
team_b_last_5_win
team_b_last_5_draw
team_b_last_5_loss
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

### Day 6: Baseline Logistic Regression Model

Trained the first baseline machine learning models.

Created:

```text
models/baseline_logistic_regression.pkl
models/baseline_logistic_regression_balanced.pkl
```

The model uses recent-form difference features to predict:

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

### Day 6 Continued: Model Refinement and Class Weight Experiments

After training the baseline logistic regression model, we investigated a major issue:

```text
The normal logistic regression model achieved the highest simple accuracy at first,
but it predicted almost no draws.
```

This matters because football has many draws, and a model that never predicts draws is not a useful full match-outcome model.

The test-set distribution was:

```text
Actual test distribution:
win     629
loss    372
draw    360
```

Several logistic-regression variations were tested:

```text
normal
balanced
mild_draw_boost
medium_draw_boost
strong_draw_boost
very_strong_draw_boost
```

The purpose was to understand how class weighting changes model behavior.

## Logistic Regression Experiment Results

| Model | Accuracy | Macro F1 | Win F1 | Loss F1 | Draw F1 | Predicted Wins | Predicted Losses | Predicted Draws |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.513 | 0.36 | 0.65 | 0.43 | 0.00 | 1081 | 280 | 0 |
| balanced | 0.484 | 0.44 | 0.60 | 0.49 | 0.22 | 650 | 475 | 236 |
| mild_draw_boost | 0.514 | 0.37 | 0.65 | 0.45 | 0.00 | 1034 | 327 | 0 |
| medium_draw_boost | 0.517 | 0.37 | 0.66 | 0.46 | 0.00 | 992 | 368 | 1 |
| strong_draw_boost | 0.447 | 0.44 | 0.54 | 0.38 | 0.39 | 461 | 185 | 715 |
| very_strong_draw_boost | 0.354 | 0.31 | 0.30 | 0.20 | 0.43 | 161 | 70 | 1130 |

## Key Findings From Day 6

The main finding was that class weighting changes the prediction distribution, but it does not fully solve the model problem.

### Normal Logistic Regression

The normal model had the best simple accuracy among the original models, but it completely ignored draws.

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

### Mild and Medium Draw Boost

The mild and medium custom class-weight models improved or matched accuracy, but still predicted almost no draws.

```text
mild_draw_boost predicted draws: 0
medium_draw_boost predicted draws: 1
```

These models are useful as accuracy baselines, but they do not solve the draw problem.

### Strong and Very Strong Draw Boost

The stronger draw-weighted models predicted many more draws, but they overcorrected.

```text
strong_draw_boost predicted draws: 715
very_strong_draw_boost predicted draws: 1130
actual draws: 360
```

These models had poor accuracy because they became too biased toward draws.

## Draw Logic Decision

A custom draw-aware prediction rule was explored. The rule attempted to predict draws only when:

```text
draw probability was high enough
win and loss probabilities were balanced
neither side was too dominant
```

However, this was removed as the main approach for now.

Reason:

```text
The draw logic changed the final prediction distribution, but it did not reliably improve accuracy or solve the underlying probability problem.
```

The project will now focus on improving the model itself rather than manually forcing draw predictions.

## Model Visualization Work

A visualization section was added/planned for the baseline model notebook to compare all logistic-regression experiments.

Visualizations include:

```text
accuracy by model
macro F1 by model
accuracy vs macro F1
F1-score by class
precision by class
recall by class
predicted outcome counts by model
actual vs predicted outcome counts
actual vs predicted outcome percentages
```

These visuals help show the main tradeoff:

```text
Higher-accuracy logistic models tend to ignore draws.
Draw-aware models become more balanced, but usually lose accuracy.
Very strong draw weighting overcorrects and predicts too many draws.
```

## Current Status

```text
Completed:
- Data exploration
- Team-perspective dataset
- Recent form features
- Matchup-level training data
- Baseline logistic regression model
- Balanced logistic regression model
- Custom class-weight experiments
- Draw logic experiment
- Model comparison table
- Model comparison visualizations

In Progress:
- Improving the model itself
- Comparing stronger model families
- Choosing the best current model for match prediction
```

## Current Limitations

The current model is only a baseline. It does not yet include many important football prediction features.

Missing features include:

```text
team historical strength
FIFA rankings
Elo ratings
opponent-adjusted form
home/neutral advantage
tournament importance
last_10 form features
absolute difference features
head-to-head history
World Cup group-stage form
player/squad data
injury data
xG or advanced match statistics
```

Because of this, the current model should be treated as an experimental baseline, not a final predictor.

## Next Steps

### Immediate Next Steps

1. Stop tuning manual draw logic for now
2. Try tree-based models:
   - Random Forest
   - Gradient Boosting
   - HistGradientBoosting
3. Compare tree-based models against logistic regression
4. Evaluate models using:
   - accuracy
   - macro F1
   - win/loss/draw F1
   - prediction distribution
   - probability quality
5. Pick the best current model

### Medium-Term Steps

1. Add team strength ratings
2. Add Elo or FIFA ranking difference
3. Add opponent-adjusted recent form
4. Add tournament importance weighting
5. Add home/neutral-site features
6. Save a clean prediction function

### Long-Term Goals

1. Add 2026 World Cup group-stage data
2. Predict knockout-stage matches
3. Convert draw probability into advancement probability for knockouts
4. Build a knockout bracket simulator
5. Run Monte Carlo simulations
6. Create a FastAPI backend
7. Create a React frontend
8. Deploy the project

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

Then run the notebooks in order:

```text
01_data_exploration.ipynb
02_team_perspective_data.ipynb
03_recent_form_features.ipynb
04_matchup_training_data.ipynb
05_baseline_model.ipynb
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

Final matchup-level dataset used for model training.

### Models

```text
models/baseline_logistic_regression.pkl
```

Initial logistic regression model.

```text
models/baseline_logistic_regression_balanced.pkl
```

Updated logistic regression model with class balancing.

## Notes

This project is still in progress. The current model is meant to establish a complete baseline pipeline, not to produce final high-confidence predictions.

The most important accomplishment so far is that the project now has a working machine learning workflow:

```text
data collection
→ data cleaning
→ feature engineering
→ training dataset creation
→ model training
→ evaluation
→ model comparison
```

Day 6 showed that manual class weighting alone is not enough to create a strong match predictor. Some logistic-regression versions achieve higher accuracy by mostly ignoring draws, while more draw-aware versions lose accuracy or overpredict draws. The next improvement should come from stronger model families and better features, not more manual threshold tuning.

Future work will focus on improving feature quality, testing tree-based models, and adapting the model for 2026 World Cup knockout-stage prediction.
