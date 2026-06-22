# FIFA World Cup 2026 Knockout Match Prediction Project

This project is a beginner-friendly sports analytics and machine learning pipeline for predicting international football match outcomes, with the long-term goal of predicting and simulating the knockout stages of the 2026 FIFA World Cup.

The project uses historical international football match data to train a baseline model. Eventually, the project will incorporate 2026 World Cup group-stage performance to make updated knockout-stage predictions.

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
models/baseline_logistic_regression_balanced.pkl
```

In plain English:

```text
Raw historical match data
→ Add match result labels
→ Convert matches into team-perspective rows
→ Add recent-form features
→ Convert into matchup-level training data
→ Train a baseline logistic regression model
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
│       └── baseline_predictions_with_draw_logic.csv
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

Trained the first baseline machine learning model.

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
classification report
confusion matrix
dumb baseline comparison
predicted probabilities
prediction counts
```

## Draw Prediction Issue

The first baseline model predicted almost no draws.

This is a known issue in football prediction because draws are harder to identify and are often less common than wins or losses.

To improve this, two changes were added:

### 1. Balanced Logistic Regression

The model was updated to use:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
```

This tells the model to give more weight to underrepresented classes like draws.

### 2. Draw-Aware Prediction Logic

Instead of always choosing the highest probability class, a custom prediction function was added.

The function predicts a draw only when:

```text
draw probability is high enough
win and loss probabilities are balanced
neither team has a dominant win probability
```

Function name:

```python
predict_with_draw_logic
```

Decision rule:

```text
Predict draw if:
draw_prob >= draw_threshold
abs(win_prob - loss_prob) <= balance_threshold
max(win_prob, loss_prob) <= max_side_threshold
```

This avoids predicting a draw in cases where one team is clearly favored.

## Current Status

```text
Completed:
- Data exploration
- Team-perspective dataset
- Recent form features
- Matchup-level training data
- Baseline logistic regression model
- Initial draw prediction improvements

In Progress:
- Improving draw prediction
- Tuning draw thresholds
- Improving model features
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

1. Finish tuning draw prediction logic
2. Add absolute-difference features for close-match detection
3. Add last-10-match form features
4. Add home/neutral-site features
5. Evaluate with accuracy, macro F1, draw precision, and draw recall

### Medium-Term Steps

1. Add team strength ratings
2. Add opponent-adjusted recent form
3. Add tournament importance weighting
4. Compare logistic regression with other models
5. Save a clean prediction function

### Long-Term Goals

1. Add 2026 World Cup group-stage data
2. Predict knockout-stage matches
3. Build a knockout bracket simulator
4. Run Monte Carlo simulations
5. Create a FastAPI backend
6. Create a React frontend
7. Deploy the project

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
```

Future work will focus on improving feature quality and adapting the model for 2026 World Cup knockout-stage prediction.