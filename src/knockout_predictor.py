import json
from pathlib import Path

import joblib
import pandas as pd


class KnockoutPredictor:
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = self.find_project_root()
        else:
            self.project_root = Path(project_root)

        self.data_processed = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models"

        self.model_path = self.models_dir / "current_best_model_with_elo.pkl"
        self.metadata_path = self.models_dir / "current_best_model_with_elo_metadata.json"

        self.model = joblib.load(self.model_path)

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]

        self.team_matches = pd.read_csv(self.data_processed / "team_matches.csv")
        self.final_elo_df = pd.read_csv(self.data_processed / "final_elo_ratings.csv")

        self.team_matches["date"] = pd.to_datetime(self.team_matches["date"])

        self.team_matches_played = self.team_matches.dropna(
            subset=["goals_for", "goals_against"]
        ).copy()

        self.team_matches_played = self.team_matches_played.sort_values(["team", "date"])

        self.final_elo_map = dict(
            zip(self.final_elo_df["team"], self.final_elo_df["final_elo"])
        )

        self.all_teams = sorted(
            set(self.team_matches_played["team"].dropna())
            | set(self.final_elo_df["team"].dropna())
        )

        self.team_lookup = {team.lower(): team for team in self.all_teams}

        self.form_features = [
            "last_5_points_per_match",
            "last_5_goals_for_per_match",
            "last_5_goals_against_per_match",
            "last_5_goal_difference_per_match",
            "last_5_win",
            "last_5_draw",
            "last_5_loss",
        ]

    def find_project_root(self):
        current = Path.cwd()

        for path in [current] + list(current.parents):
            if (path / "data").exists() and (path / "models").exists():
                return path

        raise FileNotFoundError(
            "Could not find project root. Run this from inside the project folder."
        )

    def resolve_team_name(self, team_name):
        if team_name in self.all_teams:
            return team_name

        team_name_lower = team_name.lower()

        if team_name_lower in self.team_lookup:
            return self.team_lookup[team_name_lower]

        possible_matches = [
            team for team in self.all_teams
            if team_name_lower in team.lower()
        ]

        if possible_matches:
            raise ValueError(
                f"Team '{team_name}' was not found exactly. "
                f"Did you mean one of these? {possible_matches[:10]}"
            )

        raise ValueError(f"Team '{team_name}' was not found in the dataset.")

    def get_current_team_form(self, team_name, n=5):
        team_name = self.resolve_team_name(team_name)

        team_history = (
            self.team_matches_played[self.team_matches_played["team"] == team_name]
            .sort_values("date")
            .tail(n)
        )

        matches_used = len(team_history)

        if matches_used == 0:
            return {
                "last_5_points_per_match": 0,
                "last_5_goals_for_per_match": 0,
                "last_5_goals_against_per_match": 0,
                "last_5_goal_difference_per_match": 0,
                "last_5_win": 0,
                "last_5_draw": 0,
                "last_5_loss": 0,
                "matches_used": 0,
            }

        return {
            "last_5_points_per_match": team_history["points"].sum() / matches_used,
            "last_5_goals_for_per_match": team_history["goals_for"].sum() / matches_used,
            "last_5_goals_against_per_match": team_history["goals_against"].sum() / matches_used,
            "last_5_goal_difference_per_match": team_history["goal_difference"].sum() / matches_used,
            "last_5_win": (team_history["result"] == "win").sum(),
            "last_5_draw": (team_history["result"] == "draw").sum(),
            "last_5_loss": (team_history["result"] == "loss").sum(),
            "matches_used": matches_used,
        }

    def build_match_features(self, team_a, team_b):
        team_a = self.resolve_team_name(team_a)
        team_b = self.resolve_team_name(team_b)

        team_a_form = self.get_current_team_form(team_a)
        team_b_form = self.get_current_team_form(team_b)

        row = {}

        for feature in self.form_features:
            row[f"{feature}_diff"] = team_a_form[feature] - team_b_form[feature]

        team_a_elo = self.final_elo_map.get(team_a, 1500)
        team_b_elo = self.final_elo_map.get(team_b, 1500)

        row["elo_diff"] = team_a_elo - team_b_elo

        for feature in self.features:
            if feature not in row:
                row[feature] = 0

        return pd.DataFrame([row])[self.features]

    def predict_match(self, team_a, team_b):
        team_a = self.resolve_team_name(team_a)
        team_b = self.resolve_team_name(team_b)

        X_match = self.build_match_features(team_a, team_b)

        predicted_label = self.model.predict(X_match)[0]
        probabilities = self.model.predict_proba(X_match)[0]

        probability_map = dict(zip(self.model.classes_, probabilities))

        team_a_win_prob = probability_map.get("win", 0)
        draw_prob = probability_map.get("draw", 0)
        team_b_win_prob = probability_map.get("loss", 0)

        return pd.DataFrame([
            {
                "team_a": team_a,
                "team_b": team_b,
                "predicted_outcome_from_team_a_perspective": predicted_label,
                "team_a_win_probability": team_a_win_prob,
                "draw_probability": draw_prob,
                "team_b_win_probability": team_b_win_prob,
            }
        ])

    def predict_knockout_match(self, team_a, team_b):
        prediction = self.predict_match(team_a, team_b)

        row = prediction.iloc[0]

        team_a_win_prob = row["team_a_win_probability"]
        draw_prob = row["draw_probability"]
        team_b_win_prob = row["team_b_win_probability"]

        team_a_adv_prob = team_a_win_prob + 0.5 * draw_prob
        team_b_adv_prob = team_b_win_prob + 0.5 * draw_prob

        predicted_advancing_team = (
            row["team_a"] if team_a_adv_prob >= team_b_adv_prob else row["team_b"]
        )

        prediction["team_a_advancement_probability"] = team_a_adv_prob
        prediction["team_b_advancement_probability"] = team_b_adv_prob
        prediction["predicted_advancing_team"] = predicted_advancing_team

        return prediction