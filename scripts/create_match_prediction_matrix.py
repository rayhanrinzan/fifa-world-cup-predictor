from pathlib import Path
import sys

import pandas as pd

# Add project root to Python path so scripts can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.knockout_predictor import KnockoutPredictor

ACTUAL_ROUND_OF_32 = [
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),

    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium", "Senegal"),

    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),

    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]


def main():
    project_root = PROJECT_ROOT
    output_path = project_root / "data" / "processed" / "match_prediction_matrix.csv"

    teams = sorted(set(team for matchup in ACTUAL_ROUND_OF_32 for team in matchup))

    predictor = KnockoutPredictor(project_root=project_root)

    predictions = []

    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue

            prediction = predictor.predict_knockout_match(team_a, team_b)
            predictions.append(prediction)

    prediction_matrix = pd.concat(predictions, ignore_index=True)
    prediction_matrix.to_csv(output_path, index=False)

    print("Saved match prediction matrix to:", output_path)
    print("Rows:", len(prediction_matrix))


if __name__ == "__main__":
    main()