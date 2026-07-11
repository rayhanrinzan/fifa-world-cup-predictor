from pathlib import Path

from src.knockout_predictor import KnockoutPredictor
from src.bracket_simulator import BracketSimulator


actual_round_of_32 = [
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
    project_root = Path.cwd()

    predictor = KnockoutPredictor(project_root=project_root)
    simulator = BracketSimulator(predictor)

    champion, deterministic_results = simulator.simulate_deterministic_bracket(
        actual_round_of_32
    )

    print("=" * 60)
    print("DETERMINISTIC BRACKET")
    print("=" * 60)
    print("Predicted Champion:", champion)
    print()

    output_dir = project_root / "data" / "processed"

    deterministic_path = output_dir / "actual_deterministic_bracket_results.csv"
    deterministic_results.to_csv(deterministic_path, index=False)

    print("Saved deterministic bracket results to:", deterministic_path)

    monte_carlo_results = simulator.run_monte_carlo_simulation(
        actual_round_of_32,
        n_simulations=10000
    )

    monte_carlo_path = output_dir / "monte_carlo_knockout_simulation_results.csv"
    monte_carlo_results.to_csv(monte_carlo_path, index=False)

    print("Saved Monte Carlo results to:", monte_carlo_path)
    print()

    print("=" * 60)
    print("TOP CHAMPIONSHIP PROBABILITIES")
    print("=" * 60)

    print(
        monte_carlo_results
        .sort_values("champion", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()