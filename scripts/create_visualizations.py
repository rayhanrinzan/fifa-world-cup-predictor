from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_project_root():
    current = Path.cwd()

    for path in [current] + list(current.parents):
        if (path / "data").exists():
            return path

    raise FileNotFoundError("Could not find project root. Run this from inside the project folder.")


def save_top_championship_chart(results_df, assets_dir):
    top_champions = results_df.sort_values("champion", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_champions["team"], top_champions["champion"])
    plt.xlabel("Championship Probability (%)")
    plt.ylabel("Team")
    plt.title("Top 10 World Cup Championship Probabilities")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = assets_dir / "championship_probabilities.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)


def save_top_final_chart(results_df, assets_dir):
    top_finalists = results_df.sort_values("final", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_finalists["team"], top_finalists["final"])
    plt.xlabel("Final Appearance Probability (%)")
    plt.ylabel("Team")
    plt.title("Top 10 Final Appearance Probabilities")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = assets_dir / "final_probabilities.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)


def save_semifinal_chart(results_df, assets_dir):
    top_semifinalists = results_df.sort_values("semifinals", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_semifinalists["team"], top_semifinalists["semifinals"])
    plt.xlabel("Semifinal Appearance Probability (%)")
    plt.ylabel("Team")
    plt.title("Top 10 Semifinal Appearance Probabilities")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = assets_dir / "semifinal_probabilities.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)


def save_advancement_table(results_df, assets_dir):
    advancement_table = results_df[
        [
            "team",
            "round_of_16",
            "quarterfinals",
            "semifinals",
            "final",
            "champion",
        ]
    ].copy()

    advancement_table = advancement_table.sort_values("champion", ascending=False)

    output_path = assets_dir / "advancement_probabilities.csv"
    advancement_table.to_csv(output_path, index=False)

    print("Saved:", output_path)


def main():
    project_root = find_project_root()

    data_path = project_root / "data" / "processed" / "monte_carlo_knockout_simulation_results.csv"
    assets_dir = project_root / "assets"

    assets_dir.mkdir(exist_ok=True)

    results_df = pd.read_csv(data_path)

    probability_cols = [
        "round_of_16",
        "quarterfinals",
        "semifinals",
        "final",
        "champion",
    ]

    # Convert decimals to percentages if needed
    if results_df[probability_cols].max().max() <= 1:
        results_df[probability_cols] = results_df[probability_cols] * 100

    save_top_championship_chart(results_df, assets_dir)
    save_top_final_chart(results_df, assets_dir)
    save_semifinal_chart(results_df, assets_dir)
    save_advancement_table(results_df, assets_dir)

    print()
    print("Visualization creation complete.")


if __name__ == "__main__":
    main()